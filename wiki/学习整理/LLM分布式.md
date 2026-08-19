# LLM 分布式：并行策略、通信原语与推理侧对策

单卡装不下、或者装得下但算不快时，就要把模型和数据摊到多张卡上。这篇从外往里讲四层：**四种并行策略**（DP/PP/TP/EP）与它们**共用的通信原语** → **通信代价怎么由硬件决定** → **一张卡内部的 SM/Warp 调度** → **量化与投机采样这两条不靠加卡的提速路**。

贯穿全篇的一条线：**推理场景和训练场景对这套东西的取舍完全不同**，所以每一节都会回到「在推理里它还成不成立」。

> **和 [[LLM推理的GPU硬件基础]] 的分工**：那篇讲**硬件为什么是这样**（算力/带宽失衡、显存构成、互联带宽表）；本篇讲**软件怎么在这个约束下拆模型**。两篇的接头处是同一句话——**通信代价由互联带宽决定**，所以下面每讲一种并行，都会回到「它压在哪条链路上」。

## 一、四种并行策略

### 1.1 DP（数据并行）

**每个 GPU 都拥有完整的模型副本**，把训练数据划分成多个小批次（mini-batch），每个批次分给不同 GPU 处理。

![[dist-dp-parameter-server.png]]

图里画的是**参数服务器（Parameter Server）**式的传统 DP：GPU0–2 各持一份完整模型参数 W，各算各的 mini-batch 得到梯度 G0/G1/G2，**统一推给 GPU3 做 AllReduce 聚合**，再把更新后的梯度广播回各 Worker。

**它的病也画在图里了**：GPU3 既要收三份梯度又要发三份结果，**通信负载压在一个点上**，卡数越多这个 Server 越是瓶颈。

**ZeRO**（Zero Redundancy Optimizer）是另一条解法：注意到每张卡存的**优化器状态、梯度、参数**是完全重复的，于是把这三样**切开分存**，用时再临时聚合。分三级——ZeRO-1 切优化器状态、ZeRO-2 再切梯度、ZeRO-3 连参数一起切。**它省的是显存，不是通信**（ZeRO-3 反而通信更多）。

> ⚠️ 存疑：上面「ZeRO 分三级、各切什么」是我按通行定义补的，你原笔记只写了「对优化器状态、梯度和参数进行切分」这一句。如果你看的资料对 stage 划分有不同说法，告诉我改。

### 1.2 DDP（分布式数据并行）与 Ring-AllReduce

**传统 DP 一般用于单机多卡；DDP 既能多机也能单机。** 它依赖 **Ring-AllReduce**——由百度最先提出，**有效解决数据并行中通信负载不均（Server 存在瓶颈）的问题**。

关键在于**去掉中心节点**：所有 GPU 首尾相连成环，每张卡**只和左右邻居通信**，没有谁是 Server。

**第 0 步：初始状态**，4 个进程各持一份完整数据（这里切成 4 块）。

![[dist-ring-allreduce-1-init.png]]

**第 1 阶段：Scatter-Reduce**，走 N−1 步。每步每个进程把**一块**数据发给右邻居，同时收左邻居的一块**累加**到自己对应位置上。

![[dist-ring-allreduce-2-scatter-reduce.png]]

![[dist-ring-allreduce-3-accumulate.png]]

N−1 步之后，**每个进程都持有某一块的完整规约结果**（进程 i 拿到第 i 块的总和），但还没有别人那几块。

**第 2 阶段：All-Gather**，再走 N−1 步。同样绕环传，但这次是**覆盖而不是累加**，把各自手里那块已完成的结果转一圈发给所有人。

![[dist-ring-allreduce-4-allgather.png]]

两阶段共 **2(N−1)** 步，每步每卡收发 **1/N** 的数据量。**通信量与卡数 N 无关**（每卡总收发约 2×数据量），且**带宽被均摊到每条环边上，没有热点**——这就是它取代 Parameter Server 的原因。

> 详细推导见[这篇知乎](https://zhuanlan.zhihu.com/p/69797852)。

### 1.3 PP（流水线并行）

**将模型的不同层**（单层，或连续的多层）**分配到不同 GPU 上**，按顺序处理数据，实现流水线式的并行计算。

![[dist-pp-bubble.png]]

**流水并行有点像串行**：每个 GPU 需要等待前一个 GPU 的计算结果，会导致大量 GPU 资源浪费——图中黄色部分就是 **bubble（气泡）**。朴素做法下 4 张卡里同时只有 1 张在干活。

![[dist-pp-microbatch-1f1b.png]]

两级优化，对应图 b 和图 c：

- **图 b：切 micro-batch**。把 mini-batch 进一步切成 micro-batch。GPU0 处理完一个 micro-batch 后，**紧接着开始处理下一个**，以此减少 GPU 空闲时间。气泡占比从 `(N-1)/N` 降到约 `(N-1)/(N-1+m)`（m 为 micro-batch 数）——**切得越碎气泡越小**，但每块太小又会让 kernel 效率下降。
- **图 c：1F1B 调度**。在一个 micro-batch 完成前向计算后**提前调度、完成相应的反向计算**，这样就能**释放部分显存**（前向激活值用完即弃），用以接纳新的数据，提升整体训练性能。注意图 c 的气泡总量和图 b 差不多，**它省的主要是显存**——峰值激活值从「m 份」降到「流水线深度份」。

### 1.4 TP（张量并行）

**将模型的张量（如权重矩阵）按维度切分到不同 GPU 上运行**。切分方式分按行和按列，分别对应**行并行（Row Parallelism，权重矩阵按行分割）**与**列并行（Column Parallelism，权重矩阵按列分割）**。

![[dist-tp-row-vs-column.png]]

看图上半和下半的区别，**关键在输入怎么处理、输出怎么合并**：

| | 权重切法 | 输入 X | 输出合并 |
|---|---|---|---|
| **列并行** | W 按列切成 W₁ / W₂ | **完整拷贝**给每个节点 | **拼接（All-Gather）** |
| **行并行** | W 按行切成 W₁ / W₂ | **也要切片**（X1 / X2） | **求和（All-Reduce）** |

图里的算例很清楚：列并行下节点 1 算出 `38`、节点 2 算出 `62`，拼起来就是完整的 `[38, 62]`；行并行下节点 1 得 `[5, 9]`、节点 2 得 `[33, 53]`，**必须相加**才得到 `[38, 62]`——因为每个节点算的都是**部分和**。

每个节点处理切分后的子张量，最后通过集合通信操作（**All-Gather 或 All-Reduce**）合并结果。适合单个张量过大的情况，可以**显著减少单个节点的内存占用**。但切分维度较多时**通信开销比较大**，而且实现过程较为复杂，需要仔细设计切分方式和通信策略。

> **Transformer 里为什么是「列 → 行」配对**：MLP 的两个矩阵 `W_up`（列并行）后接 `W_down`（行并行），中间那次 All-Gather 就被省掉了——列并行的输出**恰好**是行并行需要的切片形式。**一层只在最后做一次 All-Reduce**，这是 Megatron-LM 的经典设计。Attention 同理，多头天然可以按头切。

### 1.5 EP（专家并行）

**MoE（混合专家模型）中的一种并行计算策略**。通过将专家（子模型）分配到不同 GPU 上，实现计算负载的分布式处理。**最终结果按照 all-to-all 的方式通信**。

为什么必须是 all-to-all：每个 token 经过路由后要去**不同的专家**，而专家散在不同卡上，所以**每张卡都要把自己的 token 按目的地拆开发给所有卡**（dispatch），算完再**按来源收回**（combine）。这正是 all-to-all 的定义（见 §2.4）。

**EP 的特有难题是负载不均**：路由是数据决定的，某些专家可能被抢破头（hot expert），某些没人去，而 all-to-all 会**等最慢的那张卡**。所以实际系统要做冗余专家、负载均衡损失之类的补丁——[[DeepSeek-V3 架构与低成本高效训练]] 的「无辅助损失负载均衡」正是冲着这个问题去的。

## 二、通信原语

上面四种并行用到的通信操作，都是这几个**集合通信原语（collective）**的组合。按参与方数量分三类。

### 2.1 一对多

**Broadcast（广播）**：一个节点把数据发给所有节点。

![[dist-collective-broadcast.png]]

**Scatter（划分并散布）**：一个节点把数据**切开**，每个节点拿**一份不同的**。

![[dist-collective-scatter.png]]

> **两者的区别就在「切不切」**：Broadcast 后大家拿到的是**同一份完整数据**（都是 A）；Scatter 后每人拿到的是**不同的一块**（A1/B1/C1/D1）。

### 2.2 多对一

**Reduce（规约）**：在集合通信里表示「规约」运算，是**一系列简单运算操作**（SUM、MIN、MAX、PROD、LOR 等）的统称。多个节点的数据按该运算合并到一个节点上。

![[dist-collective-reduce.png]]

**Gather（反向的 Scatter）**：把各节点的数据**收集拼接**到一个节点，不做运算。

![[dist-collective-gather.png]]

> **Reduce 与 Gather 的区别是「算不算」**：Reduce 出来是 `A+B+C+D` 一块（数据量不变），Gather 出来是 `A|B|C|D` 四块摞着（数据量变成 4 倍）。

### 2.3 多对多

**All-Reduce**：**DP 中常用**。所有节点都拿到规约结果。

![[dist-collective-allreduce.png]]

**All-Gather**：**可以理解为先 Gather 再 Broadcast**，所有节点都拿到拼接后的完整数据。

![[dist-collective-allgather.png]]

**Reduce-Scatter**：**先归约（Reduce），再分散（Scatter）**——每个节点最终只拿到规约结果的**一部分**。

![[dist-collective-reduce-scatter.png]]

**All-to-All**：**将节点 i 的发送缓冲区中的第 j 块数据发送给节点 j。节点 j 将接收到的来自节点 i 的数据块，放在自身接收缓冲区的第 i 块位置。**

![[dist-collective-all-to-all.png]]

> **All-to-All 是唯一的「转置」型原语**：对比图上左右两侧，它相当于把「按行分布」的矩阵变成「按列分布」。其余原语都是复制或合并，只有它在**重新分配归属**——这正是 MoE 路由需要的语义。

### 2.4 一张表理清关系

| 原语 | = 什么组合 | 数据量变化（每卡） | 谁在用 |
|---|---|---|---|
| Broadcast | — | 1 → 1（复制到所有卡） | 参数初始化、权重分发 |
| Scatter | — | N → 1 | 切数据 |
| Reduce | — | N 份 → 1 卡 1 份 | 梯度汇总（PS 式） |
| Gather | — | 1 → N（汇到 1 卡） | 结果收集 |
| **All-Reduce** | **Reduce-Scatter + All-Gather** | 不变 | **DP 梯度同步、TP 行并行** |
| All-Gather | Gather + Broadcast | 1 → N | TP 列并行、ZeRO-3 取参数 |
| Reduce-Scatter | — | N → 1（各持一段） | ZeRO-2 切梯度 |
| **All-to-All** | — | 不变（**重新分配**） | **EP 的 dispatch / combine** |

> **第二列那行 `All-Reduce = Reduce-Scatter + All-Gather` 就是 Ring-AllReduce 的两个阶段**（§1.2）。原语之间不是并列的名词表，是可以拆解组合的。

## 三、通信代价：回到硬件

到这里把 [[LLM推理的GPU硬件基础]] §五的表接上——**同样一次 All-Reduce，走 NVLink 和走 PCIe 差一个数量级**：

| 互联 | 带宽 | 场景 |
|---|---|---|
| NVLink 4（H100） | 900 GB/s | 卡间直连 |
| PCIe 5.0 x16 | 128 GB/s | 无 NVLink 的卡 |
| InfiniBand NDR | 400 Gb/s ≈ 50 GB/s | 跨节点 |

四种并行压在链路上的**频率和位置**完全不同：

| 策略 | 通信内容 | 频率 | 在不在关键路径 | 对互联要求 |
|---|---|---|---|---|
| **TP** | 激活值 All-Reduce | **每层 2 次** | **是**，算完必须等 | **极高，必须 NVLink** |
| **PP** | 层边界激活值 | 每个 stage 边界 1 次 | 部分（可与计算重叠） | 低，PCIe 可接受 |
| **EP** | token All-to-All | 每个 MoE 层 2 次 | 是 | 高 |
| **DP** | 梯度 All-Reduce（推理时几乎不通信） | 每步 1 次 | 训练时可与反向重叠 | 低（但每卡要装下全模型） |

**关键直觉**：TP 每层都要同步，通信频繁且卡在关键路径上——**跨节点做 TP 会灾难性掉速**。所以典型部署是「**节点内 TP（走 NVLink）+ 节点间 PP/DP（走 IB）**」。

> **为什么 TP 特别怕跨节点**：它的通信量小但**次数极多**，每次都要等。而跨节点链路的问题主要是**延迟**而非带宽——每层两次同步 × 80 层，延迟被放大 160 倍。PP 恰好相反，传的数据块大但次数少，正好适合喂给带宽够、延迟高的 IB。

## 四、推理场景有什么不同

**上面的图基本都是训练视角**（有反向、有梯度、有优化器状态）。推理时这套东西要重新排位：

| | 训练 | **推理** |
|---|---|---|
| **DP** | 主力，梯度 All-Reduce 是大头 | **退化成「多副本负载均衡」**——各副本独立服务不同请求，**几乎零通信** |
| **PP** | 常用，靠 micro-batch 填气泡 | **地位下降**：decode 每步只有 1 个 token，**没有 micro-batch 可切，气泡填不满** |
| **TP** | 常用 | **主力**：单卡装不下时的首选，且能**分摊权重搬运带宽** |
| **EP** | MoE 必需 | MoE 必需，且 decode 时 all-to-all 的**延迟**比训练时更刺眼 |
| **ZeRO** | 核心技术 | **基本用不上**——推理没有优化器状态和梯度可切 |

**TP 在推理里有个训练时没有的额外好处**，这条直接来自 [[LLM推理的GPU硬件基础]] 的核心结论：

```
单流 decode 速度上限 = HBM 带宽 ÷ 权重字节
```

decode 是 **memory-bound** 的，耗时约等于「把权重从 HBM 搬一遍」的时间。TP=4 之后每张卡**只需搬 1/4 的权重**，而 4 张卡的带宽是并行叠加的——**理论上 decode 能快接近 4 倍**。这是 TP 在推理里压过 PP 的根本原因：**PP 切层虽然也减少了每卡的权重量，但 decode 时各 stage 是串行的——一步下来这些权重还是要被依次搬完，总时间没省**，TP 切张量则实打实地把带宽压力分摊了。

> **代价当然是那两次 All-Reduce**。所以 TP 的收益曲线是先升后降：卡越多每卡搬得越少，但同步开销越大。实践上单节点 8 卡以内 TP 通常划算，跨节点就要让位给 PP/DP。

## 五、单卡内部的并行：SM / Warp / Block

前四节讲的是**卡与卡之间**怎么拆。往下一层，**一张卡内部**也是大规模并行的，这层的调度单位是 SM 和 Warp。

**SM（Streaming Multiprocessor，流式多处理器）**：GPU 的主要计算单元，可以粗略理解为 GPU 内部的一个「计算集群」。

**Warp 是 CUDA 实际调度线程的基本单位**。在 NVIDIA GPU 上 **1 warp = 32 threads**，这 32 个线程通常**执行同一条指令，但处理不同的数据**，即 **SIMT（Single Instruction, Multiple Threads）**模式。

**调度关系**（从 kernel 启动到线程真正跑起来）：

1. Kernel 启动大量 **Block**；
2. GPU 把 Block 分配给不同 **SM**；
3. **一个 Block 不会拆到多个 SM 上**；
4. Block 在 SM 内被拆成多个 **Warp**；
5. **Warp Scheduler** 选择准备好的 Warp 执行。

> **第 3 条是 Block 大小的约束来源**：既然 Block 不跨 SM，Block 里的线程就能共享该 SM 的 Shared Memory、能用 `__syncthreads()` 同步——这两件事跨 SM 都做不到。反过来，Block 开太大会因为一个 SM 的寄存器 / SMEM 装不下而降低驻留数。
>
> **第 5 条是 GPU 藏延迟的核心手段**：某个 Warp 卡在等显存时，Scheduler 直接切到另一个就绪的 Warp，**不需要保存/恢复上下文**（各 Warp 的寄存器是分开常驻的）。这就是 [[LLM推理的GPU硬件基础]] §4.1 讲的 **occupancy（每 SM 驻留 warp 数 / 上限）**为什么重要——驻留的 Warp 越多，越能靠切换把访存延迟盖住。

**这一节和前四节的接头**：集合通信的 kernel（NCCL 那些）**本身也要占 SM**。所以「通信与计算重叠」并不是白拿的——重叠时通信 kernel 会和计算 kernel 抢 SM 资源，这是 §三 那张代价表之外的一笔隐性开销。

> 更细的层级（CUDA Core / Tensor Core、寄存器 / SMEM / L2 / HBM 的容量与延迟）见 [[LLM推理的GPU硬件基础]] §4.1 与 §4.3，本篇不重复。

## 六、量化推理：W8A8 / AWQ / GPTQ / FP8

模型权重默认 fp16/bf16（每个数 2 字节）。**量化 = 用更少的比特存**（int8 1 字节、int4 半字节）。收益有**两层**：

1. **省显存**：70B 从 140 GB → int4 约 35 GB，单卡能跑了 / KV cache 空间变多了；
2. **省带宽**：decode 耗时 ≈ 搬权重的时间，**权重小一半，decode 基本快一半**——这是量化在推理里如此重要的根本原因。

> 第 2 条正是 [[LLM推理的GPU硬件基础]] 的核心结论「单流 decode 速度上限 = HBM 带宽 ÷ 权重字节」的直接推论。那篇 §三 给了实测量级：70B 在 A100 上 fp16 约 23 tok/s，**量化到 INT4 后约 95 tok/s**——**不是算得快了，是读得少了**。

**命名规则：`WxAy`**——**W 是权重位宽，A 是激活位宽**。

| 方案 | 位宽 | 做法 | 特点 |
|---|---|---|---|
| **W8A8** | W8 / A8 | 权重和激活都量到 8 bit（经典做法是 **SmoothQuant** 式 int8） | **两边都是低精度，矩阵乘可以直接用 int8/fp8 Tensor Core 硬件指令，计算本身也加速** |
| **AWQ**（Activation-aware Weight Quantization） | W4 | 观察到**只有约 1% 的权重通道**对应着大幅值激活，量化坏它们伤害最大。做法：按激活统计给这些关键通道做**数值缩放加以保护**，再整体量到 4 bit | 只需**少量校准数据**过一遍，**快且效果稳**，是 4 bit 部署的主流 |
| **GPTQ** | W4 | 更「数学」的路线——**逐列量化**，每量化一列，用**二阶信息（Hessian）**去调整还没量化的列来**补偿刚造成的误差** | 精度也很好，**量化过程比 AWQ 慢** |
| **FP8**（E4M3 / E5M2） | W8 / A8 | **不是整数而是 8 bit 浮点**，保留了指数位所以**动态范围大**，对 LLM 的数值分布天然友好 | **Hopper（H100）起硬件原生支持 FP8 Tensor Core**。W8A8-FP8 **几乎无损、无需复杂校准**，是当前生产部署的主流选择；DeepSeek-V3 甚至**直接用 FP8 训练** |

> **AWQ 与 GPTQ 的分野**，一句话：**AWQ 靠「挑出重要的加以保护」，GPTQ 靠「错了再补偿回来」**。前者是事前防守，后者是事后纠偏，所以后者要解 Hessian、要慢。
>
> **W4 系（AWQ/GPTQ）和 W8A8 的定位也不同**：W4 只量权重、激活仍是 fp16，**省的主要是显存和带宽，矩阵乘还得反量化回 fp16 来算**；W8A8 两边都低精度，**能吃到 Tensor Core 的算力红利**。所以「4 bit 一定比 8 bit 快」是不成立的——decode（memory-bound）W4 更快，prefill（compute-bound）W8A8 可能反超。

**和分布式的关系**：量化和 TP 是**同一个方向上的两种手段**——都在减少「每张卡每步要搬的权重字节」。两者可以叠加（FP8 权重 + TP=4，每卡搬 1/8），但 TP 要付 All-Reduce 的通信代价，量化要付精度代价。


## 七、投机采样（Speculative Decoding）

decode 是 memory-bound 的——**搬一次权重只产出 1 个 token，算力大量闲置**。投机采样的思路：用一个**便宜的方式先猜若干个 token**，再让大模型**一次性并行验证**这几个。猜对了就白赚，猜错了从错的那个位置截断重来，**结果分布与原模型完全一致**（靠拒绝采样保证）。

四条路线：

| 路线 | 做法 | 特点 |
|---|---|---|
| **独立小模型（standalone）** | 找个**同词表**的小模型（如 1B 猜 70B） | 最直观，但小模型**「不知道大模型在想什么」**，接受率一般 |
| **N-gram** | **不用模型**，从上文里做**字符串匹配**来猜 | **零成本**，适合大量复读的场景（改代码、抽取） |
| **EAGLE** | 训一个**约一层的轻量 draft 头**，输入不只是 token，**还有大模型上一步的 hidden state** | 关键洞察：**hidden state 里已经蕴含了大模型的「意图」**，所以猜得非常准；还把草稿**长成树**（每步猜 top-k 个分支），**一次 verify 整棵树**，进一步提高期望接受长度 |
| **MTP** | **DeepSeek 训练时自带的多 token 预测头**，推理时当 EAGLE 式 draft 用 | 思路同 EAGLE，优势是**官方 checkpoint 自带、和主模型一起训过**，接受率极高（80%+） |

> 原笔记里这张表是终端里粘进来的 ASCII 制表符画的，顶边框丢失、MTP 一行的文字被截断，已重排为 markdown 表并补回被截掉的「EAGLE 式 draft 用」。内容按原意保留，未增删观点。

**为什么这条也算「分布式」的邻居**：投机采样把 decode 从「每步 1 个 token」变成「每步验证 k 个 token」，**等于人为把 batch 维度撑大**——这正好补上了 §四 说的「PP 在推理里没有 micro-batch 可切」的短板，也是 TP 的 All-Reduce 摊薄的机会（一次同步覆盖 k 个 token 而不是 1 个）。代价是**通信次数不变但每次的量变大**，对带宽更友好、对延迟更宽容。

> ⚠️ 存疑：上面「投机采样撑大 batch 维度、间接改善并行效率」是我加的串联，**你原笔记只列了四条路线的做法与特点**。方向上应该没问题（verify 阶段确实是一次 forward 过 k 个 token），但「摊薄 All-Reduce」这类量化收益我没有实测依据。

> **[[SGlang]] 里的对应**：`--speculative-algorithm` 内置 `EAGLE` / `EAGLE3` / `NEXTN` / `STANDALONE` / `NGRAM` / `DFLASH` / `DSPARK`（还可以用 `SpeculativeAlgorithm.register` 注册树外算法），配 `--speculative-num-steps`（草稿走几步）、`--speculative-eagle-topk`（每步几个分支）、`--speculative-num-draft-tokens`（一次 verify 几个）。那篇 §五 还讲到 EAGLE 路径下 FlashInfer 会额外开一条 `plan_stream`，把草稿多步的 plan 挪出关键路径。

## 八、在 SGLang 里的落点

> 以下核对自本地仓库 `d:/project/sglang`（commit `fdebc938f7`，tag `v0.5.16`）。

| 概念 | SGLang 里在哪 |
|---|---|
| 四种并行的开关 | `server_args.py`：`--tp-size` / `--pp-size` / `--dp-size` / `--ep-size`（各有 `--tensor-parallel-size` 等长别名） |
| 通信原语实现 | `srt/distributed/parallel_state.py`：`all_reduce` / `all_gather` / `reduce_scatter` / `all_to_all_single` 等方法 |
| 各家通信后端 | `srt/distributed/device_communicators/`：`pynccl.py`（NVIDIA NCCL）、`custom_all_reduce.py`（自研小消息 All-Reduce）、`npu_communicator.py`（昇腾）、`xpu_communicator.py`（Intel）、`shm_broadcast.py`（同机共享内存） |
| EP 的 all-to-all | `--moe-a2a-backend deepep` 等，`--deepep-mode` 调 normal / low_latency |
| DP attention | `--enable-dp-attention`：**attention 用 DP、FFN 用 TP**，要求 `dp_size == tp_size`，为 DeepSeek-V2 / Qwen2-3 MoE 这类模型准备 |
| PP 的 micro-batch | `--pp-max-micro-batch-size`、`--pp-async-batch-depth` |

**`custom_all_reduce.py` 值得单拎一句**：小消息 All-Reduce 用 NCCL 反而不划算（启动开销盖过传输），所以 SGLang 自己写了一套走 NVLink P2P 直写的实现——这正是「**TP 通信在关键路径上**」这条硬件事实逼出来的工程对策。

> **交叉印证**：[[SGlang]] §六讲的 `register_custom_op` 里，`inplace_all_reduce` / `outplace_all_reduce` / `reg_all_gather_into_tensor` / `reg_reduce_scatter_tensor` / `reg_all_to_all_single` 这几个算子，注册的正是本篇的通信原语——**它们要被包成 custom op，就是为了让 `torch.compile` 不去 trace 通信、把它当成不可重排的黑盒节点**（靠 `mutates_args` 声明）。

## See Also

- [[LLM推理的GPU硬件基础]] — **本篇的硬件底座**：互联带宽表、显存构成、decode 是 memory-bound 的推导。本篇 §三、§四的所有结论都由那篇的数字支撑。
- [[SGlang]] — 本篇 §八 的展开：`distributed/` 目录、PD 分离、以及通信算子如何被注册成 custom op。
- [[DeepSeek-V3 架构与低成本高效训练]] — EP 负载均衡（无辅助损失）与 DualPipe（PP 气泡优化）的工业级实例；MLA 则是从架构层减小 KV cache 而非靠并行。
- [[LLM推理压测-bench serve 与 throughput 参数详解]] — 用压测验证 TP 的实际收益。

## 备注

- 本文的并行策略、通信原语、SM/Warp、量化与投机采样各节整理自学习笔记与配图；**图为示意，非某一框架的实际实现**。
- §七 的投机采样表原为终端 ASCII 制表符绘制，顶边框缺失、MTP 一行被截断，已重排为 markdown 表并补回截断文字，**观点未增删**。
- §八 的 SGLang 参数与文件路径核对自 `d:/project/sglang`，commit `fdebc938f7`（tag `v0.5.16`）；**参数名可能随版本变化，以 `--help` 为准**。
- §1.1 的 ZeRO 三级划分见正文中的 ⚠️ 存疑标注。
