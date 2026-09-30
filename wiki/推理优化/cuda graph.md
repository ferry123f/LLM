# CUDA Graph 学习笔记

> 基于 SGLang 源码（`python/sglang/srt/model_executor/`）整理，覆盖底层机制、流程、SGLang 实现与特色设计。

---

## 一、为什么需要 CUDA Graph

### 1.1 先理解 GPU 是怎么干活的

模型虽然"跑在 GPU 上"，但真正指挥 GPU 的是 **CPU**。一次推理 forward 的大致过程：

```
CPU：请执行"矩阵乘法"kernel（函数），输入 A 和 B，输出写到 C
GPU：算完了
CPU：请执行"激活函数"kernel，输入 C，输出 D
GPU：算完了
...
```

一个 Llama-8B 模型，一次 forward 需要**几百甚至上千个 kernel**，CPU 要逐个"下发"。

### 1.2 kernel launch 的开销

CPU 每下发一个 kernel，叫 **kernel launch（内核启动）**，需要经过：

1. CPU 准备参数（输入/输出地址、网格大小等）
2. CPU 调用驱动（driver）
3. 驱动校验参数、分配资源
4. 驱动把指令放入队列（CUDA stream，指令流水线）
5. GPU 从队列取出指令，真正执行

这一整套流程，一次 launch 约 **5~10 微秒**。

### 1.3 eager 模式的困境

**eager（急切执行）** = 最朴素的方式：来一步算一步，每遇到一个算子立刻启动一个 kernel。PyTorch 默认就是 eager。

问题在两种场景下变得灾难性：

**场景一：decode（解码）阶段。**
LLM 生成文本是一个词一个词地蹦。每生成一个词都要跑一遍完整 forward。这一遍里：
- GPU 真正"算"的时间很短（batch 小、矩阵窄）
- 但 launch 开销是固定的，几百个 kernel 就是几百次 5~10 微秒

结果：**CPU 花在"下发指令"的时间，可能比 GPU 花在"计算"的时间还长**。GPU 大量时间空转等 CPU 喂 kernel。这叫 **launch-bound（启动开销受限）**——不是算得慢，是"叫"得慢。

**场景二：小模型 / 小 batch。**
模型越小、batch 越小，单 kernel 计算量越小，launch 开销占比越高。

> 比喻：让 500 个工人依次干 500 件小活，每件 3 秒，但每次交代任务要 10 秒。真正干活 1500 秒，交代任务 5000 秒。工人大部分时间在等你走过来。

### 1.4 NVIDIA 的解法：CUDA Graph

既然问题是"逐个下发太慢"，解法就是：**把要下发的指令提前打包成一份"剧本"，之后一次下发整个剧本**。

CUDA Graph（CUDA 计算图）有三个原生 API 阶段：

| 阶段 | API | 作用 |
|---|---|---|
| 捕获 Capture | `cudaStreamBeginCapture` / `cudaStreamEndCapture` | 记录这段时间内 stream 上的所有 kernel，构成 DAG |
| 实例化 Instantiate | `cudaGraphInstantiate` | 把 DAG 编译成可执行的 executable graph，做校验优化 |
| 回放 Launch | `cudaGraphLaunch` | 一条指令把整张图提交，GPU 连续执行所有 kernel |

**图（Graph）的名字由来**：捕获得到的是一张**有向无环图（DAG）**——节点 = kernel，边 = 依赖关系。

核心收益：**几百次 kernel launch 变成 1 次 graph launch**。

### 1.5 PyTorch 的封装

```python
graph = torch.cuda.CUDAGraph()
graph.capture_begin()      # 进入捕获模式
...                        # 正常写 forward，kernel 被记录
graph.capture_end()
graph.replay()             # 一次调用重放整个 forward
```

---

## 二、CUDA Graph 的代价：两条铁律（底层深入）


捕获之后，图里每个节点存的是：

- 这个 kernel 的**函数句柄**（指向哪段 GPU 代码）
- 启动参数里的**指针值**（一个 64 位地址，如 `0x00007f3c0a000000`）
- grid / block 维度、shared memory 大小等**标量**
- 节点之间的**依赖边**（谁先谁后）

注意：图里存的是**指针值本身**，不是指针指向的内存内容。这引出了两条铁律的根源：

- **铁律一（地址固定）**：图里存的是"死"的指针值，回放时不会再去问"这个地址还有效吗？里面装的是什么？"——直接拿它去读写。
- **铁律二（不能动态）**：图是**静态的有向无环图**，回放时驱动只是"按图索骥"地提交节点，**没有任何逻辑判断能力**。

---

### 2.1 铁律一：地址必须固定

#### (1) 图节点里存的是指针，不是数据

以 kernel 节点为例（对应驱动层 `cudaGraphKernelNodeParams`），它内部有个 `kernelParams` 数组，装的是**一堆指向内存的指针**。以矩阵乘法为例：

```
params[0] = 0x00007f3c0a000000   // 指向矩阵 A 的设备内存
params[1] = 0x00007f3c0b000000   // 指向矩阵 B
params[2] = 0x00007f3c0c000000   // 指向输出矩阵 C
params[3] = 512                  // M 维度
params[4] = 512                  // N 维度
```

回放时（`cudaGraphLaunch`），驱动把这些参数**原样提交**给 GPU，一个字节都不改。它**绝不会**检查"0x00007f3c0a000000 这块内存现在是不是还是 A"。

#### (2) torch 缓存分配器为什么会"偷换"地址

真正杀手是：Python/PyTorch 里 tensor 的地址极不稳定。PyTorch 用**缓存分配器（caching allocator）**：

> 释放一个 tensor 时，分配器不会立刻把内存还给系统，而是放进自己的"空闲池"缓存。下次分配大小合适的 tensor 时，**优先把刚释放的内存复用给你**。

崩溃场景：

```
捕获阶段：
  A = torch.zeros(...)    # 分配器给 A 地址 0x1000，图里记了"kernel 读 0x1000"

捕获结束后：
  del A                    # 0x1000 回到空闲池
  B = torch.zeros(...)    # 分配器把 0x1000 复用给 B

回放阶段：
  graph.replay()           # 图还是读 0x1000 —— 但现在装的是 B 的数据！
```

结果：静默算错，或者越界崩溃。所以 CUDA Graph 不能直接拿"随手 new 出来的 tensor"当输入。

#### (3) graph memory pool 解决"图内临时内存"

模型 forward 中间产生海量临时 tensor（每层激活值、GEMM 中间结果），地址怎么稳定？答案是**图专用显存池（graph memory pool）**：

1. 捕获前创建专门的图内存池（SGLang 里 `get_or_create_global_graph_memory_pool` → `graph_pool_handle()`）
2. 捕获时把池传进去（PyTorch 里 `graph.capture_begin(pool=...)`）
3. 捕获期间所有新分配的临时 tensor 都从池里分配
4. **池是"图独占"的**，不属于 torch 通用分配器，torch 后续 new 别的 tensor 不会把图池的内存复用走

于是图内临时 buffer 的地址在捕获后"冻结"，回放永远有效。这解释了为什么 `full_cuda_graph_backend.py` 捕获要写 `pool=self._pool` —— 它是保证图内临时内存稳定的关键。

#### (4) 图外输入/输出 buffer 必须手动保证

图池只管"图内新分配的临时内存"。但**图外由 Python 持有的输入/输出 buffer**（`input_ids`、`logits`）不在图池管辖内，受 torch 分配器管辖，同样面临"被复用"风险。

解决办法只有一个：**捕获时用什么 buffer，回放就必须用同一批**，且保证它们**永不释放**。这就是 SGLang"静态缓冲区（static buffer）"的由来——启动时一次性分配 `self.buffers`，捕获读它，回放前 `copy_` 进它，模块级长期持有。

| 内存类别 | 谁分配 | 地址稳定性由谁保证 |
|---|---|---|
| 图内临时 buffer | 图内存池 | 图池（自动保证） |
| 输入/输出 buffer | Python 提前 new | 你（静态 buffer 长期持有 + 回放前 copy） |
| 模型权重 | 模型加载时 | 天然稳定（权重从不换地址） |

---

### 2.2 铁律二：不能有动态行为

#### (1) 流捕获是个状态机

`cudaStreamBeginCapture` 后，stream 进入特殊模式。驱动为每个流维护**捕获状态（capture status）**：

| 状态 | 含义 |
|---|---|
| None | 正常执行，没有在捕获 |
| Active（活跃） | 正在捕获，后续 kernel 被记录进图 |
| Invalidated（已失效） | 遇到非法操作，捕获作废，必须重来 |

关键：**捕获模式下，launch 一个 kernel 的语义变了**——正常模式是"真正执行"，捕获模式是"记录进图，执行被推迟"。正是这个"执行被推迟"，引出了下面所有限制。

#### (2) 为什么同步操作致命

`torch.cuda.synchronize()` 的语义是"CPU 阻塞等这条流上所有 work 完成"。但**捕获模式下流上的 work 没被真正执行**（只是被记录），于是 CPU 傻等一个"永远不会有结果"的东西 → **死锁**。

更隐蔽的是，很多看似无害的操作内部偷偷同步：

- `tensor.item()`：值拷回 CPU，隐含同步
- `tensor.cpu()`：tensor 拷到 CPU，隐含同步
- `print(tensor)`：`__repr__` 要拿值，隐含同步
- `tensor.numpy()`：同上

所以很多捕获失败的报错看起来毫无头绪——真正原因是某个库函数内部偷偷调了 `.item()`。

#### (3) 为什么查询操作被禁止

`cudaStreamQuery` / `cudaEventQuery` 的**返回值不确定**。捕获模式下流的"完成状态"是未定义的（work 没真正执行）。如果允许你依赖"查询返回 completed 就走 A"，捕获时和回放时返回值可能不一致，图记录的逻辑和真实逻辑对不上。所以驱动直接禁止。

#### (4) 分支：捕获的是"走过的路"，不是"所有路"

```python
if x > 0:            # x 是 GPU 上的标量
    y = self.path_a(x)
else:
    y = self.path_b(x)
```

捕获时 x=3，走了 `path_a`。**图里只记录了 `path_a` 的 kernel，`path_b` 完全不在图里**。回放时哪怕 x=-1，图依然只执行 `path_a`。

结论：**数据相关的控制流，在图里被"拍扁"成一条固定路径**。所以 forward 的控制流必须形状无关、数据无关。

#### (5) 动态形状：grid/block 

kernel 启动参数里，除了指针，还有 **grid 维度、block 维度、shared memory**——决定"这个 kernel 算多大范围"。

捕获时这些值被录入图节点。回放时图**不会**根据新输入重新算 grid。举例：捕获时输入长度 128，attention kernel 的 grid 是 `(4,1,1)`；回放塞进来长度 256 的输入，图还是启动 `(4,1,1)` → **只算前 128 个位置，后 128 个静默错误**。

这就是为什么 decode 能做整图（形状 = bs × 1，固定 bs 就固定 grid）、prefill 不能（token 数每次变，grid 每次变）。

#### (6) 组合拳陷阱

真实翻车往往是几条叠加。典型坑：

```python
def forward(self, input_ids, ...):
    if input_ids.shape[0] < 256:      # 动态形状分支
        return self.small_kernel(...)  # 内部可能藏 .item()
    else:
        return self.large_kernel(...)
```

双重踩雷：动态分支被拍扁 + 内核藏同步。SGLang 的应对是把"不干净"的东西挡在门外（`can_run_graph`）或挪到图外（breakable 分段图的段间 eager 代码）。

---

### 2.3 实例化阶段做了什么（承上启下）

捕获得到的图只是"描述"（`cudaGraph`），要回放必须经过 `cudaGraphInstantiate` 得到**可执行图（`cudaGraphExec`）**。实例化做的事：

1. 遍历整张图，验证节点合法性（参数对不对、依赖有没有环）
2. 全局优化（合并相邻 memcpy、消除冗余依赖边）
3. 计算并预分配图运行资源（graph memory pool 布局）

实例化**昂贵**：图越大越慢，几千节点的模型图要几十~上百毫秒，还要占设备内存存元数据。

这解释了两件事：

- **为什么做图去重（dedup）**：结构相同的图复用 executable，省实例化时间和显存（对应 `cuda_graph_dedup_mixin.py`）
- **为什么"每张图都吃显存"**：显存开销分几块——

| 开销 | 说明 |
|---|---|
| `cudaGraph` 结构体 | 节点/边描述数据 |
| `cudaGraphExec` 元数据 | 实例化后的可执行信息 |
| graph memory pool | 图内临时 buffer（通常最大头） |
| 静态输入/输出 buffer | Python 侧持有的 buffer |

---

### 2.4 附：`cudaGraphExecUpdate` 的边界（理解 dedup 的钥匙）

`cudaGraphExecUpdate(exec, new_graph)` 允许**不重新实例化**地用新图参数更新旧 executable，但有硬边界：

> **只允许更新节点参数（指针值、标量），不允许改变图拓扑（节点数量、依赖关系）。**

- ✅ 可以改：kernel 的指针参数、memcpy 长度标量
- ❌ 不可以：增删节点、改变依赖边方向

结构变了就返回失败，必须销毁 executable 重新实例化。这正对应 SGLang dedup 逻辑：先算"拓扑签名"，**签名相同**才共用 executable，**签名不同**各自实例化。

---

### 2.5 两条铁律如何反推 SGLang 设计

| 底层铁律 | 反推出来的 SGLang 设计 |
|---|---|
| 地址固定（图外 buffer） | 静态缓冲区 + 回放前 copy（`cuda_graph_buffer_registry.py`） |
| 地址固定（图内临时内存） | 全局 graph memory pool（`runner_utils/pool.py`） |
| 不能动态形状 | 分桶机制：decode 按 bs、prefill 按 token 数 |
| 不能数据分支/同步 | 准入判定 `can_run_graph`，不干净的退回 eager |
| prefill 形状太动态 | 分段图（Breakable），把会变的部分挪到图外 eager 算 |
| 实例化昂贵 | 图去重（dedup）复用 executable |

> SGLang 的复杂不是"过度设计"，而是**被 CUDA Graph 底层铁律逼出来的**。

---

## 三、SGLang 整体架构：四层结构

```
第一层：配置层  CudaGraphConfig     —— 哪个阶段用哪种图、捕获哪些尺寸
第二层：编排层  Runner              —— 捕获时造假数据、回放时填真数据
第三层：机制层  Backend             —— 真正调用 capture/replay
第四层：基础设施 静态缓冲区 + 显存池 —— 保证地址固定、省显存
```

对应源码（都在 `python/sglang/srt/model_executor/` 下）：

- 配置：`cuda_graph_config.py`
- 编排：`runner/base_cuda_graph_runner.py`、`runner/decode_cuda_graph_runner.py`、`runner/prefill_cuda_graph_runner.py`
- 机制：`runner_backend/full_cuda_graph_backend.py`、`runner_backend/breakable_cuda_graph_backend.py`、`runner_backend/tc_piecewise_cuda_graph_backend.py`
- 基础设施：`cuda_graph_buffer_registry.py`（静态缓冲区）、`runner_utils/pool.py`（显存池）

---

## 四、核心概念：分阶段 + 分桶

### 4.1 两个阶段分开处理

模型 forward 分成两个**阶段（phase）**：

- **decode**：每次生成 1 个 token，形状可控
- **prefill**（extend）：一次性处理 prompt，token 数不定

默认策略（`cuda_graph_config.py`）：

| 阶段 | 默认策略 | 说明 |
|---|---|---|
| decode | **full（整图）** | 整个 forward 一张图，最简单最快 |
| prefill | **breakable（分段图）** | forward 切成多段，段间跑普通代码 |

### 4.2 分桶的真正动机：函数句柄一样 ≠ 图可以共用

#### (1) 一个常见误解

"不同的 bs / token 走的都是同样的 kernel 流程，一个桶不就好了吗"。这句话**算法层面对，物理层面错**。

- **对的部分**：bs=5 和 bs=8 确实是同一串算子（Embedding → 若干层 (Attention → FFN) → LM Head → Softmax），算子的**种类和顺序**一样。
- **错的部分**：CUDA Graph 记录的不是"算法流程"，而是**每个 kernel 的物理启动参数**。算法流程一样 ≠ 物理启动参数一样。

#### (2) 一个图节点里写死了三样东西

| 节点里的内容 | 对应什么 | 随形状变吗 |
|---|---|---|
| ① kernel 函数句柄 | "执行哪个算法"（GEMM / attention / softmax…） | ❌ 不变（就是"算法流程一样"） |
| ② 启动配置 grid / block / shared memory | "这个 kernel 覆盖多大范围、多少线程" | ✅ **随 bs / token 变** |
| ③ 参数指针 | "从哪个地址读、写到哪个地址" | 部分变（输入地址随前缀变） |

> **如果节点里只有 ①（函数句柄），那确实可以一个桶搞定，因为所有形状的算法流程都一样。**
> **问题就出在 ②（启动配置）——它把"覆盖多大范围"这个形状信息烧死了，所以图必须按形状分。**

#### (3) grid 维度是形状的函数，且被写死


以注意力 kernel 为例：

```
decode 时每请求生成 1 个新 token
bs = 5 → 5 个 query token → attention 需要覆盖 5 个位置 → grid = (5,1,1)
bs = 8 → 8 个 query token → attention 需要覆盖 8 个位置 → grid = (8,1,1)
```

捕获时 grid 被原封不动写进图节点。回放时图**不会**看"这次来了 8 个请求，把 grid 改成 8"——它只会按烧死的 grid=(5,1,1) 启动：

- 拿 bs=5 的图处理 bs=8 → 只算前 5 个 query，后 3 个没算 → 静默错误
- 拿 bs=8 的图处理 bs=5 → 按 8 个算，后 3 个读垃圾数据 → 也错

这就是"同样的 kernel 流程"却"必须分桶"的根本原因——**流程一样，但每个 kernel 的启动规模（grid）不一样，而 grid 是烧死的**。

> 一句话概括：
> - 人理解"流程"是算法层面的：先 GEMM、再 softmax……（一串算子名字）
> - CUDA Graph 记录"流程"是物理层面的：每个算子启动多少个线程块、读写哪些地址（一堆烧死的数字）
> - 算法流程相同，但物理启动参数随形状变化，所以图必须按形状分开。
>
> 比喻：同样的菜谱（算法流程），做 5 人份和 8 人份的"步骤"一样，但"用量"（5 个鸡蛋 vs 8 个鸡蛋）被写死在每份菜谱里。不能拿 5 人份的菜谱备 8 人的菜。

#### (4) "一个桶不就好了吗"—— 可以，但那是"用浪费换省事"

技术上完全可以只捕获一张 max_bs 的图，但代价是巨大计算浪费：

- bs=1 的请求 → 塞进 bs=8 的图 → 7 个空位白算
- bs=3 的请求 → 塞进 bs=8 → 5 个空位白算

decode 大部分时间 batch 很小（半夜就 1 个请求），如果每次 1 个请求都要按 bs=512 的图算，等于白算 511 个空位，浪费成百上千倍，launch 开销省下的早就被吞噬了。

| 方案 | 图数量 | padding 浪费 | 结论 |
|---|---|---|---|
| 只一个桶（只捕获 max_bs） | 1 张，最省显存 | 巨大（小 batch 全拉满） | ❌ 不可接受 |
| 每个 bs 一张图 | 512 张，显存/启动爆炸 | 零浪费 | ❌ 不可接受 |
| **分桶（对数式档位）** | 几十张，可控 | 少量（向上取整到最近档） | ✅ 平衡点 |

**分桶的本质，就是在"图数量"和"padding 浪费"两个极端之间取平衡。**

具体数字：`max_bs=160`，来个 bs=5 的请求——

- 只一个桶（bs=160 图）：padding 到 160，浪费 155/160 ≈ 97%
- 分桶：向上取整到 bs=8，只浪费 3/8 ≈ 37%
- 每个 bs 一张图：精确匹配，零浪费，但要 160 张图

分桶用 24 张图，把浪费从 97% 压到 37%。

prefill 按 token 分桶的动机**完全一样**，只是"形状"从 batch_size 换成 token 数（token 数从几十到几万，跨度几何级）。

### 4.3 分桶机制（bucket）

分桶思想本身不是 SGLang 独有（vLLM 也用），但 SGLang 做成了一套精细的、贯穿始终的机制。

**decode 按 batch_size 分桶：**

- 选定一档捕获尺寸列表，如 `[1, 2, 4, 8, 16, 32, ...]`
- 每个尺寸捕获一张图
- 回放时真实 bs=5，向上取整到 8，用 bs=8 的图
- 多出的 3 个空位用 padding 假数据填，最后把输出切回前 5 个

向上取整用二分查找（`base_cuda_graph_runner.py` 的 `_pad_to_bucket`）：

```python
def _pad_to_bucket(raw_size, buckets):
    index = bisect.bisect_left(buckets, raw_size)  # 第一个 >= raw_size
    return buckets[index]
```

**prefill 按 token 数分桶：**

分桶维度从 batch_size 换成 token 数。但 prefill 有 decode 没有的难题——**padding 浪费**（真实 257 token 补到 512，多算近一倍）。

两个保护：

1. **padding 倍数上限**：补完后/真实大小 > 2 倍就退回 eager。定义在 `prefill_cuda_graph_runner.py` 的 `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2`。
2. **细密档位**：token 小的区间档位密，大的档位疏：

| token 范围 | 档位间隔 |
|---|---|
| 4 – 32 | 4 |
| 48 – 256 | 16 |
| 288 – 512 | 32 |
| 576 – 1024 | 64 |
| 1280 – 4096 | 256 |
| 4096+ | 512 |

### 4.4 分桶的资源消耗：多张图会不会显存爆炸？

#### (1) 先破除误解：图里存的是地址，不是数据

关键认知：**CUDA Graph 不存任何中间激活值的"数值"，只存"指令 + 地址"**。

中间激活值（每层 attention 输出、GEMM 结果）的**数值**是回放时现场算出来的，算完被下次回放覆盖，**从不持久保存**。图里只存这些中间 tensor 的**地址（指针）**：

```
节点 3：执行 GEMM kernel
    参数1 = 0x1000   ← 输入矩阵地址（指针）
    参数2 = 0x2000   ← 权重地址
    参数3 = 0x3000   ← 输出写到这个地址
```

存的是 `0x3000` 这个数字（指针值），不是"0x3000 里装了什么数据"。

所以"中间 tensor 形状不同"只影响"地址空间需要多大"，不影响"要不要存数据"。数据永远不存，存的只是地址，而**地址可以跨图复用**。

#### (2) 为什么地址能跨图复用：因为从不同时回放

假设 bs=8 图中间 tensor 需要 0.5GB，bs=16 图需要 1GB。它们形状不同，需要的内存大小不同，但**从不同时运行**：

```
显存池按最大的图（bs=16）分配一次，地址范围 [0x1000 ~ 0x1000 + 1GB]

bs=8 的图：  中间 tensor 用 [0x1000 ~ 0x1000 + 0.5GB]  ← 只用前缀一半
bs=16 的图： 中间 tensor 用 [0x1000 ~ 0x1000 + 1GB]    ← 用全部
```

两张图中间 tensor 的地址重叠（都从 0x1000 开始）。安全，因为回放 bs=8 时只有 bs=8 的图在跑，回放 bs=16 时只有 bs=16 在跑，**永远不会同时**。

所以池只需分配 1GB（取最大的），而不是 0.5+1=1.5GB。

> 比喻：显存池是一间固定大小的仓库。bs=8 图是"用 8 个货架中转"的方案，bs=16 图是"用 16 个货架中转"的方案。方案里写的是货架编号（地址），不是货架上摆的东西（数据）。因为从不同时执行，可以共用一间仓库、货架编号重叠。仓库按最大方案（16 个货架）建一次。

#### (3) 反向捕获（大→小）让地址复用真正发生

地址重叠是"理想情况"，怎么保证它真发生？答案就是反向捕获（源码注释直白：`Capture the large shapes first so that the smaller shapes can reuse the memory pool`）。

**顺序反着（小→大）会怎样：**

1. 先捕获 bs=8 图：池里分配 0.5GB（0x1000 起），用完释放回池
2. 再捕获 bs=16 图：需要 1GB，池里空闲只有 0.5GB，**不够** → 另外申请 1GB（0x120000000 起）
3. 结果：两图地址不重叠，池共占 1.5GB

**顺序正着（大→小）：**

1. 先捕获 bs=16 图：分配 1GB（0x1000 起），用完释放（空闲 1GB）
2. 再捕获 bs=8 图：需要 0.5GB，池里正好有 1GB 空闲块，直接复用（拿 0x1000 起的前 0.5GB）
3. 结果：两图地址重叠，池只占 1GB

所以 `reversed(self.capture_bs)` 不是可有可无，它是让地址复用真正发生的关键。

#### (4) 三块资源的账本

一张 CUDA Graph 占的资源分三块，增长规律完全不同：

| 资源 | 是什么 | 随图数量怎么变 |
|---|---|---|
| ① 显存池 | 图内临时激活值（中间结果） | **几乎不随图数量增长**（共享） |
| ② 静态缓冲区 | 输入/输出 tensor 固定地址 | **完全不随图数量增长**（只按 max_bs 分一套） |
| ③ 可执行图元数据 | 图节点/边描述结构 | **线性增长，但极小** |

**① 显存池（大头，共享）**：prefill 图和 decode 图共用同一个池，同 phase 内所有档位也共用。因为任何时刻只有一张图在回放。所以显存池 ≈ 最大一张图的中间激活显存，而不是"所有图之和"。

**② 静态缓冲区（只分一套）**：`DecodeInputBuffers.create(max_bs=...)` 按最大 bs 分一套，每个档位只是这套 buffer 的不同前缀切片（bs=8 读 `buffers[:8]`，bs=16 读 `buffers[:16]`），共享同一块物理内存。

**③ 可执行图元数据（唯一真正"每张图一份"）**：存"这张图有几个节点、每个节点是什么 kernel、参数指针多少、依赖边怎么连"的结构描述。一个 kernel 节点约几十~几百字节，一张模型图几百~上千节点，**一张图元数据 ≈ 几十 KB ~ 几百 KB**。捕获 52 张图总共几十 MB，相比显存池几个 GB，**占比不到 1%**，还能 dedup 去重。

> 真正随图数量线性增长的，只有 ③ 可执行图元数据和实例化时间（都很小，且启动时一次性付清）。显存大头（① 中间激活显存 + ② 静态 buffer）是全局共享的，只按最大尺寸分一份。

#### (5) SGLang 到底捕获多少张图

档位是**对数式分布**（小档密、大档疏），目的就是控制图数量。

decode 档位生成（`server_args.py` 的 `_generate_decode_cuda_graph_batch_sizes`）：

```python
capture_bs = (
    [1, 2, 4, 8, 12]
    + list(range(16, 257, 8))      # 16~256 步长 8
    + list(range(272, 512, 16))    # 272~512 步长 16
    + list(range(512, max_bs + 1, 32))  # 512+ 步长 32
)
```

- `max_bs=160` → 约 24 张
- `max_bs=512` → 约 52 张

prefill 档位生成（`_generate_prefill_cuda_graph_batch_sizes`）：

```python
capture_sizes = (
    list(range(4, 33, 4))
    + list(range(48, 257, 16))
    + list(range(288, 513, 32))
    + list(range(576, 1025, 64))
    + list(range(1280, 4097, 256))
    + list(range(4608, max_bs + 1, 512))
)
```

- `max_bs=8192` → 约 58 张

所以正常情况下是几十张图，不是几百张。**对数式分布是在"padding 浪费"和"图数量"之间找平衡**。

#### (6) SGLang 缓解资源消耗的四个手段

1. **全局共享显存池**（`runner_utils/pool.py`）：prefill + decode + 同 phase 所有档位共用，显存从"所有图之和"变"最大一张图"，砍掉几十倍。
2. **反向捕获**：大→小，让地址复用真正发生。
3. **图去重 dedup**（`cuda_graph_dedup_mixin.py`）：结构相同的图复用同一个 executable，用 `cudaGraphExecUpdate` 更新参数，砍元数据和实例化时间。
4. **max_bs 由显存自动决定**（`server_args.py` 的 `_handle_gpu_memory_settings`）：防止图数量失控，并启动时预估算图占显存。

第 4 点的关键注释（源码 `_handle_gpu_memory_settings`）：

```
GPU memory = 模型权重 + KV cache + 激活值 + cuda graph buffers
reserved_mem = chunked_prefill_size * 1.5 + max_bs * 2   （单位 GB）
mem_fraction_static = (总显存 - reserved_mem) / 总显存
```

SGLang 启动时**预估 cuda graph 吃多少显存**（用 `max_bs * 2` 启发式系数估），再压缩 KV cache 池大小（`mem_fraction_static`），确保不会因图占显存而 OOM。

`max_bs` 按显存自动分档：

| 显存 | 默认 max_bs |
|---|---|
| < 20GB（T4/4080） | 8 |
| 20~35GB（A10/4090） | 24 或 80 |
| 35~60GB（A100 40GB） | 32 或 160 |
| 60~90GB（H100/A100） | 256 或 512 |
| 90~160GB（H20/H200） | 256 或 512 |
| >160GB（B200/MI300） | 512 |

#### (7) 极端情况：`disable_cuda_graph_padding`

关闭 padding 时（`_generate_decode_cuda_graph_batch_sizes` 第一句）：

```python
if self.disable_cuda_graph_padding:
    capture_bs = list(range(1, max_bs + 1))   # 每个 bs 一张图！
```

`max_bs=512` 时会捕获 512 张图（1,2,3,...,512）。这是"用资源换零浪费"的极端权衡：padding 浪费为 0，但图数量爆炸 → 元数据涨到 512 份、实例化时间 512 次、启动变慢。

但注意：**即使 512 张图，显存池和静态 buffer 依然共享**，不会显存爆炸，真正涨的是元数据 + 实例化时间 + 启动时间。所以 SGLang 默认不关闭 padding。

---



## 五、捕获流程（以 decode 整图为例）

### 5.1 启动入口

`cuda_graph_setup.py` 的 `capture_cuda_graphs()`，顺序有讲究：

1. 先建 **eager runner**（普通执行器，作兜底）
2. 再捕获 **prefill 图**
3. 再捕获 **decode 图**

顺序是"eager 先、prefill 次、decode 后"，让后建的缓冲区复用先建的显存。

### 5.2 进入图模式

`decode_cuda_graph_runner.py` 的 `capture()`：

```python
with freeze_gc(...):              # 冻结 Python 垃圾回收
    with graph_capture() as ctx:  # 进入图捕获上下文
        self.stream = ctx.stream
        with self.backend.capture_session(self.stream):
            self._capture_one_stream()
```

两个关键动作：

**动作一：`freeze_gc` 冻结垃圾回收。**
捕获中若 GC 回收了中间 tensor，地址可能被复用，导致图里记录的地址变悬空地址，回放崩溃。

**动作二：`graph_capture()` 开独立流 + 通信组进入图模式。**
（`parallel_state.py`）

1. 开一条和默认流隔离的独立 stream 专门捕获，后台代码的 kernel 不会混进图。
2. 让 TP/PP/DP/MoE 通信组进入图捕获模式。

为什么通信组特殊？**图捕获期间 NCCL 集合通信（all-reduce）不能真通信**（捕获是排练，真通信会死锁）。所以通信组捕获时用占位代替真通信，捕获结束后一次性初始化通信 buffer。这也是捕获时每一步之间要 `tp_group.barrier()` 对齐所有 GPU 的原因。

### 5.3 从大到小捕获

```python
capture_range = tqdm.tqdm(list(reversed(self.capture_bs)))
```

**先捕获最大的 bs，再捕获小的**。原因：大图先占大块显存池，小图复用同一块池，避免碎片化。

### 5.4 单尺寸捕获：假数据 + 预热 + 真捕获

`capture_one_shape()` 干三件事：

**第一件：`capture_prepare(bs)` 造假输入。**
关键不是数值对不对，而是**地址必须固定**。所以从预先分配的静态缓冲区切片：

```python
input_ids = _slot("input_ids")   # 从静态缓冲区取
seq_lens  = _slot("seq_lens")
positions = _slot("positions")
```

**第二件：预热两遍（warmup）。**
（`full_cuda_graph_backend.py` 的 `capture_one`）

```python
for _ in range(2):
    synchronize()
    tp_group.barrier()
    forward_fn()   # 跑一遍，但还没捕获
```

第一次跑 forward 触发一堆一次性行为：cuBLAS 惰性加载、workspace 首次分配、JIT 首次编译、cuDNN 算法选择。这些有副作用/动态决策，**不能出现在图里**。跑两遍付清，第三遍是纯 kernel 序列。

**第三件：真捕获。**

```python
graph = torch.cuda.CUDAGraph()
with graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream):
    out = forward_fn()          # 这一遍被记录进图
self._graphs[shape_key] = graph
self._outputs[shape_key] = out
```

`pool=self._pool` 指定显存池，捕获期间分配的中间 tensor 都落在这个池里，回放也复用。这是地址固定的根基。

---

## 六、回放流程

服务运行时每个 decode 步走 `execute()` → `load_batch()` → `replay()`。

### 6.1 准入判定 `can_run_graph()`

不是每个 batch 都能用图。满足所有条件才走图，否则**退回 eager**（慢但永远正确）。典型检查：

- bs 是否超上限
- 有没有动态 token embedding（`replace_embeds`）
- 是不是 encoder-decoder 混合 batch
- 投机解码宽度是否匹配
- TBO、ngram、隐藏态模式是否兼容

### 6.2 填数据 `load_batch()`

铁律一"地址固定"的落地：

1. 向上取整：真实 `raw_bs` → 最近的捕获尺寸 `bs`
2. 填静态缓冲区：`buffer_registry.fill_from(...)` 把真实 batch 的数据拷进固定地址 buffer
3. 构造回放 batch 视图：组合"真实运行期字段" + "静态 buffer 补全字段"

### 6.3 重放 `replay()`

```python
self._graphs[shape_key].replay()   # 就这一句，一次 cudaGraphLaunch
return self._outputs[shape_key]
```

### 6.4 切输出

```python
next_token_logits = output.next_token_logits[: self.raw_num_token]
```

---

## 七、SGLang 的特别之处

### 7.1 静态缓冲区 + Padding 策略

代码在 `cuda_graph_buffer_registry.py`。

补全出来的空位（padding）图会**真的计算**（kernel 规模是捕获时的尺寸），所以空位必须填无害值，否则除零/越界/野地址。

抽象出 `GraphSlot`：每个 ForwardBatch 字段对应一个静态缓冲区槽位，带**填充策略（PaddingPolicy）**：

| 填充策略 | 含义 | 用在哪些字段 |
|---|---|---|
| 直接拷贝头部 | 只拷真实部分，尾部不管 | `input_ids` |
| 填哨兵值 | 尾部填安全值（序列长度填非零避免除零） | `seq_lens` |
| 填零 | 尾部填 0（假注意力读安全地址） | `out_cache_loc`、`positions` |
| 只填一次 | 分配时填一次，之后不重置 | `encoder_lens` |

这是 SGLang 能稳定跑大规模混合 batch 的基础。

### 7.2 全局共享显存池

代码在 `runner_utils/pool.py`。

prefill 图和 decode 图共用同一个显存池。因为两个阶段**从不同时回放**，只需保留较大阶段的显存，省一半。

### 7.3 反向捕获

`reversed(self.capture_bs)`。大图先捕获占大块池，小图复用，显存不碎片化。

### 7.4 三种图策略（Backend）并存

把"图怎么切"抽象成可插拔后端，同一套 runner 逻辑可换不同图策略：

**(1) Full（整图）—— decode 主力。**
整个 forward 一张图，注意力元数据也捕进图里。最快最省。代价是元数据不能变，只能 padding 掩盖差异。

**(2) Breakable（分段图）—— prefill 主力。**
（`breakable_cuda_graph.py`）

- 在 attention / mamba 层边界把 forward 切开
- 每段捕获一张小图，段间跑普通代码（eager）
- 关键：注意力元数据计算放在段间普通代码里，不进图，回放时用真实 batch 现场算 → 支持多请求 prefill 一次算完

机制：装饰器 `eager_on_graph` 标记断点函数：

```
开始捕获第1段 → 可图化层 → [断点] 结束第1段 → 普通执行断点函数 → 开始捕获第2段 → ...
```

回放：

```python
for i, seg in enumerate(segments):
    seg.replay()       # 重放图段 i
    if i < len(break_fns):
        break_fns[i]() # 普通执行断点函数，结果拷回静态 buffer
```

还 hook 了 `torch.cuda.Stream.wait_stream`，跟踪 fork 出去的副流，保证断点处把副流收回（否则 `capture_end` 失败）。

**(3) TcPiecewise（编译分段图）。**
用 `torch.compile` 的 FX 图拆图（在 MoE 分发等分割点切开），每段编译后各自捕获。主要服务 ROCm / NPU 等非 NVIDIA 平台。

### 7.5 图去重（dedup）

代码在 `cuda_graph_dedup_mixin.py`。

不同尺寸的图若结构完全一样只是参数不同，用 CUDA driver API 遍历节点和边算"拓扑签名"。签名相同的图用 `cudaGraphExecUpdate` 复用同一个可执行图，省显存和初始化时间。

### 7.6 预热的全局门控

`base_runner.py` 的 `warmup()` 有 `_kernel_warmed_up` 标志：**整个进程只预热一次**（decode 和 prefill 两个 runner 共享）。两个阶段用的 kernel 大部分相同，预热一次即可，避免重复 JIT 编译和 autotune。

### 7.7 兜底机制：eager runner 永远在场

`cuda_graph_setup.py` 里 eager runner 永远先创建，两个角色：

1. CUDA Graph 完全禁用时直接用它
2. 某 batch 不满足图条件时退回 eager

保证 CUDA Graph 是纯加速——**最坏情况只是变慢，不影响正确性**。

---

## 八、一张图总结

```
【启动阶段：捕获】
配置(哪个阶段用哪种图、捕获哪些尺寸)
  → 进入图模式(独立流 + 通信组占位 + 冻结GC)
  → 按尺寸从大到小:
       造假数据(静态缓冲区) → 预热两遍 → 真捕获一遍 → 存"尺寸→图"表

【服务阶段：回放】
每个 batch:
  → 准入检查(不满足就退回 eager 兜底)
  → 向上取整到最近的捕获尺寸
  → 把真实数据填进静态缓冲区(按字段的填充策略处理补全尾部)
  → 一次 replay 重放整张图
  → 把输出切回真实长度
```

---

## 九、源码阅读

1. `runner_backend/full_cuda_graph_backend.py` —— 最短最纯粹，建立"预热→捕获→回放"直觉
2. `runner/decode_cuda_graph_runner.py` —— 看 `capture` / `capture_one_shape` / `load_batch` / `execute` 串起全流程
3. `cuda_graph_buffer_registry.py` —— 静态缓冲区和填充策略
4. `runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py` —— 分段图 API 层实现
5. `runner_utils/pool.py` + `cuda_graph_dedup_mixin.py` —— 显存优化

---
