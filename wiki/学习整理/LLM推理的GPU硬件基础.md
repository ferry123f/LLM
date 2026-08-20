# LLM 推理的 GPU 硬件基础

> 一句话：现代 GPU 的**算力远远超过带宽**（H100 约 295 FLOP/Byte），所以 LLM 推理里「更快」几乎总是「**搬更少字节**」的同义词——理解显存怎么被吃掉、带宽怎么成为上限，就理解了 continuous batching、KV cache 优化、量化、PD 分离这一整套技术为什么存在。

## 一、总纲：算力与带宽的失衡

先记住一个数——**H100 的算力带宽比**：

```
989 TFLOPS (BF16 稠密)  ÷  3.35 TB/s  ≈  295 FLOP/Byte
```

意思是：**从显存每读 1 字节，必须做够 295 次浮点运算，才能喂饱算力。** 达不到，GPU 就在干等数据。

这个比值是 **roofline 模型的脊点（ridge point）**，它把所有 kernel 分成两类：

| | 算术强度（FLOP/Byte） | 瓶颈 | 优化方向 |
|---|---|---|---|
| **compute-bound** | > 295 | 算力 | 更好的 kernel、更低精度 |
| **memory-bound** | < 295 | 带宽 | **减少搬运字节** |

### LLM 推理的两个阶段分踞两端

这是全篇最重要的一点：

| 阶段 | 一次处理 | 算术强度 | 性质 |
|---|---|---|---|
| **Prefill** | 几百~几万 token 一起 | 高 | **compute-bound** |
| **Decode** | 每个请求 1 个 token | ≈ batch size | **memory-bound** |

decode 时每个权重从显存读出来，只服务 batch 里的 batch_size 个 token，所以**算术强度约等于 batch size**。想让 decode 摆脱带宽瓶颈，batch 得堆到脊点（H100 约 295）以上。

> **这就是 continuous batching 存在的根本原因**，也是 [[SGlang]] 里 `get_next_batch_to_run` 拼命组批的意义：不是为了省事，是为了把算术强度从个位数拉到几百，让 GPU 从「等数据」变成「在算」。
>
> 也解释了为什么 **prefill 和 decode 该分开部署**（PD 分离）——两个阶段的硬件诉求根本相反。

## 二、显存：装什么，怎么算

```
总显存 = 模型权重 + KV cache + 激活值 + 框架/CUDA 开销
         └ 固定 ┘   └ 随并发和长度增长 ┘  └ 临时 ┘  └ ~1GB ┘
```

### 2.1 权重

```
权重字节 = 参数量 × 每参数字节数
```

| 精度 | 字节/参数 | 7B | 70B | 671B |
|---|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB | — |
| **FP16/BF16** | 2 | **14 GB** | **140 GB** | 1.3 TB |
| FP8 | 1 | 7 GB | 70 GB | 671 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB | 336 GB |

**MoE 要特别注意**：DeepSeek-V3 是 671B 总参数、37B 激活参数。**显存按总参数算（全都要装下），算力按激活参数算**。所以 MoE 是「吃显存、省算力」的架构，推理特别依赖大显存和高带宽。详见 [[DeepSeek-V3 架构与低成本高效训练]]。

### 2.2 KV cache（真正的变量）

```
每 token 字节 = 2(K和V) × 层数 × KV头数 × head_dim × 精度字节
```

以 Llama-3-70B 为例（80 层、8 个 KV 头 GQA、head_dim 128、FP16）：

```
2 × 80 × 8 × 128 × 2 = 327,680 B ≈ 320 KB / token
```

- 8K 上下文的**一个**请求：**2.5 GB**
- 并发 **32** 个这样的请求：**84 GB** —— 比模型权重（140GB）已是同一量级

**注意力架构直接决定这个数**，差距是数量级的：

| 架构 | KV 头数 | 相对 KV 大小 | 代表模型 |
|---|---|---|---|
| **MHA** | = 查询头数（如 64） | 1× | Llama-2 |
| **GQA** | 分组共享（如 8） | **1/8** | Llama-3、Qwen |
| **MQA** | 1 | 1/64 | Falcon |
| **MLA** | 压成低秩潜向量 | **~1/10 甚至更低** | DeepSeek |

> KV cache 是长上下文和高并发的**头号瓶颈**。整个推理优化史的一大半都在跟它较劲：
> - **PagedAttention** —— 分页消除碎片
> - **RadixCache** —— 前缀去重复用（见 [[SGlang]]）
> - **KV 量化** —— FP8/INT8 存 KV
> - **offload / HiCache** —— 挪到 CPU 内存或 SSD
> - **MLA** —— 从架构层面把它压小（见 [[DeepSeek-V3 架构与低成本高效训练]]）

### 2.3 实践：显存怎么划

SGLang 的 `--mem-fraction-static` 就是划这条线的：**先给权重和静态结构留出比例，剩下的全部作为 KV pool**。设太高会 OOM，设太低则 KV pool 小、并发上不去。部署时第一个要调的参数。

## 三、带宽与内存层级

### 3.1 单流 decode 速度有硬上限

```
单请求 decode 速度上限 (tok/s) = HBM 带宽 ÷ 模型权重字节
```

因为**每生成一个 token，全部权重都要从显存过一遍**。

70B FP16（140GB）在 H100（3.35 TB/s）上：

```
3350 GB/s ÷ 140 GB ≈ 24 tok/s
```

这个估算和实测非常接近。三个有用的推论：

- **换更快的卡**（H200，4.8 TB/s）→ ~34 tok/s
- **量化到 INT4**（35GB）→ ~95 tok/s ← **量化加速 decode 的真正原理：不是算得快了，是读得少了**
- **堆 batch 不提升单流速度**，但总吞吐线性涨（权重只读一次，服务更多请求）

### 3.2 内存层级

| 层级 | 容量（H100） | 延迟 | 带宽 |
|---|---|---|---|
| 寄存器 | 256 KB/SM | ~1 cycle | 极高 |
| **Shared Memory / L1** | 228 KB/SM | ~20-30 cycle | ~10+ TB/s |
| **L2** | 50 MB | ~200 cycle | ~5 TB/s |
| **HBM（显存）** | 80 GB | ~400-800 cycle | 3.35 TB/s |
| CPU 内存（PCIe 5.0） | TB 级 | ~微秒 | ~64 GB/s |
| SSD | 更大 | 毫秒 | ~7 GB/s |

**每往下一级，带宽掉一个量级。** 这解释了两件事：

1. **FlashAttention 为什么快** —— 它不减少计算量，只是把注意力计算切块塞进 SMEM，避免把巨大的中间矩阵写回 HBM 再读出来。**省的是 HBM 往返**。
2. **KV offload 的账** —— 挪到 CPU 内存要过 PCIe（64 GB/s，比 HBM 慢 50 倍），但**仍远快于重新 prefill**。这就是分层缓存成立的理由。

### 3.3 附：为什么索引用 int32 而非 int64

memory-bound 的机器上，「快」= 「搬更少字节」。**int32 并不比 int64「每次访问更快」——单位带宽相同；快在同样一批数据只占一半字节。** 收益有四处：

1. **DRAM 流量减半** —— 最主要
2. **访存事务减半** —— 一个 warp 读 32 个 int32 = 128B = 1 个事务；int64 = 256B = 2 个事务
3. **L2 命中率提升** —— 同样的 L2 能装两倍条目
4. **寄存器压力降低** —— int64 占 2 个 32-bit 寄存器，int32 占 1 个，**occupancy 更高**

另外 **64-bit 整数算术本身也可能慢**：加乘要拆成多条 32-bit 指令走进位链，**除法/取模尤其慢**——而地址计算里 `idx / page_size`、`idx % page_size` 极常见。

所以 SGLang 里索引类张量（`kv_indptr`、`kv_indices`、`req_to_token`）全是 int32：

```python
# memory_pool.py — req→token 索引表
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len), dtype=torch.int32, device=device
)
```

4096 slot × 128K 上下文下，int32 是 2.1 GB，int64 要 4.3 GB —— **省下的 2.1 GB 全部可以拿去放 KV cache**。这里 int32 的收益其实是**显存容量 > 访存速度**。

**但该用 int64 的地方仍要用**：单调递增的代际计数器（`req_generation`）有溢出风险，且数组本身很小，省不出什么。

> **判断方法**：这个张量是不是「**大 + 被反复访问**」？是则 int32 收益明显；否则不值得为它引入溢出风险。

## 四、算力与精度

### 4.1 计算单元层级

```
GPU → SM（H100 有 132 个）→ warp（32 线程，调度的最小单位）→ 线程
                            ↘ Tensor Core（矩阵乘专用单元）
```

- **CUDA Core** 做通用标量运算
- **Tensor Core** 做矩阵乘累加，是算力数字的来源。**LLM 推理 99% 的算力来自 Tensor Core**
- **occupancy（占用率）** = 每 SM 驻留 warp 数 / 上限。warp 越多越能靠切换掩盖访存延迟；寄存器用太多会压低 occupancy

### 4.2 精度格式

| 格式 | 位宽 | 用途 |
|---|---|---|
| FP32 | 32 | 基本不用于推理 |
| TF32 | 19 | NVIDIA 私有，训练用 |
| **BF16** | 16 | **推理默认**，动态范围同 FP32 |
| FP16 | 16 | 动态范围小，易溢出 |
| **FP8 (E4M3/E5M2)** | 8 | Hopper 起支持，推理主流方向 |
| INT8 / INT4 | 8/4 | 权重量化（GPTQ/AWQ） |
| FP4 | 4 | Blackwell 新增 |

> **读规格书的坑**：厂商标的 TFLOPS 常常是「**含稀疏（with sparsity）**」的数字，实际稠密算力**要除以 2**。看到「H100 1979 TFLOPS BF16」，稠密是 989。

## 五、多卡互联与并行策略

单卡装不下时要拆模型，而**拆法的代价完全由互联带宽决定**：

| 互联 | 带宽 | 场景 |
|---|---|---|
| **NVLink 4（H100）** | 900 GB/s | 卡间直连 |
| **NVSwitch** | 全互联 | 节点内 8 卡任意两两 |
| **PCIe 5.0 x16** | 128 GB/s | 无 NVLink 的卡 |
| **InfiniBand NDR** | 400 Gb/s ≈ 50 GB/s | 跨节点 |

### 并行策略的通信代价

| 策略 | 怎么拆 | 通信 | 对互联要求 |
|---|---|---|---|
| **TP（张量并行）** | 每层内部横切 | **每层 2 次 all-reduce** | **极高，必须 NVLink** |
| **PP（流水并行）** | 按层纵切 | 层边界传激活值 | 低，PCIe 可接受 |
| **EP（专家并行）** | MoE 专家分卡 | all-to-all 分发 token | 高 |
| **DP（数据并行）** | 整个模型复制 | 几乎不通信 | 低（但每卡要装下全模型） |

**关键直觉**：TP 每层都要同步，通信频繁且在关键路径上——**跨节点做 TP 会灾难性掉速**。所以典型部署是「**节点内 TP（走 NVLink）+ 节点间 PP/DP（走 IB）**」。

## 六、常见卡对照

| 型号 | 显存 | 带宽 | BF16 稠密算力 | 互联 |
|---|---|---|---|---|
| **B200** | 192 GB HBM3e | ~8 TB/s | ~2.2 PFLOPS | NVLink 5 |
| **H200** | 141 GB HBM3e | 4.8 TB/s | 989 TF | NVLink 900 GB/s |
| **H100 SXM** | 80 GB HBM3 | 3.35 TB/s | 989 TF | NVLink 900 GB/s |
| **H100 PCIe** | 80 GB HBM2e | 2.0 TB/s | ~756 TF | PCIe |
| **A100 80G** | 80 GB HBM2e | 2.0 TB/s | 312 TF | NVLink 600 GB/s |
| **L40S** | 48 GB GDDR6 | 864 GB/s | ~181 TF | **无 NVLink** |
| **RTX 4090** | 24 GB GDDR6X | 1.0 TB/s | ~165 TF | **无 NVLink** |
| **RTX 5090** | 32 GB GDDR7 | 1.79 TB/s | — | 无 NVLink |

> **消费卡的两个致命短板**：显存小 + 没有 NVLink。4090 跑 70B 要 4 张，而 TP 只能走 PCIe，通信开销会吃掉大部分收益。
>
> 国产卡（昇腾 910B 等）各 SKU 差异大、公开数据口径不一，此处不列具体数字，以厂商文档为准。SGLang 对国产芯片的适配情况见 [[SGlang]] 的「国产 GPU/NPU 适配文件清单」章节。

## 七、实战：容量规划怎么算

**问：Llama-3-70B、8K 上下文、要支持 32 并发，需要什么配置？**

```
① 权重  70B × 2 B                    = 140 GB
② KV    320 KB/tok × 8192 × 32       ≈  84 GB
③ 激活 + 框架开销                     ≈  10 GB
                                  总计 ≈ 234 GB
```

→ **4×H100（320 GB）**，TP=4 走 NVLink。剩余约 86 GB 作为 KV pool 冗余。

**追问：2×H100（160 GB）能跑吗？**

权重 140 GB 就几乎占满，KV 只剩不到 20 GB → 并发撑死几个。**要么量化到 FP8**（权重 70 GB，KV 剩 90 GB，可行），**要么加卡**。

这套算法就是容量规划的全部。

## 八、速查公式

```
权重字节             = 参数量 × 精度字节
KV/token 字节        = 2 × 层数 × KV头数 × head_dim × 精度字节
单流 decode 速度上限  = HBM 带宽 ÷ 权重字节
脱离带宽瓶颈的 batch  ≈ 算力 ÷ 带宽（H100 BF16 约 295）
Prefill FLOPs        ≈ 2 × 参数量 × token 数
```

## 九、硬件事实 → 框架对策

| 硬件事实 | SGLang 里的对策 |
|---|---|
| decode 是 memory-bound，算术强度 ≈ batch | `get_next_batch_to_run` 连续组批 |
| KV cache 是显存头号消耗 | **RadixCache 前缀复用**、分页分配、`evict` |
| 显存有限、超了就崩 | `--mem-fraction-static`、`retract`（踢回请求） |
| HBM 往返昂贵 | FlashAttention / FlashInfer 等 attention backend |
| 索引搬运也占带宽 | 到处的 **int32** |
| TP 通信在关键路径上 | 节点内 TP、`distributed/device_communicators/` |
| prefill 与 decode 性质相反 | **PD 分离**（`srt/disaggregation/`） |
| CPU 内存比重算便宜 | HiCache 分层 KV offload |

## See Also

- [[SGlang]] — 本文的硬件约束在框架层的具体实现：组批、RadixCache、attention backend、PD 分离。
- [[LLM推理压测-bench serve 与 throughput 参数详解]] — 用压测验证本文的估算：TTFT 反映 prefill（compute-bound），TPOT/ITL 反映 decode（memory-bound）。
- [[DeepSeek-V3 架构与低成本高效训练]] — MLA 如何从架构层面把 KV cache 压小；MoE「吃显存省算力」的典型。

> 三篇构成闭环：**硬件原理（本文）→ 框架实现（SGlang）→ 压测验证（压测详解）**。

## 备注

- 规格数字取自厂商公开资料，**均为稠密算力**（已剔除 sparsity 翻倍）；具体 SKU 与固件版本会有差异，以实测为准。
- 延迟 cycle 数为量级参考，随架构与访问模式变化较大。
- 容量规划的估算未计入 CUDA context、通信 buffer 等零碎开销，实际部署建议再留 5–10% 余量。
