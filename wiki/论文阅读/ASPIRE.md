
# ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference

- **论文**：[ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference](https://github.com/Amir-zsh/ASPIRE)
- **作者**：Amir Ziashahabi*, Hossein Entezari Zarch*, Lei Gao, Murali Annavaram, Salman Avestimehr（* 共同一作，USC）
- **出处**：SCALE Workshop @ ICML 2026
- **核心结论**：3 个模型、5 个基准上，解码吞吐是自回归的 **1.70–4.58×**，一致优于此前的自推测方法（MagicDec、SpecAttn）。

---



LLM 生成 token 时，每生成一个新 token 都要把**之前所有 token 的 KV cache 读一遍**。上下文越长，这一步读的数据越多，越慢（因为卡在显存带宽，不是卡在算力）。

**投机解码（Speculative Decoding）** 的核心思想：用"便宜的猜测"换"少读几次贵的数据"。

```
① 起草(Draft)：用一个"便宜的模型"连续猜 γ 个 token
② 验证(Verify)：用"真正的模型"一次性检查这 γ 个猜得对不对
③ 一次验证能验收 γ 个 token = 用 1 次全量读取换 γ 个 token
```

ASPIRE 属于**自推测**：起草和验证用的是**同一个模型**，区别只是——起草时让它只读一小部分 KV（稀疏注意力，便宜），验证时读全部 KV（全注意力，贵）。

**ASPIRE 要解决两个问题**：

| 问题 | 解决它的组件 |
|---|---|
| 一个 batch 里所有请求被迫"同步锁步"（要么一起起草、要么一起验证） | **§4.1 统一混合前向** |
| 每个请求该起草多长再验证（草稿太长会猜错、太短又没省够时间） | **§4.2 推测调度器** |
| 起草时用的稀疏上下文越用越"陈旧"，导致越猜越不准 | **§4.3 起草内刷新层** |

下面把这三块**一步一步拆到底**。

---

## 1. 背景：为什么长上下文推理慢

### 1.1 compute-bound 与 memory-bound

- **算术强度 = FLOPs / 访存字节数**。
- 算术强度高 → **compute-bound**（卡算力）；低 → **memory-bound**（卡显存带宽）。
- 解码中：**MLP 是 compute-bound**（权重读取能被 batch 摊销 → GEMV 变 GEMM）；**attention 是 memory-bound**（每个请求的 KV 都不同，无法摊销）。

### 1.2 量化估算（8B 模型，示意）

```
每个 token 每层的 KV = 8头 × 128维 × 2(K,V) × 2字节 ≈ 4 KB
每个 token 总 KV     = 4KB × 36层 ≈ 144 KB
100K 上下文          = 100000 × 144KB ≈ 14.4 GB   ← 每生成 1 个 token 的 attention 读取量
```

算 1 个 token 的注意力只要 ~30 GFLOP（约 0.03ms），读 14.4GB 却要 ~4.4ms → **访存是算力的上百倍**，而且 batching 救不了 attention（每请求 KV 独立）。

### 1.3 自推测解码的完整流程

1. **起草**：用稀疏 attention 连续生成 γ 个草稿 token。
2. **验证**：用全 attention 一次性验证这 γ 个草稿。
3. 一次验证验收 γ 个 token = 用 1 次全量 KV 读取换 γ 个 token。

**无损性**：验证时只接受"与全 attention 输出一致"的 token，所以输出分布严格等于目标模型（贪心下逐 token 一致；采样下分布等价）。这个"无损"是拒绝采样保证的，**与接受率、草稿模型好坏都无关**。

### 1.4 已有方法的两个缺陷

- **MagicDec**：window attention 起草，稀疏规则死板。
- **SpecAttn**：用验证时的 attention 分数选稀疏上下文，但（1）稀疏上下文只在验证边界更新一次、起草越久越陈旧；（2）batch 内所有请求共用同一套 draft/verify 节奏，锁步前进。

---

## 2. 三个动机实验（为什么需要 ASPIRE）

| 观察 | 做法 | 结论 |
|---|---|---|
| ① 最优草稿长度因请求而异 | 256 个 LongBench 样本，测每个请求"能接受多长的草稿" | 范围 2.56–20.00，均值 11.51，分布很宽 → 需要 per-request 调度 |
| ② 草稿长度随时间变化 | 热力图看每请求跨验证轮次的波动 | 同一请求内也大幅波动 → 需要 per-step 调度 |
| ③ 跨层信号比时间复用稳定 | 比较"时间复用 Δt"vs"跨层 Δℓ"预测注意力 | 跨层退化慢得多 → 可用单层全注意力刷新上下文 |

---

## 三个核心组件总览（论文 Figure 2）

![[Pasted image 20260921163603.png|700]]

```
§4.2 调度器 = 大脑：决定谁起草、谁验证、起草多长
§4.1 混合前向 = 执行器：两类请求同一次前向跑完
§4.3 刷新层 = 眼睛：起草时睁全眼看一眼，选出该看的page
```

---

## 3. 组件一：统一混合前向（§4.1 Unified Mixed Forward）

### 3.1 它解决什么问题：同步锁步

没有它之前，旧方法（MagicDec/SpecAttn）的 batch 是**齐步走**的：

```
阶段1：全体起草（全体做稀疏 attention）
阶段2：全体验证（全体做全量 attention）
阶段1：全体再起草
阶段2：全体再验证
...
```

两个坏处：
1. **GPU 利用率差**：起草阶段只干便宜活，验证阶段只干贵活，两段时间被硬切开；
2. **节奏不匹配**：但每个请求最优草稿长度不同（§2 实测 2.56~20 不等），锁步逼所有人用同一节奏，必然有人欠起草、有人过起草。

### 3.2 核心思想：两类请求塞进"同一次前向"

统一混合前向让**同一时刻，有的请求在起草、有的请求在验证**：

```
一次前向里：
  请求 A（起草中）→ 稀疏 attention，解码 1 个草稿 token
  请求 B（验证中）→ 全量 attention，解码 d_i+1 个 token
  请求 C（验证中）→ 全量 attention，解码 d_j+1 个 token
  ...
```

**为什么能做到**：因为起草和验证用的是**同一个模型、同一套权重**，唯一的区别只有两点——attention 上下文不同（稀疏 vs 全量）、解码长度不同（1 vs d_i+1）。这两个区别用元数据就能表达，所以能合并成一次前向，MLP/权重加载完全共享。

### 3.3 具体做法：token 怎么排平

假设一个 batch 有 B 个请求，每个请求本轮解码长度不同：

```
请求 0（起草）：  解码 1 个 token
请求 1（验证）：  解码 d_1+1 个 token
请求 2（验证）：  解码 d_2+1 个 token
...
```

把所有要解码的 token **首尾相接**拼成一条 1D 序列：

```
idx (T,) = [请求0的1个] ++ [请求1的d_1+1个] ++ [请求2的d_2+1个] ++ ...
```

用一个**边界数组**记清楚"哪个 token 属于哪个请求"：

```
qo_indptr (B+1,) = [0, 1, 1+d_1+1, 1+d_1+1+d_2+1, ...]
                    ↑   ↑        ↑
                 请求0  请求1    请求2 的起点
```

比如 B=3，请求 0 解码 1 个、请求 1 解码 4 个、请求 2 解码 2 个：

```
idx       = [t0, t1, t2, t3, t4, t5, t6]    （共 T=7 个 token）
qo_indptr = [0, 1, 5, 7]
              ↑  ↑  ↑  ↑
              |  请求0起=0，请求1起=1，请求2起=5，末尾=7
```

这样请求 1 的 token 就是 `idx[1:5]`（t1~t4），请求 2 的 token 就是 `idx[5:7]`（t5~t6）。

### 3.4 四份元数据（区分两类请求的关键）

| 元数据          | 形状       | 回答的问题                        |
| ------------ | -------- | ---------------------------- |
| `mode_flags` | `(B,)`   | 这行是起草(0)还是验证(1)？             |
| `qo_indptr`  | `(B+1,)` | 哪个 token 属于哪个请求？             |
| `offsets`    | `(T,)`   | 每个 token 的绝对位置（喂给 RoPE）      |
| page 表       | `(B, ·)` | 注意力读哪几页（稀疏 70 页 / 全量 1000 页） |

### 3.5 注意力怎么路由：page 表

**page 表（page table）是什么**：KV cache 按"页"（每页 16 个 token）组织，存在一个连续的 KV 池里。page 表就是一个**存"页号"的整数数组**——它本身不是 K/V 数据，只是"借书单"，告诉注意力内核"去书架（KV 池）拿哪几页"。

- **全量上下文**：page 表里存全部 1000 个页号；
- **稀疏上下文**：page 表里只存选出来的 70 个页号。

注意力内核的流程：对每个 token 查 `qo_indptr`（属于哪个请求）→ 查 `mode_flags`（起草还是验证）→ 决定查哪张 page 表：

```
起草行 ── draft_page_table ──► 稀疏 70 页
验证行 ── target_page_table ─► 全量 1000 页
```

### 3.6 一个具体例子串起来

batch 里 3 个请求：请求 0 在起草，请求 1、2 在验证（分别已起草 3、6 个 token）。

```
mode_flags      = [0, 1, 1]
verify_dec_lens = [1, 4, 7]          （起草行=1，验证行=d_i+1）
idx             = [t0, t1 t2 t3 t4, t5 t6 t7 t8 t9 t10 t11]   （共 12 个）
qo_indptr       = [0, 1, 5, 12]
offsets         = [100, 101 102 103 104, 200 201 202 203 204 205 206]
```

GPU 里注意力内核：
- t0 属于请求 0（起草）→ 查 draft_page_table → 只读 70 页；
- t1~t4 属于请求 1（验证）→ 查 target_page_table → 读全部 1000 页；
- t5~t11 属于请求 2（验证）→ 读全部 1000 页。

---

## 4. 组件二：推测调度器（§4.2 Speculation Scheduler）

### 4.1 它解决什么问题

每个请求每步该"继续起草"还是"该验证"——本质是**选一个最优草稿长度 γ**。γ 太短：没省够时间；γ 太长：草稿猜错的多，白验证。

### 4.2 目标：单位时间产出多少 token

调度器想让**吞吐最大**，而吞吐 = 平均每个时间单位产出的 token 数：

```
效率 = 期望产出的 token 数 ÷ 期望花费的时间
```

所以要算两件事：**分子（起草 γ 个能验收几个 token）**、**分母（要花多少时间）**。

### 4.3 分子：E[tokens] 怎么一步步推出来

**先彻底搞懂"一次验证到底产出几个 token"**：

规则：起草 γ 个 token，验证后提交的数量 = **被接受的草稿数 + 1**。那个"+1"是验证器自己算出的下一个 token，**永远会被提交**（因为它是真模型亲手算的）。

用 γ=3 的例子，把每个结局摆出来（α = 每个草稿被接受的概率）：

| 结局 | 接受数 | 提交数 | 概率 |
|---|---|---|---|
| d₁ 就错了 | 0 | 0+1=1 | 1−α |
| d₁ 对，d₂ 错 | 1 | 1+1=2 | α(1−α) |
| d₁、d₂ 对，d₃ 错 | 2 | 2+1=3 | α²(1−α) |
| d₁、d₂、d₃ 全对 | 3 | 3+1=4 | α³ |

**用"每个位置贡献多少"算期望（最省力的理解）**：

一次验证有 γ+1 个"候选提交位置"，逐个问"会不会被提交"：

```
位置 0（验证器给的 bonus token）：永远提交          → 贡献 1
位置 1（草稿 d₁）：d₁ 被接受才提交                  → 贡献 α
位置 2（草稿 d₂）：d₁ 和 d₂ 都被接受才提交          → 贡献 α²
位置 3（草稿 d₃）：d₁、d₂、d₃ 都被接受才提交        → 贡献 α³
```

期望值 = 每个位置的贡献之和（期望的线性性）：

```
E[tokens] = 1 + α + α² + α³
```

**严谨验证（逐项代数）**：用上面的结局表直接算期望 = Σ(提交数 × 概率)：

```
E = 1·(1−α) + 2·α(1−α) + 3·α²(1−α) + 4·α³
```

逐项拆开（把 k·α^{k-1}(1−α) 拆成 k·α^{k-1} − k·α^k）：

```
= (1 − α) + (2α − 2α²) + (3α² − 3α³) + 4α³
```

合并同次幂的系数：

```
α⁰：1          = 1
α¹：−1+2       = 1
α²：−2+3       = 1
α³：−3+4       = 1
```

**每一项系数都是 1**，所以 `E = 1 + α + α² + α³` ✓

**推广到任意 γ**：

```
E[tokens] = 1 + α + α² + ... + α^γ = (1 − α^{γ+1}) / (1 − α)   （等比数列求和，α≠1）
```

### 4.4 分母：E[time]

一个完整周期 = **γ 次起草 + 1 次验证**：

```
总时间 = γ·T_draft + T_verify = T_verify · (1 + γ·T_draft/T_verify) = T_verify · (1 + c·γ)
```

其中 `c = T_draft / T_verify`（起草一次相当于验证一次的几分之一）。c 越小，起草越便宜。

### 4.5 合起来：效率公式 + argmax + 算例

```
效率(γ) = E[tokens] / E[time]
        = [ (1 − α^{γ+1}) / (1 − α) ] / [ T_verify · (1 + c·γ) ]
```

`T_verify` 是公共常数（对所有 γ 都一样），比大小时约掉：

```
γ* = argmax_{γ ∈ {0,…,d_max}}   (1 − α^{γ+1}) / (1 − α)
                                ─────────────────────────
                                      1 + c·γ
```

**完整算例（α=0.9, c=0.1）**：

| γ | 分子（E[tokens]） | 分母（1+cγ） | 效率 |
|---|---|---|---|
| 0 | 1.00 | 1.0 | 1.000 |
| 2 | 2.71 | 1.2 | 2.258 |
| 4 | 4.10 | 1.4 | 2.925 |
| 6 | 5.22 | 1.6 | 3.261 |
| 8 | 6.13 | 1.8 | 3.403 |
| **10** | **6.86** | **2.0** | **3.431 ← 最大** |
| 12 | 7.46 | 2.2 | 3.390 |

γ 从 0 增大，效率**先升后降**，峰值在 γ=10。为什么先升后降：分子（token 数）涨得越来越慢（α^γ 指数衰减），分母（时间）线性涨，最后分子几乎不涨了、分母还在涨，效率掉头。

### 4.6 α 怎么在线估计（每次验证后更新）

```
α̂ = a_i / min(a_i + 1, d_i)      # 部分接受：a_i 次成功 + 1 次失败；全接受：=1
α ← ω·α + (1−ω)·α̂                 # 指数平滑，ω=0.8
```

- 例：起草 10 个、接受 9 个 → `α̂ = 9/min(10,10) = 0.9`；
- 起草 10 个、接受 5 个 → `α̂ = 5/min(6,10) = 5/6 ≈ 0.833`。

### 4.7 完整决策循环（Algorithm 1）

```
1. if d_i ≥ d_max → VERIFY
2. if 没见过验证结果 且 d_i < γ_init → DRAFT
3. 用成本模型算 T_draft_i、T_verify_i
4. c_i ← T_draft_i / T_verify_i
5. γ_i ← argmax_γ 效率公式
6. if d_i ≥ γ_i → VERIFY 否则 → DRAFT
```

**一个请求从头 trace 一遍**（α=0.9, c=0.1, γ_init=3, d_max=16）：

- 先算 γ*：代入公式 → **γ\*=10**。
- 第 1~3 步：没见过验证结果且 d_i<3 → 强制起草（d_i: 0→1→2→3）。
- 第 4~10 步：d_i < 10 → 继续起草（d_i: 3→…→10）。
- 第 11 步：d_i=10 ≥ 10 → **验证**。假设接受 9 个 → 提交 10 个 token → 更新 α 仍是 0.9 → γ* 仍是 10 → 循环。

**α 一变 γ 就跟着变**（自适应的精髓）：如果后来进入难段，起草 10 个只接受 5 个 → α 降到 0.887 → γ* 变 9；接受率继续掉 → γ* 继续缩到 8、4……反过来全接受则 γ* 变长（最多到 d_max）。

### 4.8 思考：IID α 假设的局限

公式假设"每个位置接受率都是 α、相互独立"。**严格不成立**：

- 接受率真实地**随起草位置衰减**（第 1 个草稿最准、第 10 个最不准）。论文自己的图 3 就证明了这点：γ=1 时接受率 ~95%，γ=10 时 76.6%（有刷新层）/ 60.3%（无刷新层）。
- 但该假设**只影响 γ 选多大（启发式），不影响无损性**（无损靠拒绝采样，与 α 无关）。
- 代价：会轻微系统性偏"过度起草"，靠反馈闭环 + d_max + 刷新层兜底。

---

## 5. 成本模型：c 从哪来

### 5.1 为什么需要它 + 三步线性模型

调度器需要 `c_i = T_draft/T_verify`，但这两个耗时取决于 GPU/模型/batch/上下文长度，不能写死。所以建一个**能预测一次前向耗时的模型**：

```
τ_t = β_model + β_mlp·n_t + β_attn·( Σ_{j∈D_t}|S_j| + Σ_{j∈V_t}ℓ_j )
```

| 项 | 含义 | 为什么线性 |
|---|---|---|
| β_model | 固定开销（内核启动等） | 与负载无关 |
| β_mlp·n_t | MLP 耗时，n_t = 本步前向 token 总数 | compute-bound，∝ FLOPs ∝ token 数 |
| β_attn·(总KV) | attention 耗时，总KV = 本步读的 KV 长度 | memory-bound，∝ 字节 ∝ KV 长度 |

### 5.2 怎么标定（两步法，离线最小二乘）

```bash
# 第一遍：跑 FSM 采集每步耗时（FSM 不需要成本模型）
python benchmark.py --method aspire-fsm --fit-calibration calibration.json
# 第二遍：加载标定跑完整 ASPIRE
python benchmark.py --method aspire --calibration calibration.json
```

第一遍每步记 5 个数（`TimingSample`）：

```
forward_ms          这一步实际花了多少毫秒      ← 目标 y
active_requests     活跃请求数 B
extra_decode_tokens 验证行多出来的 token 数 = Σ(d_i)
draft_requests      起草行数量
verify_kv_tokens    验证行读的 KV 总量
```

每个样本构造成一行特征 `[1, n_t, 总KV]`：

```
特征1 = 1                                   → β_model
特征2 = active + extra = n_t                → β_mlp（前向 token 总数）
特征3 = verify_kv + draft × draft_cache     → β_attn（总 KV 读取）
```

（验证特征2：`active + extra = n_draft + n_verify + Σ(d_i) = n_draft·1 + n_verify·(d_i+1)` = 前向 token 总数 ✓）

所有样本堆成矩阵 X、y，解最小二乘：

```
β = (XᵀX)⁻¹ Xᵀ y     （代码里 np.linalg.lstsq(x, y)）
```

校验：rank=3（特征要线性无关）、β 非负。再算 r²、rmse，连同 β 存进 `calibration.json`。

### 5.3 怎么算 c_i（同质 batch 近似）

假设"B 个槽位全是请求 i 的副本"，代入模型：

```
T_draft_i  = β_model + β_mlp·B         + β_attn·B·|S_i|
T_verify_i = β_model + β_mlp·B·(d_i+1) + β_attn·B·ℓ_i
c_i = T_draft_i / T_verify_i
```

随 `d_i`、`ℓ_i`、`|S_i|` 变化，**每步在线重算**。

**完整算例**（示意 β）：β_model=0.3ms、β_mlp=0.02ms/token、β_attn=0.0001ms/KV-token；B=24、d_i=8、|S_i|=70页×16=1120、ℓ_i=16000：

```
T_draft  = 0.3 + 0.02×24     + 0.0001×24×1120  = 3.468 ms
T_verify = 0.3 + 0.02×24×9   + 0.0001×24×16000 = 43.02 ms
c_i = 3.468 / 43.02 ≈ 0.081
```

看清结构：T_verify 里 attention 项 38.4ms 占 89%——这就是"验证贵在 KV 读取"，也是长上下文 c 变小的原因。

### 5.4 校准绑定（防拿错机器）

`CalibrationKey` 记录 model/gpu/tp/dtype/max_batch_size/page_size/top_k_pages/graph_buckets，加载时逐项校验，**换模型/硬件/配置都要重新标定**。

---

## 6. 组件三：起草内刷新层（§4.3 Refresh Layer）

### 6.1 它解决什么问题：稀疏上下文陈旧

起草时模型用稀疏上下文 S_i（选出来的少量 KV page）。问题是这个 S_i 什么时候更新？

- MagicDec：固定规则，死板；
- SpecAttn：只在**验证时**更新一次。验证一结束 S_i 就冻结，起草越久越陈旧 → 草稿质量下降 → 接受率掉。

**目标：让 S_i 在起草过程中也能更新，而不是只在验证边界更新一次。**

### 6.2 刷新层机制

依据 §2 的观察③（跨层注意力高度相关：浅层注意力能预测深层注意力）。所以：

> 起草时，只有指定刷新层 ℓr = N−2（倒数第二层）做**全注意力**，其他层照常用稀疏注意力。ℓr 顺手抓出它的注意力 logits，用来**给下一步重新挑选 S_i**。

```
起草一个 token 时逐层前向：
  第 0~33 层：稀疏注意力（读 S_i 的 70 页）
  第 34 层（ℓr）：全注意力（读全部 1000 页）+ 抓 logits
  第 35 层：稀疏注意力（读 S_i）
```

成本只多"一层全注意力"，很轻。抓到的 logits 用来刷新**下一步**的 S_i（存在一个 token 的滞后，跨层信号退化慢，可接受）。

### 6.3 选页的完整公式链

```
Z^src_{i,t,h}(u)   ℓr 层对上下文位置 u 的注意力 logit
  → online softmax → A^src_{i,t,h}(u)   注意力权重（每个头对 16000 个位置的概率分布）
  → r_{i,t}(u) = max_h A^src_{i,t,h}(u)   token 重要性（max over 32 头）
  → s_{i,t}(P) = Σ_{u∈P} r_{i,t}(u)      page 得分（每 16 个求和）
  → 预算 k_i = max(k_min, ⌈ρ·P_i⌉)
  → 最近 L 页保底 + 其余按得分取 top 页 = 新 S_i
```

### 6.4 真实数字：1000 页怎么选出 70 页

以 16k 上下文为例（page_size=16 → P_i = 16000/16 = **1000 页**）：

```
第 1 步：32 个头 × 16000 位置 = 32×16000 的注意力表
第 2 步：每列取 max（32 个头里谁最关注这个位置）→ 16000 个 token 重要性
第 3 步：每 16 个求和 → 1000 个 page 得分
第 4 步：预算 k_i = max(32, ⌈7%×1000⌉) = max(32, 70) = 70 页
第 5 步：最近 8 页无条件保留 + 其余 992 页取 top-62 = 70 页
```

三步压缩：`32×16000 →(max)→ 16000 →(每16求和)→ 1000 →(保底+top)→ 70`。

**为什么"最近 8 页无条件保留"**：保证局部连续性——刚生成的 token 通常和紧邻上文最相关，即使得分不高也强制留下。

> [!note] 数据来源说明
> "1000 页"和"70 页"是**代入论文参数算出的示例值**，不是论文原文直接印出的数字。论文原文只给了参数（§5：page size 16、ρ=7%、k_min=32、L=8），公式是 §4.3 的 `k_i = max(k_min, ⌈ρ·P_i⌉)`。代入 16k 上下文：`⌈16000/16⌉=1000` 页 → `max(32,⌈7%×1000⌉)=70` 页。

**16k 只是论文的入门档**，稀疏页数随上下文长度线性增长：

| 上下文档位 | 页数 P_i | 稀疏页数 k_i（≈⌈7%⌉，≥32） |
|---|---|---|
| 16k-18k | 1000~1125 | 70~79 |
| 30k-40k | 1875~2500 | 132~175 |
| 80k-100k | 5000~6250 | 350~438 |

（"70 页"只对应 16k 档，别的档位是别的数；机制完全一样，只是数字变大。）

### 6.5 验证时的处理（零成本）

验证行**本来就要做全注意力**，所以 ℓr 层在验证时抓 logits 是免费的。但验证有 d_i+1 个 query token，全抓太重，所以**只抓第一和最后一个验证 token 的注意力，取平均**，用平均值选页。

### 6.6 为什么这样能对抗陈旧的注意力

S_i 不是"验证时才更新一次"，而是**每个起草步都由 ℓr 重新选一遍**：

```
起草步 t：  ℓr 全注意力 → 抓 logits → 选新 S_i
起草步 t+1：用新 S_i 做稀疏注意力；ℓr 又全注意力 → 再选新 S_i
...
```

S_i 永远跟着当前 token 走。附录 C（图 3）证明：有刷新层时起草 10 个 token 接受率还能保持 76.6%，无刷新层只有 60.3%。

---

## 7. 端到端流程（一次循环）

```
① 调度器(CPU)：每请求比 d_i vs γ_i → 产出 mode_flags、verify_dec_lens
      ↓
② 混合前向(GPU)：token 排平 → 逐层×36 → 注意力按 page 表分流
   （起草行读稀疏 70 页 / 验证行读全量 1000 页；第 34 层全注意力 + dump）
      ↓
③ 验证比对：数接受数 a_i → 提交 a_i+1 → 更新 α → d_i 归零
   ③' 重选页：dump logits → 70 页 → 写回 draft_page_table
      ↓
④ 重算 c_i、γ_i → 回到 ①
```

退出条件：EOS 或达到 max_new_tokens。

**大小限制**：单请求解码 ≤ d_max+1=17 token；一批总 token ≤ B×17；批大小 B 由显存卡死（5~42）；起草:验证比例无固定限制。

---

## 8. 实验结果

**加速比（相对自回归，吞吐 tok/s）**：

| 模型          | 方法         | AIME25    | CodeElo   | LB[16k-18k] | LBv2[30-40k] | LBv2[80-100k] | 平均        |
| ----------- | ---------- | --------- | --------- | ----------- | ------------ | ------------- | --------- |
| Qwen3-1.7B  | MagicDec   | 1.93×     | 1.94×     | 2.52×       | 3.17×        | 3.53×         | 2.62×     |
|             | SpecAttn   | 2.10×     | 1.96×     | 2.62×       | 3.07×        | 3.36×         | 2.62×     |
|             | **ASPIRE** | 2.06×     | 2.12×     | **3.78×**   | 4.17×        | **4.58×**     | **3.34×** |
| Qwen3-8B    | MagicDec   | 1.33×     | 1.24×     | 1.30×       | 1.63×        | 2.03×         | 1.51×     |
|             | SpecAttn   | 1.40×     | 1.28×     | 1.66×       | 1.74×        | 2.05×         | 1.63×     |
|             | **ASPIRE** | **1.86×** | **1.70×** | 1.81×       | **2.30×**    | **2.75×**     | **2.08×** |
| DS-Llama-8B | **ASPIRE** | 1.80×     | 1.80×     | 1.94×       | 2.49×        | 2.66×         | **2.14×** |

**三条规律**：
1. 上下文越长加速越大（1.7B 从 2.06× → 4.58×）——memory-bound 越严重越划算；
2. 小模型收益最大（3.34×）——MLP 占比小、attention 占比大；
3. SpecAttn 是最强基线，但 ASPIRE 在 8B 模型上全面胜出。

**为什么上下文越长加速越大（机制）**：

加速本质是"起草便宜、验证贵"，而验证贵在 attention 读 KV——上下文越长，"贵"越被放大：

```
起草一步 = MLP + attention(读稀疏 ≈7% 上下文)
验证一步 = MLP + attention(读全量 100% 上下文)

短上下文(16k)：attention 占比小 → 起草省的钱相对少 → c 不算小
长上下文(100k)：attention 占主导 → 起草省的钱巨大 → c 非常小
```

`c` 越小，代入 `γ* = argmax E[tokens]/(1+cγ)` 算出的 γ 越大、加速越大。所以 ASPIRE 的价值随上下文变长**单调上升**：

| 档位 | c 相对大小 | 加速比（Qwen3-8B） |
|---|---|---|
| 16k-18k | 最大（收益最温和） | 1.81× |
| 30k-40k | 中等 | 2.30× |
| 80k-100k | 最小（收益最大） | 2.75× |

（1.7B 更明显：16k 档 3.78× → 100k 档 4.58×。）**所以 16k 是论文最弱的入门档，80k-100k 才是重头戏；方法的甜蜜区在 30k 以上，越短收益越缩水。**

**消融**：ASPIRE-FSM（±1 反馈）平均 1.93–1.94×，ASPIRE-Fixed（固定 γ=5）平均 1.78–1.80×，均低于完整 ASPIRE。

**附录证据**：
- 附录 C（图 3）：固定 γ 从 1 扫到 10，无刷新层接受率 95%→60.3%，有刷新层保持 76.6%（差 16.4pp）→ 刷新层有效对抗陈旧。
- 附录 D（图 4）：刷新层扫 0–35 层，N−2 层最优（88.2% 接受率），N−1 层骤降（84.5%，其注意力被"预测下一个 token"特异化）。

---

## 9. 关键参数汇总

| 参数 | 默认值 | 含义 |
|---|---|---|
| page_size | 16 | 每页 token 数 |
| ρ | 7% | 稀疏率 |
| k_min | 32 | 稀疏页保底 |
| L | 8 | 强制保留的最近页数 |
| ℓr | N−2 | 刷新层位置 |
| d_max | 16 | 最大草稿长度 |
| ω | 0.8 | α 的 EMA 平滑系数 |
| γ_init | 3 | 首次验证前强制起草长度 |

---

## 10. 批判性分析

1. **IID α 假设**：接受率真实地随起草位置衰减（图 3 自证），公式会系统性偏"过度起草"；但不影响无损性，只影响 γ 启发式的精度。
2. **成本模型近似**：三步线性 + 离线拟合 + 同质 batch 近似，换硬件/模型需重新标定，泛化性弱。
3. **代码 vs 论文差异**：`fit_latency_model` 的 `draft_cache_tokens` 写死为 `k_min×page_size=512`，而论文用真实 `|S_i|`（长上下文可达 ~7000）→ 低估 T_draft → 偏过度起草。
4. **无损 vs 采样**：正文"只接受一致 token"对应贪心；实际配置用 temperature=0.7 采样（投机采样，分布等价而非逐 token 一致），表里 Avg Gen Len 差异很大正是采样方差。
5. **对比公平性**：SpecAttn 需存所有层 attention score，batch 更小（混杂变量）。
6. **评测单一**：只报吞吐、无延迟（TTFT/TPOT）、无质量/准确率对照；单卡、3 模型、TP=1，规模外推待验证。

---

## 11. 代码对应（复现仓库）

| 论文组件 | 代码位置 |
|---|---|
| §4.2 调度器 | `aspire/policies/core.py`（AspirePolicy）、`engine/scheduler.py`（ReplicatedPlanner） |
| §4.2 成本模型 | `aspire/policies/timing.py`（LatencyModel、fit_latency_model） |
| §4.1 混合前向 | `aspire/models/mixed_transformer.py` + `mixed_runner.py`、`sparse_draft_runner.py` |
| §4.3 刷新层 | `aspire/backends/base.py`（_bind_dump_attention_ops）、`backends/page_selector.py` |
| 入口 | `benchmark.py` |

**运行**：`python benchmark.py --model qwen3-8b --dataset longbench --method aspire ...`（方法：autoregressive / aspire-fixed / aspire-fsm / aspire）。

---

## 12. 一条线记住全文

```
长上下文解码慢在 attention 读 KV（memory-bound）
  → 自推测：起草(稀疏读少) + 验证(全量读多) 一次验收 γ 个
  → 问题1：batch 锁步 → §4.1 混合前向（两类请求一次前向）
  → 问题2：γ 选多大 → §4.2 调度器（argmax E[tokens]/(1+cγ)）
  → 问题3：稀疏上下文陈旧 → §4.3 刷新层（ℓr 全注意力每步重选 70 页）
  → 结果：1.70–4.58× 吞吐
```

---

## 13. 与 DSpark（SGLang）的对比

> 两个都解决"batch 自回归/投机解码的异构调度"，但走的是**两条正交的技术路线**。DSpark 源码走读见 [[sglang中的dspark]]。

### 13.1 一句话定位

|         | ASPIRE                                            | DSpark                                  |
| ------- | ------------------------------------------------- | --------------------------------------- |
| 解决的核心问题 | **长上下文** decode 的 attention 内存墙                   | **batch decode** 的投机调度效率                |
| 投机技术    | self-speculative：稀疏 attention 起草 + 全 attention 验证 | block-level（EAGLE/MTP 族）：dense block 起草 |
| 形态      | 研究代码（gpt-fast 风格）                                 | SGLang 生产组件                             |

### 13.2 最根本区别：草稿怎么产生

- **ASPIRE 是 self-speculative**，起草和验证用**同一个 target 模型**，差别只在"读多少 KV"：
  - draft：同一模型 + 稀疏 attention（读选出的 KV 子集 `S_i`，默认 ρ=7%），一次只前进 1 个 token。
  - verify：全 attention 读完整 KV，并行解码 `d_i + 1` 个 token。
  - 不需要额外草稿模型/头，加速来源是"省掉重复读满 KV"。

- **DSpark 是 block-level speculative**，草稿来自专门 draft 结构：
  - 从 target hidden 抽特征，用 **markov head + confidence head**（或 DeepSeek-V4 的 MoE draft）一次生成 `gamma` 个 token 的 block。
  - draft 是 **dense** 的，GPU block attention，全程不涉及稀疏 KV 选择。

> 所以 DSpark 属 EAGLE/MTP 线（靠额外 draft 头提质量），ASPIRE 属 MagicDec/SpecAttn 线（靠稀疏 attention 省 KV 读取）。**两条完全不同的投机技术路线。**

### 13.3 "ragged" vs "去同步化"：异构性处理方式相反

两者最像、也最易混淆——都在解决"不同请求最优草稿深度不同"，但手段相反：

- **ASPIRE：把"draft vs verify"这个状态本身去同步化（§4.1 混合前向）**
  - 同一 forward 里，batch 中一部分请求在 draft、另一部分在 verify，各处于不同 spec 状态。
  - 每请求独立维护 `d_i`（当前草稿长度）和稀疏上下文 `S_i`，各自决定何时 verify。
  - 打破"全 batch 一起 draft、一起 verify"的全局 phase barrier。

- **DSpark：保留两阶段（全 batch draft → 全 batch verify），但把 verify 长度做 ragged（参差）**
  - `_forward_decode` 里仍先 `propose`（整批一起 draft block）再 verify（整批一起验证）。
  - 区别在 verify 阶段每请求 `verify_len` 不同（`RaggedVerifyLayout`），预算按置信度 top-k 分配。

> **一句话**：ASPIRE 让"谁来 draft、谁来 verify"不同步；DSpark 让"每个请求 verify 多长"不同步。DSpark 的 draft 阶段仍是 batch 级同步的。

### 13.4 成本模型：都离线 profile，但建模对象不同

| | ASPIRE | DSpark |
|---|---|---|
| 形式 | 白盒线性模型 | 黑盒吞吐曲线 |
| 公式 | `τ = β_model + β_mlp·n + β_attn·Σkv` | `batch_tokens → steps_per_sec`（1D 查表 / 2D additive） |
| 是否拆 MLP/attention | ✅ 显式拆（compute-bound vs memory-bound） | ❌ 不拆，整批一个吞吐 |
| 决策粒度 | per-request（算 `c_i`，得 `γ_i`） | batch 级（算总 budget） |
| 目标函数 | `γ_i = argmax (1-α^{γ+1})/(1-α) / (1+c_i·γ)` | `budget = argmax τ(k)·sps(k)` |

### 13.5 ASPIRE 独有、DSpark 完全没有的概念

1. **Refresh layer（起草内上下文刷新）**：因稀疏草稿上下文 `S_i` 会随起草变长而陈旧，故在固定层（N−2）做 full attention 重选稀疏页。DSpark 是 dense draft，无稀疏上下文，**不存在 stale 问题，也无 refresh 概念**。
2. **接受率在线估计 `α_i`（指数平滑）**：每请求维护平滑标量 `α_i`。DSpark 用 confidence head 直接输出**逐位置生存概率**（cumprod），信息更丰富，但需专门训练的 confidence head。

### 13.6 相同点

1. **都无损**：拒绝采样/一致接受保证，与接受率、草稿质量无关。
2. **都承认"固定草稿长度次优"**：ASPIRE 图 1 是动机（per-request 接受长度 2.56~20.00）；DSpark 整个 ragged verify + SPS 调度基于同一动机。
3. **都需离线 profile/calibration**，且都强调"针对同一 task/GPU/batch 配置重新校准"。
4. **都有退化对照**：ASPIRE 有 `aspire-fixed` / `aspire-fsm`；DSpark 有 `RAGGED_VERIFY_MODE=static` 和未初始化 SPS 表退化为 verify-all。

---

## 14. 对 DSpark 的成本模型启发（后续优化点）

> 只聚焦**成本模型**这一个点。ASPIRE 最有价值的启发不是招牌的"混合前向"，而是它的**白盒成本模型**——尤其是对 attention/KV 那一项的显式建模。混合前向（去同步 draft/verify）和刷新层是 self-speculation 独有的，DSpark 的 block-draft 架构搬不动。

### 14.1 问题：DSpark 的 SPS 表缺失 KV 长度维度

DSpark 的 SPS 表查表键只有一个 `batch_tokens`（= 这一步 verify 的 token 数），`lookup(batch_tokens)` 只吃这一个数：

- 1D 表：`lookup` 按 `batch_tokens` 二分 clamp。
- 2D additive 表：`step_time = bias + alpha(bs) + theta(M)`，`M = num_reqs + budget`，同样没有 KV 长度。

而 profiler 测表时把 prefix 固定死了：`DEFAULT_INPUT_LEN = 16`，注释明说 "the table is conditioned on the decode-heavy regime"。

**问题在哪**：decode 的 step time 在长上下文下由 full-attention 读完整 KV 主导，而 DSpark 的 verify 恰恰是 full attention。SPS 表是在"KV 长度固定（≈短 prefix）"下测的，一旦 batch 里 prefix 长或异构，`steps_per_sec` 被系统性高估 → 预算决策失真。而 `compute_verify_token_budget` 从头到尾只看 `num_requests` 和 `batch_tokens`，完全没有 prefix/KV 信息。

### 14.2 ASPIRE 的启发：显式拆出 attention/KV 项

ASPIRE 的成本模型显式拆了这项：

```
τ_t = β_model + β_mlp·n_t + β_attn·( Σ_{j∈D_t}|S_j| + Σ_{j∈V_t} ℓ_j )
```

其中 `β_attn · Σℓ_j` 就是"attention 随读入的 KV 总长度线性增长"这一物理规律——`β_mlp` 对 compute-bound 的 MLP（∝ token 数）、`β_attn` 对 memory-bound 的 attention（∝ KV 长度）。

### 14.3 迁移方向

1. **给 additive 表补 KV 项**：DSpark 的 `SpsAdditiveCostTable` 已经是 additive 形态（bias/alpha/theta），加一项 `γ(Σkv_len)` 很自然：
   ```
   step_time = bias + alpha(bs) + theta(M) + gamma(Σkv_len)
   ```
2. **或把查表键扩成二维** `(batch_tokens, total_kv_len)`，让 profiler 支持长 prefix sweep，而不是钉死 16 token。
3. **用线性拟合生成/平滑 SPS 表**：保留 SPS 表快速查询，但用 ASPIRE 式线性模型生成表值、或作采样点之外的外推 fallback，让表对"没 profile 过的 batch 组合"更鲁棒。

### 14.4 一句话

> ASPIRE 证明了"成本模型不应拍平成 batch_tokens→sps 的黑盒，而应显式拆出 attention/KV 这一项"。DSpark 的 SPS 表当前最大短板就是缺失 KV 长度维度（长上下文下失真），而 `SpsAdditiveCostTable` 已走在 additive 白盒化的路上，补上 `β_attn·Σkv_len` 这一项就齐了。
