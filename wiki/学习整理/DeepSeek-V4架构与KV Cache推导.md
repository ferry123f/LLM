# DeepSeek-V4 架构与 KV Cache 推导

> 基于 sglang 源码逐行推导（`python/sglang/srt/models/deepseek_v4.py`、`mem_cache/deepseek_v4_memory_pool.py`、`layers/attention/dsv4/`），已与官方口径对账一致（见文末）。
> 相关笔记：[[SGlang]]、[[SGlang中的DeepSeek.py]]、[[LLM推理的GPU硬件基础]]

---

## 1. 整体架构：三种注意力分工

DeepSeek-V4 Pro（1.6T A49B，1M 上下文，61 层 = 58 MoE + 3 Hash）。核心矛盾：1M 上下文的全量 KV cache 存不下。解法是**近处精确、远处压缩**：

| 机制 | 压缩率 | 覆盖 | 读取方式 |
|---|---|---|---|
| SWA 滑动窗口 | 不压缩 | 最近 128 token | 全读（环形缓冲，固定大小） |
| CSA (Compressed Sparse Attention) | 4:1 | 中距离 | Indexer 打分选 top-1024 稀疏读 |
| HCA (Heavily Compressed Attention) | 128:1 | 超长距离 | 密集全读（压完已经很少） |

每层走哪种由 `config.compress_ratios[layer_id] ∈ {0, 4, 128}` 决定。**官方层分布：30 层 c4 + 31 层 c128**。

### 注意力主体是 MQA over shared-KV（不是 V3 的 MLA）

- `num_key_value_heads == 1`：128 个 query 头共享**一个 512 维 KV 向量**
- KV 只有一级投影 `wkv: 7168 → 512`，**没有上投影/没有权重吸收**——V3 的 "latent + 展开" 机制在 V4 已删除
- 512 维 = 448 nope + 64 rope（decoupled RoPE：nope 段位置无关，rope 段单独承载位置）
- attention 调用时 k 和 v 传同一个张量
- config 里的 `kv_lora_rank=512` 是 V3 继承下来的死字段，代码不读；算显存认 `qk_nope_head_dim=448` / `qk_rope_head_dim=64`

---

## 2. 权重命名规则（w 系列速查）

```
w + 流向 + a/b
  流向: q=Query  kv=Key/Value  o=Output  gate=打分(过softmax)
  a = 降维进瓶颈   b = 从瓶颈升维
  成对出现 = 低秩分解省参数；单个名字 = 一步到位无瓶颈
```

单层 attention 数据流（hidden=7168）：

```
x [7168] ── attn_norm ──┬─ wq_a → [1536] ─ q_norm ─┬─ wq_b → 128头×512 → Q
                        │        (q_lora_rank)     └─ indexer.wq_b → 64头×128 → 打分Q
                        ├─ wkv → [512] ─ kv_norm ─ RoPE → SWA cache
                        ├─ compressor(wkv+wgate+ape) → 4合1 → [512] → c4/c128 池
                        ├─ indexer.compressor → 4合1 → [128] → indexer 池
                        └─ indexer.weights_proj → [64] 打分头权重
                                 ↓
  打分Q × indexer池 → top-1024 → Q × (SWA + 选中c4条目) + attn_sink
                                 ↓
         输出 128头×512 ─ 分16组 ─ wo_a(每组4096→1024) ─ wo_b(16384→7168)
```

### 与 V3 MLA 的结构对比（关键记忆点）

| 路径 | V3 (MLA) | V4 | 变化 |
|---|---|---|---|
| Q | 两级 q_a/q_b | 两级 wq_a/wq_b | 不变 |
| KV | 两级（latent 在中间，权重吸收） | **一级 wkv，512 维即 KV 本身** | 删掉上投影 |
| O | 一个大矩阵 | wo_a/wo_b 两级 + 分 16 组 | 新加分解 |

一句话：**低秩瓶颈从 KV 侧搬到了 O 侧**；显存压缩改由序列方向的 Compressor 负责。

### 单层参数表（layers.N.attn，CSA 层实例）

| 参数 | 形状 | 精度 | 含义 |
|---|---|---|---|
| `attn_norm.weight` | [7168] | BF16 | 块入口 RMSNorm |
| `q_norm.weight` | [1536] | BF16 | Q 瓶颈处 RMSNorm |
| `kv_norm.weight` | [512] | BF16 | KV 入 cache 前 RMSNorm |
| `attn_sink` | [128] | F32 | 每头一个 softmax 泄压标量 |
| `wq_a.weight` | [1536, 7168] | F8 | Q 降维 |
| `wq_b.weight` | [65536, 1536] | F8 | Q 升维 = 128头×512 |
| `wkv.weight` | [512, 7168] | F8 | 唯一 KV 投影 |
| `wo_a.weight` | [16384, 4096] | F8 | 输出降维（16组×8头×512→16组×1024） |
| `wo_b.weight` | [7168, 16384] | F8 | 输出升维回 hidden |
| `compressor.wkv/.wgate` | [1024, 7168]×2 | BF16 | 压缩的内容/打分投影（1024=2×512，overlap 双份；运行时拼成 wkv_gate[2048,7168]） |
| `compressor.ape` | [4, 1024] | F32 | 窗口 4 座位 × 双角色位置偏置 |
| `compressor.norm.weight` | [512] | BF16 | 压缩产物 RMSNorm |
| `indexer.wq_b.weight` | [8192, 1536] | F8 | 打分 Q = 64头×128（蹭主路 wq_a） |
| `indexer.weights_proj.weight` | [64, 7168] | BF16 | 64 个打分头的重要性权重 |
| `indexer.compressor.*` | wkv/wgate [256,7168]、ape [4,256]、norm [128] | BF16/F32 | 迷你压缩器（head_dim=128，rotate=True 做 Hadamard） |

参数量分布：wo_b 35% + wq_b 30% + wo_a 20% ≈ 85% 在 Q/O 两条路；整套压缩+索引机制仅占 ~10%。

### CSA / HCA 层 30 秒鉴别口诀

| 看什么 | CSA (ratio=4) | HCA (ratio=128) | ratio=0 |
|---|---|---|---|
| `indexer.*` | 有 | 无 | 无 |
| `compressor.ape` | [4, 1024] | [128, 512] | 无 compressor |
| `compressor.wgate/wkv` | [1024, 7168] | [512, 7168] | — |

---

## 3. Compressor 机制（4 个 token 怎么变 1 个）

不是取平均，是**逐维度可学习加权平均**，五步流水（kernel 注释原文："APE-add + overlap-transform + softmax-pool + RMSNorm + RoPE"）：

```
每 token:  kv[t] = wkv @ x[t]（内容）   score[t] = wgate @ x[t]（打分）
攒满窗口:  w[0..3] = softmax( score_state[k][d] + ape[k][d] )   ← 每维独立
           out[d]  = Σ w[k]·kv_state[k][d]
收尾:      RMSNorm → RoPE(compress_rope_theta=40000) → 写入压缩池
```

### ape = [ratio, coff × head_dim] 的每个维度

- **第一维 4**：窗口内座位数（= compress_ratio）。softmax 池化本身不分先后，ape 给每个座位一条可学习偏置，是池化感知语序的唯一来源。名字里的 absolute position 指**窗口内**位置（0~3），全局位置由 RoPE 管
- **第二维 1024 = 2 × 512**：`coff = 1 + overlap`。CSA 窗口重叠——每个条目看 8 个 token（前窗 4 + 当前窗 4，stride 4），每个 token 要扮演"当前窗/前窗"两个角色，投影和 ape 各给双份。HCA 无重叠，coff=1
- **权重形状永远不含 seq_len**：ape 像卷积核，被所有窗口复用。"seq_len/4" 是压缩**产物条数**（激活/缓存），不是 ape 的形状

验证：`attn.compressor.ape [4, 2×512=1024]` ✓ `indexer.compressor.ape [4, 2×128=256]` ✓ HCA 层 `[128, 1×512=512]` ✓

### 压缩累积态（kv_state / score_state）

decode 逐 token 到来，凑不满窗口的半成品必须**跨 step 存活**（CompressStatePool 环形槽）：

- c4：8 槽；c128：128 槽（投机解码下 16/256）
- `ONLINE_C128` 开关：改为维护 (max, sum, kv) 在线 softmax 三元组，128 槽塌缩为 1，数学等价（暂不兼容 MTP）
- **正确性状态而非缓存**：抢占/PD 迁移必须随 KV 池一起搬，否则窗口错位后续全错

---

## 4. Indexer（CSA 专属的 top-k 检索）

一套独立迷你注意力：64 头 × 128 维（主注意力 128 头的一半）。

```
打分: scores = ReLU( q_indexer · k_indexer ) · weights_proj(x) 各头求和 × kv_scale
选择: top-1024 个压缩条目 → 稀疏 attention 只读这 1024 条
```

indexer 的 K 单独存一份（不复用主 cache）的三个原因：
1. 只需 128 维（打分不用 512 维，省全量扫描带宽）
2. 过了 Hadamard 旋转（rotate=True，让 FP8 内积误差均匀），数值空间与主 cache 不兼容
3. 布局为 fp8_paged_mqa_logits kernel 定制（64 token/页）

**反直觉结论**：1M 上下文时 Indexer 全量扫描 250K 条 × 132B ≈ 33 MB/层/步，而选完后 attention 只读 1024×584B ≈ 0.6 MB——**扫描比注意力本身贵 55 倍**。CSA 省的是 attention，省不掉扫描 → FP4 indexer（132→68B）是收益最大的单一开关。

---

## 5. FP8 量化：weight 与 scale 成对出现

**weight 存压缩后的"码"，scale 存还原倍率**，128×128 块量化：

- E4M3（权重码）：1+4+3 位，范围 ±448，约 2 位有效数字——存缩放后的值
- E8M0（scale）：纯 2 的幂——还原零舍入误差，对齐 Blackwell microscaling；sglang 内叫 `weight_scale_inv`
- 分块原因：① 离群值只污染本块 ② 与 DeepGEMM 128-tile 对齐，还原在累加器层面顺手完成
- **scale 形状 = weight 形状 ÷ 128**（逐一验证成立，如 wo_b [7168,16384] → [56,128]）
- 推理走 W8A8：激活运行时 per-token-group 量化，FP8×FP8 GEMM，FP32 累加，不先反量化

精度分配三档逻辑：大 GEMM → FP8（量化收益大）；compressor 小矩阵/norm → BF16（softmax 敏感且小）；ape/attn_sink → F32（直接进 softmax 指数位，误差被指数放大）。

同一"码+分块倍率"模式出现三次：权重（128×128 块）、KV cache nope 段（64 维/块）、indexer 条目（整行一个 scale）。

---

## 6. KV Cache 逐字节推导（核心）

### 6.1 主 KV 条目 = 584 B

源码断言 `assert bytes_per_token == 448 + 64*2 + 8`：

| 段 | 字节 | 推导 | 为什么这么存 |
|---|---|---|---|
| nope | 448 | 448 维 × 1B (FP8) | 内容段，量化损伤可控 |
| rope | 128 | 64 维 × 2B (BF16) | 旋转后编码相位，FP8 舍入=位置漂移，**不量化** |
| scale | 7 | 448÷64 块 × 1B | 每 64 维一个还原倍率，丢了码无法还原 |
| pad | 1 | 583→584 对齐 | 工程开销 0.17% |

### 6.2 indexer 条目 = 132 B（FP4 时 68 B）

128 维 × 1B (FP8) + 4B (F32 per-token scale)。FP4：128÷2 + 4 = 68。

### 6.3 摊到每原始 token（除以压缩率）

```
c4 层:   584/4 + 132/4 = 146 + 33 = 179 B/token/层   （FP4 indexer: 163）
c128 层: 584/128       = 4.56 B/token/层             （无 indexer）
ratio=0: 0（只有固定 SWA）
```

c4 比 c128 贵 39 倍仍保留：128:1 太狠会糊掉中距离细节，c4+top-1024 负责把这段捞回来。层配比 = 质量-显存旋钮。

### 6.4 总量公式

$$\text{每请求} \approx S \times (L_4 \times 179 + L_{128} \times 4.56 + 4_{\text{页表}}) + 17\,\text{MB}_{\text{固定}}$$

代入官方分布 L4=30, L128=31：

```
线性: 30×179 + 31×4.56 + 4 ≈ 5515 B ≈ 5.4 KB/token
  → 4K: 22 MB   128K: 0.69 GB   1M: 5.4 GB（FP4 indexer 时 ≈5.0 KB → 4.9 GB）

固定（不随长度涨，迁移时必须搬）:
  SWA 环:   61 层 × 128 token × 512 维 × 2B ≈ 7.6 MB
  累积态:   30×40KB + 31×256KB ≈ 9.1 MB（ONLINE_C128 后 c128 项基本消失）
  合计 ≈ 17 MB/请求（1M 时占 0.3% 可忽略；4K 时占 43% 别忘）
```

### 6.5 不保存什么（有损、不可逆）

超出 128 窗口的**原始 KV 永久丢弃**（环覆盖写）。推论：
- 无法事后用原始 KV 重算精确 attention
- prefix caching 复用 = 压缩池条目 + 边界时刻 SWA 环 + 累积态，**三样一套缺一不可**
- PD 分离/HiCache 换入换出搬的也是这一整套

### 6.6 与 V3 对比

V3 (MLA)：(512+64)×2B = 1152 B/层/token × 61 层 ≈ 70 KB/token → 1M ≈ 70 GB。
V4 fp8 口径 5.4 KB/token = **1/13**。分解：FP8 量化 ×1.97；c4 序列压缩 ×3.3；c128 ×126；按层配比加权。

---

## 7. 与官方口径对账（全部命中）

| 项 | 代码推导 | 官方 | 判定 |
|---|---|---|---|
| 层分布 | 假设 30 c4 + 31 c128 | vLLM: "30 c4a and 31 c128a layers" | ✅ |
| CSA overlap | coff=2，看 8 token | "weighted sum of 8 tokens, stride 4" | ✅ |
| HCA | 128:1 无重叠 | "128 tokens, stride 128" | ✅ |
| SWA | 128 | "sliding window of size 128" | ✅ |
| 量化 | 主 FP8 / indexer FP4 | "fp4 for indexer, fp8 for attention, ~2x" | ✅（**FP4 indexer 是生产默认**，非可选） |
| 1M 总量 | bf16 口径重算 ≈9.2 GiB / fp8+fp4 ≈4.9 GB | bf16 9.62 GiB / "再省 ~2x" ≈4.8 GB | ✅ 差 5% 为 SWA/padding 边角 |
| vs V3.2 | 同 bf16 口径 ≈7.1x | 8.7x (83.9→9.62 GiB) | ✅ 同档（官方基线含 V3.2 indexer） |

**报数字带口径**：bf16 理论 ≈9.6 GB/1M；fp8+fp4 实际 ≈5 GB/1M。"2% of GQA-8" 是另一个更宽松基线。

来源：vLLM Blog (2026-04-24 deepseek-v4)、arXiv 2606.19348、DeepSeek V4 Model Card。

---

## 8. 推理优化视角（心智模型）

**容量与带宽是两个独立指标**：
- 容量：5.4 KB/token × 上下文 × 并发 → 决定塞多少请求
- decode 带宽（1M 单请求每步）：30 层 × ~34 MB (indexer 扫描主导) + 31 层 × 4.7 MB ≈ **1.17 GB/step**

**KV 与权重耦合**：1.6T A49B FP8 每步读 ~49 GB 激活专家权重，但权重全 batch 共享、KV 每请求独立。小 batch 瓶颈在权重带宽，大 batch KV 反超。**KV 压小的真正回报 = 抬高可行 batch → 摊薄权重读取**。

优先级清单：
1. c4 双池是容量+带宽双料大头；FP4 indexer 是现成减半开关
2. c128 几乎免费（4.6 B/token），不用花精力
3. SWA/累积态是常数，只在抢占/迁移/前缀复用的**正确性**上重要
4. 拿到 checkpoint 先确认 `compress_ratios` 真实分布（c4 比 c128 贵 40 倍）

---

## 9. MoE 速记（同一模型的另一半）

- 384 routed + 1 shared，每 token 激活 6；`moe_inter_dim=3072`
- 打分 `sqrt(softplus(x))`（V3 是 sigmoid）；**ungrouped** top-6（V3 分组）；`/=sum` 归一化让 top-6 权重和为 1（shared 列不参与归一化）
- `e_score_correction_bias`：选专家时加、加权求和时不加——推理期可调的负载均衡旋钮（noaux_tc）
- 前 3 层 Hash 路由：`tid2eid[vocab_size, topk]` 查表按 token_id 固定选专家，分数只当权重用——训练初期负载均衡 + dispatch 可预计算
