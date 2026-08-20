# DeepSeek-V4 config.json 逐项详解与显存估算

> 一句话：拿 V4-Pro 的 config.json 当解剖标本，把每个字段放回 Transformer 骨架里的位置，再一步步算出「1.6T 总参 / 49B 激活 / 1M 上下文只花 5.2 GB KV cache」这三笔账。面向初学者，公式不跳步。
>
> 素材：对话中用户提供的 V4-Pro 与 V4-Flash 两份 config.json（HF 格式）。机制背景见 [DeepSeek-V4 架构与百万上下文效率](DeepSeek-V4%20架构与百万上下文效率.md)，通用公式见 [LLM 推理的 GPU 硬件基础](../学习整理/LLM推理的GPU硬件基础.md)。

## 〇、心智模型：config.json 是什么

LLM = **同一种积木块（Transformer 层）叠 N 层**。每块只有两个部件：

```
输入向量 → [注意力：看上下文，搬运信息] → [FFN/MoE：对每个位置做变换] → 输出向量
```

config.json 就是这套积木的**图纸尺寸表**：每个字段回答一个「多大 / 几个 / 什么形状」。读 config 的正确姿势：拿到字段先问**它是骨架里哪个部件的尺寸**。

## 一、骨架四件套

| 字段 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 7168 | 每个 token 在模型内部的向量维度，「信息容量」。几乎所有矩阵尺寸都挂在这个数上 |
| `num_hidden_layers` | 61 | 积木叠 61 层，每个 token 依次穿过被加工 61 次 |
| `vocab_size` | 129280 | 词表大小。入口查表 token 编号 → 7168 维向量，这张表叫 embedding，形状 `129280 × 7168` |
| `tie_word_embeddings` | false | 输出头（7168 维 → 129280 个 token 的概率）**不**与 embedding 共用权重，各存一份 |

第一笔账：

```
embedding = 129280 × 7168 ≈ 0.93B
输出头   = 7168 × 129280 ≈ 0.93B
合计 ≈ 1.85B —— 相对 1.6T 九牛一毛
```

其余常规字段：`initializer_range 0.02`（初始化标准差）、`rms_norm_eps 1e-6`（RMSNorm 防除零）、`use_cache true`（生成时启用 KV cache）、`bos/eos_token_id`、`transformers_version`——都无信息量。

## 二、注意力参数

### 基础概念

- **Q/K/V**：每个 token 生成三个向量——Q（我在找什么）、K（我是什么）、V（我携带什么）。拿 Q 和前面所有 token 的 K 做点积算相关度，按相关度加权求和 V = 「看一次上下文」。
- **头（head）**：把大向量切成多组小向量并行做注意力，每组一个头，各自关注不同模式。
- **KV cache**：生成第 1000 个 token 需要前 999 个的 K/V；它们不变，算一次存显存反复用。上下文越长越大——**长上下文贵的根源**。

### 逐字段

| 字段 | 值 | 含义 |
|---|---|---|
| `num_attention_heads` | 128 | Q 侧 128 个头 |
| `num_key_value_heads` | **1** | KV 只存 **1 份**，128 个 Q 头共享 = **MQA**。对比：MHA 每头一份（128 份）、GQA 分组共享（如 8 份）。KV cache 直接 ÷128，**省显存第一板斧** |
| `head_dim` | 512 | 那份共享 KV 的维度。不是普通「每头维度」，而是继承 V3 MLA 思路的**压缩 latent**（不存原始 K/V，存低维浓缩包，用时展开） |
| `qk_rope_head_dim` | 64 | **部分 RoPE**：位置信息只编码在最后 64 维。RoPE = 按位置把向量旋转不同角度来注入顺序信息。每条 KV 实际 = 512（内容）+ 64（位置）= **576 维**（KV cache 账的关键数） |
| `q_lora_rank` | 1536 | Q 的生成走低秩分解：`7168 → 1536 → 展开`。低秩 = 大矩阵拆成两个小矩阵，参数大省、表达力略降 |
| `o_lora_rank` / `o_groups` | 1024 / 16 | **Grouped Output Projection**：128 头分 16 组共享输出投影，且投影做 rank=1024 低秩分解——治「头太多导致 W_O 巨大」 |
| `attention_bias` / `attention_dropout` | false / 0.0 | 无偏置、无 dropout，现代大模型标配 |

### 压缩注意力：V4 的杀手锏（「少存」而非「省着存」）

| 字段 | 值 | 含义 |
|---|---|---|
| `compress_ratios` | `[128,128,4,128,4,…,4,0]` | **逐层压缩比**，62 项 = 61 层 + 1 MTP 层。`4` = CSA 层（每 4 token 压 1 条，再稀疏选择，共 30 层）；`128` = HCA 层（每 128 token 压 1 条，稠密注意，共 31 层）；`0` = MTP 层不压。CSA/HCA 交错排布：细粒度与粗粒度轮流 |
| `index_n_heads` / `index_head_dim` | 64 / 128 | Lightning Indexer（CSA 的打分器）自己的多头配置 |
| `index_topk` | 1024 | 每个 Q 从 ~26 万条压缩条目里只挑 top-1024 来看。1024 条 × 4 token/条 ≈ 覆盖 4096 个原始 token |
| `sliding_window` | 128 | 压缩块内 token 互相看不清，补一路最近 128 token 的**原始未压缩**注意力。近处高清、远处低清 |
| `compress_rope_theta` | 160000 | 压缩条目自己的 RoPE 基数（1 条顶 4/128 个 token，位置尺度变了，需独立一套；`rope_theta 10000` 是原始 token 用的） |

## 三、长上下文：YaRN 16 倍外推

| 字段 | 值 | 含义 |
|---|---|---|
| `max_position_embeddings` | 1048576 | 1M token（2²⁰） |
| `rope_scaling` | yarn / factor 16 / original 65536 | 原生只训到 64K，YaRN 把位置编码拉伸 16 倍外推：65536 × 16 = 1048576。`beta_fast 32 / beta_slow 1` 控制各频率维度的插值边界，标准旋钮 |

为什么不直接训 1M：注意力计算量随长度平方涨，「短训长用 + 外推」是行业通用做法。

## 四、MoE：1.6T 总参为什么只算 49B

概念：**MoE** = 与其一个大 FFN 共用，不如养一堆小 FFN（专家），路由器给每个 token 挑几个最合适的——**参数总量巨大、每 token 只动用一小撮**。显存装全部，算力只花激活的。

| 字段                            | 值            | 含义                                                                           |
| ----------------------------- | ------------ | ---------------------------------------------------------------------------- |
| `n_routed_experts`            | 384          | 每层 384 个专家                                                                   |
| `num_experts_per_tok`         | 6            | 每 token 挑 6 个，稀疏率 6/384 ≈ 1.6%                                               |
| `n_shared_experts`            | 1            | 1 个必经的共享专家（学通用知识），路由专家学专门知识。每 token 实际过 7 个                                  |
| `moe_intermediate_size`       | 3072         | 单专家中间维。一个专家 = SwiGLU 三矩阵（gate/up/down），各 `7168 × 3072`                       |
| `scoring_func`                | sqrtsoftplus | 路由亲和度打分函数，V3 的 Sigmoid → V4 的 √Softplus(·)                                   |
| `topk_method`                 | noaux_tc     | **无辅助损失负载均衡**（V3 首创）：给专家挂动态偏置，谁太忙调低谁的中选率，不用会伤主任务的辅助 loss                     |
| `norm_topk_prob`              | true         | 选出的 6 个门控权重归一化到和为 1                                                          |
| `routed_scaling_factor`       | 2.5          | 路由专家合并输出 × 2.5，配平与共享专家的贡献                                                    |
| `num_hash_layers`             | 3            | 前 3 层用**哈希**定死 token→专家映射（不学习）。浅层学拼写级通用特征，定死更稳；V3 前 3 层是稠密 FFN，V4 连它们也 MoE 化 |
| `hidden_act` / `swiglu_limit` | silu / 10.0  | SwiGLU 激活；输出钳 ±10 = **SwiGLU Clamping**，消 MoE 层异常值、防 loss spike              |
| `num_nextn_predict_layers`    | 1            | MTP：顶上多接 1 层预测「下下个 token」，逼模型学前瞻表示；推理可做投机解码或直接丢弃                             |

## 五、mHC：残差连接的升级版

残差连接 =「输入 + 修正量」，深网络能训的生命线。HC 把单车道拓成 `hc_mult: 4` 条，车道间用可学习矩阵混合；不加约束时 61 层乘下来数值爆炸。mHC 强制混合矩阵为**双随机矩阵**（行和=列和=1、非负）→ 映射永不放大，叠多深都稳。

| 字段 | 值 | 含义 |
|---|---|---|
| `hc_mult` | 4 | 残差流宽度 ×4 |
| `hc_sinkhorn_iters` | 20 | Sinkhorn 算法（交替行/列归一化）把矩阵投影到双随机流形，最多 20 次 |
| `hc_eps` | 1e-06 | 归一化防除零（推断） |

原理与图解见 [Hyper-Connections 超连接](Hyper-Connections%20超连接.md)。

## 六、精度：按「部件多敏感」分三档

| 字段 | 部件 | 精度 | 理由 |
|---|---|---|---|
| `torch_dtype: bfloat16` | norm、embedding 等零碎 | BF16（2 字节） | 敏感、量小，不值得省 |
| `quantization_config: fp8` | 主干 | FP8 e4m3（1 字节） | 每 128×128 块共享一个 scale（`weight_block_size`）；激活动态量化；scale 用 ue8m0（2 的幂，硬件友好） |
| `expert_dtype: fp4` | MoE 专家 | FP4（0.5 字节） | 专家占 1.6T 绝大头，砍这里收益最大；QAT 保精度 |

**越占地方压得越狠**——精度设计的全部逻辑。

## 七、算账（一）：1.6T 总参 / 49B 激活

```
单个专家    = 3 × 7168 × 3072 ≈ 66M
一层全部专家 = 385 × 66M ≈ 25.4B          （384 路由 + 1 共享）
61 层       = 61 × 25.4B ≈ 1.55T
+ 注意力（低秩后每层 ~0.3B × 61 ≈ 20B）+ embedding/输出头 1.85B
≈ 1.6T ✓
```

激活（每 token 实际用到）：

```
每层：7 专家 × 66M + 注意力 ~0.3B ≈ 0.76B
61 层 ≈ 46B，+ 输出头 0.93B → 49B 的量级 ✓
```

**总参数 97% 是专家，激活时 384 个里只用 6 个**——MoE 魔术的全部。

## 八、算账（二）：KV cache，V4 全部设计的靶心

### 基线：传统模型要多少

通用公式（出处：[LLM 推理的 GPU 硬件基础](../学习整理/LLM推理的GPU硬件基础.md)）：

```
每 token KV 字节 = 2 × 层数 × KV头数 × head_dim × 精度字节
                  ↑ K、V 各一份
```

假想同级传统配置（GQA8、head_dim 128、BF16、61 层）：

```
2 × 61 × 8 × 128 × 2 = 249,856 字节 ≈ 244 KB/token
× 1,048,576 token ≈ 250 GB —— 三张 H100 存不下一条请求的缓存
```

### V4 三板斧逐级砍

**第一斧：MQA + 压缩 latent（存的形状变了）**。每层每条 KV = 1 条 576 维（K/V 合一，不乘 2）。混合精度：内容 512 维 FP8、RoPE 64 维 BF16：

```
每条 = 512×1 + 64×2 = 640 字节
只有这一斧（每 token 每层一条）：640 × 61 ≈ 39 KB/token → 1M ≈ 41 GB
```

（V3 的 MLA 基本停在这一档。）

**第二斧：压缩条目（存的数量变了）**。按 `compress_ratios`：

```
30 个 CSA 层：每层 1,048,576 ÷ 4   = 262,144 条
31 个 HCA 层：每层 1,048,576 ÷ 128 =   8,192 条
总条数 = 30×262,144 + 31×8,192 ≈ 8.12M 条   （不压缩是 61×1M ≈ 64M 条，砍 87%）

总量 = 8.12M × 640 字节 ≈ 5.2 GB / 一条 1M 请求
```

**验证**：5.2 ÷ 250 ≈ **2.1%**，与论文「KV 缓存约为基线 2%」吻合 ✓（滑窗 128 token 的原始 KV 与索引器缓存另加零头，不改量级）。

**第三斧在算力侧**：`index_topk 1024` 让每个 Q 只对 1024 条做注意力而非 26 万条——KV cache 是显存账，top-k 是计算账，两本都省。

### 为什么这笔账重要

decode 每生成一个 token 都要把 KV cache 完整从显存读一遍：**decode 是 memory-bound，速度上限 = 带宽 ÷ 每步搬的字节**。250 GB → 5 GB 意味着同带宽下生成速度和单卡并发数翻着倍涨——**KV cache 不只是省显存，它直接就是速度和吞吐**。

## 九、算账（三）：要几张卡

```
权重：专家 1.55T × 0.5 字节(FP4) ≈ 775 GB
      其余 ~50B × 1 字节(FP8)  ≈  50 GB
      合计 ≈ 825 GB
```

单机 8×H100（640 GB）装不下 → 至少 8×H200（1128 GB）或两台 8×H100。跨机用 PP 或 EP（384 个专家天然适合分家），机内 NVLink 才跑 TP。装完权重，8×H200 剩 ~300 GB 做 KV pool ≈ 同时挂 **50 条**满 1M 请求；没有三板斧（每条 250 GB）连 1 条都挂不住。

## 十、附：V4-Flash config 对比速览

Flash（284B / 激活 13B）的缩法非常克制——**三大创新一个字段没动，砍的全是尺寸**：

| 字段 | Pro | Flash | 备注 |
|---|---|---|---|
| `hidden_size` | 7168 | 4096 | 宽度 −43% |
| `num_hidden_layers` | 61 | 43 | 层数 −30%；KV cache 比例 43/61 ≈ 0.7，正好对应论文 Flash 7% vs Pro 10% |
| `num_attention_heads` / `o_groups` | 128 / 16 | 64 / 8 | 头减半，组内仍 8 头/组 |
| `q_lora_rank` | 1536 | 1024 | 随宽度缩 |
| `n_routed_experts` / `moe_intermediate_size` | 384 / 3072 | 256 / 2048 | **精确回到 V3 的 MoE 尺寸** |
| `routed_scaling_factor` | 2.5 | 1.5 | 输出配平重调 |
| `index_topk` | 1024 | 512 | 注意力「视野」减半 |
| `compress_ratios` 排布 | 开头 128,128 | 开头 **0,0**；结尾连续多个 0 | （推断）小模型前两层压不起，跑全量注意力；结尾疑为 dspark 让路 |

参数量验证：单专家 3×4096×2048 ≈ 25M，×256×43 层 ≈ 277B + 零碎 ≈ 284B ✓；激活 7×25M×43 + 注意力 ≈ 13B ✓。

**两个未解之谜**（均需 modeling 代码裁决）：

1. Flash 独有的 `dspark_block_size 5 / dspark_noise_token_id 128799 / dspark_target_layer_ids [40,41,42] / dspark_markov_rank 256`——形态上像安在最后三层的块级草稿生成（一次 5 token + rank-256 低秩转移模型辅助的投机解码变体），**纯字段名反推，低置信度**，论文与本 wiki 均无此机制记载。
2. Flash 的 `compress_ratios` 数了是 46 项，但 43 层 + 1 MTP = 44，多 2 项对不上（Pro 的 62 = 61+1 严丝合缝）。可能是转贴出错，也可能索引规则与「一层一项」不同。

## 十一、速查：从 config 到三笔账

| 你想知道 | 看哪些字段 | 公式 |
|---|---|---|
| 模型多大 | 专家数 × 专家尺寸 × 层数 | `385 × 3×7168×3072 × 61 ≈ 1.6T` |
| 每 token 算多少 | 激活专家数 | `7 × 66M × 61 ≈ 49B 量级` |
| KV cache 多大 | 每条字节 × 条数 | `640 B × Σ(T ÷ 压缩比) ≈ 5.2 GB @1M` |
| 权重占多少显存 | 参数 × 精度字节 | `1.55T×0.5 + 50B×1 ≈ 825 GB` |
| 能外推多长 | YaRN factor | `65536 × 16 = 1M` |

遇到不懂的字段问三个问题：**它是哪个部件的尺寸？出现在哪笔账里？改大改小谁付代价？**

## See Also

- [DeepSeek-V4 架构与百万上下文效率](DeepSeek-V4%20架构与百万上下文效率.md) — 本文所有字段背后的**机制**：CSA/HCA、mHC、MTP、训练稳定性技巧的完整消化。
- [Hyper-Connections 超连接](Hyper-Connections%20超连接.md) — `hc_*` 三字段的理论出处。
- [LLM 推理的 GPU 硬件基础](../学习整理/LLM推理的GPU硬件基础.md) — KV cache 公式、memory-bound 结论与卡对照表的出处；本文第八、九节的账都建立在它之上。

## Sources

- 对话归档（2026-08-14）：用户提供的 V4-Pro / V4-Flash config.json 逐项分析。字段语义大部分可与 [DeepSeek-V4 论文](../../raw/deepseek/deepseek%20v4.pdf) 互证；标「推断」处（`compress_ratios` 的 4/128 归属、`o_lora_rank`、`hc_eps`、`compress_rope_theta`、`dspark_*`）为从字段名与论文结构反推，待 modeling 代码确认。
