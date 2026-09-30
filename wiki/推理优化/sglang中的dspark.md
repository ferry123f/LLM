# sglang 中的 DSpark：源码走读

> 一句话：DSpark 在 SGLang 里的落地 = **「并行前向 + 串行小 head」的半自回归 draft（Markov head）+ 置信度驱动的变长 verify（Confidence head + SPS 预算调度）**。本文只讲**实现层**——代码怎么组织、一个 decode step 的完整数据流、预算数学、accept 语义、KV 注入与 CUDA graph 折叠。算法/论文层见 [[投机采样]] §2.7，训练层见 [[DeepSpec（Qwen3_8B+Dspark）]]。

与已有笔记的分工

| 层次 | 笔记 | 内容 |
|---|---|---|
| 算法 / 论文 | [[投机采样]] §2.7 | Markov head + Confidence head 的原理、公式、训练 loss |
| 训练落地 | [[DeepSpec（Qwen3_8B+Dspark）]] | 用 DeepSpec 训 Qwen3-8B 的 DSpark draft 头全流程 |
| **实现走读（本文）** | 本篇 | SGLang 源码里 DSpark 怎么注册、调度、verify、accept |

---

## 一图串起整个流程

> 这一节把后面"按文件走读"的内容串成一条线，用大白话讲一遍。想看代码细节回到第二～八章。

**DSpark 在干嘛（一句话）**：让一个**小模型**一次性猜出一串 token，让**大模型**一次性验这一串里哪些是对的——猜得越准、验得越省，就越快。

**一个 decode step 的旅程**（顺着一个 token 走）：

```
上一步 bonus ──锚点──► draft 猜一串 ──► 预算决定验几个 ──► target 验 ──► accept 数对的 ──► 写 KV ──► 下一步
```

1. **继承锚点**：上一步结束时，每个请求都多出 1 个 "bonus token"（target 亲自补的那个），它就是这一拍的**锚点 anchor**。
2. **draft 猜一串（半自回归）**：
   - **大头并行**：draft 模型**一次 forward**，同时算出 γ（默认 7）个位置的基础 logits——"7 个位置同时起跑"。
   - **小头串行**：但"第 2 个猜什么"依赖"第 1 个猜了什么"，所以用一个极小的 **Markov head** 逐位置加一点低秩修正、再采样（`vocab→rank→vocab` 的矩阵乘，rank 很小，几乎免费）。
   - 同时 **confidence head** 给每个 draft 打一个"会被接受"的分（STS 逐位置温度校准过）。
3. **预算：验多少**（DSpark 的灵魂）：
   - "连续接受 k 个"的概率 = `survival = 前面所有 confidence 连乘`。
   - 多花 k 个验证 token 的期望收益 = `τ(k)`；代价 = 验证变慢（查 **SPS 成本表**）。
   - 找 `θ(k) = τ(k) · SPS` 最大的 k = budget。**在"多验的收益"与"多验的耗时"之间取最优。**
4. **verify：target 验**：target 对 `[anchor, d0..d6]` 跑一次前向。变长模式只验每请求前 `verify_lens` 个（紧凑打包省算力）。
5. **accept：数连续对几个**：d0 对没对上？对上就看 d1，断了就停。`correct_len`=连续对的个数；`bonus`=第一个断掉处 target 自己的预测。
6. **落地 + 写 KV**：`commit_lens = correct_len + 1` 个 token 落地；target 的 hidden 投影成 KV 写进 cache（标准注意力通用 / MLA 专用两条路）。
7. **循环**：bonus 变下一拍的锚点，回到第 1 步。


---

## 一、代码地图

### 1.1 核心目录：`python/sglang/srt/speculative/dspark_components/`

| 文件 | 职责 |
|---|---|
| `dspark_config.py` | 从 checkpoint 解析 DSpark 配置（gamma、markov_rank、mask_token_id 等） |
| `dspark_worker_v2.py` | **顶层 worker**（`DSparkWorkerV2`），编排整个 decode 循环 |
| `dspark_draft.py` | draft 阶段：跑 draft 模型、Markov 采样、折叠进 CUDA graph |
| `dspark_verify.py` | target verify 阶段：ragged/compact verify、accept、commit |
| `dspark_planner.py` | **调度核心**：置信度预算规划 + verify 长度 top-k 调度 |
| `dspark_kv_inject.py` | 把目标模型 hidden states 注入 draft KV cache |
| `dspark_sts.py` | STS（Sequential Temperature Scaling）标定数据 |
| `dspark_sps.py` | SPS（steps-per-second）成本表 |
| `dspark_observability.py` | 观察性 / 指标 / 调试 dump |
| `dspark_block_accept_estimator.py` | block accept 概率离线估计 |
| `kernels/` | Triton / Torch 双实现算子（accept、schedule、verify_window、draft_model、attn_metadata） |

### 1.2 模型定义

- `models/dspark.py` —— **标准注意力** draft 模型（示例为 Qwen3 dense）：`VanillaMarkov` / `GatedMarkovHead` / `RNNHead` + `DSparkConfidenceHead`
- `models/deepseek_v4_dspark.py` —— **MLA 注意力** draft 模型（示例为 DeepSeek-V4 MoE）：`DeepseekV4ForCausalLMDSpark`，走 MLA + hc head

### 1.3 注册与参数

- **注册**：`speculative/spec_info.py` —— `SpeculativeAlgorithm.DSPARK` 是内置枚举；`is_dflash_family()` 判定与 DFLASH 同族；**`supports_ragged_verify()` 只有 DSPARK 返回 True**（这是它与 DFLASH 的关键差异：携带每请求不同 verify 长度的 `RaggedVerifyLayout`）。
- **参数**：`arg_groups/speculative_hook.py::_handle_dspark` —— 强制 `cuda`、`pp_size == 1`、`speculative_num_steps == 1`、`speculative_eagle_topk == 1`；解析 gamma（`--speculative-dspark-block-size`）；若 target checkpoint 里捆绑了 draft（`dspark_*` 前缀 key）则自动默认 `speculative_draft_model_path = model_path`。

### 1.4 复用 DFLASH 的零件

DSpark 大量复用 DFLASH 的基础设施：

- `dflash_info_v2.py::DFlashDraftInputV2` —— 跨 step 的 draft 状态（`bonus_tokens`、`new_seq_lens`）
- `dflash_utils.py` —— verify logits 调整、`compute_dflash_correct_drafts_and_bonus`（accept 语义唯一真源）
- `draft_worker_common.py` —— draft TP worker 构建、block 位置偏移

---

## 二、一个 decode step 的完整数据流（五步）

入口 `DSparkWorkerV2._forward_decode`（`dspark_worker_v2.py:477`）。设 batch 大小 `bs`，各请求已提交长度 `prefix_lens`，`gamma` = 每步提出的 draft 数，`verify_num_draft_tokens = gamma + 1`。

### ① 构造 draft 输入块

```python
draft_block_ids = torch.full((bs, gamma), mask_token_id)   # 全填 noise token
draft_block_ids[:, 0] = draft_input.bonus_tokens            # 第0列 = 上一步的 bonus（锚点）
```

draft 前向得到 hidden `(bs, gamma, hidden_size)`。

### ② 半自回归采样（Markov head）

```python
base_logits = lm_head(hidden)          # (bs, gamma, vocab) —— 并行一次性算
for step in range(gamma):              # 唯一"串行"的循环，每步只是小 head
    step_logits = base_logits[:, step] + markov_bias(prev_token)
    next_token  = sampler(step_logits)  # argmax 或采样
    prev_token  = next_token
```

产出 `draft_tokens (bs, gamma)` 与 `corrected_logits (bs, gamma, vocab)`（后者供采样模式 accept 用）。

### ③ 算置信度（confidence head）

```python
confidence_raw = confidence_head(draft_hidden, markov_embed)   # (bs, gamma) logit
confidence     = sigmoid(confidence_raw / sts_temperatures)    # 每个位置的接受概率
```

### ④ 预算规划 + 调度 verify 长度

```python
budget = planner.resolve_verify_token_budget(...)   # 标量：总 verify token 预算
layout = planner.schedule_layout(...)               # RaggedVerifyLayout，含每请求 verify_lens
```

### ⑤ 变长 verify + accept + 提交

```python
verify_ids_2d = cat([draft_block_ids[:, :1], draft_tokens], dim=1)   # (bs, gamma+1) = [锚点, d0..d_{γ-1}]
# target 模型对 gamma+1 个位置跑一次前向 → logits (bs, gamma+1, vocab)
correct_len, bonus, cap_trim_lens = accept(...)
commit_lens  = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
```

`new_seq_lens` 是下一个 step 的 `prefix_lens`；`bonus` 是下一个 step 的锚点（`make_next_draft_input` 把它写回 `DFlashDraftInputV2.bonus_tokens`）。

---

## 三、半自回归 draft：Markov head
| 文件                                              | 职责                                                          |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `models/dspark.py`                              | 三种 Markov head 实现 + 串行采样主循环 + 用共享 lm_head 算 base_logits     |
| `models/deepseek_v4_dspark.py`                  | V4(MLA) 版的 Markov head（带 TP 分片优化）+ 它的 `compute_base_logits` |
| `speculative/dspark_components/dspark_draft.py` | 调度编排：draft 前向、采样器、fold/eager 两条路径                           |

### 3.1 先想清楚"半自回归"为什么成立

**普通投机 draft（如 EAGLE）为什么慢**：要猜 γ 个 token，得跑 **γ 次串行 forward**——第 i 个 token 依赖第 i-1 个猜出什么，谁也没法跳步。γ 次 forward 每次都要过完整主干（attention + FFN），这就是投机采样开销的大头。

**DSpark 的关键观察**：draft 主干算的是"**每个位置的上下文表示**"，这件事**不需要知道前一个位置具体猜出了哪个 token**——它只依赖"前缀 + 锚点"，而这些上一步就定死了。真正依赖"上一个 token"的，只有最后那一小步——"用上一个 token 的 embedding 去微调当前 logits"，而这个微调只是一次低秩矩阵乘。

于是把一步拆成两部分：

| | 干什么 | 依赖 | 怎么算 |
|---|---|---|---|
| **大头（贵）** | 主干 forward + `lm_head` 得到 γ 个位置的基础 logits | 只依赖前缀，不依赖彼此 | **并行，1 次 forward** |
| **小头（便宜）** | 用上一个 token 微调当前 logits，再采样 | 依赖上一个 token | **串行，每步一个小矩阵乘** |

这就是"**半自回归**"：**大头并行、小头串行**——γ 个位置同时起跑，只有"第 2 个猜什么"依赖"第 1 个猜了什么"，这个依赖用小 head 串起来，几乎免费。

### 3.2 入口 `run_markov_block`（主循环）（models/dspark.py:32）

```python
def run_markov_block(head, base_logits, *, first_prev_tokens, hidden_states, sampler):
    prev_tokens = first_prev_tokens.long()               # 锚点 = 上一步 bonus
    for step_idx in range(proposal_len):                 # proposal_len == gamma
        step_hidden = hidden_states[:, step_idx, ...]
        step_logits = head.apply_step_logits(base_logits[:, step_idx, :],
                                              token_ids=prev_tokens, hidden_states=step_hidden)
        next_tokens = sampler(step_logits, step_idx)
        prev_tokens = next_tokens                        # 用刚采的 token 当下一步 prev
```

**逐行读**：

- `base_logits`：`(bs, gamma, vocab)`，是**循环外**就并行算好的——γ 个位置的基础 logits，互不依赖。
- 循环里每步只做两件便宜事：`apply_step_logits`（小 head 微调）+ `sampler`（argmax 或采样）。
- `prev_tokens` 从锚点开始，每步被刚采的 token 覆盖——**这是全流程唯一真正串行的依赖链**。

**要点**：贵的（attention/FFN）只算一次；串行的只有小 head。这就是"半"自回归的落地。

### 3.3 三种 head（共享 `apply_step_logits` 接口）

| head | `apply_step_logits` 做的事 | 何时用 |
|---|---|---|
| `VanillaMarkov`（dspark.py:65） | `logits + W2(W1(prev_token))` —— 纯一阶马尔可夫偏置 | 最轻，默认 |
| `GatedMarkovHead`（dspark.py:131） | `sigmoid(gate([hidden, prev_emb])) * prev_emb` 再投影 | 让 hidden 决定"信多少上一 token" |
| `RNNHead`（dspark.py:162） | 维护 RNN 状态，bias 累积整条链而非只看上一 token | 想要更长程依赖 |

**VanillaMarkov 的本质（低秩分解）**：`bias = W2 @ W1[prev]` 是一个 `V × V` **一阶马尔可夫转移矩阵**的低秩分解——`W1 = Embedding(V, r)`、`W2 = Linear(r, V)`。为什么这样省：

- 直接存 `V × V` 转移矩阵：Qwen3 的 `V ≈ 151k`，`V² ≈ 2.28×10¹⁰`（228 亿参数）——根本存不下。
- 低秩分解后：`2·V·r`，取 `markov_rank=256`（默认）即 `2×151k×256 ≈ 7.7×10⁷`（7700 万参数）。
- **压缩比 ≈ 300 倍**，而 rank=256 已足够刻画"常见 token 的转移倾向"。

**GatedMarkovHead 的直觉**：VanillaMarkov 的偏置是"固定的一阶转移概率"（不看当前语境，`W1[prev]` 对谁都一样）。Gate 版用 `gate([hidden, prev_emb])` 算一个 0~1 的权重，**hidden 决定"这个位置该信多少上一 token"**——比如上一个 token 是"the"，后面跟名词的偏置就该强一点、跟标点的就该弱一点。

**RNNHead 的直觉**：Vanilla 只看**上一个** token；RNN 维护一个 state，把**整条 draft 链**的信息累积进 bias，能捕捉"两个 token 之前"的影响。代价是循环里要更新 state，略重。

### 3.4 复用与采样

- **复用**：draft 模型**复用 target 的 `embed_tokens` 和 `lm_head`**（`attach_shared_modules`，dspark_worker_v2.py:148），只存自己的主干层 + markov head + confidence head——这也是 draft 能这么小的原因。
- **采样**：`sample_draft_block`（dspark_draft.py:159）里 `greedy_mask` 区分每个请求贪心/采样；贪心走 `argmax`，采样走 Gumbel 噪声 + `SampleStepTokens` kernel，**同一 batch 混合处理**（不同请求可以一个贪心一个采样）。

---

## 四、置信度 head 与 STS
| 文件                                                      | 职责                                                     |
| ------------------------------------------------------- | ------------------------------------------------------ |
| `models/dspark.py`                                      | `DSparkConfidenceHead` 定义 + `build_confidence_head` 工厂 |
| `models/deepseek_v4_dspark.py`                          | V4 版置信度构建 + `compute_confidence`（用 `x_post_hc` tap）    |
| `speculative/dspark_components/dspark_planner.py`       | 置信度调度入口 + STS 加载 + `build_markov_embed_stack`          |
| `speculative/dspark_components/dspark_sts.py`           | STS 校准数据结构 + 离线数据采集器                                   |
| `benchmark/dspark_sts_fit.py`                           | **STS 拟合脚本**（网格搜索最小化 ECE）                              |
| `speculative/dspark_components/dspark_observability.py` | 置信度指标探针（算 survival + 前缀标签）                             |

### 4.1 为什么需要"置信度"

预算规划（下一章）要回答一个问题：**"这个 draft 有多大概率会被 target 接受？"** 只有知道每个位置的接受概率，才能算出"多验证一个位置划不划算"。所以 draft 除了猜 token，还要**给每个猜出来的 token 打一个"会被接受"的分**——这就是 confidence head。

### 4.2 `DSparkConfidenceHead`（models/dspark.py:287）

```python
input_dim = hidden_size + (markov_rank if with_markov else 0)
self.proj = nn.Linear(input_dim, 1, bias=bias)     # 就一个线性层

def forward(self, hidden_states, markov_embed_stack=None):
    features = cat([hidden_states, markov_embed_stack], dim=-1)
    return self.proj(features).squeeze(-1)          # (bs, gamma) 的 logit

def apply_sts(self, confidence_raw):
    return torch.sigmoid(confidence_raw / self.sts_temperatures)
```

**逐点读**：

- 输入：`hidden_states`（`(bs, gamma, hidden_size)`）+ 可选 `markov_embed_stack`（`(bs, gamma, markov_rank)`）——后者是 Markov head 里那个低秩 embedding，让置信度也能"看到"上一个 token 的信息。
- 结构：**就一个线性层** `Linear(hidden_size + markov_rank, 1)`，输出 `(bs, gamma)` 的 logit——每个位置一个标量。
- `squeeze(-1)` 把最后一维 1 去掉，得到 `(bs, gamma)`。

### 4.3 从 logit 到概率：STS 温度

```python
confidence = sigmoid(confidence_raw / sts_temperatures)   # (bs, gamma)，每个位置一个概率
```

- `sts_temperatures`（STS = **Sequential Temperature Scaling**）默认 `ones`，是**按位置**的温度（离线标定后是 `gamma` 个值，位置 j 一个温度 `T_j`）。
- 作用：`sigmoid(raw/T_j)` 把原始 logit 缩放到"真实的接受概率"。温度越大，sigmoid 越平缓（越保守）；温度越小越陡。

### 4.4 STS 校准的是"survival"，不是"单个 confidence"（关键）

预算公式（下一章）用的不是单个 `confidence`，而是**连乘**：

```python
survival[b, k] = cumprod(confidence)[b, k]   # P(前 k 个 draft 全被接受)
```

**为什么这决定了 STS 怎么标定**：连乘会把每个位置的误差**放大**（`0.9⁷ ≈ 0.48`，单个都挺高、连乘就掉下来了）。所以 STS 不追求"每个位置校准准"，而是**直接对准连乘尾概率 `survival`**——用网格搜索找一组温度 `{T_j}`，使 `survival` 的预测和"实际连续接受长度"的 ECE（期望校准误差）最小。

- 标定入口：`dspark_sts.py::DSparkStsCalibration`（离线），拟合逻辑在 `dspark_sts_fit.py`（网格搜索最小化 ECE）。
- 标定用的 mask：`prefix_mask[i, j] = 1 iff j < num_correct_drafts[i]`——只用"真正被连续接受"的那段历史来拟合，让 `survival` 对齐真实接受分布。

---

## 五、预算规划（灵魂）
| 文件                                             | 职责                                                |
| ---------------------------------------------- | ------------------------------------------------- |
| `dspark_components/dspark_sps.py`              | SPS 成本表（吞吐随 batch token 数怎么变）                     |
| `dspark_components/dspark_planner.py`          | 核心公式 `compute_verify_token_budget` + relay lag 包装 |
| `dspark_components/kernels/dspark_schedule.py` | 把 budget 切成每请求 `verify_lens`（含 Triton kernel）     |

### 5.1 为什么"预算"是 DSpark 的灵魂

普通投机解码：draft 猜 γ 个，target **固定验证全部 γ+1 个**。DSpark 问了一个更细的问题：**"多验证一个 draft，到底划不划算？"**——因为：

- **多验证的收益**：这个 draft 有多大概率被接受（= 置信度 survival），接受一个就多落地一个 token。
- **多验证的代价**：target 前向的 token 数变多，batch 变大，**每步更慢**（steps-per-second 下降）。

预算规划就是**在"多验的收益"和"多验的耗时"之间找最优平衡点**，动态决定本步一共验证多少个 token，再把这些 token 分给各个请求。

### 5.2 生存概率 survival

```python
# dspark_planner.py:1101
k_survival = torch.cumprod(confidence, dim=1)   # survival[b,k] = P(前 k 个 draft 全被接受)
```

**直觉**：第 k 个 draft 只有在前 k-1 个都被接受了才"有用"（否则 accept 早就断了）。所以"多验证第 k 个位置"的边际价值 = `survival[b,k]`。

例子：某请求 7 个位置的 confidence 都是 0.9，则 `survival = [0.9, 0.81, 0.729, ...]`——越靠后的位置，边际价值越低（因为大概率前面就断了）。

### 5.3 `compute_verify_token_budget`（dspark_planner.py:942）

非可加表分支：

```
candidates        = survival[:, :max_len].flatten()   # 展平所有 (请求,位置) 的边际价值
candidates_sorted = 降序排序
prefix_sum        = cumsum(candidates_sorted)          # 花 k 个额外 token 的最优期望收益
tau(k)            = num_requests + prefix_sum[k]       # 期望"有效"token 数
batch_tokens(k)   = num_requests + k                  # verify 前向总 token 数
sps(k)            = SPS_table.lookup(batch_tokens)     # 该 batch 大小的吞吐(steps/sec)
theta(k)          = tau(k) * sps(k)                    # 目标函数
budget            = argmax_k theta(k)
```

**逐项读**：

- `candidates`：把 `(bs, max_len)` 的 survival 拉平成一条，每个元素 = "多验证这个 (请求,位置) 的边际价值"。
- `prefix_sum[k]`：**贪心地**挑边际价值最高的 k 个，其和就是"花 k 个额外 token 的最优期望收益"。
- **`tau(k) = num_requests + prefix_sum[k]`**：期望"有效"token 数。**`num_requests` = bs，不是 1**——每个请求至少会落地 1 个 bonus（保底），这是白送的收益，跟验不验证无关。
- **`batch_tokens(k) = num_requests + k`**：verify 前向实际要算的 token 数（bs 个锚点 + k 个额外 draft）。
- **`sps(k)`**：查 SPS 表——verify 前向在 `batch_tokens` 个 token 下的吞吐（steps/sec）。batch 越大，SPS 越低（越慢）。
- **`theta(k) = tau(k) * sps(k)`**：目标 = 期望收益 × 吞吐 = **"每秒能落地几个有效 token"**。取最大的 k = budget。

**一句话直觉**：`tau(k)` 随 k 增加而**递增但趋缓**（边际价值递减），`sps(k)` 随 k 增加而**递减**（batch 变大变慢），乘积 `theta(k)` 是**先升后降的山峰**，山顶就是最优 budget。这就是 DSpark 区别于其他投机解码的根本：**它不固定 verify 全部 `gamma+1`，而是动态找最优 k**。

（`SpsAdditiveCostTable` 分支本质相同，只是 `sps` 换成解析式 `1/step_time`，`step_time = bias + α(bs) + θ(bs+k)`——用解析函数逼近 SPS 曲线，省去查表。）

### 5.4 把预算分给每个请求（`ScheduleVerifyLensTopk`，kernels/dspark_schedule.py:53）

算出总预算 budget 后，要把它拆成**每个请求验证几个**：

```
每请求先保底 min_verify_len（默认 1，即至少验证锚点）
剩余 budget 按"全局 top-k 生存概率"分配给 (请求,位置) 对
verify_len[i] = min_verify_len + 被选中次数，满足 1 ≤ verify_len[i] ≤ gamma+1
```

**直觉**：总预算有限，就优先给"最有希望被接受"的 (请求,位置)——即 survival 最高的那些。一个请求的 draft 越自信，分到的验证长度越长。

### 5.5 lag 机制（HostConfidenceBudgetPlanner，dspark_planner.py:1011）

置信度是**上一拍** draft 算的，预算给**下一拍**用，中间隔 overlap 流水线：

- `lag_steps`（默认 2）：按 request 索引从 carry buffer 取出"lag 步前"的置信度（因为算完置信度到用它之间隔了几拍）。
- **freshness 过滤**：`fresh = (current_gen >= 1) & (lagged_generation == current_gen)`——旧 generation 的置信度置 1（无信息 → 全验证），防脏数据污染预算。
- `forced_budget_frac`：`dspark_sps_profiler.py` 用来钉住预算比例做 profiling 的钩子（`set_dspark_forced_budget_frac`）。

### 5.6 SPS 表没 profile 会怎样（退化行为）

SPS 表是离线 profile 出来的；如果没 profile，`build_uninitialized_sps_table` 会退化成 `[1]→[1.0]`（SPS 恒为 1）：

- 于是 `theta(k) = tau(k) * 1 = tau(k)`，而 `tau(k)` 是**单调递增**的（边际价值恒为正）。
- `argmax` 就永远取最大的 k → **budget = 全部 → verify-all**。
- 结论：**没 profile 时 DSpark 保守退化成"验证全部"，正确性没问题，只是丢掉了"动态省算力"的收益**。

---

## 六、变长 verify 与 accept 语义
| 文件                                | 职责                                                      |
| --------------------------------- | ------------------------------------------------------- |
| `dspark_planner.py`               | `schedule_layout`：verify_lens → RaggedVerifyLayout      |
| `ragged_verify.py`                | `RaggedVerifyLayout`：变长窗口的元数据（indptr 等）                 |
| `kernels/dspark_verify_window.py` | `BuildRaggedVerifyWindow`：把变长 token 打包成紧凑窗口             |
| `dspark_verify.py`                | `TargetVerifyExecutor`：跑 target 前向 + accept + finalize  |
| `dflash_utils.py`                 | `compute_dflash_correct_drafts_and_bonus`：accept 规则（右移） |
| `kernels/dspark_accept.py`        | accept/finalize/cap 的 torch + Triton 实现                 |
|                                   |                                                         |
|                                   |                                                         |

### 6.1 为什么"变长"

上一章算出每个请求的 `verify_lens`（有的验证 2 个、有的验证 8 个）。如果还按"每个请求都验证 gamma+1 个"来跑，就等于没做预算。所以 verify 阶段要支持**每个请求验证不同长度**，并把它**紧凑打包**（只算实际要验证的 token，不浪费算力）——这就是 ragged/compact verify。

### 6.2 RaggedVerifyLayout（ragged_verify.py:46）

承载变长 verify 的元数据：

| 字段 | 含义 |
|---|---|
| `verify_lens` | 每请求 verify 长度（`(bs,)`） |
| `extend_start_loc` | 每请求在紧凑 buffer 里的起始偏移 |
| `qo_indptr` | 变长 attention 的 indptr（每个请求 query 区间的起止） |
| `graph_num_tokens` | 向上取整到 CUDA graph token 桶 |

三种模式：

| 模式 | 行为 |
|---|---|
| `static` | 每请求固定 verify `gamma+1`（verify-all，无 confidence head） |
| `compact` | 按 `verify_lens` 只 verify 需要的 token，打包进紧凑前向 |
| `cap-accept` | 中间态 |

**compact 落地**（dspark_verify.py::run_compact）：`BuildRaggedVerifyWindow` 把各请求 verify_ids 按 verify_lens 拼成紧凑 `(total_tokens,)`，target 只算 `total_tokens` 个位置而非 `bs × (gamma+1)`；`ScatterCompactToStrided` 再把紧凑 logits 散射回规整布局。

### 6.3 accept 语义（dflash_utils.py:547）——唯一真源

```python
matches     = candidates[:, 1:] == target_predict[:, :-1]   # d_i 与 target 预测 d_i 的位置比
correct_len = matches.cumprod(dim=1).sum(dim=1)             # 连续匹配长度
bonus       = target_predict[arange(bs), correct_len]        # 第一个不匹配处的 target 预测
```

**先讲右移对齐**：`candidates[:, 1:]` 是 draft 猜的 `[d0, d1, ..., d_{γ-1}]`，`target_predict[:, :-1]` 是 target 在处理完锚点、d0、d1… 之后**分别**的预测。所以：

- `matches[:, 0]` 比的是 `d0`（draft 猜的第 1 个）vs target 处理完锚点后预测的下一个——两者对齐。
- 这就是 `[:, 1:]` 和 `[:, :-1]` 错一位的原因：**target 的第 i 个预测，对应 draft 的第 i+1 个 token**。

**具体例子**（单请求，gamma=3）：

```
锚点 = 你
draft 猜：  [好, 吗, ？]            → candidates = [你, 好, 吗, ？]
target 预测：[好, 吗, ？, ！]        → target_predict（处理完每个输入后的预测）

matches = [好==好, 吗==吗, ？==？] = [True, True, False]
cumprod = [True, True, False]      # 一断全断（False 之后全 False）
correct_len = 2                    # 连续对了 2 个
bonus = target_predict[2] = ？      # 第一个断掉处的 target 预测
```

| 量 | 含义 |
|---|---|
| `correct_len` | 连续被接受的 draft 数（∈ [0, gamma]） |
| `bonus` | 第一个被拒绝位置处 target 自己的预测（额外 append 的 token） |
| `commit_lens = correct_len + 1` | 本步真正落地的 token 数（draft + bonus） |

上面例子里：`commit_lens = 3`，落地 `[好, 吗, ？]`（2 个 draft + 1 个 bonus）。

**为什么 `candidates[:, 0]`（锚点）不参与比较**：锚点本身就是上一步已经提交的 bonus token，它只作为 verify 前向的输入 token，不参与"是否被接受"的判断。

### 6.4 变长 accept 的 cap 语义（关键细节）

若某请求 `verify_lens = k < gamma+1`，只验证了前 k 个位置：

```python
capped        = min(correct_len, k - 1)     # 最多接受 k-1 个 draft
cap_trim_lens = correct_len - capped        # "本来能接受、但没验证"的部分
```

worker 里（dspark_worker_v2.py:666）：

```python
accept_lens       = commit_lens              # 真正落地 = capped_correct + 1
block_accept_lens = commit_lens + cap_trim_lens   # 不设 cap 时的"满血"接受长度
```

**恒等式（记这个就够）**：

```
block_accept_lens = commit_lens + cap_trim_lens = correct_len + 1
```

含义：`block_accept_lens` 是"如果不设验证上限、本应接受的完整长度"；`cap_trim_lens` 是"因为验证窗口没开满而被截掉的部分"。`cap_trim_lens` 由 `dspark_block_accept_estimator.py` 离线估计"verify 窗口全开会多接受多少"，是调 SPS/预算的依据。

### 6.5 verify-window 的三个 kernel（compact 的骨架）

| kernel | 职责 |
|---|---|
| `BuildRaggedVerifyWindow` | 按 verify_lens 把每请求的 verify_ids 拼成紧凑 buffer（`compact_row_index` 用 cumsum + searchsorted 算每请求起止） |
| `ScatterCompactToStrided` | 把紧凑 logits 散射回 `(bs, gamma+1)` 规整布局（用 **+1 sink 技巧**：散射目标偏移 +1，空位自动沉到下一行） |
| `BuildOutTokens` / `BuildCommitInjectLayout` | 由 correct_len/bonus 组装落地 token 序列 + KV 注入布局 |

---

## 七、KV 注入
| 文件                                      | 职责                                             |
| --------------------------------------- | ---------------------------------------------- |
| `dspark_components/dspark_kv_inject.py` | `TargetHiddenKvInjector`：分叉两条路径的总开关            |
| `models/dspark.py`                      | 标准注意力路径的 `write_target_hidden_kv`              |
| `models/deepseek_v4_dspark.py`          | MLA 路径的 `write_target_hidden_kv`               |
| `mem_cache/deepseek_v4_memory_pool.py`  | MLA 融合 kernel `fused_k_norm_rope_flashmla` 的入口 |
### 7.1 为什么需要 KV 注入

下一个 step 的 draft 前向，需要"前缀 + 上一个 draft 块"的 KV。draft 自己重算前缀的 KV 就浪费了（前缀在 target 里已经算过）。所以把 **target verify 阶段的 hidden states 投影成 draft 的 KV**，直接写进 draft 的 KV pool——省掉 draft 重算前缀。

### 7.2 两条路径的分流（dspark_kv_inject.py:59）

```python
if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
    _inject_mla(...)              # MLA 路径（DeepSeek V4）
else:
    write_target_hidden_kv(...)   # 标准注意力路径（Qwen3 等）
```

**分叉键是「注意力类型」，不是「dense / MoE」**（dense/MoE 是 FFN 维度，MLA/标准注意力是 KV 维度，两者正交）。`commit_lens` 用来"只注入前 `commit_lens` 个位置的 KV"，被 cap 掉的 draft 位置不写。

### 7.3 标准注意力路径（models/dspark.py:461 `write_target_hidden_kv`）

```python
x  = target_hidden                                  # (bs, commit_len, hidden)
x  = self.target_norm(x)                            # ① norm
kv = self.kv_proj(x)                                # ② KV 投影（一次算出 k 和 v）
k  = apply_k_norm(apply_k_rope(pos, kv[..., :d]))   # ③ k 还要 norm + rope
v  = kv[..., d:]
pool.set_kv_buffer(k, v)                            # ④ 写进 draft KV pool
```

**顺序**：`norm → kv_proj → k_norm → rope → set_kv_buffer`。全是通用 torch 算子，**标准注意力模型（无论 dense 还是 MoE）走这条路**。

### 7.4 MLA 路径（models/deepseek_v4_dspark.py:654 `write_target_hidden_kv`）

```python
main_x = self.project_target_hidden(main_hidden)          # ① 投影 target hidden
kvs    = CommitKvProj.execute(main_x, wkv_linears)        # ② 压缩 KV 投影
for stage, kv in zip(self.stages, kvs):
    pool.set_swa_key_buffer_radix_fused_norm_rope(        # ③ fused norm+rope+写缓存
        layer_id, swa_loc, kv, kv_weight, eps, freqs_cis, positions)
```

第 ③ 步调的是 CUDA-only kernel **`fused_k_norm_rope_flashmla`**（mem_cache/deepseek_v4_memory_pool.py:1187）。它一个 kernel 干 5 件事（`kHeadDim=512`、`kRopeDim=64`）：

```
① RMSNorm：k = kv / sqrt(mean(kv²) + eps) * kv_weight
② 跳过判断：out_loc < 0（-1 哨兵）→ 不写
③ 算页地址：page = out_loc >> log2(page_size)，offset = out_loc % page_size
④ decoupled RoPE：只对后 64 维旋转，前 448 维不转
⑤ 量化 + 打包写缓存：前 448 维 → FP8(E4M3)+UE8M0 scale；后 64 维 → BF16；
   每 token 占 576 字节（448 字节 FP8 + 128 字节 BF16）写进 radix-tree SWA 缓存
```

**这就是难以迁移的来源**：MLA 路径的 ⑤ 绑死了 NVIDIA 的 FP8 scale 格式（UE8M0）和 FlashMLA 的字节布局（radix-tree + FP8/BF16 交错），不是通用算子能替代的（详细拆解见 §十 迁移思考）。

### 7.5 正交表（分叉键为什么是注意力类型）

|           | 标准注意力（MHA / GQA）    | MLA 注意力（KV 压缩）        |
| --------- | ------------------- | --------------------- |
| dense FFN | Qwen3、Llama 等       | /                     |
| MoE FFN   | Qwen3-MoE、Mixtral 等 | DeepSeek V2 / V3 / V4 |

KV 注入只碰 norm + KV 投影 + RoPE + 写 cache，**不碰 FFN**，所以这条分叉只看注意力：标准注意力（无论 dense 还是 MoE）走通用路径，MLA 走专用路径。DeepSeek V4 之所以难迁，是因为它「MoE + MLA」里真正卡 KV 注入的是 **MLA**。

---

## 八、CUDA graph 折叠

| 文件                                      | 职责                                           |
| --------------------------------------- | -------------------------------------------- |
| `dspark_components/dspark_draft.py`     | ① draft 采样折叠（`DsparkDraftSampler`）           |
| `dspark_components/dspark_verify.py`    | ② verify epilogue 折叠（`DsparkVerifyEpilogue`） |
| `dspark_components/dspark_worker_v2.py` | 折叠条件的层层判定                                    |
| `dspark_components/dspark_planner.py`   | ③ 变长验证的分桶录图                                  |
### 8.1 为什么需要折叠

DSpark 每步跑 **draft 前向 + target 前向 + accept + KV 注入** 四段。如果不折叠，每段都是"Python 发指令 → GPU 执行 → 结果回 Python"，中间有大量 **Python 往返 + 每次 kernel launch 的开销**。

CUDA graph 的妙处：**把"形状确定"的一长串 kernel 提前录制下来，之后一次 replay 跑完**——省掉 Python 往返和 launch 开销。

### 8.2 折叠了两段

1. **draft 采样折叠**：`DsparkDraftSampler`（dspark_draft.py:61）在 draft graph 尾部跑 `compute_base_logits + markov_head.sample_block`，写 `out` buffer（`maybe_build_draft_sampler` 决定能否折叠，否则 eager）。
2. **accept/commit 折叠**：`DsparkVerifyEpilogue`（dspark_verify.py:448）在 target-verify graph 尾部跑 `scatter + accept + finalize + KV 注入`，graph 重放后 `read_accept()` 直接读 buffer。

### 8.3 折叠条件（dspark_worker_v2.py:571）

```python
fold_eligible = (verify_epilogue is not None) and proposal.folded \
                and verify_logits_adjustments_are_noop(sampling_info) and simulate_acc_len <= 0
```

**逐项读**：

- `verify_epilogue is not None`：能构建出折叠用的 epilogue（否则只能 eager）。
- `proposal.folded`：draft 采样那端也折叠了。
- `verify_logits_adjustments_are_noop`：**没有任何 logits 后处理**——自定义 processor、penalty、vocab mask、logit bias 都会让 accept 无法折叠（这些调整没法预先录进 graph）。
- `simulate_acc_len <= 0`：没在模拟 accept 长度。

任何一项不满足 → **退回 eager**（每段单独 launch，慢但正确）。

### 8.4 ragged verify 的 graph 折叠（迁移重点）

compact ragged verify 的 graph 折叠依赖 **CUDA graph 的 token-keyed 桶**（decode_cuda_graph_runner.py:481 `_capture_ragged_verify_layout`）——因为变长 verify 的 token 数每次不同，要按 token 数分桶、每桶录一个 graph。

**这正是迁国产卡时要重点补的一块**：CUDA graph 是 NVIDIA 机制，海光有 hipGraph（移植）、昆仑芯没有（eager 降级）。短期可用 `--disable-cuda-graph` + `SGLANG_RAGGED_VERIFY_MODE=static` 走 eager 保证正确性。

---

## 九、串讲：一个 decode step 走完全部模块

> 前几章把每个模块单独拆开讲了。这一节把三~八章重新串成**一条数据流**：每个张量从哪个模块产出、形状是什么、又喂给哪个模块。第二章"五步"是骨架，这一节是它的**"带形状 + 带交叉引用"** 版本。

设 `bs=3`（3 个请求）、`gamma=7`（默认）、`hidden_size=4096`、`vocab≈151k`。数值示例只展开请求 0。

### 9.1 数据流总图

```
上一步 bonus
    │
    ▼
① 构造 draft 输入块 ─────────────────（三章）
    │ draft_block_ids (bs, gamma)
    ▼
② draft 前向（大头并行）──────────────（三章）
    │ hidden (bs, gamma, hidden_size)
    ├──────────────┬──────────────┐
    ▼              ▼              │
③ Markov 采样   ④ 置信度 head      │
（小头串行）     （四章）           │
    │              │              │
    │   draft_tokens(bs,gamma)   confidence(bs,gamma)
    ▼              ▼
⑤ 预算 + 调度 ──────────────────────（五章）
    │ budget(标量)、verify_lens(bs,)
    ▼
⑥ 变长 verify ──────────────────────（六章）
    │ target_logits + target_hidden
    ▼
⑦ accept ─────────────────────────（六章）
    │ correct_len、bonus、commit_lens
    ▼
⑧ KV 注入 ────────────────────────（七章）
   （②③⑥⑦⑧ 折叠进 CUDA graph）───（八章）
    │ 写 draft KV
    ▼
下一 step（bonus = 新锚点，回到①）
```

### 9.2 逐步串（shape + 来处 + 去向）

**① 构造 draft 输入块**（三章）

```python
draft_block_ids = full((bs, gamma), mask_token_id)  # (3,7) 全填 mask_token
draft_block_ids[:, 0] = bonus_tokens                 # 第0列 = 上一步 bonus（锚点）
```

例：请求 0 上一拍 bonus 是 token `<你>`，则 `draft_block_ids[0] = [你, mask, mask, ...]`。

**② draft 前向 → hidden**（三章：大头并行）

```python
hidden = draft_model(draft_block_ids)   # (3,7,4096)
```

这一步**并行**算 7 个位置的基础表示，互不依赖。产出 `hidden` 后**分叉两条路**：③和④都吃它。

**③ Markov 采样 → draft_tokens**（三章：小头串行）

```python
base_logits = lm_head(hidden)          # (3,7,151k) 并行
for step in range(7):                  # 串行：每步一个小矩阵乘
    step_logits = base_logits[:, step] + markov_bias(prev_token)
    prev_token = sampler(step_logits)
# → draft_tokens (3,7)
```

**④ 置信度 → confidence**（四章，与③共用 hidden）

```python
confidence = sigmoid(confidence_head(hidden, markov_embed) / sts_T)  # (3,7)
```

例：请求 0 → `[0.95, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]`。

**⑤ 预算 + 调度**（五章）——confidence 的唯一消费者

```python
survival = cumprod(confidence, dim=1)      # 请求0: [0.95,0.76,0.46,0.18,...]
τ(k) = bs + prefix_sum(排序后的 survival)    # 期望收益（bs = 每请求保底1个bonus）
θ(k) = τ(k) · SPS(k)                        # 目标函数
budget = argmax θ(k)                        # → 比如 12
layout = schedule(budget)                   # verify_lens (3,) 比如 [8,3,5]
```

含义：总预算 12 个 verify token，按 survival 高低分给 3 个请求（分别验 8 / 3 / 5 个）。

**⑥ 变长 verify**（六章）——verify_lens 的唯一消费者

```python
verify_ids = [bonus, d0..d6] 按 verify_lens 紧凑打包成 (total_tokens,)
target_logits, target_hidden = target(verify_ids)   # 只算需要的位置
```

**⑦ accept**（六章）

```python
correct_len, bonus = accept(draft_tokens, target_predict)  # 连续对几个
commit_lens = correct_len + 1                              # 落地几个
```

例：请求 0 连续对 4 个 + 1 个 bonus = 落地 5 个 token。

**⑧ KV 注入 + CUDA graph 折叠**（七章 + 八章）

```python
write_target_hidden_kv(target_hidden, commit_lens)  # target hidden → draft KV
```

- **KV 注入**（七章）：Qwen3 走 `norm→kv_proj→k_norm→rope→set_kv_buffer`（通用算子）；DeepSeek V4 走 `fused_k_norm_rope_flashmla`（CUDA-only）。
- **CUDA graph 折叠**（八章）：②③ 采样折叠进 draft graph，⑥⑦⑧ 折叠进 verify graph；满足 `fold_eligible` 就一次 replay，否则 eager。

**循环**：`bonus` 变成下一拍的锚点（`make_next_draft_input`），回到①。

### 9.3 数据依赖一句话总结

```
hidden ──┬──► ③ Markov 采样 ──► draft_tokens ──► ⑦ accept
         └──► ④ confidence ──► ⑤ budget ──► verify_lens ──► ⑥ verify ──► target_hidden ──► ⑧ KV 注入
```

**两条主线**：

1. **token 线**（产出/落地）：`hidden → Markov → draft_tokens → verify → accept → commit_lens`。
2. **调度线**（决定验几个）：`hidden → confidence → budget → verify_lens`（反作用于⑥）。

---

## 十、国芯迁移思考



### 10.1 依赖分层（按迁移难度递进）

| #   | 类别            | 组件                                     | 依赖什么                               | 海光             | 昆仑芯      |
| --- | ------------- | -------------------------------------- | ---------------------------------- | -------------- | -------- |
| 1   | 设备校验          | 硬性启动校验 + 静默降级                          | `startswith("cuda")` / `is_cuda()` | 必改             | 必改       |
| 2   | 纯 PyTorch     | Markov head、confidence head            | 无                                  | ✅ 直接用          | ✅ 直接用    |
| 3   | 纯 CPU         | 预算规划、SPS/STS 表                         | 无                                  | ✅ 直接用          | ✅ 直接用    |
| 4   | Triton 算子     | accept / schedule / verify-window      | Triton（有 torch 兜底）                 | ⚠️ 中           | ⚠️ 中     |
| 5   | CUDA graph 折叠 | draft 采样、verify epilogue、ragged 桶      | CUDA graph                         | ⚠️ 中(hipGraph) | ❌ 难(无等价) |


**两处设备校验**：

| 检查 | 位置 | 性质 | 迁移动作 |
|---|---|---|---|
| 硬性启动校验 | `speculative_hook.py:277` 的 `startswith("cuda")` | **raise**，非 CUDA 直接拒启动 | **必须改**，放行 `hip` / `xpu` |
| 静默降级×2 | worker `dspark_worker_v2.py:195/298` 的 `is_cuda()` | 跳过 graph 折叠 → eager | 不拆也能跑（慢），建议显式打日志 |

### 10.2 分阶段迁移流程

**阶段 1：改设备校验**（两卡通用，量级：半天）

- 改硬性启动校验：放行 `hip` / `xpu`（或改设备白名单）。
- 静默降级可暂不拆，但建议显式处理。
- **退出条件**：服务能启动、能走进 DSpark 代码路径。

**阶段 2：标准注意力模型 + eager 跑通正确性**（两卡通用，先做这个）

- 用 Qwen3 这类 dense + 标准注意力模型（**避开 MLA**）。
- `--disable-cuda-graph` + `SGLANG_RAGGED_VERIFY_MODE=static` 走 eager。
- 第 4 类 Triton 算子靠 **torch 兜底**。
- **退出条件**：对拍 NVIDIA，输出逐 token 一致。
- 难度：海光中低、昆仑芯中低（前提：厂商 PyTorch 覆盖基础算子）。

**阶段 3：恢复 Triton 算子性能**（accept / schedule / verify-window）

- 海光：试 Triton AMD backend 编译，不支持的退回 torch 版。
- 昆仑芯：用 **FlagGems**（昆仑芯的 Triton 工具链）重写/适配；或保持 torch 版。

**阶段 4：恢复 CUDA graph 折叠**（性能关键）

- 海光：**hipGraph 移植**（token-keyed 桶 `decode_cuda_graph_runner.py:481` + 两个 epilogue 折叠）。
- 昆仑芯：无 CUDA graph 等价 → **只能 eager**（这是昆仑芯的性能天花板）。



### 10.3 两卡关键差异

|         | 海光 DCU               | 昆仑芯 XPU             |
| ------- | -------------------- | ------------------- |
| 生态      | AMD ROCm / HIP       | 百度自研                |
| CUDA 生态 | HIP （类CUDA）          | 无                   |
| Triton  | mainline AMD backend | FlagGems（自研 Triton） |
| graph   | hipGraph（可移植）        | 无等价 → eager         |
| FP8     | 有（AMD 格式）            | 有（自家格式）             |
| 迁移性质    | 翻译为主                 | 重写为主                |








---

## See Also

- [[投机采样]] —— §2.7 是本文的算法/论文层（Markov head、Confidence head 的公式与训练 loss），§2.9 是 DSpark 的实测 accept_len 对照
- [[DeepSpec（Qwen3_8B+Dspark）]] —— DSpark draft 头的训练全流程（本文不涉及训练）
- [[SGlang]] —— 本文依赖的 SGLang 调度/CUDA Graph/RadixCache 背景
- [[国产GPU与NPU适配]] —— DSpark 迁国产卡时，`dspark_components/kernels/` 的 Triton 算子与 CUDA graph 折叠是要重点攻的两块

## 备注

- 本文是**源码走读**，代码基准为 SGLang commit `fdebc938f7`（`release/v0.5.16`，2026-07-24），路径均相对于 `python/sglang/`。
- 文中「文件:行号」为实扫记录，跨版本可能漂移，以函数名/类名为准。
- 预算公式、accept 的 `correct_len/bonus/cap_trim_lens` 语义、KV 注入两条路径，均为从源码 `dflash_utils.py`、`dspark_planner.py`、`dspark_kv_inject.py` 逐行推导所得，未参照官方文档。
