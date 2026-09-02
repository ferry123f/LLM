## 八、投机采样（Speculative Decoding）

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

