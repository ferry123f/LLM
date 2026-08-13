# Knowledge Base Index

## DeepSeek

- [DeepSeek-V3 架构与低成本高效训练](deepseek/DeepSeek-V3%20架构与低成本高效训练.md) — DeepSeek-V3（671B / 激活 37B）用 MLA + DeepSeekMoE + 无辅助损失负载均衡 + MTP，配合 FP8 训练与 DualPipe，把完整训练成本压到 2.788M H800 GPU 时（约 $5.576M），性能追平 GPT-4o 与 Claude-3.5-Sonnet，是当时最强开源模型（含论文官方架构图 Fig2/3/6 与核心公式）。Updated: 2026-08-07
- [DeepSeek-V4 架构与百万上下文效率](deepseek/DeepSeek-V4%20架构与百万上下文效率.md) — DeepSeek-V4（Pro 1.6T / Flash 284B）用混合注意力 CSA+HCA、mHC 超连接、Muon 优化器把 1M token 上下文推理成本压到 V3.2 的 10–27%，附完整架构、训练与评测消化（含论文官方架构图 Fig2/3/4 与核心公式）。Updated: 2026-08-12
- [Hyper-Connections 超连接](deepseek/Hyper-Connections%20超连接.md) — 残差连接的可学习替代（ByteDance, ICLR 2025）：用深度连接 + 宽度连接（扩展率 n）化解 Pre-/Post-Norm 的「梯度消失↔表示崩塌」跷跷板，DHC×4 收敛快 1.8×、ARC +6 分；是 DeepSeek-V4 mHC 的理论基础（含论文 Figure 2 示意图与核心公式）。Updated: 2026-08-12

## 学习整理

- [Docker 常用命令](学习整理/docker.md) — Docker 镜像/容器核心区分 + 镜像/容器/run 参数/日志/清理/Compose 常用命令速查（已纠正镜像与容器混淆）。Updated: 2026-08-10
- [Git 常用命令](学习整理/git.md) — Git 首次配置、初始化关联远程、日常 add/commit/pull/push 上传流程，外加查历史/撤销回退/分支/.gitignore/rebase 冲突处理等高频救急命令速查。Updated: 2026-08-12
- [LLM 推理压测：bench serve 与 throughput 参数详解](学习整理/LLM推理压测-bench%20serve%20与%20throughput%20参数详解.md) — SGLang 与 vLLM 两框架的在线压测（serve）与离线吞吐（throughput）逐参数详解，含 TTFT/TPOT/ITL/goodput 指标解释、四命令对照表与上手示例。Updated: 2026-08-10
- [SGLang 架构与调度循环源码走读](学习整理/SGlang.md) — SGLang 三进程模型（Tokenizer/Scheduler/Detokenizer）+ 三层架构走读，逐步拆解 Scheduler 事件循环 recv→分发→组批→run→result：含组批优先级策略（FCFS/LPM/DFS-Weights/LOF…）、采样算法、grammar 约束、eager/CUDA Graph、RadixCache 前缀复用与注意力 backend 选型；含国产 GPU/NPU 适配文件清单（昇腾树内全栈 / 摩尔线程 MUSA / 昆仑芯等走树外插件）与 Detokenizer 增量解码（BPE 边界 / UTF-8 半字符 / trim_matched_stop）+ SSE 流式输出协议。Updated: 2026-08-13
