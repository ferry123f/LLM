# Knowledge Base Index

## DeepSeek

- [DeepSeek-V3 架构与低成本高效训练](DeepSeek-V3%20架构与低成本高效训练.md) — DeepSeek-V3（671B / 激活 37B）用 MLA + DeepSeekMoE + 无辅助损失负载均衡 + MTP，配合 FP8 训练与 DualPipe，把完整训练成本压到 2.788M H800 GPU 时（约 $5.576M），性能追平 GPT-4o 与 Claude-3.5-Sonnet，是当时最强开源模型（含论文官方架构图 Fig2/3/6 与核心公式）。Updated: 2026-08-07
- [DeepSeek-V4 架构与百万上下文效率](DeepSeek-V4%20架构与百万上下文效率.md) — DeepSeek-V4（Pro 1.6T / Flash 284B）用混合注意力 CSA+HCA、mHC 超连接、Muon 优化器把 1M token 上下文推理成本压到 V3.2 的 10–27%，附完整架构、训练与评测消化（含论文官方架构图 Fig2/3/4 与核心公式）。Updated: 2026-08-06

## 学习整理

- [Docker 常用命令](学习整理/docker.md) — Docker 镜像/容器核心区分 + 镜像/容器/run 参数/日志/清理/Compose 常用命令速查（已纠正镜像与容器混淆）。Updated: 2026-08-10
- [Git 常用命令](学习整理/git.md) — Git 首次配置、初始化关联仓库、日常 add/commit/pull/push 上传流程速查。Updated: 2026-08-10
- [LLM 推理压测：bench serve 与 throughput 参数详解](学习整理/LLM推理压测-bench%20serve%20与%20throughput%20参数详解.md) — SGLang 与 vLLM 两框架的在线压测（serve）与离线吞吐（throughput）逐参数详解，含 TTFT/TPOT/ITL/goodput 指标解释、四命令对照表与上手示例。Updated: 2026-08-10
