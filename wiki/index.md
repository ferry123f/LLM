# Knowledge Base Index

## DeepSeek

- [DeepSeek-V3 架构与低成本高效训练](deepseek/DeepSeek-V3%20架构与低成本高效训练.md) — DeepSeek-V3（671B / 激活 37B）用 MLA + DeepSeekMoE + 无辅助损失负载均衡 + MTP，配合 FP8 训练与 DualPipe，把完整训练成本压到 2.788M H800 GPU 时（约 $5.576M），性能追平 GPT-4o 与 Claude-3.5-Sonnet，是当时最强开源模型（含论文官方架构图 Fig2/3/6 与核心公式）。Updated: 2026-08-07
- [DeepSeek-V4 架构与百万上下文效率](deepseek/DeepSeek-V4%20架构与百万上下文效率.md) — DeepSeek-V4（Pro 1.6T / Flash 284B）用混合注意力 CSA+HCA、mHC 超连接、Muon 优化器把 1M token 上下文推理成本压到 V3.2 的 10–27%，附完整架构、训练与评测消化（含论文官方架构图 Fig2/3/4 与核心公式）。Updated: 2026-08-12
- [DeepSeek-V4 config.json 逐项详解与显存估算](deepseek/DeepSeek-V4%20config.json%20逐项详解与显存估算.md) — [Archived] 面向初学者：拿 V4-Pro 的 config.json 逐字段放回 Transformer 骨架（骨架/注意力/压缩/MoE/mHC/精度六组），再一步步算出 1.6T 总参、49B 激活、1M 上下文 KV cache 从基线 250 GB 砍到 5.2 GB（≈2%，与论文互证）的完整账，附 Flash 对比速览与两个未解之谜（dspark_* 字段、compress_ratios 长度不符）。Updated: 2026-08-14
- [Hyper-Connections 超连接](deepseek/Hyper-Connections%20超连接.md) — 残差连接的可学习替代（ByteDance, ICLR 2025）：用深度连接 + 宽度连接（扩展率 n）化解 Pre-/Post-Norm 的「梯度消失↔表示崩塌」跷跷板，DHC×4 收敛快 1.8×、ARC +6 分；是 DeepSeek-V4 mHC 的理论基础（含论文 Figure 2 示意图与核心公式）。Updated: 2026-08-12

## 学习整理

- [Docker 常用命令](学习整理/docker.md) — Docker 镜像/容器核心区分 + 镜像/容器/run 参数/日志/清理/Compose 常用命令速查（已纠正镜像与容器混淆）。Updated: 2026-08-10
- [Git 常用命令](学习整理/git.md) — Git 首次配置、初始化关联远程、日常 add/commit/pull/push 上传流程，外加查历史/撤销回退/分支/.gitignore/rebase 冲突处理等高频救急命令速查。Updated: 2026-08-12
- [LLM 推理的 GPU 硬件基础](学习整理/LLM推理的GPU硬件基础.md) — 从「算力/带宽失衡」（H100 约 295 FLOP/Byte）出发讲清 LLM 推理的硬件约束：prefill 是 compute-bound、decode 是 memory-bound；显存四块构成与 KV cache 计算公式（MHA/GQA/MQA/MLA 对比）；单流 decode 速度上限 = 带宽÷权重；内存层级、精度格式、TP/PP/EP 互联代价、常见卡对照与容量规划实例；附 int32 vs int64 的四点收益与速查公式。Updated: 2026-08-13
- [LLM 推理压测：bench serve 与 throughput 参数详解](学习整理/LLM推理压测-bench%20serve%20与%20throughput%20参数详解.md) — SGLang 与 vLLM 两框架的在线压测（serve）与离线吞吐（throughput）逐参数详解，含 TTFT/TPOT/ITL/goodput 指标解释、四命令对照表与上手示例；新增 Qwen3-0.6B 四组实测记录与分析：总吞吐被输入灌水（=Output×9）、同命令两跑差 1.68×（warmup/前缀缓存实证）、离线比在线快 35%、ITL Max 1817ms 暴露 prefill 抢占 decode、retokenized 少 20% 揭示 random-ids 代价，附「不可比」五条质疑与复现清单。Updated: 2026-08-13
- [SGLang 架构与调度循环源码走读](学习整理/SGlang.md) — SGLang 三进程模型（Tokenizer/Scheduler/Detokenizer）+ 三层架构走读，逐步拆解 Scheduler 事件循环 recv→分发→组批→run→result：含组批优先级策略（FCFS/LPM/DFS-Weights/LOF…）、采样算法、grammar 约束、eager/CUDA Graph；**RadixCache 与 KV 内存池深挖**——树的增删改查、7 种驱逐策略统一为 `get_priority` 排序键（含级联向上驱逐）、`lock_ref` 如何把前缀移出可驱逐账本、前缀匹配的指数搜索+二分、Token-to-page 两级映射（`req_to_token` 表的 padding 行由来、三个 alloc 入口分工）；**注意力 Backend 深挖**——**Ragged KV vs Paged KV 两种布局的区别与分工**（零间接的连续内存 vs 过页表的离散页、为何 extend 必须两者兼用、分算后按 LSE 合并）、基类 `AttentionBackend` 的三组契约（forward 按 mode 分发 / metadata 三方法与 CUDA Graph 可录制约束 / 能力声明类属性），Triton（CSR 索引 kv_indptr+kv_indices+qo_indptr）、FlashInfer（wrapper + plan/run 两阶段 + IndicesUpdater 适配层 + plan_stream + ragged/paged 双路 `causal` 取值分裂）、TorchNative（逐请求 SDPA 循环）三者实现与横向对比；含国产 GPU/NPU 适配文件清单（昇腾树内全栈 / 摩尔线程 MUSA / 昆仑芯等走树外插件）与 Detokenizer 增量解码（BPE 边界 / UTF-8 半字符 / trim_matched_stop）+ SSE 流式输出协议。Updated: 2026-08-18
