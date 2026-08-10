# Operation Log

## [2026-08-04] ingest | DeepSeek-V4 架构与百万上下文效率
- Source: raw/deepseek v4/deepseek v4.pdf (DeepSeek-AI, arXiv:2606.19348v1, preview)
- New article: 全新概念，无同主题已有文章可合并
- Updated: （无级联更新，wiki/deepseek v4 首篇文章）

## [2026-08-05] ingest | DeepSeek-V3 架构与低成本高效训练
- Source: raw/deepseek/deepseek v3.pdf (DeepSeek-AI, arXiv:2412.19437v1, 2024-12-27)
- New article: 全新概念（与 V4 同系列、不同代），新建独立文章
- Updated: DeepSeek-V4 架构与百万上下文效率（补 See Also 交叉引用 + 修复失效的 Sources 链接 raw/deepseek v4/ → raw/deepseek/）

## [2026-08-06] edit | DeepSeek-V4 架构与百万上下文效率（补图与公式）
- 加入 Mermaid 图解：Transformer 块数据流、CSA+HCA 协作、OPD 两阶段后训练
- 加入 LaTeX 公式：KV 压缩 / CSA top-k 选择 / Attention Sink、mHC 双随机矩阵约束与 Sinkhorn-Knopp、Muon Newton-Schulz 正交化
- 环境无 pdfimages/pdftoppm/pdftocairo，无法抽 PDF 原图；改用 Obsidian 原生 Mermaid + LaTeX（离线渲染、不依赖外部图片）

## [2026-08-06] edit | DeepSeek-V4 架构与百万上下文效率（改用官方原图）
- 更正上一条结论：pdfimages/pdftoppm 确实缺失，但 miniconda Python 可装 pypdfium2（自带 pdfium.dll）渲染 PDF 页面，原图**可抽**
- 抽取论文 Figure 2/3/4 存入 assets/：deepseek-v4-overall-arch.png、deepseek-v4-csa.png、deepseek-v4-hca.png
- 用官方图替换自画 Mermaid：总体架构（Fig2）、CSA（Fig3）、HCA（Fig4，原 Mermaid 未画，新增）
- 保留：mHC / Muon 的 LaTeX 公式（论文无独立示意图）、OPD 两阶段 Mermaid（论文无对应图，属自行梳理）
- raw 未改动，仅渲染读取

## [2026-08-07] edit | DeepSeek-V3 架构与低成本高效训练（补官方图与公式）
- 用 pypdfium2 抽取论文 Figure 2/3/6 存入 assets/：deepseek-v3-arch-mla-moe.png、deepseek-v3-mtp.png、deepseek-v3-fp8-framework.png
- 嵌入官方图：基本架构 MLA+DeepSeekMoE（Fig2）、MTP 串行因果链（Fig3）、FP8 混合精度框架（Fig6）
- 加入 LaTeX 公式：MLA 低秩压缩/上投影/解耦 RoPE、无辅助损失均衡的偏置门控 + 偏置更新
- raw 未改动，仅渲染读取

## [2026-08-10] edit | docker 常用命令（纠错 + 丰富）
- 修正概念错误：`docker ps` 查的是容器不是镜像；`docker exec` 参数是容器名不是镜像名
- 从 4 行扩充为分组速查：镜像操作 / 容器操作 / run 参数 / 日志文件状态 / 清理 / Compose / 常见易错点
- 补登 index.md：新增"学习整理"分组，收录 docker.md
- 注：wiki/学习整理/git.md 也未登记 index、SGlang.md 为空文件（本次未动，待确认）

## [2026-08-10] ingest | LLM 推理压测：bench serve 与 throughput 参数详解
- 来源：SGLang 与 vLLM main 分支源码（bench_serving / bench_offline_throughput / vllm bench serve / vllm bench throughput）
- 新建独立文章：跨 SGLang+vLLM 两框架的参数对照学习资料，含 serve/throughput 区别、延迟指标（TTFT/TPOT/ITL/E2EL/goodput）、四命令速查表、示例与避坑
- 记录版本差异：SGLang 老命令路径已改为弃用 shim（实现移至 sglang.benchmark.*）；vLLM 已统一为 `vllm bench` 子命令
- 补登 index.md 学习整理分组

## [2026-08-10] lint | 补登 git.md 到 index
- wiki/学习整理/git.md 已存在但未在 index 登记 → 补条目（内容未改动）
- 剩余：SGlang.md 仍为空文件，待用户确认写入或删除
