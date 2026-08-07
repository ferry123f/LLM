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
