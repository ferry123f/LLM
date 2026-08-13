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

## [2026-08-12] lint | 6 issues found, 3 auto-fixed
- 自动修：index.md 两条 DeepSeek 死链（链接缺 `deepseek/` 子目录前缀，指向 wiki 根不存在的文件）→ 修正为 deepseek/DeepSeek-V3、deepseek/DeepSeek-V4
- 自动修：SGlang.md 已有内容（不再是空文件）但未登记 index → 补登「学习整理」分组
- 报告（未自动改）：SGlang.md 系用户手写半成品（无标题/frontmatter，有笔误 recv/processs/行号 L4574r），建议完善或授权 AI 整理成规范文章
- 报告：raw/deepseek/HC.pdf 未被任何 wiki 文章消化，也未在 Sources 引用（疑似 mHC/超连接原始来源）→ 可 ingest 或在 V4 mHC 章节补引
- 报告：benchmark 文章链向 [[SGlang]]，但 SGlang.md 无回链 See Also（可选补充双向交叉引用）

## [2026-08-12] edit | SGLang 架构与调度循环源码走读（半成品 → 规范文章）
- 把用户手写走读笔记整理成规范文章：分层架构 / 三进程模型 / 事件循环五步 / 组批优先级策略 / 采样算法 / grammar / eager-CUDA Graph / RadixCache / 注意力 backend 选型 / 两种 loop 对比，加 2 张 Mermaid 图
- 整理期间用户又扩充了笔记（新增优先级调度、run_batch 三步、采样算法、backend 表等）→ 一并结构化纳入
- 修正笔误：rece_requests→recv_requests、processs_input_requests→process_input_requests、handle_generate_reques→request、events_loop→event_loop；淡化不确定行号 L4574r
- 补 See Also 双向回链 benchmark 文章（解决本日 lint 报告的无回链项）
- 加版本免责声明 + 架构图溯源（Awesome-ML-SYS-Tutorial）；raw 未涉及
- 更新 index.md 摘要（去掉「半成品待完善」标注）

## [2026-08-12] ingest | Hyper-Connections（超连接）
- Source: raw/deepseek/HC.pdf（Zhu et al., ByteDance Seed, ICLR 2025, arXiv:2409.19606v3）
- 新建独立文章 wiki/deepseek/Hyper-Connections 超连接.md：全新通用概念（残差连接的替代），按概念命名而非 raw 文件名
- 抽取论文 Figure 2（残差 / HC / 深度连接 / 宽度连接 对比示意）→ assets/hc-connections.png
- 级联更新：DeepSeek-V4 mHC 章节补 HC 溯源引用与交叉链接；V4 Updated 2026-08-06 → 2026-08-12
- 补登 index.md（DeepSeek 分组）
- raw 未改动，仅渲染读取；已清理临时提取文件（_tmp_hc_text.txt / _probe_hc_p2.png）

## [2026-08-12] normalize | Git 常用命令
- 规范化 wiki/学习整理/git.md：裸文本流水账 → 加 # 标题 + > 一句话摘要
- 排版重构：按「首次配置 / 初始化提交 / 关联远程 / 日常上传 / 流程速记」分节，命令全部包进 ```bash 代码块（行内 # 注释对齐）
- 补 #git 标签；补 See Also 双链 [[docker]]（同属命令速查系列）
- 事实未改动，仅排版与结构规范化
- 首次由 normalize-note skill 执行

## [2026-08-12] normalize | Git 常用命令（内容补全，第二轮）
- 按 skill 新取向（内容优先）二次加工：原笔记只是「最小上手路径」，补齐新手日常必撞的高频缺口
- 补充章节：查看状态与历史（log/diff）、撤销与回退（restore/reset/amend）、分支操作（switch/merge）、.gitignore 用法、pull --rebase 冲突处理
- 补充说明：GitHub 已不支持密码推送需用 PAT/SSH（依据：GitHub 2021 起停用密码认证）；restore 为 Git 2.23+ 命令，老命令 checkout -- 仍等价可用
- 安全标注：git reset --hard 标 ⚠️ 不可恢复慎用；.gitignore 只对未跟踪文件生效需 rm --cached
- 依据：Git 官方文档 + 通用常识；未改动用户原有命令与流程结论
- 更新 index.md 摘要

## [2026-08-13] normalize | SGLang 架构与调度循环源码走读（整合 Detokenizer 笔记）
- 用户在上次规范化后，把一段 Detokenizer 走读笔记裸贴在文末（错落在「备注」之后），本轮结构化整合
- 新增「七、Detokenizer 事件循环：增量解码回文本」：主循环三步（sock_recv → _request_dispatcher → sock_send）+ Mermaid 流程图
- 结构化零散笔记：DecodeStatus 偏移量表（surr_offset/read_offset/decoded_text_len/sent_offset/pending）、增量=全量段−上下文段的取差逻辑、三个难点（增量翻译 / UTF-8 半字符边界 / _grouped_batch_decode + trim_matched_stop）
- 重建被压成一行的 SSE 对照表（普通 HTTP / SSE / WebSocket / HTTP chunked）
- 修正笔误：Detokenilzer→Detokenizer；顺手理顺 §五 attention-backend 引言断句
- 在 §二 Detokenizer 行加「详见 §七」前向索引；raw 未涉及（本主题无 raw 素材）
- 更新 index.md 摘要与 Updated 日期 2026-08-12 → 2026-08-13
