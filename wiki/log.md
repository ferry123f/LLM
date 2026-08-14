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

## [2026-08-13] ingest | SGLang 国产 GPU/NPU 适配文件清单
- 素材来源：本地仓库 d:/project/sglang（commit fdebc938f7，release/v0.5.16 线）实扫，非训练知识
- 新增「六、国产 GPU/NPU 适配文件清单」，原「六、两种事件循环对比」→ 七，「七、Detokenizer」→ 八
- 内容：适配三落点（is_npu/is_musa 探测 → hardware_backend/<device>/ 实现 → attention_registry 注册）
- 昇腾 Ascend：树内全栈（attention 6 文件 / graph_runner 6 / MoE 7 / 量化 4 / dsv4 4 / 显存 3 + PD 分离、通信、LoRA、torch 补丁等散点），5 条 CI workflow，16 篇文档
- 摩尔线程 MUSA：树内浅适配（attention + topk + patch_torch），特点是 sgl-kernel 里自带独立 C++ 编译链（setup_musa.py / csrc/musa/ 等），2 条 CI，1 篇文档
- 昆仑芯等其余国产芯片：树内零命中（实扫 cambricon/biren/metax/hygon/iluvatar/tecorigin 全 0），走树外插件 SGLANG_PLATFORM + srt/plugins/
- 纠错提示：xpu 是 Intel 独显（hardware_backend/xpu/__init__.py 明写），不是昆仑芯，避免望名生义
- 更新 index.md 摘要

## [2026-08-13] ingest | LLM 推理的 GPU 硬件基础
- 新建 wiki/学习整理/LLM推理的GPU硬件基础.md（对话合成，无 raw 素材）
- 主线：以 roofline 脊点（H100 约 295 FLOP/Byte）统摄全篇，prefill=compute-bound / decode=memory-bound 是理解一切优化的钥匙
- 内容：显存四块构成与权重/KV 计算公式（MHA/GQA/MQA/MLA 对比）、单流 decode 上限=带宽÷权重、内存层级与 FlashAttention 原理、int32 vs int64 四点收益、算力精度格式与 sparsity 读数坑、TP/PP/EP/DP 通信代价、常见卡对照、容量规划实例、速查公式、硬件事实→框架对策对照表
- 交叉引用：新增 See Also 指向 SGlang / 压测详解 / DeepSeek-V3；并**反向**给 SGlang.md 与压测详解补上指回本文的条目
- 三篇构成闭环：硬件原理 → 框架实现 → 压测验证
- 更新 index.md（学习整理组新增条目，按字母序置于压测详解之前）

## [2026-08-13] normalize | LLM 推理压测（整合实操记录 + 结果分析）
- 用户把 4 条实操命令与 4 张结果截图裸贴在文末（错落在 See Also 之后，命令被压成单行、缩进用了不间断空格 U+00A0）
- 截图归位：4 张 `Pasted image 2026081316*.png` 从 vault 根 → `assets/`，改描述性名 bench-vllm-serve-run1/run2、bench-sglang-serve、bench-sglang-offline
- 新增「七、实操记录与分析：Qwen3-0.6B 四组压测」：7.1 命令（还原成规范 bash 多行）/ 7.2 结果汇总表 + 原图 / 7.3 六条分析 / 7.4 可比性质疑 / 7.5 存疑 / 7.6 复现清单
- **六条分析**（读图逐项核对数字得出，非套话）：
  ① 总吞吐被输入灌水——四组 Total 精确 = Output×9（因 input:output=8:1），跨配置只能比 Output token throughput
  ② 同一条 vLLM 命令两次运行差 1.68×（TTFT 均值 3.9× 降、P99 7.4× 降；分布由右偏转左偏）→ 冷启动 + 前缀缓存双重污染，两次数字都不算数
  ③ TTFT 1.6s 是 inf rate 一把梭的排队产物：实测 E2E 均值 4961.93 ≈ TTFT 1262.36 + 127×TPOT 29.13 = 4961.87 完全自洽，而全场仅 5.13s → 每个请求都横跨整个窗口
  ④ 离线比在线快 35.5%（6766 vs 4993 tok/s，两比值 1.357/1.355 互证）= 服务化（HTTP+SSE+ZMQ 跨进程）的定量标价
  ⑤ SGLang ITL Max 1817.85ms = 中位数的 99 倍 → prefill 优先调度抢占 decode 的现场；教训是 TPOT 均值掩盖抖动，必看 ITL P99/Max
  ⑥ retokenized 20327 vs generated 25600（少 20.6%）→ random-ids 纯随机 token id 的代价，该指标可当「数据集真实度」探针
- **明确否定错误结论**：7.4 列五条原因说明这四组数据不能用来判「SGLang vs vLLM 谁强」（数据集不同 random vs random-ids、冷热不一致、卡未隔离未记录、服务端参数全缺、单次无方差）；唯一站得住的是 ④ 的同框架纵向对比
- **⚠️ 发现文档冲突（标注未自动改）**：实测 `--random-range-ratio 1` 得到精确定长 204800，与本文 §2.1/§2.2 表格「0 表示定长」矛盾 → 7.5 标存疑 + 待办查源码，给出「用 Total input tokens 反查」的保险做法
- 顺带修 §2.1 数据集行：补 `random-ids` 条目并说明与 `random` 的区别（原表未收录，而用户实操正用了它）
- 补 See Also：把 [[SGlang]] 的描述从「SGLang 使用笔记」改为指明 §7.3 ④⑤⑥ 三条分析在那篇的机制出处
- raw 未涉及（本次素材为用户实操产出，非 raw 素材）
- 更新 index.md 摘要与 Updated 日期 2026-08-10 → 2026-08-13

## [2026-08-13] refactor | assets/ 重排为主题子目录 + 止住粘贴图乱落的根因
- 起因：用户指出 assets/ 12 张图平铺、还混着未改名的 `Pasted image 20260806170532.png`，「后面东西会越来越多，不能直接这么就扔进去」
- **根因定位**：`.obsidian/app.json` 未设 `attachmentFolderPath` → Obsidian 粘贴图默认落 **vault 根目录**、名字带时间戳。不改设置则每次都得手动搬，治标不治本
- **前置核查**：grep 全库 11 处图片嵌入**全是纯文件名短链**（不带 `assets/` 路径）→ Obsidian 靠文件名全库检索，移进子目录**不会断链**（前提是文件名全局唯一）
- **目录重排**（方案：镜像 wiki 主题分子目录）：
  - `assets/deepseek/` ← deepseek-v3-arch-mla-moe / -fp8-framework / -mtp、deepseek-v4-overall-arch / -csa / -hca、hc-connections（7 张）
  - `assets/学习整理/` ← bench-vllm-serve-run1 / -run2、bench-sglang-serve / -offline（4 张）
  - `assets/_inbox/` 新建暂存区（`.gitkeep` 占位）
  - 全部用 `git mv`，保留文件历史
- **改名遗留图**：`Pasted image 20260806170532.png` → `assets/deepseek/deepseek-v4-mhc.png`。读图确认内容为三联对比（a) 标准残差 / (b) HC / (c) mHC），故按 mHC 命名，与同目录 deepseek-v4-* 系列对齐
- 同步修 V4 笔记 L79 唯一引用：换新文件名，补前后空行（原图挤在列表与公式之间无空行），并按本文既有图注体例补一句三代残差演进的说明
- **改设置（治本）**：`.obsidian/app.json` 加 `attachmentFolderPath: "assets/_inbox"`（粘贴图落暂存区，不再污染 vault 根）+ `newLinkFormat: "shortest"`（锁死短链生成，保证以后挪目录仍不断链）
  - ⚠️ 该文件在 `.gitignore:8`，属机器专属配置**不随 git 同步**，另一台电脑需手动设一次：设置 → 文件与链接 → 新附件的默认位置 → 「在以下文件夹中」→ `assets/_inbox`
- **固化约定**：CLAUDE.md 新增「assets 规矩」5 条（落点镜像主题 / 文件名全局唯一且描述性 / 引用一律短链不写路径 / _inbox 是暂存区且每次 ingest-normalize 必须清空 / 改名必须连引用一起改），让后续录入自动遵守，不再靠临时判断
- raw 未涉及

## [2026-08-13] normalize | SGLang 架构与调度循环源码走读（RadixCache 与 KV 内存池深挖）
- 用户在 §四 RadixCache 后面裸贴了一段生笔记（radix tree 增删改查 + Token-to-page 映射），制表符缩进、无结构、有笔误
- **所有说法均实扫本地仓库 `d:/project/sglang`（commit `fdebc938f7`，release/v0.5.16）核对后才落笔**，非训练知识
- 章节改名：「四、关键数据结构：RadixCache」→「四、关键数据结构：RadixCache 与 KV 内存池」（内容已超出单一数据结构）
- **结构化**：裸文本拆成 4.1 树的增删改查（表格）/ 4.2 驱逐策略 / 4.3 lock_ref / 4.4 前缀匹配 / 4.5 Token-to-page 两级映射（含 Mermaid 图）
- **核对属实、予以保留的**：`RadixKey.match` 的「指数搜索+二分 O(log n)」（源码注释明写 gallop，无逐 token Python 循环）；`req_to_token` 形状 `[size+1, max_context_len]`；`_store_kv_layer` 确属 `MHATokenToKVPool`（曾疑心张冠李戴，查证后确认无误）
- **补全（原笔记缺失）**：
  - 驱逐策略由 3 种补到**全部 7 种**（lru 默认 / lfu / fifo / mru / filo / priority / slru），并点出统一抽象——「策略」只是一个返回排序键的 `get_priority()`，最小堆弹出最小者；开关是 `--radix-eviction-policy`
  - `evict` **会顺分支向上级联**：删掉叶子后父节点若变成无子且 lock_ref==0，立刻压回堆成为新候选（原笔记只写「驱逐叶子」）
  - `lock_ref` 的**目的**：inc/dec 真正做的是在 `evictable_size_` 与 `protected_size_` 之间搬账，使锁住的前缀不可能被驱逐；漏 dec 会导致该分支永久占显存（原笔记只写 ++/--）
  - `req_to_token` 第一维 `+1` 的**原因**：第 0 行是 CUDA Graph 定长批次凑数假请求的 padding 行（`req_pool_indices` 默认 0），`free_slots` 从 1 开始
  - allocator **三个入口分工**：`alloc`（页对齐批量）/ `alloc_extend`（prefill，复用命中前缀的半页，靠 `last_loc` 对齐，内部走 Triton kernel `alloc_extend_kernel`）/ `alloc_decode`（每步 1 token）——原笔记只提了 alloc_extend
  - `set_kv_buffer` 是基类 `KVCache` 的统一接口，MLA/FP8/FP4/page-major 各自实现 → 换 KV 布局不动 backend 与调度层
- **纠正**：`priority` 策略原写「综合方法」，实为**按请求优先级**低者先删、同级内 LRU；`ReqToTokenPool. init初始话` → `__init__` / 初始化；`-》` → `→`
- **修失效交叉引用**：§二 Detokenizer 行写「详见 §七」，但上次插入「六、国产适配」后 Detokenizer 已后移为 §八，改指 §八（上次改标题时漏改正文指路）
- 版式：把「前缀缓存虚高吞吐」的提示从内存池细节之后挪回节首（属 RadixAttention 概念范畴），消掉两个相邻引用块
- 补版本基准到「备注」，作为 §四 与 §六 共用的单一出处
- 呼应 `req_to_token` 用 int32 → 补链 [[LLM推理的GPU硬件基础]] 索引位宽一节（See Also 原本就有该条，此处为正文内联呼应）
- raw 未涉及；`assets/_inbox` 本次为空，无需清理
- 更新 index.md 摘要（Updated 仍为 2026-08-13）

## [2026-08-14] query | Archived: DeepSeek-V4 config.json 逐项详解与显存估算
- 素材：用户在对话中提供的 V4-Pro 与 V4-Flash 两份 config.json，追问「面向小白逐项讲清 + 公式与 KV cache 估算」
- 新建 wiki/deepseek/DeepSeek-V4 config.json 逐项详解与显存估算.md（归档合成答案，不合并进已有文章）
- 内容：字段按骨架/注意力/压缩注意力/长上下文/MoE/mHC/精度七组逐项解读；三笔账全程手算——总参 1.6T（385 专家×66M×61 层）、激活 49B、KV cache 基线 250 GB → MQA+latent 41 GB → 压缩条目 5.2 GB（≈2%，与论文数字互证）；部署估算 825 GB 权重需 8×H200；附 Flash 对比速览（三大创新未动、只砍尺寸、MoE 精确回退 V3 配置）
- 存疑已标注：compress_ratios 的 4/128 归属、o_lora_rank、hc_eps、compress_rope_theta 为字段名推断；Flash 独有 dspark_* 四字段不明（疑似块级投机解码，待查 modeling 代码）；Flash compress_ratios 46 项 vs 44 层对不上
- Updated: DeepSeek-V4 架构与百万上下文效率（See Also 补反向链接）
