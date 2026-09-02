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

## [2026-08-18] normalize | SGLang 架构与调度循环源码走读（注意力 Backend 深挖）
- 用户在 §五 表格后裸贴了三个 backend 的走读笔记（制表符缩进、无结构、句末带「-》」箭头），本轮结构化并大幅补全
- **所有说法均实扫本地仓库 `d:/project/sglang`（commit `fdebc938f7`，tag `v0.5.16`）核对后才落笔**，非训练知识
- 章节改名：「五、注意力 Backend 选型」→「五、注意力 Backend：基类契约与三种实现」（原标题只覆盖选型表，已容不下内容）
- **结构化**：裸文本拆成 5.1 可选 Backend 一览 / 5.2 基类契约 / 5.3 Triton / 5.4 FlashInfer / 5.5 TorchNative / 5.6 三者对比
- **核对属实、予以保留的**：Triton 三索引数组语义（`kv_indices` 长条 / `kv_indptr` 段落分界 / `qo_indptr` extend 时 Q 分界）；Triton 构造四步；FlashInfer 构造四步（workspace / indptr buffer / wrapper / IndicesUpdater 适配层）与 plan/run 两阶段；TorchNative 逐请求 gather + SDPA、`_run_sdpa_forward_extend` 的七步循环（含「造带空位前缀的完整 Q」）
- **补全（原笔记完全缺失——用户明说基类没写，要求补）**：
  - **基类 `AttentionBackend` 三组契约**（`base_attn_backend.py`，260 行全读）：
    ① `forward()` 由基类实现、按 `forward_mode` 分发到 `forward_decode` / `forward_extend` / `forward_mixed`（后者仅 `is_npu()` 走），idle 直接返回空张量——**三个钩子才是子类要填的空**
    ② metadata 三方法契约 `init_forward_metadata` / `_out_graph(fb, in_capture)` / `_in_graph`，按「能否录进 CUDA Graph」切分；`_in_graph` 的 lint 契约明写禁用 `.item()` / `.cpu()` / `.tolist()` / 动态 shape `torch.empty()`
    ③ 能力声明类属性 `needs_cpu_seq_lens`（Triton 覆盖为 `False`）/ `supports_ragged_verify_graph` / `use_captured_forward_metadata_for_breakable_cuda_graph` / `support_triton()`（TorchNative 覆盖为 `False`）/ `get_cuda_graph_seq_len_fill_value` / `get_indexer_metadata`
  - 注册机制：`ATTENTION_BACKENDS` 字典 + `@register_attention_backend` 挂的是**工厂函数不是类**，故 `flashinfer` 可按 `use_mla_backend` 分流到 `FlashInferAttnBackend` / `FlashInferMLAAttnBackend`
  - 点明 Triton 三索引数组即 **CSR（压缩稀疏行）布局**，`indptr` 长度恒 `bs+1`、由 `cumsum` 填出；补 `ForwardMetadata` 其余字段（`attn_logits`/`attn_lse`/`num_kv_splits`/`custom_mask`/`mask_indptr`/`window_*`）
  - 补 Triton `init_forward_metadata` 三条 mode 分支的差异（decode 无 `qo_indptr`；target_verify 用 `arange` 等步长而非 cumsum，因草稿 token 数恒定）
  - 补「为什么 plan/run 要分」：规划只依赖 batch 形状与 K/V 数值无关，plan 一次、几十层 run 复用 → 天然属「每批一次」
  - 补 FlashInfer 三类 wrapper（Ragged / Paged-prefill+verify / Decode）与 `num_wrappers` 由滑动窗口决定
  - **`plan_stream`（原笔记只留「关于 plan_stream」一句没写完）**：实为 `attention_registry.py:51-56` 里**仅当 `speculative_algorithm == "EAGLE"`** 才创建的独立 CUDA Stream，用于让 plan 与主 stream 计算重叠，使用点在 `speculative/eagle_worker_v2.py`
  - 补 TorchNative 的定位（无编译期依赖 → 兜底与对拍基准）与源码自带的 `TODO: this loop process a sequence per iter, this is inefficient`
  - 补「造带空位前缀的完整 Q」的**动因**：SDPA `is_causal=True` 按 Q/K 等长对齐推导掩码，故补齐后算完再裁掉前缀段
- **新增 5.6 三者对比表**（用户明确要求「三者的一些对比学习」）：11 个维度横向对比 + 三条结论（复杂度与性能正相关 / 越快的 backend 越多工作被挪出 forward / 基类抽象经受住三种极端实现）
- **纠正**：`init_forward_metedata` → `init_forward_metadata`（笔误）；`TorchNativeAttnBackend` 的 `init_forward_metadata` 原写「如果启用了 SWA kv pool…」属实但漏了它**几乎是空方法**这一关键对比点，已补
- **交叉呼应**：§5.2 能力声明呼应 §4.2 `get_priority` 与 §4.5 `set_kv_buffer`（同为「变化点收进窄接口」）；§5.4 `plan_stream` 呼应 §三 `event_loop_overlap` 多 stream；§5.5 冗余 Q 呼应 §八 Detokenizer「全量减上下文」（同为「宁可重算不维护复杂状态」）；§5.6 结论 3 呼应 §六 国产卡注册机制
- **存疑（标注未自动改）**：§5.4 末尾标 ⚠️——`plan_stream` 仅 EAGLE 路径创建，若用户当时读的是 `speculative/dflash_info_v2.py` 的 `_get_overlap_plan_stream` 则语境不同，待用户确认
- 版本基准从「§四与§六」扩为「§四、§五与§六」；`release/v0.5.16 线` 改为准确的 `tag v0.5.16`
- raw 未涉及；`assets/_inbox` 本次为空，无需清理
- 更新 index.md 摘要与 Updated 日期 2026-08-13 → 2026-08-18

## [2026-08-18] normalize | SGLang §五 补 Ragged KV 与 Paged KV 的区别
- 用户读 §5.5 时注意到 FlashInfer 同时持有 ragged 与 paged 两种 wrapper，要求在合适位置补两者区别
- **落点选择**：没有塞进 FlashInfer 小节，而是**新建 §5.2「前置概念：Ragged KV 与 Paged KV」**置于基类之前——因为这是「两种 KV 内存布局」的通用概念（Triton 的 CSR 也是 ragged），不是 FlashInfer 私有实现细节；原 5.2～5.6 顺延为 5.3～5.7
- **新增 §5.2 内容**（均实扫核对，commit `fdebc938f7` / tag `v0.5.16`）：
  - 两种布局 5 维对比表（物理形态 / 定位方式 / 间接层数 / 能否复用 / 类比）——核心是 **ragged 零间接、paged 过一层页表**
  - 点明 **ragged 就是 §5.4 Triton 的 CSR 布局**，并区分「ragged」一词的两层用法（数据布局 vs FlashInfer wrapper 名）
  - **为什么必须两者兼用**：extend 时 KV 有两个来源——历史前缀（早已在池中、可被共享，必然 paged）与本次新 token（刚算出的连续张量，天然 ragged）；只用 paged 会多一次「写入+间接寻址」往返，只用 ragged 则 RadixCache 前缀共享失效
  - **代价与解法**：两半 softmax 分母不同不能直接相加，需各返回 LSE 再加权合并（`forward_return_lse` / `merge_state`）；点明这与 Triton split-KV（`attn_logits`/`attn_lse`/`num_kv_splits`）是同一套在线 softmax 机制
- **§5.5 FlashInfer 内新增「extend 时 ragged 与 paged 如何分工」**：
  - `use_ragged` 的成立条件（非确定性模式 且 不在 piecewise cuda graph 且 未设 `SGLANG_FLASHINFER_USE_PAGED`；多模态与 multi-item scoring 强制 `False`）
  - **最硬的实证**：`update_single_wrapper` 里 `paged_kernel_lens` 在 `use_ragged=True` 时取 `prefix_lens`、否则取 `seq_lens`——一行代码即证明「开 ragged 时 paged 只管旧的那半截」
  - 两条子路径：`extend_no_prefix=True` 时只调 ragged 一次；有前缀命中时两 wrapper 各算一半再 `_safe_merge_state` 合并
  - **点出 `causal` 取值分裂**：ragged 半 `causal=True`（新 token 互相看，存在未来需遮）、paged 半 `causal=False`（新 token 看历史，不存在未来，遮反而错）——用以说明两半是数学上精确切开的
  - 补 **KV 写入时机差异**：`use_ragged=False` 必须先 `set_kv_buffer` 再算（paged wrapper 要从池中读新 token KV）；`use_ragged=True` 则算完才写（写池仅为后续请求留缓存）
- 细化 §5.5 构造函数第 3 步的 wrapper 清单，补 **decode 只有 paged wrapper** 及其原因（每步仅加 1 token，无「一批新 token」可言）
- §5.7 对比表新增「KV 布局」一行
- 修正 §六 引言遗留的 `release/v0.5.16 线` → `tag v0.5.16`，与文末备注统一
- 全文 573 → 643 行；§五 交叉引用（§5.2→§5.5/§5.4、§4.5、§三、§六、§八）全部复核自洽
- 更新 index.md 摘要（补 ragged/paged 要点）

## [2026-08-19] normalize | SGLang 新增 §六 算子层：custom_op 注册与多平台分派

用户在文末粘了一段 custom_op 注册机制的原始笔记（tab 缩进、无结构）。全部实扫 `d:/project/sglang`（commit `fdebc938f7`，tag `v0.5.16`）核对后，重写成独立的 §六，放在 §五（注意力 Backend）之后、原 §六（国产 GPU/NPU 适配）之前——理由：它是「注意力这一类算子」再往下一层的**通用算子**抽象，且直接给下一节的硬件适配打底（dispatch key / OOT 注册口）。原 §六/七/八 顺延为 §七/八/九，全文 §x 交叉引用与备注版本基准行同步改过。

- 纠正拼写：`register_custom_op_form_extern` → **`register_custom_op_from_extern`**（`utils/custom_op.py:197`），按错名字 grep 不到。
- 纠正表述：笔记写「注册机制在**导入时**根据当前平台绑到不同实现」。实为**两条独立的线、两个不同时机**——`direct_register_custom_op` 在注册那刻（import 期）挑 dispatch key；`MultiPlatformOp` 在对象 `__init__` 那刻定 `_forward_method`。共同点是都**不在每次 forward 现判**，运行时零分支。已拆成 6.1 的表 + 提示块讲清。
- 补全笔记未写的部分：`out_shape` 可以是 `int` 也可以是 `str`（源码 `bound.args[i]` vs `bound.arguments[name]` 两条取值路径）；4 个 `@overload` 只为类型检查器存在；`real_impl` 最终返回的是 `debug_torch_op(...)`，日志关闭时**它就是 `torch.ops.sglang.<op_name>` 本身**——这才是「对 torch.compile 是黑盒」的落点；`mutates_args` 的真实作用是喂给 `infer_schema` 让编译器不敢重排/消除该 op。
- 补全笔记只写了名字的 `register_custom_op_from_extern`：它**不是装饰器是普通函数**；两个独有参数 `out_dtype`（fp8 入 bf16 出这类）与 `computed_args`（把随 batch 变的参数移出 schema，**防 torch.compile 反复重编译**，为此要手工搬 `__name__`/`__signature__`/`__annotations__`）；幂等、无懒注册；树内仅 5 处调用。
- 补全笔记只写了一句的 `direct_register_custom_op`：不在 `custom_op.py` 而在 `utils/common.py:2514`；docstring 明说「优先用 register_custom_op」；为什么不用 `torch.library.custom_op`（通用分派开销大）；`sglang_lib = Library("sglang", "FRAGMENT")` 与算子生命周期绑定；五步（查重→infer_schema→define→impl→_register_fake）；**第四步按平台挑 dispatch key 的四行分支**（npu→`PrivateUse1` / xpu→`XPU` / musa→`MUSA` / 其余→`CUDA`），这条是和 §七 昇腾适配的接头；错误处理只吞「多引擎重复注册」那一种 RuntimeError、AttributeError 一律重抛。
- 补全笔记只有一行的 `MultiPlatformOp`：完整钩子回退表（**hip/musa 默认落 `forward_cuda`，其余落 `forward_native`**，方向不对称是因为 HIP/MUSA 的 API 贴近 CUDA）；`dispatch_forward()` 先判树外平台再走树内链；CPU 那支要求 `_is_cpu_amx_available`；`register_oot_forward` 是给树外插件的注入口（对应 §七 的 `SGLANG_PLATFORM` entry_point）；树内 21 个子类清单；`SiluAndMul` 六套实现的实例；子类可在 `__init__` 直接改写 `_forward_method` 覆盖分派（RL 复现、aiter）。
- **新增笔记完全没有的 6.7**：`MultiPlatformOp` 的第二身份是 `torch.compile` 开关（`enter/leave_torch_compile` + `torch_compile_decoration.py:_to_torch` 递归遍历）。由此点出两条线对编译器的**相反策略**——底层把 kernel 藏成不透明黑盒**绕开**编译器，`MultiPlatformOp` 反而把厂商 kernel 换成 `forward_native` **迎合**编译器，判据是该算子值不值得被 Inductor 融合。附两处补丁痕迹：FusedMoE/TopK 仅 `num_tokens == 1` 才切 native；`is_torch_compile` 幂等守卫是为 `RotaryEmbedding` 这类被多层复用的对象准备的。
- 级联：`wiki/index.md` 条目补 custom_op 段并把 Updated 改到 2026-08-19。
- 无新增存疑项；§五 里 `plan_stream` 那条旧的 ⚠️ 仍待用户确认。

## [2026-08-19] normalize | LLM分布式：结构化重写 + 与 GPU 硬件基础打通
- 原文 52 行无标题的流水笔记 → 301 行八节结构：并行策略 / 通信原语 / 通信代价 / 训练 vs 推理 / SM-Warp / 量化 / 投机采样 / SGLang 落点
- **assets 归位**：`_inbox` 20 张图 → 16 张按 `dist-<对象>-<内容>.png` 改名进 `assets/学习整理/`，笔记内 16 处引用同步换成短链；3 张 md5 相同且无人引用的重复图删除，`_inbox` 清空
- **补全 Ring All-Reduce**：原文只有一个知乎裸链接，用 4 张原本没被引用的 `_inbox` 图补出 scatter-reduce + all-gather 两阶段完整讲解，并点出「通信量与卡数无关」
- **修复投机采样表**：原为终端 ASCII 制表符粘贴，顶边框丢失、MTP 一行文字截断，重排为 markdown 表并补回「EAGLE 式 draft 用」，观点未增删
- **与 [[LLM推理的GPU硬件基础]] 结合**：重叠部分（SM/warp 层级、精度格式、互联带宽、并行通信代价）改为交叉引用而非重述；新增该篇没有的「训练 vs 推理地位重排」对照表，并把 TP 的推理收益接到那篇「decode 速度上限 = 带宽 ÷ 权重字节」的结论上
- 补充（原文只有名词或一句话的）：参数服务器瓶颈成因、ZeRO 三级、气泡占比公式、1F1B 省的是显存不是气泡、Megatron「列→行」配对、EP 负载不均、八原语的组合关系表、AWQ 与 GPTQ 的路线分野、W4 与 W8A8 在 decode/prefill 上的不同
- 核对：SGLang 参数（`--tp/pp/dp/ep-size`、`--enable-dp-attention`、`--moe-a2a-backend`、`--pp-max-micro-batch-size`、`--speculative-*`）与 `distributed/` 目录均实扫自 `d:/project/sglang` commit fdebc938f7（tag v0.5.16）；`--speculative-algorithm` 实际内置 7 种（含 NEXTN/DFLASH/DSPARK），已按源码写全
- 级联：`wiki/index.md` 补上该篇条目（此前完全缺失）；`LLM推理的GPU硬件基础.md` See Also 加反向链接
- 新增 2 处 ⚠️ 存疑：ZeRO 三级划分的出处、「先量化再加卡更划算」与「投机采样摊薄 All-Reduce」两条是我推的串联而非笔记原意

## [2026-08-19] normalize | 国产 GPU/NPU 适配独立成篇
- 新建 `wiki/学习整理/国产GPU与NPU适配.md`（213 行），内容抽自 `SGlang.md` §七
- `SGlang.md` §七 由 91 行压成 26 行的概览（三个落点 + 现状表 + 与 §6.5/§6.6 的接头），保留章节号使 §八/§九 不移位；818 → 755 行
- **抽出时核对源码，修正两处旧数字**：昇腾文档 16 篇 → 18 篇 .mdx + 3 子目录；NPU CI「5 条」→ 8 条（原文正文已列 8 条、汇总表写 5 条，自相矛盾）
- **补上原先漏记的两块**：`srt/platforms/` 平台抽象层（`interface.py` + `device_mixin.py`，约 30 个方法，树内实现 cuda/rocm/cpu）；`srt/models/mindspore.py` + `mindspore_backend.mdx` 揭示的 **CANN + MindSpore** 框架接入线（实现在独立包 `sgl-mindspore`，支持 Qwen3 与 DeepSeek V3/R1）
- **纠正一处路径不准**：树外插件加载器原写作 `srt/plugins/`，实际选择逻辑在 `srt/platforms/__init__.py::_resolve_platform()`；`srt/plugins/__init__.py` 只定义两个 entry_point 组名（`sglang.srt.platforms` / `sglang.srt.plugins`）
- 新增（原文没有的）：`SGLANG_PLATFORM` 设置/未设置两条分支的完整流程与六档 fallback 链、front-loading filter 为什么要「先按名过滤再 import」、MUSA 走「重编译 CUDA」vs 昇腾「重写实现」的路线对比、§八 待补充清单
- 级联：`index.md` 补新条目并改写 SGlang 条目末段；`SGlang.md` 内 5 处 §七 交叉引用改为指向新篇；两篇 See Also 互链
- 边界声明：新篇 §一（CUDA 三层生态）标注为背景铺垫非实扫；全篇只有代码量证据，**无任何性能实测**
## [2026-08-20] normalize | LLM分布式：新增 §五 KV 传输（NIXL / Mooncake TE）
- 文末两段裸笔记（NIXL、Mooncake TE 各一段，贴在 §备注 之后）扩写成独立的 §五，插在 §四（推理场景）与原 §五（SM/Warp）之间——理由：它和前四节同属「卡间通信」，但走的是另一条链路，接在推理场景之后最顺；原 §五～§八 顺延为 §六～§九，全文 §x 交叉引用、开篇导语（四层→五层）与备注同步改过
- **原文观点全部保留**：Dynamo 出身、模块化后端（文件系统/POSIX/RDMA）、UCX 与 Mooncake TE 都是其 RDMA 后端、UCX 支持 AMD GPU、read/write 而非 NCCL 风格、元数据走带外（TCP/ETCD）、GPU-Direct RDMA 不占 SM、Kimi 出身、按 PCIe 拓扑自动检测网卡亲和性
- **补上原文没有的源码落点**（实扫 `d:/project/sglang`，commit `fdebc938f7`，tag `v0.5.16`）：`disaggregation/{nixl,mooncake}/conn.py` 的注册/元数据/传输三步 API；NIXL 默认后端是 **UCX**（`SGLANG_DISAGGREGATION_NIXL_BACKEND`）；`MooncakeTransferEngine.initialize(..., "P2PHANDSHAKE", protocol, ...)` 默认 `rdma`；`get_kv_class()` 要求每个后端提供 KVManager/Sender/Receiver/BootstrapServer 四件套
- **实扫发现的两条原文没有的细节**：① NIXL 路径下 `initialize_xfer`/`make_prepped_xfer` 的方向参数**全是 `"WRITE"`，`"READ"` 零命中**——SGLang 走 prefill 主动推；② mooncake 的 `session_id` 是 `host:rpc_port` 而非 rank 号，印证「点对点、非集合通信」
- **把「自动检测网卡亲和性」接到参数层**：`--disaggregation-ib-device` 留空（默认 None）才触发 mooncake 自动探测，显式指定支持单设备/逗号列表/每 GPU JSON 映射/JSON 文件四种写法
- 补全后端全表：`mooncake`（默认）/ `mooncake_tcp`（`MC_FORCE_TCP=1` 且清空 ib_device 的降级路径）/ `nixl` / `mori` / `ascend` / `fake`，另有 `add_disagg_transfer_backend_choices()` 供树外插件追加
- 补 5.5 与集合通信的六项对照表（参与方 / API 形态 / 元数据 / 占不占 SM / 典型链路 / 怕什么），并点出「NCCL 占 SM、KV 传输不占」是 PD 分离能成立的前提
- 级联：`index.md` 条目补 §五 段并把「四层」改成「五层」、Updated 改到 2026-08-20；两处 §x 备注跟着renumber
- 新增 1 处 ⚠️ 存疑：5.1 的「三步」归纳与「collective 要求固定 rank 步调一致所以不能复用 NCCL」是我按 API 形态推的解释，非笔记原意
- 未决：`mori` 后端的厂商归属在本仓库内没有直接证据（只有 `mori.io` / `mori.cpp` 的 import），故正文只写「依赖外部 mori 包」，未标厂商

## [2026-08-28] normalize | 量化感知训练（QAT）
- 内容：由半成品草稿（量化定义 + 公式 + 两段实验摘抄 + 运维命令）就地重写为完整文章。修正 S/z 公式的括号缺失（s=(rmax−rmin)/(qmax−qmin)）；补齐量化基础（对称/粒度/per-group、W×A× 记号、"低比特存储 vs 低比特计算"与 W4A16 反量化机理）；补精度补救梯子（RTN→GPTQ/AWQ→QAT，含 fake quant + STE 代码）；两阶段实验按 背景→设置→五臂设计→结果→结论 完整成文（依据：用户 2026-08 在 A100 上的实测 results.json / phase2_results.json / vllm bench 日志，数字全部来自实测原值）；收束为"速度账先行 + 精度梯子"选型方法论。原运维命令收进"附：复现"。按用户要求不写试错过程，torchao 0.16 纯 Python 路径的数值误差以"数字有效范围注记"一句标注（依据：手写 fp32-scale int8 复核 23.85 vs 基线 23.88）。
- 格式：# 标题 + > 一句话 + 五章分节；公式 LaTeX 化；结果表格化；补双链 [[LLM推理的GPU硬件基础]] [[LLM分布式]] [[投机采样]] 与标签；配图改名为 quant-fp32-int8-symmetric-mapping.png。
- 级联：index.md 新增"推理优化"分组（本篇 + 投机采样 两条目）。
- 存疑：无。

## [2026-08-28] assets | 清空 _inbox（11 张截图归位）
- 读图辨认 → 描述性改名 → 移入主题目录 → 更新引用（tracked 用 git mv，untracked 用 mv）：
  - 推理优化/：quant-fp32-int8-symmetric-mapping（QAT 笔记）；specdec-suffix-tree-draft-pipeline、specdec-medusa-heads、specdec-eagle1-feat-extrapolator、specdec-eagle1-draft-tree、specdec-eagle3-multilayer-fusion、specdec-mtp-modules、specdec-dflash-arch（投机采样笔记，7 张）
  - deepseek/：deepseek-v4-hca-shared-kv-mqa、deepseek-v4-csa-lightning-indexer（DeepSeek-V4 笔记）
  - 学习整理/：deepseek-v4-pro-infratech-full-arch（SGlang中的DeepSeek.py 笔记）
- 引用更新覆盖 4 篇笔记，grep 复核 wiki/ 下无残留 "Pasted image" 活引用。

## [2026-09-02] normalize | 投机采样（wiki/推理优化/投机采样.md）+ LLM分布式 §八
- **可回退**：改动前完整备份在 `.backup/2026-09-02-pre-spec/`，含
  - `wiki/推理优化/投机采样.md`（116 行原文）
  - `wiki/学习整理/LLM分布式.md`（402 行原文）与单独切出的 `section-八-投机采样.md`
  - `_inbox-原样/`（23 张改名前的 `Pasted image *.png` 快照）
  - 回退方式：`cp .backup/2026-09-02-pre-spec/wiki/推理优化/投机采样.md wiki/推理优化/` 即可；
    `LLM分布式.md` 亦可用 `git show fffab70a33531f2529a33584d8df8a335b14c578:"wiki/学习整理/LLM分布式.md"` 取改前版本。
- **投机采样.md 116 → 290 行**：原文三个 `#` 一级标题各是一整句话，改成「零/一/二/三/四」五节 + 二级子节；
  修复被截断的粗体（`α **个 to****ken**`、`γ  **个 token**`）、标题行内嵌图片、混用的列表符号。
  **用户原有观点一字未删**（安装前逐条 grep 校验 21 个承载论点的短语，全部留存）。
- 新增（均以引用块或 ⚠️ 标注为「非原文」）：§零 共同骨架与三类分野、bonus token 白送的解释、
  「为什么在特征空间外推」、EAGLE-3 多层融合的图读、DFlash↔EAGLE 分野、DSpark Confidence Head 是调度用的、
  DFlash 2 路径选择器的相邻对打分/深度消融读数（5 层+转换器 +3% 参数 ≈ 15 层）、三组 benchmark 转写、§四 选型表。
- **对原文的更正 2 处**（在 LLM分布式 §8.4，源码核对自 `d:/project/sglang` commit `fdebc938f7` / tag v0.5.16）：
  ① `NEXTN` 不在 `SpeculativeAlgorithm` 枚举里，只是 `EAGLE` 的别名（`spec_registry.py:162` 的 `_RESERVED_ALIASES`
     + `arg_groups/speculative_hook.py:52`）；`server_args.py:1717` 的 help 文本仍列着它，help 与枚举不一致。
  ② `--speculative-accept-threshold-single/-acc`（默认均 1.0）调低会把「分布无损」变成有损档位；
     `speculative_use_rejection_sampling` 默认 False。
- **删除 1 句**（LLM分布式 §8.5）：原文「补上 §四 说的『PP 在推理里没有 micro-batch 可切』的短板」——
  投机采样并不产生可流水的 micro-batch，理由已写在该处引用块里。
- **配图归位**：`assets/_inbox/` 23 张 → 17 张 `git mv` 进 `assets/推理优化/`，按 `specdec-<方法>-<内容>.png` 命名；
  6 张经 md5 核对为重复/内容已转写进正文的图 `git rm`（原件仍在上述备份中）。`_inbox` 已清空至仅剩 `.gitkeep`。
- 校验：全库 129 条 wikilink/图片嵌入 **0 坏链**，图片文件名全局唯一（短链前提），代码围栏配平，无 tab/nbsp。
- 更新 index.md（投机采样条目重写，Updated → 2026-09-02）。
- **未处理**：`wiki/推理优化/Vibe coding赋能推理优化.md`（用户新建的 2 行素材，index 中尚无条目），本轮未动。
