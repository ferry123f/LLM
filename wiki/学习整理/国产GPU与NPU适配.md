# 国产 GPU/NPU 适配：生态、路径与 SGLang 实证

记录国产 AI 芯片在推理框架里怎么被接进来。**主体是从 SGLang 源码实扫出来的证据**（commit `fdebc938f7`，tag `v0.5.16`），因为「某厂商支持得怎么样」这种话，看代码量、CI 条数和文档厚度比看新闻稿准得多。

> 本篇从 [[SGlang]] §七 独立出来，方便后续持续补充。框架内部机制（算子注册、多平台分派）仍在那篇，本篇只在需要时引用。

## 一、难在哪：要替的不是一块芯片，是三层生态


「国产替代」在推理框架这一层，真正要替的是 CUDA 拖着的一整条链：

| 层 | NVIDIA | 替代它意味着 |
|---|---|---|
| **硬件 / 驱动** | CUDA Driver、NVLink | 厂商自己的驱动与互联，**这层通常最不成问题** |
| **算子库 / 运行时** | cuBLAS、cuDNN、NCCL、CUDA Runtime | 要有对等的 GEMM / 注意力 / 集合通信实现，**性能差距主要在这层** |
| **框架接入** | PyTorch 原生 `cuda` 后端 | 要么改 PyTorch，要么走它的外挂设备通道（见 §三） |
| **上层生态** | FlashAttention、flashinfer、vLLM/SGLang… | **最难的一层**：这些库是照着 CUDA 写的，每一个都要重新适配 |

**最后一层才是瓶颈**：芯片能跑矩阵乘不难，难的是几十个手写 CUDA kernel 的推理库要一个个重来。所以看一家国产卡的成熟度，**看它在主流推理框架里的代码量，比看峰值算力实在**。

## 二、框架侧的三个适配落点

SGLang 把硬件适配收敛到三个层次，**看任何一家的适配都可以按这个顺序找**：

| 落点 | 位置 | 作用 |
|---|---|---|
| **设备探测** | `srt/utils/common.py` 的 `is_npu()` / `is_musa()` / `is_hip()` / `is_xpu()` | 全局开关，决定走哪条分支 |
| **后端实现** | `srt/hardware_backend/<device>/` | 各设备自己的 attention / MoE / 量化 / graph runner |
| **注册分发** | `srt/layers/attention/attention_registry.py` | `@register_attention_backend("ascend")` 把实现挂到 `--attention-backend` 上 |

`hardware_backend/` 目录本身就是这套设计的产物，同级并列六家：`gpu`（NVIDIA）、`cpu`、`mlx`（Apple）、`xpu`（Intel）、**`npu`（昇腾）**、**`musa`（摩尔线程）**。

> **`xpu` 是 Intel 独显**（`hardware_backend/xpu/__init__.py` 明写 "XPU (Intel GPU)"），**不是昆仑芯**——名字容易想当然，实际昆仑芯走的是 §六 的插件。

还有一层更细的抽象在 `srt/platforms/`：`interface.py` + `device_mixin.py` 定义了一个平台要实现的完整 API（`is_cuda()` / `is_npu()` / `get_device_total_memory()` / `get_communicator_class()` / `verify_quantization()` / `get_torch_distributed_backend_str()` 等约 30 个方法），树内实现了 `cuda.py` / `rocm.py` / `cpu.py`。厂商要做的就是实现这套接口。**

## 三、PyTorch 的接入口：PrivateUse1

国产卡接进 PyTorch，靠的是 **dispatch key**。SGLang 注册自定义算子时按平台挑 key（`utils/common.py` 的 `direct_register_custom_op`）：

| 平台判定 | dispatch key |
|---|---|
| `is_npu()`（昇腾） | **`PrivateUse1`** |
| `is_xpu()`（Intel） | `XPU` |
| `is_musa()`（摩尔线程） | `MUSA` |
| 其余 | `CUDA` |

**`PrivateUse1` 是 PyTorch 专门给树外设备预留的通用 key**。这里有个值得记的反差：**昇腾在 SGLang 里代码最多、CI 最全，但在 PyTorch 眼里仍是「外挂设备」**——走的是通用外挂通道，而 Intel XPU、摩尔线程 MUSA 反倒有自己的专属 key。

> 这说明**「树内 / 树外」在不同项目里含义不同**：在 SGLang 树内，不代表在 PyTorch 树内。判断生态位置要分清是哪一层的树。


## 四、案例 A：华为昇腾 Ascend NPU —— 唯一全栈

唯一的**一等公民级**国产适配：覆盖 attention、MoE、量化、图捕获、投机解码、PD 分离、LoRA，并有独立 CI 与完整文档。

### 4.1 核心实现 `srt/hardware_backend/npu/`

| 子模块 | 文件 |
|---|---|
| **attention** | `ascend_backend.py`（主）、`ascend_dsv4_backend.py`（DeepSeek V4）、`ascend_gdn_backend.py`（GDN）、`ascend_hybrid_linear_attn_backend.py`（混合线性注意力）、`ascend_torch_native_backend.py`（兜底）、`mla_preprocess.py` |
| **graph_runner**（对标 CUDA Graph） | `npu_graph_runner.py`、`npu_cudagraph_backend.py`、`vit_npu_graph_runner.py`、`eagle_draft_npu_graph_runner.py`、`eagle_draft_extend_npu_graph_runner.py`、`multi_layer_eagle_draft_extend_npu_graph_runner.py` |
| **MoE** | `moe/` 下 `topk.py`、`init_routing.py`、`finalize_routing.py`、`matmul.py`、`activation.py`、`hidden_states_quant.py`、`fuseep.py` |
| **量化** | `quantization/` 下 `linear_method_npu.py`、`moe_methods.py`、`awq_kernels.py`、`gptq_kernels.py` |
| **显存管理** | `allocator_npu.py`、`memory_pool_npu.py`、`cmo.py` |
| **DeepSeek V4 专用** | `dsv4/` 下 `dsv4_allocator.py`、`dsv4_memory_pool.py`、`dsv4_req_to_token_pool.py`、`dsv4_common_hooks.py` |
| **模型模块** | `modules/` 下 `deepseek_v2_attention_mla_npu.py`、`qwen_vl_processor.py`、`glm46v_processor.py` |
| **其他** | `utils.py`、`batch_invariant_ops/npu_batch_invariant_ops.py` |

> **有独立的 graph_runner 和显存池，是「深度适配」的硬指标**：这两块最贴硬件，能自己写说明厂商吃透了内存模型与图捕获机制，不是只把算子换个名字。

### 4.2 散落在主干里的 NPU 分支

| 领域 | 文件 |
|---|---|
| PD 分离 | `srt/disaggregation/ascend/`：`conn.py`、`transfer_engine.py` |
| 通信 | `srt/distributed/device_communicators/npu_communicator.py` |
| 图编译 | `srt/compilation/npu_piecewise_backend.py` |
| MoE 分发 | `srt/layers/moe/moe_runner/ascend.py`、`srt/layers/moe/token_dispatcher/ascend_tp.py` |
| 量化 | `srt/layers/quantization/npu_mxfp4.py`、`npu_mxfp4_w4a4.py` |
| LoRA | `srt/lora/backend/ascend_backend.py` |
| torch 补丁 | `srt/utils/torch_npu_patch_utils.py` |
| 多模态生成 | `multimodal_gen/runtime/platforms/npu.py`、`layers/attention/backends/ascend_fa.py`、`layers/quantization/mxfp4_npu.py`、`mxfp8_npu.py` |

### 4.3 测试 / CI / 文档 / 镜像

- **测试**：`python/sglang/test/ascend/`（`e2e/` 下精度、多机、性能三套 utils；`gsm8k_ascend_mixin.py`、`test_mmlu.py`、`test_ascend_utils.py`、`test_npu_logging.py`）
- **CI：8 条 workflow** —— `pr-test-npu.yml`、`full-test-npu.yml`、`nightly-test-npu.yml`、`nightly-test-npu-e2e-single-node.yml`、`nightly-test-npu-e2e-multi-node.yml`、`release-docker-npu.yml`、`release-docker-npu-nightly.yml`、`diffusion-ci-gt-gen-npu.yml`
- **镜像**：`docker/npu.Dockerfile`
- **文档**：`docs_new/docs/hardware-platforms/ascend-npus/` 共 **18 篇 .mdx + 3 个子目录**（`best_practice` / `diffusion` / `model-tutorials`），含快速上手、支持的模型/特性、量化、性能测试、精度评估、profiling、环境变量、**算子开发与调优（两篇）**、Ring SP 性能、FAQ、贡献指南

### 4.4 MindSpore：接进来的不只是芯片，还有整套框架

`docs_new/.../ascend-npus/mindspore_backend.mdx` 揭示了一条容易漏掉的线——**SGLang 能跑 MindSpore 模型**：

- 树内有 `srt/models/mindspore.py`，`configs/model_config.py` 里有 `MINDSPORE = "mindspore"` 枚举
- 但**模型实现在独立包 `sgl-mindspore` 里**，树内只留接入点（`import_model_classes("sgl_mindspore.models")`）
- 依赖**昇腾 CANN 软件栈**（文档推荐 8.3.RC2），要先装好 Ascend NPU 版 SGLang 再装 `sgl-mindspore`
- 目前支持 **Qwen3（Dense 与 MoE）、DeepSeek V3/R1**

**这条的意义**：国产化不只是「换块卡」，还可以是**换掉 PyTorch 这一层**（CANN + MindSpore 是华为的完整自研栈）。而 SGLang 的接法依然是「树内留接口、实现放外部包」——**和 §六 的插件哲学完全一致**。

## 五、案例 B：摩尔线程 MUSA —— 浅，但自带 kernel 编译链

覆盖面窄（attention + 少量算子），特别之处是**在 `sgl-kernel` 里有独立的 C++ 编译入口**，即自带一套 kernel 构建体系。

| 层 | 文件 |
|---|---|
| **后端实现** | `srt/hardware_backend/musa/`：`attention/flashattention_backend.py`、`kernels/topk.py`、`layers/utils/cp_utils.py`、`utils/patch_torch.py` |
| **kernel 编译链** | `sgl-kernel/` 下 `setup_musa.py`、`pyproject_musa.toml`、`csrc/musa/`、`csrc/common_extension_musa.cc`、`include/musa/`、`include/sgl_kernel_musa_ops.h`、`python/sgl_kernel/musa.py` |
| **多模态生成** | `multimodal_gen/runtime/platforms/musa.py` |
| **测试** | `multimodal_gen/test/server/musa/`（1/2 卡 server 测试 + `perf_baselines_musa.json`）、`test/unit/musa/layers/`（rmsnorm、silu_and_mul）、`test/registered/musa/test_llm_server_smoke_musa.py` |
| **CI** | `.github/workflows/pr-test-musa.yml`、`nightly-test-musa.yml`（2 条） |
| **CI 脚本** | `scripts/ci/musa/musa_install_dependency.sh`、`rename_wheels_musa.sh` |
| **文档** | `docs_new/docs/hardware-platforms/mthreads_gpu.mdx`（1 篇） |

**两个和昇腾不同的做法**：

1. **attention 入口是用 `_is_musa` 全局开关拦截替换**（`attention_registry.py` 里 `if not _is_musa: ...` 那段），而不像 ascend 那样用 `@register_attention_backend` 注册成具名 backend。**这是更"侵入"但更省事的接法**——不用改调用方，但在主干留下了分支。
2. **算子层默认回退到 `forward_cuda` 而不是 `forward_native`**（见 [[SGlang]] §6.6 的回退表），因为 MUSA 的 API 和 CUDA 高度相似，**很多 kernel 源码一字不改就能编**。这解释了为什么它能用较少的适配代码跑起来，也解释了为什么它需要自己的 C++ 编译链——**走的是「重编译 CUDA 代码」而非「重写实现」的路线**。

## 六、其余厂商：树外插件路径

昆仑芯、寒武纪、海光、壁仞、天数、沐曦等在本仓库**没有任何代码**（实扫 `cambricon` / `biren` / `metax` / `hygon` / `iluvatar` / `tecorigin` **全部零命中**）。它们通过**插件机制**适配。

### 6.1 两个 entry_point group

`srt/plugins/__init__.py` 定义了两个标准 setuptools entry_point 组：

| 组名 | 用途 |
|---|---|
| `sglang.srt.platforms` | **平台插件**——注册一整个硬件平台 |
| `sglang.srt.plugins` | 通用插件——注册其他扩展钩子 |

厂商在自己的包里这样声明（`plugin.mdx` 原文示例）：

```toml
[project.entry-points."sglang.srt.platforms"]
my_device = "my_platform_plugin:activate"
```

### 6.2 发现与激活流程

逻辑在 `srt/platforms/__init__.py` 的 `_resolve_platform()`，**分两条路**：

**`SGLANG_PLATFORM` 已设置**（front-loading filter）：
1. 枚举 entry_points，**但不 import 任何插件模块**（只读元数据）
2. 只 `load()` + `activate()` 指定的那一个
3. **其余插件永不被 import** —— 避免拉进无关厂商的 SDK 依赖
4. 名字找不到 → `RuntimeError`（错误信息会列出已发现的插件名）
5. `activate()` 返回 `None` → `RuntimeError`（硬件不在这台机器上）

**`SGLANG_PLATFORM` 未设置**（auto-discover，全部激活后看结果）：

| 激活数 | 结果 |
|---|---|
| 0 且 `SGLANG_USE_CPU_ENGINE=1` | 回退 `CpuSRTPlatform`（**先查这条**，显式 opt-in 优先于 CUDA 可用性） |
| 0 且 CUDA 可用 | 回退 `CudaSRTPlatform` |
| 0 且 ROCm 可用 | 回退 `RocmSRTPlatform` |
| 0 且都不满足 | 回退基类 `SRTPlatform` |
| 1 | 就用它 |
| **N（多于一个）** | **`RuntimeError`，强制要求设 `SGLANG_PLATFORM`** |

> **第 3 条「其余插件永不被 import」是这套设计的精髓**。如果 import 所有插件再挑，一台只装了昆仑芯的机器会因为某个插件 `import torch_npu` 失败而起不来。**先按名字过滤、再 import**，让「装了多家插件」成为可能。
>
> **auto-discover 里「先查 `SGLANG_USE_CPU_ENGINE`」也有讲究**：让 GPU 机器上的开发者能主动走 CPU 路径调试，显式意图压过自动探测。

### 6.3 算子层的注入口

厂商拿到平台后，还要为具体算子挂实现。`MultiPlatformOp.register_oot_forward(op_cls, fn, platform_key)` 就是这个注入口——**为已有的算子类挂上自己平台的 forward，不用改主仓库一行代码**。`dispatch_forward()` 会**先查树外注册表**（`current_platform.is_out_of_tree()` 为真时），再退回树内链。详见 [[SGlang]] §6.6。

> **整体是个很聪明的解耦**：厂商把适配代码放自己的 pip 包里，主仓库不用为每家芯片背维护成本。**代价是外部无法评估**——想了解某家国产卡的支持情况，得去它自己的 fork 或 pip 包里找，**在主仓库 grep 是找不到的，零命中不等于不支持**。



## See Also

- [[SGlang]] — 本篇的来源：§6.6 讲 `MultiPlatformOp` 的平台分派与回退链、§6.5 讲 dispatch key 怎么选，是本篇 §三 与 §六.3 的机制细节。
- [[LLM分布式]] — §八 列了各家的通信后端（`npu_communicator.py` / `xpu_communicator.py` 等），是本篇在通信层的对应。
- [[LLM推理的GPU硬件基础]] — 硬件指标的通用框架；评估国产卡时同样看算力/带宽比、显存容量与互联。
- [[Vibe coding赋能推理优化]] — 本篇 §五 说摩尔线程走「重编译 CUDA」路线、§4.3 记昇腾有算子开发指南；那篇讲的是同一道题的第三种解法：让模型自动生成算子（MusaCoder）。

## 备注

- **§一 是背景铺垫，非实扫**，无源码支撑，仅供建立框架。
- **§二～§七 全部实扫自本地仓库** `d:/project/sglang`，commit `fdebc938f7`（tag `v0.5.16`）。文件清单会随版本变化。
- 本篇只有**代码量与工程投入的证据**，**不含任何性能实测**，不能据此判断某家卡跑得快不快。
- 从 [[SGlang]] §七 抽出时修正了两处旧数字：昇腾文档 16 篇 → **18 篇 .mdx + 3 子目录**、NPU CI「5 条」→ **8 条**（原文正文已列 8 条，汇总表写 5 条，自相矛盾）；并补上了原先漏记的 `srt/platforms/` 平台抽象层与 MindSpore 后端。
