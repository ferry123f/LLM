# LLM 推理压测：bench serve 与 throughput 参数详解

覆盖 **SGLang** 和 **vLLM** 两个框架的两类基准测试命令，逐参数说明含义。数据取自两个项目 main 分支源码（2026-08）。

---

## 一、先搞懂：serve 和 throughput 到底测什么

这是两个**目标完全不同**的测试，别混：

| | **serve（在线服务压测）** | **throughput（离线吞吐）** |
|---|---|---|
| 测什么 | 模拟真实并发流量下的**延迟**（TTFT/TPOT/ITL）+ 吞吐 | 一把梭灌入全部请求，测**极限吞吐**（token/s、req/s） |
| 需要 server 吗 | **要**。先另起一个推理服务，再对它发请求（client） | **不要**。在进程内直接起引擎，跑完即退 |
| 核心控制量 | 到达速率 `--request-rate` + 并发上限 `--max-concurrency` | 批量大小 / prompt 数，不模拟"到达过程" |
| 回答的问题 | "这套服务在 X QPS 下用户体验（首字延迟等）如何" | "这张卡这个模型最多能榨出多少 token/s" |

一句话：**serve 关心"用户等多久"，throughput 关心"总共多快"。**

### 关键延迟指标（serve 会报，务必理解）

- **TTFT**（Time To First Token，首 token 延迟）：从发请求到吐出第一个 token 的时间。**决定"感觉卡不卡"**，受 prefill 和排队影响。
- **TPOT**（Time Per Output Token）：进入 decode 后，平均每个输出 token 的耗时。**决定"打字速度"**。
- **ITL**（Inter-Token Latency，token 间延迟）：相邻两个输出 token 的间隔，比 TPOT 更细（能看抖动）。
- **E2EL**（End-to-End Latency）：单请求端到端总耗时。
- **Goodput**（有效吞吐）：只统计**满足 SLO**（如 TTFT<200ms 且 TPOT<50ms）的请求吞吐，比裸吞吐更贴近真实可用性。

---

## 二、SGLang

> ⚠️ 路径变更：老命令 `python -m sglang.bench_serving` / `sglang.bench_offline_throughput` 现在只是**弃用转发壳（deprecation shim）**，真正实现已移到 `sglang.benchmark.serving` / `sglang.benchmark.offline_throughput`。老命令暂时仍能用，但会告警。

### 2.1 `sglang.bench_serving`（在线服务压测）

**前置**：先起服务 `python -m sglang.launch_server --model-path <model> --port 30000`，再跑压测 client。

#### 后端 / 连接
| 参数 | 默认 | 含义 |
|------|------|------|
| `--backend` | `sglang` | 后端类型：sglang / sglang-oai / vllm / lmdeploy / trt 等。决定用哪套 API 协议发请求 |
| `--base-url` | 无 | 直接给服务完整 URL（替代 host+port） |
| `--host` / `--port` | `0.0.0.0` / 按引擎 | 服务地址 |
| `--ready-check-timeout-sec` | `60` | 等服务就绪的最长秒数，`0` 跳过 |

#### 数据集
| 参数 | 默认 | 含义 |
|------|------|------|
| `--dataset-name` | `sharegpt` | 数据集：`sharegpt`（真实对话）/`random`（从真实语料采样再截断到目标长度，**是自然文本**）/`random-ids`（**纯随机 token id**，天然绕开前缀缓存但「不像人话」，代价见 §7.3 ⑥）/`generated-shared-prefix`（测前缀缓存命中）/`mmmu`/`image` 等 |
| `--dataset-path` | `""` | 数据集文件路径 |
| `--num-prompts` | `1000` | 总请求数 |
| `--sharegpt-output-len` | 无 | 覆盖 ShareGPT 自带的输出长度 |

#### 随机数据集专用（`--dataset-name random`）
| 参数 | 默认 | 含义 |
|------|------|------|
| `--random-input-len` | `1024` | 每请求输入 token 数 |
| `--random-output-len` | `1024` | 每请求输出 token 数 |
| `--random-range-ratio` | `0.0` | 输入/输出长度的随机浮动比例，`0` 表示定长 |

#### 流量模式（serve 的灵魂）
| 参数 | 默认 | 含义 |
|------|------|------|
| `--request-rate` | `inf` | **每秒请求数（QPS）**。`inf`=瞬间全发（压满）；有限值=按泊松过程生成到达时刻，模拟真实流量 |
| `--max-concurrency` | 无 | **并发上限**，模拟上层限流。与 request-rate 配合可精确控制"排队" |

> 💡 常见做法：固定 `--max-concurrency` 扫不同值，或固定并发扫 `--request-rate`，画出"吞吐 vs 延迟"曲线找拐点。

#### 采样 / 生成
| 参数 | 默认 | 含义 |
|------|------|------|
| `--temperature` | `0.0` | 采样温度 |
| `--top-p` | `1.0` | 核采样 |
| `--disable-stream` | off | 关闭流式（关了就测不了 TTFT/ITL） |
| `--disable-ignore-eos` | off | 默认忽略 EOS 强制生成满 output-len；加此项则遇 EOS 就停 |
| `--apply-chat-template` | off | 套用聊天模板 |
| `--extra-request-body` | 无 | 追加 JSON 到请求体（塞额外采样参数） |
| `--seed` | `42` | 随机种子 |

#### 输出 / 观测
| 参数 | 默认 | 含义 |
|------|------|------|
| `--output-file` | 无 | 结果写入 JSONL |
| `--output-details` | off | 输出每请求明细 |
| `--disable-tqdm` | off | 关进度条 |
| `--cache-report` | off | 附带缓存命中统计（仅 sglang 后端） |
| `--flush-cache` | off | 压测前清空缓存（避免前缀缓存污染结果） |
| `--profile` | off | 开 Torch Profiler（服务需设 `SGLANG_TORCH_PROFILER_DIR`） |
| `--warmup-requests` | `1` | 正式测量前的预热请求数 |

其它进阶：`--return-logprob` / `--top-logprobs-num`（测 logprob）、`--lora-name` + `--lora-request-distribution`（LoRA 压测）、`--pd-separated`（测 PD 分离服务）、`--tokenize-prompt`（用 token id 精确控长）。

---

### 2.2 `sglang.bench_offline_throughput`（离线吞吐）

**无需起服务**，进程内直接拉引擎。除下列 BenchArgs 外，还接受全部 `ServerArgs`（`--model-path`、`--tp-size`、`--mem-fraction-static` 等引擎参数）。

| 参数 | 默认 | 含义 |
|------|------|------|
| `--backend` | `engine` | `engine`（进程内引擎，标准离线）或 `runtime` |
| `--dataset-name` | `sharegpt` | `sharegpt` / `random` / `generated-shared-prefix` |
| `--dataset-path` | `""` | 数据集路径 |
| `--num-prompts` | `1000` | 灌入的总 prompt 数 |
| `--random-input-len` | `1024` | 随机集：输入 token 数 |
| `--random-output-len` | `1024` | 随机集：输出 token 数 |
| `--random-range-ratio` | `0.0` | 随机集：长度浮动比例 |
| `--sharegpt-output-len` | 无 | 覆盖 ShareGPT 输出长度 |
| `--result-filename` | `""` | 结果输出文件 |
| `--seed` | `42` | 随机种子 |
| `--skip-warmup` | off | 跳过预热批次 |
| `--disable-ignore-eos` | off | 遇 EOS 即停（否则强制生成满长度） |
| `--extra-request-body` | 无 | 追加 JSON 采样参数 |
| `--apply-chat-template` | off | 套聊天模板 |
| `--profile` | off | Torch Profiler |
| `--profile-steps` | 无 | profile 多少步 |

**共享前缀集专用**（`--dataset-name generated-shared-prefix`，测前缀缓存收益）：
`--gsp-num-groups`(64) 组数、`--gsp-prompts-per-group`(16) 每组请求数、`--gsp-system-prompt-len`(2048) 共享系统提示长度、`--gsp-question-len`(128) 问题长度、`--gsp-output-len`(256) 输出长度。

---

## 三、vLLM

> vLLM 现在统一为 `vllm bench <子命令>`。老脚本 `python benchmarks/benchmark_serving.py` 已并入 `vllm bench serve`。

### 3.1 `vllm bench serve`（在线服务压测）

**前置**：先起服务 `vllm serve <model> --port 8000`。

#### 连接 / 后端
| 参数 | 默认 | 含义 |
|------|------|------|
| `--backend` | `openai` | 后端协议 |
| `--base-url` | 无 | 完整服务 URL |
| `--host` / `--port` | `127.0.0.1` / `8000` | 服务地址 |
| `--endpoint` | `/v1/completions` | API 路径 |
| `--header` | 无 | 追加自定义请求头（`KEY=VALUE`） |
| `--max-concurrency` | 无 | 并发上限（同 SGLang） |

#### 模型 / 分词
| 参数 | 默认 | 含义 |
|------|------|------|
| `--model` | 无 | 模型名/路径；不填则查 `/v1/models` |
| `--tokenizer` | 无 | 分词器路径 |
| `--served-model-name` | 无 | 服务端注册的模型名 |
| `--logprobs` | 无 | 每 token 返回的 logprob 数 |
| `--use-beam-search` | off | 用 beam search |

#### 流量 / 速率控制（比 SGLang 更细）
| 参数 | 默认 | 含义 |
|------|------|------|
| `--request-rate` | `inf` | QPS；`inf`=全发，有限值=泊松/伽马到达 |
| `--burstiness` | `1.0` | **突发度**。仅当 rate≠inf 生效；`1`=泊松（标准），`<1` 更突发，`>1` 更均匀 |
| `--ramp-up-strategy` | 无 | 速率**爬坡**：`linear`/`exponential`，配合下面两个用 |
| `--ramp-up-start-rps` / `--ramp-up-end-rps` | 无 | 爬坡起止 QPS，测服务在负载渐增下的表现 |
| `--probe-request-rate` | `0.0` | 额外发单 token 探针请求（绕过并发上限），单独报延迟 |

#### 采样参数（仅 openai 兼容后端）
`--top-p`、`--top-k`、`--min-p`、`--temperature`、`--frequency-penalty`、`--presence-penalty`、`--repetition-penalty`，均默认 `None`（用服务端默认）。

#### 指标报告
| 参数 | 默认 | 含义 |
|------|------|------|
| `--percentile-metrics` | `ttft,tpot,itl` | 要算分位数的指标，可选 `e2el` |
| `--metric-percentiles` | `99` | 报哪些分位，如 `"50,90,99"` |
| `--goodput` | 无 | 设 SLO 算有效吞吐，格式 `ttft:200 tpot:50 e2el:5000`（毫秒） |

#### 执行 / 输出
| 参数 | 默认 | 含义 |
|------|------|------|
| `--num-warmups` | `0` | 预热请求数 |
| `--ignore-eos` | off | 忽略 EOS 强制生成满长度（控变量必开） |
| `--disable-tqdm` | off | 关进度条 |
| `--profile` | off | vLLM profiling（服务需配 `--profiler-config`） |
| `--save-result` | off | 保存结果 JSON |
| `--save-detailed` | off | 存每请求明细 |
| `--append-result` | off | 追加到已有 JSON |
| `--result-dir` / `--result-filename` | 无 | 结果目录/文件名 |
| `--metadata` | 无 | 往结果里塞运行元信息（`KEY=VALUE`） |
| `--plot-timeline` | off | 生成 HTML 时间线图 |

> 数据集参数（`--dataset-name`、`--input-len` 等）来自共享的 `add_dataset_parser`：支持 `sharegpt`/`random`/`sonnet`/`burstgpt`/`hf` 等。

---

### 3.2 `vllm bench throughput`（离线吞吐）

**无需起服务**，进程内 `LLM` 类。除下列外，还接受全部 `AsyncEngineArgs`（`--model`、`--tensor-parallel-size`、`--max-num-seqs`、`--gpu-memory-utilization`、`--quantization` 等）。

#### 数据 / 长度
| 参数 | 默认 | 含义 |
|------|------|------|
| `--backend` | `vllm` | `vllm` / `hf` / `mii` / `vllm-chat` |
| `--dataset-name` | `sharegpt` | `sharegpt`/`random`/`sonnet`/`burstgpt`/`hf`/`prefix_repetition` 等 |
| `--dataset-path` | 无 | 数据集路径 |
| `--input-len` | 无 | 每请求输入长度（随机集） |
| `--output-len` | 无 | 每请求输出长度，覆盖数据集自带 |
| `--num-prompts` | `1000` | 总请求数 |
| `--n` | `1` | 每 prompt 生成几条序列 |
| `--prefix-len` | `0` | 随机上下文前的固定前缀 token 数（测前缀缓存） |
| `--num-warmups` | `0` | 计时前预热的 prompt 数 |
| `--no-oversample` | off | 数据不足 num-prompts 时不重复采样 |

#### 引擎行为 / 输出
| 参数 | 默认 | 含义 |
|------|------|------|
| `--async-engine` | off | 用异步引擎（`AsyncLLMEngine`）而非同步 `LLM` |
| `--disable-detokenize` | off | 不解码输出（把 detokenize 时间排除出测量） |
| `--prequeue-requests` | off | 先把所有请求入队再启动调度，**提升可复现性**但可能压低测得吞吐 |
| `--output-json` | 无 | 结果存 JSON |
| `--profile` | off | vLLM profiling |

#### LoRA / HF
`--lora-path` + `--lora-assignment`(`random`/`round-robin`)；HF 后端：`--hf-max-batch-size`、`--hf-enable-torch-compile`、`--hf-subset`/`--hf-split`/`--hf-name`。

---

## 四、四命令速查对照

| 维度 | SGLang serving | SGLang offline_tp | vLLM serve | vLLM throughput |
|------|---------------|-------------------|-----------|-----------------|
| 起服务 | 要 | 不要 | 要 | 不要 |
| 调用 | `python -m sglang.bench_serving` | `python -m sglang.bench_offline_throughput` | `vllm bench serve` | `vllm bench throughput` |
| 请求数 | `--num-prompts` | `--num-prompts` | `--num-prompts` | `--num-prompts` |
| 随机输入长 | `--random-input-len` | `--random-input-len` | `--input-len`(数据集) | `--input-len` |
| 随机输出长 | `--random-output-len` | `--random-output-len` | `--output-len` | `--output-len` |
| 流量速率 | `--request-rate` | —（一把梭） | `--request-rate`(+`--burstiness`) | —（一把梭） |
| 并发上限 | `--max-concurrency` | — | `--max-concurrency` | `--max-num-seqs`(引擎) |
| 强制满长度 | `--disable-ignore-eos` 反向 | `--disable-ignore-eos` 反向 | `--ignore-eos` | `--ignore-eos`(引擎侧) |
| 有效吞吐 SLO | — | — | `--goodput` | — |

---

## 五、上手示例

```bash
# ① SGLang 在线压测：先起服务，再固定 16 并发、定长 1024/512 压测
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
python -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input-len 1024 --random-output-len 512 \
  --max-concurrency 16 --num-prompts 500

# ② SGLang 离线吞吐：无需起服务，直接测极限吞吐
python -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random --random-input-len 1024 --random-output-len 512 --num-prompts 1000

# ③ vLLM 在线压测：扫 QPS，带 SLO 有效吞吐
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
vllm bench serve --model meta-llama/Llama-3.1-8B-Instruct --dataset-name random \
  --random-input-len 1024 --random-output-len 512 \
  --request-rate 8 --goodput ttft:500 tpot:50 --save-result

# ④ vLLM 离线吞吐
vllm bench throughput --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 --output-len 512 --num-prompts 1000
```

## 六、避坑

- **测延迟一定要流式**：SGLang 别加 `--disable-stream`，否则 TTFT/ITL 失真。
- **控变量要强制满输出长度**：否则不同请求提前 EOS，输出长度不齐没法比。SGLang 默认就 ignore-eos，vLLM 要显式 `--ignore-eos`。
- **前缀缓存会"作弊"**：重复 prompt 命中 KV 缓存会虚高吞吐。要么 `--flush-cache`（SGLang），要么用带随机性的数据集。
- **serve 的吞吐 ≠ throughput 的吞吐**：serve 受到达过程和并发限制，天然低于离线一把梭；两者不能直接对比。
- **warmup 不能省**：首批请求含编译/显存分配开销，用 `--warmup-requests`/`--num-warmups` 排除。

## 七、实操记录与分析：Qwen3-0.6B 四组压测

> 2026-08-13 本机实测，Docker 内跑。模型 **Qwen3-0.6B**，统一 **输入 1024 / 输出 128 / 200 请求**。
> ⚠️ **GPU 型号、显存、框架版本、服务端启动参数均未记录**——这限制了结论的可复现性，改进清单见 §7.6。

### 7.1 实际执行的命令

**① vLLM 在线压测**（同一条命令跑了两次，结果差异见 §7.3 ②）

```bash
vllm bench serve \
  --host 127.0.0.1 --port 8000 \
  --model Qwen3-0.6B \
  --tokenizer /home/models/Qwen3-0.6B \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 200 \
  --ignore-eos                      # 强制生成满 128 token，控变量必开
```

**② vLLM 离线吞吐**（已执行，但**未留结果截图**）

```bash
# 随机集
CUDA_VISIBLE_DEVICES=2 vllm bench throughput \
  --model /home/models/Qwen3-0.6B \
  --dataset-name random \
  --input-len 1024 --output-len 128 \
  --num-prompts 200

# 换真实语料 ShareGPT（长度由数据集决定，不再是定长 1024/128）
CUDA_VISIBLE_DEVICES=1 vllm bench throughput \
  --model /home/models/Qwen3-0.6B \
  --dataset-name sharegpt \
  --dataset-path /vllm-workspace/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 200
```

**③ SGLang 在线压测**

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model /home/models/Qwen3-0.6B \
  --dataset-name random-ids \        # 注意：不是 random，差异见 §7.3 ⑥
  --random-input-len 1024 \
  --random-output-len 128 \
  --random-range-ratio 1 \           # 实测=定长，与本文 §2.1 表述冲突，见 §7.5
  --num-prompts 200
```

**④ SGLang 离线吞吐**

```bash
CUDA_VISIBLE_DEVICES=<实际卡号> python3 -m sglang.bench_offline_throughput \
  --model-path /home/models/Qwen3-0.6B \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --random-range-ratio 1 \
  --num-prompts 200
```

> 原命令还带了 `--dataset-path /cjw/ShareGPT_V3_unfiltered_cleaned_split.json`，但 `--dataset-name random` 下**该路径不生效**，已省去以免误导。另注意这个路径与 vLLM 那条用的 `/vllm-workspace/...` 不是同一份文件。

**语料下载**（ShareGPT V3，约 600 MB，hf-mirror 国内镜像）：

```bash
wget https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 7.2 结果汇总

四组的 workload 完全一致：200 请求 / 204800 输入 token / 25600 输出 token，全部 0 失败。

| 指标 | vLLM serve ①第一次 | vLLM serve ②第二次 | SGLang serve | SGLang offline |
|---|---|---|---|---|
| 耗时 (s) | 5.36 | **3.19** | 5.13 | **3.78** |
| 请求吞吐 (req/s) | 37.30 | 62.71 | 39.01 | 52.86 |
| **输出吞吐 (tok/s)** | 4774.88 | **8027.06** | 4993.44 | 6766.34 |
| 峰值输出 (tok/s) | 8695.00 | 9500.00 | **10395.00** | —（报 Last gen 11317.84） |
| 总吞吐 (tok/s) | 42973.91 | 72243.58 | 44941.00 | 60897.05 |
| TTFT 均值 (ms) | 1643.30 | **418.58** | 1262.36 | — |
| TTFT 中位 / P99 (ms) | 1485.45 / 3528.89 | 448.09 / **475.78** | 1228.38 / 1931.55 | — |
| TPOT 均值 (ms) | 25.05 | 21.32 | 29.13 | — |
| ITL 均值 / 中位 (ms) | 25.09 / 21.53 | 21.34 / 20.97 | 28.53 / **18.32** | — |
| ITL P99 / Max (ms) | 47.45 / 未报 | 28.86 / 未报 | 81.37 / **1817.85** | — |
| E2E 均值 (ms) | 未报 | 未报 | 4961.93 | — |

**原始输出：**

| vLLM serve ①第一次 | vLLM serve ②第二次 |
|---|---|
| ![[bench-vllm-serve-run1.png]] | ![[bench-vllm-serve-run2.png]] |

| SGLang serve | SGLang offline throughput |
|---|---|
| ![[bench-sglang-serve.png]] | ![[bench-sglang-offline.png]] |

### 7.3 六条分析

#### ① 别看「总吞吐」——它被输入长度灌了水

四组的 Total token throughput **精确等于 Output × 9**：

| | Output × 9 | 报告的 Total |
|---|---|---|
| vLLM ① | 4774.88 × 9 = 42973.9 | 42973.91 |
| vLLM ② | 8027.06 × 9 = 72243.5 | 72243.58 |
| SGLang serve | 4993.44 × 9 = 44940.9 | 44941.00 |
| SGLang offline | 6766.34 × 9 = 60897.1 | 60897.05 |

因为 `Total = input + output`，而本次 input : output = 1024 : 128 = **8 : 1**，所以 Total = 9 × Output——**其中 8/9 全是 prefill 在灌水**。把 `--input-len` 从 1024 改成 2048，Total 立刻接近翻倍，但服务能力一点没变。

> **跨配置比较只认 Output token throughput。** 报告里那个五位数的 Total 好看，但只要输入输出比一变就没有可比性。

#### ② 同一条命令跑两次差 1.68 倍——warmup 与前缀缓存的实证

图①②是同一条 vLLM 命令的两次运行（请求数、输入/输出 token 数完全一致，可确认 workload 相同），但：

| | 第一次 | 第二次 | 变化 |
|---|---|---|---|
| 耗时 | 5.36 s | 3.19 s | **1.68×** |
| TTFT 均值 | 1643 ms | 419 ms | **3.9× 降** |
| TTFT P99 | 3529 ms | 476 ms | **7.4× 降** |
| TPOT 均值 | 25.05 ms | 21.32 ms | 1.18× 降 |

**P99 降得比均值还狠**，说明第一次跑里有一批请求被硬拖了 3.5 秒。分布形状也印证：第一次 中位(1485) < 均值(1643)，**右偏、有掉队者**；第二次 中位(448) > 均值(419)，**左偏、无长尾**。

两个推断原因（本次未做对照实验，标为**推断**）：

- **首次运行含一次性开销**：CUDA graph 捕获、kernel autotune、显存池分配。而 `vllm bench serve` 的 `--num-warmups` **默认就是 0**（见 §3.1），等于把编译开销原封不动算进了第一次的成绩。
- **前缀缓存命中**：两次用同样 seed 生成同样的 prompt，vLLM 默认开启 automatic prefix caching，第二次 prefill 大面积命中 → TTFT 暴跌。

> 这一条同时踩中 §六 避坑里的**两条**：「warmup 不能省」和「前缀缓存会作弊」。
> **结论很扎心：第一次的数字被冷启动污染，第二次的数字被缓存污染——两个都不算数。** 正确做法见 §7.6。

#### ③ TTFT 1.6 秒不是「服务慢」，是压测姿势决定的

两边都没设 `--request-rate`（默认 `inf`）也没设 `--max-concurrency`，于是 200 个请求**同一瞬间**全部到达（Peak concurrent = 200），20 万 token 的 prefill 全堆在开头排队。

SGLang 那组给了铁证——它报了 vLLM 没报的 **E2E**，而且指标之间完全自洽：

```text
E2E 均值 4961.93 ms  ≈  TTFT 1262.36 + 127 × TPOT 29.13 = 4961.87 ms   ✓
```

（输出 128 token，TPOT 按定义不含第 1 个，故乘 127）

关键在于：整场测试只跑了 **5.13 秒**，而单请求 E2E 均值就有 **4.96 秒**——**几乎每个请求都横跨了整个测试窗口**，所有人同时开始、同时结束。vLLM 那两组同样如此（①约 90%、②约 98% 的窗口占比）。

> 这种模式测的是「200 并发下把这批活干完要多久」。**TTFT 在这里只反映排队深度，不代表任何真实用户的等待体验。**
> 要测体验，必须给有限的 `--request-rate` 或 `--max-concurrency`，扫出一条「吞吐 vs 延迟」曲线找拐点——正是 §2.1 那条 💡 说的做法。

#### ④ 离线比在线快 35%——这就是「服务化」的标价

同为 SGLang、同一份 workload，唯一区别是要不要过 HTTP：

| | 耗时 | 输出吞吐 |
|---|---|---|
| serve（在线） | 5.13 s | 4993.44 tok/s |
| offline（离线） | 3.78 s | 6766.34 tok/s |
| **差距** | **−26%** | **+35.5%** |

两个比值互相自洽（5.13/3.78 = 1.357，6766/4993 = 1.355），可信。这 35% 花在了：HTTP 请求/响应的序列化往返、SSE 逐 token 推送、tokenizer ↔ scheduler ↔ detokenizer 的跨进程 ZMQ 通信（见 [[SGlang]] §二 三进程模型）。离线模式在进程内直接拉引擎，这些全省，还能一把梭让调度器自由组批。

> **这是 §六 那条「serve 的吞吐 ≠ throughput 的吞吐」的定量版：在这个配置下，差距是 35%。**
> 拿离线数字去承诺线上 SLA，等于超卖三分之一的容量。

另外 offline 独有一个指标 **Last generation throughput 11317.84 tok/s**，值得单独看：那是最后一批（所有请求都已进入 decode、batch 最满）的瞬时速度，可视为**纯 decode 的峰值能力**；而均值 6766 里掺了开头 prefill 的「空窗期」。两者比值 11317 / 6766 ≈ **1.67**，大致反映 prefill 阶段吃掉了多少整体时间。

#### ⑤ ITL 最大 1817 ms——prefill 抢占 decode 的现场

SGLang 那张图信息量最大，因为它**报了 Max**（vLLM 没报）：

| ITL 分位 | 值 | 相对中位数 |
|---|---|---|
| 中位 | 18.32 ms | 1× |
| 均值 | 28.53 ms | 1.6× |
| P95 | 21.81 ms | 1.2× |
| P99 | 81.37 ms | 4.4× |
| **Max** | **1817.85 ms** | **99×** |

中位只有 18 ms 说明**绝大多数 token 吐得很顺**，是少数超长间隔把均值拉到了 28.5。那接近 **1.8 秒**的卡顿是什么？

**SGLang 是 prefill 优先调度**（见 [[SGlang]] §三 ③：「先看有没有新请求要 prefill……没有才继续跑 decode」）。200 请求一把梭时，前期不断有新的 prefill 批插队，正在 decode 的请求就得干等一轮。**用户侧的体感是：字正往外蹦，突然卡住将近两秒。**

对比 vLLM 的 ITL P99 只有 47.45（①）/ 28.86（②），离散度小得多——很可能是 **chunked prefill** 在起作用：把长 prefill 切片、穿插进 decode，牺牲一点 prefill 延迟换 decode 平滑。但注意 **vLLM 没报 Max，「没报」不等于「没有」**。

> **教训：TPOT 是平均值，天然掩盖抖动；必须看 ITL 的 P99 和 Max。**
> SGLang 的 TPOT 均值(29.13) 和 ITL 均值(28.53) 看着差不多，但 ITL 中位只有 18.32——只盯均值，你永远发现不了那 1.8 秒。这正是 §一 说的「ITL 比 TPOT 更细（能看抖动）」的实证。

#### ⑥ retokenized 比 generated 少 20%——`random-ids` 的隐性代价

SGLang 报了个 vLLM 没有的指标：

```text
Total generated tokens:               25600
Total generated tokens (retokenized): 20327    ← 少了 20.6%
```

含义：模型生成 25600 个 token id，解码成文本、再重新分词回去只剩 20327 个。

根因在数据集选型：`--dataset-name random-ids` 直接**随机生成 token id**，而不是像 `random` 那样从真实语料采样文本。模型在一段乱码上下文里输出的 token 序列也是「非规范」组合；retokenize 时 BPE 会按最优方式重新合并，token 数自然变少（这个边界问题的机制见 [[SGlang]] §八 增量解码）。

> **这个指标其实是「数据集有多真实」的探针**：自然文本压测下 retokenized 应 ≈ generated，差得越远，说明输入越不像人话、结果越不代表真实负载。
> 本次差 20%，意味着**这组 SGLang 数字的代表性弱于 vLLM 那组**（后者用 `random`，是真实语料截断）。

### 7.4 ⚠️ 这四组数据**不能**用来判「SGLang vs vLLM 谁更强」

表面上看：SGLang serve（4993 tok/s）比 vLLM 第一次（4775）快 4.6%，又被 vLLM 第二次（8027）反超 61%。**这三个数字互相都不可比**，至少五条原因：

1. **数据集不是一个东西**：vLLM 用 `random`（真实语料截断），SGLang 用 `random-ids`（纯随机 id）。后者绕开了前缀缓存（更「干净」），但也牺牲了真实性（见 ⑥）。
2. **冷热状态不一致**：vLLM 两次自己就差 1.68 倍，SGLang 只跑了一次——不知道它落在自己的哪一端。
3. **卡未隔离、未记录**：vLLM throughput 用了 `CUDA_VISIBLE_DEVICES=2` 和 `=1`，SGLang offline 原文写的是 `<空卡号>` 占位符，两条 serve 则完全没指定。**共享机器上别的进程占卡会严重干扰**，而且事后无法追溯到底用了哪张。
4. **服务端启动参数完全没记录**：`--mem-fraction-static`、`--max-num-seqs`、chunked prefill 开关、attention backend、是否开 CUDA graph——**这些对结果的影响远大于框架差异本身**。
5. **只跑了一次，没有方差**：5.36 vs 5.13 只差 4.6%，很可能落在运行间噪声里。

> **目前唯一站得住脚的结论，是那条纵向对比：同框架、同 workload 下，离线比在线快 35%（§7.3 ④）。**
> 因为那两次的框架、数据集、模型、机器全都一致，只差一个「要不要过 HTTP」——这才叫控变量。

### 7.5 ⚠️ 存疑：`--random-range-ratio` 的语义与本文 §2.1 表格冲突

本次 SGLang 两条命令都加了 `--random-range-ratio 1`，实测得到 **Total input tokens = 204800 = 200 × 1024，精确定长**。

但本文 §2.1 / §2.2 的表格写的是「输入/输出长度的随机浮动比例，**`0` 表示定长**」。若该描述成立，`ratio 1` 应当是**最大浮动**，不该得到精确定长。**实测与本文表述矛盾。**

两种可能：

- SGLang 实现里 `range_ratio` 是**采样区间的下界系数**——长度采样自 `[input_len × ratio, input_len]`，于是 `1` → `[1024, 1024]` 定长、`0` → `[0, 1024]` 全随机。**本次实测支持这个解释。**
- 或该参数语义在版本间变更过。

同时注意 vLLM 那条**没加**任何 range 参数，也得到了精确的 204800——说明**两个框架同名参数的默认行为不同**，不能照搬。

> 🔧 **待办**：查源码确认 `--random-range-ratio` 的实际语义，确认后修正 §2.1 / §2.2 的表格描述。
> 在此之前的保险做法：**要定长就照抄本次的 `--random-range-ratio 1`，然后用报告里的 `Total input tokens` 反查是不是真的定长**（应精确等于 `num-prompts × input-len`）。

### 7.6 下次怎么测才算数

按本次踩到的坑，一份可信的对比至少要补齐：

| 要补的 | 怎么做 |
|---|---|
| **记录环境** | GPU 型号 + 显存 + 驱动/CUDA 版本 + 框架版本 + **实际卡号**（别再写 `<空卡号>`） |
| **记录服务端启动命令** | 两边 `launch_server` / `vllm serve` 的全部参数，尤其 max-num-seqs、chunked prefill、attention backend |
| **独占 GPU** | 每次显式 `CUDA_VISIBLE_DEVICES=<确定卡号>`，并先确认卡上没有别的进程 |
| **消除冷启动** | SGLang `--warmup-requests 10`；vLLM `--num-warmups 10`（**默认是 0！**） |
| **消除缓存作弊** | SGLang 加 `--flush-cache`；vLLM 起服务时关前缀缓存，或每轮重启服务 |
| **对齐数据集** | 两边都用 `random`（或都用 sharegpt），别一边 `random` 一边 `random-ids` |
| **跑够次数** | 每组至少 3 次取中位数，并报出波动范围 |
| **换个测法** | 别只跑 `inf`：扫 `--request-rate` 或 `--max-concurrency`，画吞吐-延迟曲线，才看得出拐点和真实 TTFT |
| **加 SLO** | vLLM 用 `--goodput ttft:500 tpot:50` 看有效吞吐，比裸吞吐贴近可用性 |

> 另外：**vLLM 的两条 `bench throughput` 已执行但没留结果截图**。补跑并记录后，才能和 SGLang offline 的 6766 tok/s 凑成完整的 2×2 对照（两框架 × 在线/离线）。

## See Also

- [[SGlang]] — 本文压测的对象：SGLang 三进程模型与 Scheduler 事件循环源码走读。§7.3 ④ 的「服务化 35% 开销」、⑤ 的「prefill 抢占 decode」、⑥ 的「retokenize 边界」都能在那篇找到机制解释。
- [[LLM推理的GPU硬件基础]] — 压测数字背后的硬件解释：TTFT 反映 prefill（compute-bound），TPOT/ITL 反映 decode（memory-bound）；单流 decode 速度上限 = 带宽 ÷ 权重字节，可用来判断实测有没有跑到硬件天花板。
