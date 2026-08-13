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
| `--dataset-name` | `sharegpt` | 数据集：`sharegpt`（真实对话）/`random`（随机长度，最常用于控变量）/`generated-shared-prefix`（测前缀缓存命中）/`mmmu`/`image` 等 |
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

## See Also

- [[SGlang]] — SGLang 使用笔记
- [[LLM推理的GPU硬件基础]] — 压测数字背后的硬件解释：TTFT 反映 prefill（compute-bound），TPOT/ITL 反映 decode（memory-bound）；单流 decode 速度上限 = 带宽 ÷ 权重字节，可用来判断实测有没有跑到硬件天花板。
Vllm
root@3dd8d07a7ec5:/vllm-workspace# vllm bench serve \    --host 127.0.0.1 --port 8000 \

  --model Qwen3-0.6B \   --tokenizer /home/models/Qwen3-0.6B \ --dataset-name random \

  --random-input-len 1024 \   --random-output-len 128 \   --num-prompts 200 \

  --ignore-eos
  ![[Pasted image 20260813163349.png]]
  ![[Pasted image 20260813163354.png]]
  Vllm
  root@3dd8d07a7ec5:/vllm-workspace# CUDA_VISIBLE_DEVICES=2 vllm bench throughput \

  --model /home/models/Qwen3-0.6B \   --dataset-name random \

  --input-len 1024 --output-len 128 \   --num-prompts 200
  CUDA_VISIBLE_DEVICES=1 vllm bench throughput \  --model /home/models/Qwen3-0.6B \  --dataset-name sharegpt \  --dataset-path /vllm-workspace/ShareGPT_V3_unfiltered_cleaned_split.json \  --num-prompts 200
  SGlang
  root@e6adbbdf4cf8:/sgl-workspace/sglang# python3 -m sglang.bench_serving \   --backend sglang \

  --host 127.0.0.1 --port 30000 \   --model /home/models/Qwen3-0.6B \   --dataset-name random-ids \

  --random-input-len 1024 \   --random-output-len 128 \   --random-range-ratio 1 \   --num-prompts 200
  ![[Pasted image 20260813163446.png]]
  sglang
  root@3dd8d07a7ec5:/vllm-workspace# CUDA_VISIBLE_DEVICES=<空卡号> python3 -m sglang.bench_offline_throughput \  --model-path /home/models/Qwen3-0.6B \  --dataset-name random \  --dataset-path /cjw/ShareGPT_V3_unfiltered_cleaned_split.json \  --random-input-len 1024 \  --random-output-len 128 \  --random-range-ratio 1 \  --num-prompts 200
  下载一个600MB左右语料，"https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
  ![[Pasted image 20260813163510.png]]