# DeepSpec 训练 DSpark draft 头：Qwen3-8B 全流程

> 用 DeepSpec 框架给 Qwen3-8B 训一个 **DSpark** draft 头的完整跑通记录（smoke 规模，1053 条样本 / 2 epoch），从装环境一路到验证接受长度。DSpark 本身的原理见 [[投机采样]] §2.7。

> [!note] 流水线一览
> 数据切分 → 用 target 自己生成答案（蒸馏语料）→ 预抽 target 的 hidden state 存成 cache → 拿 cache 训 draft 头 → 评接受长度。
> 第 3、4 步是这套流程的关键：draft 头学的不是「预测下一个 token」，而是「拟合 target 的中间表征」，所以必须先把 target 的输出和特征落盘。

## 1. 装环境

```bash
source ~/venvs/deepspec/bin/activate
deactivate
```

## 2. 数据切分

```bash
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --sample-size 1053 \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

## 3. 启动 sglang 推理服务，生成答案

```bash
cd /home/DeepSpec
mkdir -p train_datasets/qwen3_8b
python3 scripts/data/generate_train_data.py \
    --model /models/Qwen3-8B \
    --server-address 127.0.0.1:30000 \
    --concurrency 32 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --max-tokens 512 \
    --disable-thinking --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_8b/perfectblend_train_regen.jsonl
```

![[deepspec-step3-generate-train-data.png|664]]

## 4. 准备 Target Cache

```bash
PYTHONPATH=/home/DeepSpec CUDA_VISIBLE_DEVICES=1 python3 scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_8b.py \
    --train-data-path train_datasets/qwen3_8b/perfectblend_train_regen.jsonl \
    --output-dir /home/DeepSpec/runs/cache/smoke_cache_8b \
    --local-batch-size 4 \
    --opts "data.max_length=512" \
    --opts "model.target_model_name_or_path=/home/models/Qwen3-8B"
```

![[deepspec-step4-prepare-target-cache.png]]

产物体积与语料条数：

```bash
(deepspec) root@server107:/home/DeepSpec# du -sh /home/DeepSpec/runs/cache/smoke_cache_8b
19G     /home/DeepSpec/runs/cache/smoke_cache_8b
wc -l train_datasets/qwen3_8b/perfectblend_train_regen.jsonl
1000 train_datasets/qwen3_8b/perfectblend_train_regen.jsonl
```

> 1000 条样本、`max_length=512`，target cache 就占了 **19G**——这一步是整条流水线的磁盘大头，跑之前先看空间。

## 5. 训练

```bash
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --config config/dspark/dspark_qwen3_8b.py \
    --opts "data.target_cache_path=/home/DeepSpec/runs/cache/smoke_cache_8b" \
    --opts "data.max_length=512" \
    --opts "model.target_model_name_or_path=/home/models/Qwen3-8B" \
    --opts "exp_name=smoke_8b_1gpu" \
    --opts "train.global_batch_size=16" \
    --opts "train.local_batch_size=2" \
    --opts "train.num_train_epochs=2" \
    --opts "train.torch_compile=False" \
    --opts "logging.checkpointing_steps=200" \
    2>&1 | tee runs/train_smoke.log
```

![[deepspec-step5-train-log.png]]

## 6. 验证

```bash
cd /home/DeepSpec
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
    --target_name_or_path /home/models/Qwen3-8B \
    --draft_name_or_path /home/DeepSpec/runs/checkpoints/deepspec/smoke_8b_1gpu/step_latest \
    --max-new-tokens 256 \
    --temperature 0
```

![[deepspec-step6-eval-gsm8k.png]]

> 截图读数（gsm8k，`step_latest`，propose `7.00+1`）：**accept_len 1.05**，verify_rate 0.1308，accept_rate@0 0.0458。
>
> ⚠️ accept_len ≈ 1 意味着**草稿基本全被拒**，等于没有加速——对 1053 条样本、2 epoch 的 smoke 跑这是预期结果，只能证明流程通了，不能用来评价 DSpark 本身。对照 [[投机采样]] §2.9 里正式 checkpoint 在 Qwen3.8-27B 上的 accept_len 4.7 左右。

## 7.SpecForge（Qwen3_8B+Dflash2）
7.1安装
NVIDIA CUDA、AMD ROCm、Ascend NPU
7.2数据准备
Dataset Presets：
	# ultrachat python scripts/prepare_data.py --dataset ultrachat
	# sharegpt python scripts/prepare_data.py --dataset sharegpt
	本地数据：
		python scripts/prepare_data.py \
	    --dataset sharegpt \
	    --data-path ./raw_sharegpt.jsonl \
	    --output-path ./cache/dataset
Regenerate Datasets：
	启动sglang服务：
		python3 -m sglang.launch_server \
	    --model-path meta-llama/Llama-3.1-8B-Instruct \
	    --cuda-graph-max-bs 128 \
	    --dtype bfloat16 \
	    --mem-fraction-static 0.8 \
	    --port 30000
	使用脚本重新生成数据集`regenerate_train_data.py‘:
		python scripts/regenerate_train_data.py \
	    --model meta-llama/Llama-3.1-8B-Instruct \
	    --concurrency 128 \
	    --max-tokens 98304 \
	    --server-address localhost:30000 \
	    --temperature 0.8 \
	    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
	    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
7.3获取中间层隐特征
离线：
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --strategy eagle3 \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3.json \
    --data-path ./your_preformatted_dataset.jsonl \
    --output-path ./cache/hidden_states/llama3.1-8b-eagle3 \
    --chat-template llama3 \
    --is-preformatted \
    --max-length 2048
在线：
让目标模型以 patched SGLang 服务的形式在线运行，负责前向并捕获 hidden states，写入 Mooncake 共享存储。训练侧不直接拿大张量，而是只收到 `SampleRef` 引用（ key、shape、dtype 等），再由 `RefDistributor` （引用分发器，是生产者和消费者之间的调度组件）分发到各 trainer rank，训练时凭引用通过 RDMA 去 Mooncake 取特征。生产者和消费者是两个独立进程池，可以分别扩缩容，并通过窗口同步和背压控制保证训练步一致、缓存不堆积。它不提前落全量 target cache，适合边生成边训练、和 SGLang serving 联动的场景，但系统复杂度比离线模式高。
7.4训练
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3-disaggregated.yaml
## See Also

- [[投机采样]] —— DSpark / DFlash / MTP 的原理与实测对比
- [[SGlang]] —— 第 3 步起服务用的推理框架

## 备注

- 本篇是**自测流水账**，命令与输出均为实机记录，未做改写。
- 2026-09-10 规范化：仅做排版——补标题、命令套 bash 代码块、步骤标题由 H1 降为 H2、图片引用改为描述性文件名并取消缩进；命令、路径、参数一字未改。新增的只有开头的流水线说明、19G 那条提醒、以及第 6 步截图读数与解读。
