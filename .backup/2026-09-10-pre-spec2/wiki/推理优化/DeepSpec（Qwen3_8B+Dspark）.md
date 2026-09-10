# 1.装环境

source ~/venvs/deepspec/bin/activate
deactivate
# 2.数据切分
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --sample-size 1053 \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
# 3.启动sglang推理服务，生成答案
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
    ![[Pasted image 20260910151254.png|664]]
# 4.准备Target Cache
PYTHONPATH=/home/DeepSpec CUDA_VISIBLE_DEVICES=1 python3 scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_8b.py \
    --train-data-path train_datasets/qwen3_8b/perfectblend_train_regen.jsonl \
    --output-dir /home/DeepSpec/runs/cache/smoke_cache_8b \
    --local-batch-size 4 \
    --opts "data.max_length=512" \
    --opts "model.target_model_name_or_path=/home/models/Qwen3-8B"
![[Pasted image 20260910152904.png]]

(deepspec) root@server107:/home/DeepSpec# du -sh /home/DeepSpec/runs/cache/smoke_cache_8b
19G     /home/DeepSpec/runs/cache/smoke_cache_8b
wc -l train_datasets/qwen3_8b/perfectblend_train_regen.jsonl
1000 train_datasets/qwen3_8b/perfectblend_train_regen.jsonl
# 5.训练
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
    ![[Pasted image 20260910154636.png]]
# 6.验证
 cd /home/DeepSpec
CUDA_VISIBLE_DEVICES=1 python3 eval.py \
    --target_name_or_path /home/models/Qwen3-8B \
    --draft_name_or_path /home/DeepSpec/runs/checkpoints/deepspec/smoke_8b_1gpu/step_latest \
    --max-new-tokens 256 \
    --temperature 0
    ![[Pasted image 20260910164432.png]]