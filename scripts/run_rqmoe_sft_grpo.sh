#!/bin/bash
# =============================================================================
# RQ-MoE V2 + SFT + GRPO
# =============================================================================
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$ROOT/logs/rqmoe_grpo_${TIMESTAMP}"
mkdir -p "$LOGDIR"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate MiniOneRec
export NCCL_IB_DISABLE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_DISABLED=true
CATEGORY="Industrial_and_Scientific"
EMB="$ROOT/data/Amazon/index/${CATEGORY}.emb-qwen-td.npy"
INDEX="$ROOT/data/Amazon/index/${CATEGORY}.index.json"
echo "=== RQ-MoE V2 + SFT + GRPO ==="

# Stage 1: RQ-MoE V2 SID (3-phase: pretrain+KMeans+STE)
if [ -f "$INDEX" ]; then echo "[skip] index.json exists"; else
    python rq/train_rqmoe_v2.py --data_path "$EMB" --ckpt_dir "$ROOT/checkpoints/rqmoe_v2" \
        --num_emb_list 256 256 256 --e_dim 2560 --moe_N 2 --moe_L 4 --moe_H 256 \
        --batch_size 256 --lr_p3 1e-4 --epochs_p1 1 --epochs_p2 0 --epochs_p3 2000 --device cuda:0 2>&1 | tee "$LOGDIR/s1_train.log"
    CKPT=$(ls -t "$ROOT"/checkpoints/rqmoe_v2/best_model.pth 2>/dev/null | head -1)
    python rq/generate_indices_rqmoe.py --data_path "$EMB" --ckpt_path "$CKPT" \
        --num_emb_list 256 256 256 --e_dim 2560 --layers 512 256 128 \
        --moe_N 2 --moe_L 4 --moe_H 256 --moe_dropout 0.1 --batch_size 2048 --device cuda:0 2>&1 | tee "$LOGDIR/s1_indices.log"
fi

# Stage 2: SFT
SFT_OUT="$ROOT/output/sft/${CATEGORY}_rqmoe"
if [ ! -f "$SFT_OUT/final_checkpoint/config.json" ]; then
    python convert_dataset.py --dataset_name "$CATEGORY" --data_dir data/Amazon/index --output_dir data/Amazon --category "$CATEGORY" --seed 42 2>&1 | tee "$LOGDIR/s2_convert.log"
    TRAIN=$(ls -f ./data/Amazon/train/${CATEGORY}*11.csv | head -1)
    EVAL=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv | head -1)
    torchrun --nproc_per_node 1 sft.py --base_model ./models/Qwen2.5-1.5B --batch_size 1024 --micro_batch_size 16 --train_file "$TRAIN" --eval_file "$EVAL" --output_dir "$SFT_OUT" --category "$CATEGORY" --train_from_scratch False --seed 42 --sid_index_path "./data/Amazon/index/${CATEGORY}.index.json" --item_meta_path "./data/Amazon/index/${CATEGORY}.item.json" --freeze_LLM False 2>&1 | tee "$LOGDIR/s2_sft.log"
fi

# Stage 3: RL GRPO
RL_OUT="$ROOT/output/rl/${CATEGORY}_rqmoe_grpo"
TRAIN_CSV=$(ls -f ./data/Amazon/train/${CATEGORY}*.csv | head -1)
EVAL_CSV=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv | head -1)
INFO=$(ls -f ./data/Amazon/info/${CATEGORY}*.txt | head -1)
HF_ENDPOINT=https://hf-mirror.com accelerate launch --config_file ./config/zero2_opt.yaml --num_processes 1 --main_process_port 29505 rl.py --model_path "$SFT_OUT" --train_batch_size 32 --eval_batch_size 32 --num_train_epochs 2 --gradient_accumulation_steps 32 --train_file "$TRAIN_CSV" --eval_file "$EVAL_CSV" --info_file "$INFO" --category "$CATEGORY" --sample_train False --eval_step 0.0999 --reward_type ranking --num_generations 16 --sync_ref_model True --beam_search True --temperature 1.0 --learning_rate 1e-5 --beta 1e-3 --gdpo False --output_dir "$RL_OUT" --sid_index_path "./data/Amazon/index/${CATEGORY}.index.json" --item_meta_path "./data/Amazon/index/${CATEGORY}.item.json" 2>&1 | tee "$LOGDIR/s3_rl.log"

echo "[DONE] RQ-MoE+SFT+GRPO complete"
