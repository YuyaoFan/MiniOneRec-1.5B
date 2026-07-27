#!/bin/bash
# =============================================================================
# RQ-VAE + SFT + GRPO (Baseline — 最优方案)
# =============================================================================
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="$ROOT/logs/rqvae_grpo_${TIMESTAMP}"
mkdir -p "$LOGDIR"
source /root/miniconda3/etc/profile.d/conda.sh && conda activate MiniOneRec

export NCCL_IB_DISABLE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
CATEGORY="Industrial_and_Scientific"
INDEX="$ROOT/data/Amazon/index/${CATEGORY}.index.json"
CODEBOOK="$ROOT/data/Amazon/index/${CATEGORY}.codebooks_constrained.npz"
EMB="$ROOT/data/Amazon/index/${CATEGORY}.emb-qwen-td.npy"

echo "=== RQ-VAE + SFT + GRPO ==="

# ── Stage 1: RQ-VAE SID ──
if [ -f "$INDEX" ]; then
    echo "[skip] index.json exists"
else
    cd "$ROOT/rq"
    python rqkmeans_constrained.py --dataset "$CATEGORY" --root "../data/Amazon/index" --k 256 --l 3 --max_iter 100 --seed 42 --verbose 2>&1 | tee "$LOGDIR/s1_constrained.log"
    python rqkmeans_plus.py --data_path "../data/Amazon/index/${CATEGORY}.emb-qwen-td.npy" --pretrained_codebook_path "../data/Amazon/index/${CATEGORY}.codebooks_constrained.npz" --num_emb_list 256 256 256 --e_dim 2560 --lr 1e-4 --epochs 10000 --batch_size 2048 2>&1 | tee "$LOGDIR/s1_train.log"
    CKPT_DIR=$(ls -td "$ROOT/rq"/*/ 2>/dev/null | head -1)
    BEST=$(ls "$CKPT_DIR"/best_collision_model.pth 2>/dev/null || ls "$CKPT_DIR"/best_loss_model.pth)
    python generate_indices_plus.py --data_path "../data/Amazon/index/${CATEGORY}.emb-qwen-td.npy" --ckpt_path "$BEST" --num_emb_list 256 256 256 --device cuda:0 2>&1 | tee "$LOGDIR/s1_indices.log"
    cd "$ROOT"
fi

# ── Stage 2: SFT ──
SFT_OUT="$ROOT/output/sft/${CATEGORY}_rqvae"
if [ ! -f "$SFT_OUT/final_checkpoint/config.json" ]; then
    python convert_dataset.py --dataset_name "$CATEGORY" --data_dir data/Amazon/index --output_dir data/Amazon --category "$CATEGORY" --seed 42 2>&1 | tee "$LOGDIR/s2_convert.log"
    TRAIN=$(ls -f ./data/Amazon/train/${CATEGORY}*11.csv | head -1)
    EVAL=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv | head -1)
    torchrun --nproc_per_node 1 sft.py --base_model ./models/Qwen2.5-1.5B --batch_size 1024 --micro_batch_size 16 --train_file "$TRAIN" --eval_file "$EVAL" --output_dir "$SFT_OUT" --category "$CATEGORY" --train_from_scratch False --seed 42 --sid_index_path "./data/Amazon/index/${CATEGORY}.index.json" --item_meta_path "./data/Amazon/index/${CATEGORY}.item.json" --freeze_LLM False 2>&1 | tee "$LOGDIR/s2_sft.log"
fi

# ── Stage 3: RL GRPO ──
RL_OUT="$ROOT/output/rl/${CATEGORY}_rqvae_grpo"
TRAIN_CSV=$(ls -f ./data/Amazon/train/${CATEGORY}*.csv | head -1)
EVAL_CSV=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv | head -1)
INFO=$(ls -f ./data/Amazon/info/${CATEGORY}*.txt | head -1)
HF_ENDPOINT=https://hf-mirror.com accelerate launch --config_file ./config/zero2_opt.yaml --num_processes 1 --main_process_port 29503 rl.py --model_path "$SFT_OUT" --train_batch_size 32 --eval_batch_size 32 --num_train_epochs 2 --gradient_accumulation_steps 32 --train_file "$TRAIN_CSV" --eval_file "$EVAL_CSV" --info_file "$INFO" --category "$CATEGORY" --sample_train False --eval_step 0.0999 --reward_type ranking --num_generations 16 --sync_ref_model True --beam_search True --temperature 1.0 --learning_rate 1e-5 --beta 1e-3 --gdpo False --output_dir "$RL_OUT" --sid_index_path "./data/Amazon/index/${CATEGORY}.index.json" --item_meta_path "./data/Amazon/index/${CATEGORY}.item.json" 2>&1 | tee "$LOGDIR/s3_rl.log"

# ── Stage 4: Eval ──
python tools/eval_all.py --batch_size 8 2>&1 | tee "$LOGDIR/s4_eval.log"
echo "[DONE] RQ-VAE+SFT+GRPO complete"
