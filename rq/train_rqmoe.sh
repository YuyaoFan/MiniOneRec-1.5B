#!/bin/bash
# Train RQ-MoE V2 for SID codebook generation (MiniOneRec).
#
# V2 architecture (see rq/rqmoe_collapse_analysis.md):
#   - STE gradient: encoder receives training signal
#   - e_dim=256: reduced compression (10x vs 40x)
#   - layers=[512,256,128]: shallower encoder for 256-dim
#   - moe_H=512: wider experts
#   - K-Means codebook init: prevents initial collapse
#
# Usage:  bash rq/train_rqmoe.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_PATH="data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"
CKPT_DIR="./checkpoints/rqmoe_v2"

python rq/train_rqmoe.py \
    --data_path "${DATA_PATH}" \
    --ckpt_dir "${CKPT_DIR}" \
    --num_emb_list 256 256 256 \
    --e_dim 256 \
    --layers 512 256 128 \
    --moe_N 2 --moe_L 4 --moe_H 512 --moe_dropout 0.1 \
    --quant_loss_weight 1.0 \
    --lr 1e-3 --learner adam \
    --epochs 10000 --batch_size 1024 \
    --warmup_epochs 5 --eval_step 10 \
    --kmeans_init --kmeans_iters 20 \
    --save_limit 5 --device cuda:0
