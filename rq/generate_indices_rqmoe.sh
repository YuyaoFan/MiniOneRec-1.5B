#!/bin/bash
# Generate SID indices using trained RQ-MoE V2 model.
#
# Update CKPT_PATH to the best checkpoint path after training completes.
#
# Usage:  bash rq/generate_indices_rqmoe.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_PATH="data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy"
CKPT_PATH="./checkpoints/rqmoe_v2/best_collision_model.pth"

python rq/generate_indices_rqmoe.py \
    --data_path "${DATA_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --num_emb_list 256 256 256 \
    --e_dim 256 \
    --layers 512 256 128 \
    --moe_N 2 --moe_L 4 --moe_H 512 --moe_dropout 0.1 \
    --batch_size 2048 --num_workers 4 --device cuda:0
