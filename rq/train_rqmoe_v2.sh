#!/bin/bash
# RQ-MoE V2: 3-Phase EMA + STE Training
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python rq/train_rqmoe_v2.py \
    --data_path data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_dir ./checkpoints/rqmoe_v2 \
    --num_emb_list 256 256 256 \
    --e_dim 2560 \
    --moe_N 2 --moe_L 4 --moe_H 256 --moe_dropout 0.1 \
    --batch_size 512 \
    --lr 1e-3 \
    --epochs_p1 100 \
    --epochs_p2 5000 \
    --epochs_p3 1000 \
    --rqvae_codebook data/Amazon/index/Industrial_and_Scientific.codebooks_constrained.npz \
    --device cuda:0
