"""
RQ-MoE V2 training entry: 3-phase EMA + STE training.

Phase 1: EMA codebook update (epochs_p1)
Phase 2: Joint STE training (epochs_p2)
Phase 3: Fine-tune projections (epochs_p3)
"""

import argparse, logging, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqmoe_v2 import RQMoEV2Wrapper
from utils import ensure_dir, set_color

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints/rqmoe_v2")
    p.add_argument("--num_emb_list", type=int, nargs='+', default=[256, 256, 256])
    p.add_argument("--e_dim", type=int, default=2560)
    p.add_argument("--moe_N", type=int, default=2)
    p.add_argument("--moe_L", type=int, default=4)
    p.add_argument("--moe_H", type=int, default=256)
    p.add_argument("--moe_dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_p2", type=float, default=1e-5)
    p.add_argument("--lr_p3", type=float, default=1e-4)
    p.add_argument("--epochs_p1", type=int, default=100)
    p.add_argument("--epochs_p2", type=int, default=5000)
    p.add_argument("--epochs_p3", type=int, default=1000)
    p.add_argument("--rqvae_codebook", type=str, default="")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    logger = logging.getLogger()
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("RQ-MoE V2: 3-Phase EMA + STE Training")
    logger.info(f"  e_dim={args.e_dim}, K={args.num_emb_list[0]}, M={len(args.num_emb_list)}")
    logger.info(f"  Phases: EMA={args.epochs_p1} + STE={args.epochs_p2} + FT={args.epochs_p3}")

    # Data
    ds = EmbDataset(args.data_path)
    logger.info(f"Data: {len(ds)} × {ds.dim}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Model
    m = RQMoEV2Wrapper(
        in_dim=ds.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        moe_N=args.moe_N, moe_L=args.moe_L, moe_H=args.moe_H,
        moe_dropout=args.moe_dropout,
    ).to(device)
    logger.info(f"Params: {sum(p.numel() for p in m.parameters()):,}")

    # Init from RQ-VAE codebooks
    if args.rqvae_codebook and os.path.exists(args.rqvae_codebook):
        m.rq.init_from_rqvae(args.rqvae_codebook, device)

    # Phase 0: Pretrain projection with MSE reconstruction
    opt_p0 = torch.optim.Adam(m.parameters(), lr=1e-3)
    m.phase0_pretrain_projection(dl, opt_p0, 200, device)

    # Phase 1: sklearn K-Means on trained projection
    m.phase1_kmeans_init(dl, device)

    # Phase 2: K-Means codebooks are frozen, skip EMA (causes drift)
    # Phase 3: STE projection training (codebooks frozen)
    opt_p3 = torch.optim.Adam(m.parameters(), lr=args.lr_p3)
    m.phase3_ste_train(dl, opt_p3, args.epochs_p3, device)

    # Save
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({"state_dict": m.state_dict()}, os.path.join(args.ckpt_dir, "best_model.pth"))
    logger.info(f"Saved to {args.ckpt_dir}/best_model.pth")


if __name__ == "__main__":
    main()
