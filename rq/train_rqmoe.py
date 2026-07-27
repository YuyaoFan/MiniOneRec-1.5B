"""
RQ-MoE V2 training entry point for MiniOneRec SID codebook generation.

V2 fixes: STE gradient, e_dim=256, K-Means init, shallower encoder.
"""

import argparse, logging, os, sys
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqmoe import RQMoEWrapper
from rqmoe_trainer import RQMoETrainer
from utils import ensure_dir, set_color

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-MoE V2 for SID generation")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/rqmoe_v2")

    # Model (V2 defaults)
    parser.add_argument("--num_emb_list", type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument("--e_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", action="store_true", default=False)

    # RQ-MoE (V2 defaults)
    parser.add_argument("--moe_N", type=int, default=2)
    parser.add_argument("--moe_L", type=int, default=4)
    parser.add_argument("--moe_H", type=int, default=512)
    parser.add_argument("--moe_dropout", type=float, default=0.1)

    # Loss
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--quant_loss_weight", type=float, default=1.0)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--learner", type=str, default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=20480)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "constant"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--save_limit", type=int, default=5)

    # K-Means init for codebook
    parser.add_argument("--kmeans_init", action="store_true", default=True)
    parser.add_argument("--kmeans_iters", type=int, default=10)
    parser.add_argument("--no_kmeans", action="store_true", default=False,
                        help="Disable K-Means init")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = logging.getLogger()

    logger.info(set_color("=" * 60, "blue"))
    logger.info(set_color("RQ-MoE V2 Training (STE + e_dim=256 + K-Means)", "blue"))
    logger.info(set_color("=" * 60, "blue"))
    logger.info(f"Config: M={len(args.num_emb_list)}, K={args.num_emb_list[0]}, "
                f"e_dim={args.e_dim}, N={args.moe_N}, L={args.moe_L}, H={args.moe_H}")
    logger.info(f"Data: {args.data_path}")

    # Load data
    logger.info("Loading embeddings...")
    dataset = EmbDataset(args.data_path)
    logger.info(f"Loaded {len(dataset)} embeddings of dim {dataset.dim}")

    # Build model
    logger.info("Building RQ-MoE V2 model (STE forward)...")
    model = RQMoEWrapper(
        in_dim=dataset.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init and not args.no_kmeans,
        kmeans_iters=args.kmeans_iters,
        moe_N=args.moe_N, moe_L=args.moe_L,
        moe_H=args.moe_H, moe_dropout=args.moe_dropout,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # K-Means init on projected latent space
    if args.kmeans_init and not args.no_kmeans:
        logger.info("Running K-Means codebook init on projected latent...")
        model = model.to(args.device)
        all_data = torch.tensor(dataset.embeddings[:], dtype=torch.float32, device=args.device)
        with torch.no_grad():
            projected = model.input_proj(all_data)
        model.rq.init_codebooks_kmeans(data=projected, kmeans_iters=args.kmeans_iters)

    # Trainer
    data_num = len(data_loader)
    trainer = RQMoETrainer(args, model, data_num)

    # Train
    logger.info("Starting training (bf16 autocast for memory efficiency)...")
    best_loss, best_collision = trainer.fit(data_loader)

    logger.info(set_color(f"Training complete. Best loss: {best_loss:.6f}, "
                          f"Best collision: {best_collision:.6f}", "green"))


if __name__ == "__main__":
    main()
