"""
TensorBoard logging utility for MiniOneRec training stages.

Provides drop-in loggers for each training stage that write to
logs/tensorboard/<stage>/ for real-time visualization.

Usage:
    from tools.tb_logger import TBSFTLogger, TBRLMetricsLogger

    # SFT: log loss, learning rate, epoch
    logger = TBSFTLogger("output/sft/Industrial_and_Scientific_plus")
    logger.log_metrics({"loss": 2.34, "eval_loss": 2.15, "lr": 3e-4}, step=100)

    # RL: log rewards, diversity, NDCG, HR
    rl_logger = TBRLMetricsLogger("output/rl/Industrial_and_Scientific_plus")
    rl_logger.log_step(step=100, metrics_dict={...})
"""

import os
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Base TensorBoard logger with common functionality."""

    def __init__(self, log_dir: str, tag: str = "train"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.tag = tag
        self._step = 0

    def log_scalar(self, name: str, value: float, step: int = None):
        if step is None:
            step = self._step
        self.writer.add_scalar(f"{self.tag}/{name}", value, step)

    def log_scalars(self, prefix: str, metrics: dict, step: int = None):
        if step is None:
            step = self._step
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{self.tag}/{prefix}/{k}", v, step)

    def step(self):
        self._step += 1
        return self._step

    def close(self):
        self.writer.close()


class TBStage1Logger(TBLogger):
    """Stage 1 (SID Construction) TensorBoard logger.

    Logs:
        - train/loss (total)
        - train/recon_loss (reconstruction MSE)
        - train/quant_loss (NRL for RQ-MoE, quant loss for RQ-VAE)
        - eval/collision_rate
    """

    def __init__(self, log_dir: str = "logs/tensorboard/stage1"):
        super().__init__(log_dir, tag="stage1")

    def log_train(self, total_loss: float, recon_loss: float,
                  quant_loss: float = None, epoch: int = None):
        step = epoch if epoch is not None else self.step()
        self.log_scalar("train/loss", total_loss, step)
        self.log_scalar("train/recon_loss", recon_loss, step)
        if quant_loss is not None:
            self.log_scalar("train/quant_loss", quant_loss, step)

    def log_eval(self, collision_rate: float, epoch: int):
        self.log_scalar("eval/collision_rate", collision_rate, epoch)


class TBSFTLogger(TBLogger):
    """Stage 2 (SFT) TensorBoard logger.

    Logs:
        - train/loss
        - train/learning_rate
        - eval/loss
    """

    def __init__(self, log_dir: str = "logs/tensorboard/sft"):
        super().__init__(log_dir, tag="sft")

    def log_train(self, loss: float, lr: float, step: int = None):
        s = step if step is not None else self.step()
        self.log_scalar("train/loss", loss, s)
        self.log_scalar("train/lr", lr, s)

    def log_eval(self, eval_loss: float, step: int = None):
        s = step if step is not None else self._step
        self.log_scalar("eval/loss", eval_loss, s)


class TBRLMetricsLogger(TBLogger):
    """Stage 3 (RL) TensorBoard logger.

    Logs:
        - rewards/<reward_name>
        - metrics/reward, metrics/reward_std
        - metrics/categorical_diversity, metrics/token_diversity
        - metrics/completion_length, metrics/kl
        - eval/NDCG@K, eval/HR@K
    """

    def __init__(self, log_dir: str = "logs/tensorboard/rl"):
        super().__init__(log_dir, tag="rl")

    def log_step(self, metrics: dict, step: int = None):
        """Log a dictionary of RL metrics."""
        s = step if step is not None else self.step()

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(key.replace("/", "_"), value, s)
            elif isinstance(value, dict):
                self.log_scalars(key.replace("/", "_"), value, s)

        self.writer.flush()

    def log_eval(self, ndcg: dict, hr: dict, step: int):
        """Log evaluation NDCG and HR scores."""
        for k, v in ndcg.items():
            self.log_scalar(f"eval/NDCG@{k}", v, step)
        for k, v in hr.items():
            self.log_scalar(f"eval/HR@{k}", v, step)


# ── Convenience: auto-detect stage and create logger ──

def create_logger(stage: str, run_name: str = None) -> TBLogger:
    """Factory: create appropriate logger for training stage.

    Args:
        stage: one of "stage1", "sft", "rl"
        run_name: optional subdirectory for multiple runs

    Returns:
        TBLogger subclass instance
    """
    base = "logs/tensorboard"
    if run_name:
        base = os.path.join(base, run_name)

    if stage == "stage1":
        return TBStage1Logger(os.path.join(base, "stage1"))
    elif stage == "sft":
        return TBSFTLogger(os.path.join(base, "sft"))
    elif stage == "rl":
        return TBRLMetricsLogger(os.path.join(base, "rl"))
    else:
        raise ValueError(f"Unknown stage: {stage}")
