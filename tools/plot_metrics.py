#!/usr/bin/env python
"""读取 results/eval/metrics_all.json，画 NDCG@K / Recall@K 随 RL 训练步的变化曲线。
SFT(step=0) 作为水平虚线基线，RL 各快照连成实线。

用法： python tools/plot_metrics.py
输出： results/eval/plot_ndcg.png, plot_recall.png
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "results/eval")
KS = [5, 10, 20]          # 展示这几档（最常报告）
COLORS = {5: "#d62728", 10: "#1f77b4", 20: "#2ca02c"}


def main():
    data = json.load(open(os.path.join(OUTDIR, "metrics_all.json")))
    sft = next((r for r in data if r["name"] == "sft"), None)
    rl = [r for r in data if r["name"] != "sft"]
    rl.sort(key=lambda r: r["step"])
    steps = [r["step"] for r in rl]

    for metric, fname, title in [("ndcg", "plot_ndcg.png", "NDCG@K"),
                                 ("recall", "plot_recall.png", "Recall@K (=HR@K)")]:
        plt.figure(figsize=(8, 5))
        for k in KS:
            ys = [r[metric].get(str(k), r[metric].get(k)) for r in rl]
            plt.plot(steps, ys, marker="o", color=COLORS[k], label=f"RL @{k}")
            if sft is not None:
                base = sft[metric].get(str(k), sft[metric].get(k))
                plt.axhline(base, color=COLORS[k], ls="--", alpha=0.6,
                            label=f"SFT @{k}={base:.4f}")
        plt.xlabel("RL training step")
        plt.ylabel(title)
        plt.title(f"{title}: RL snapshots vs SFT baseline\n(Industrial_and_Scientific, Qwen2.5-1.5B)")
        plt.legend(fontsize=8, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUTDIR, fname)
        plt.savefig(out, dpi=150)
        print("saved", out)


if __name__ == "__main__":
    main()
