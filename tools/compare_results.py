#!/usr/bin/env python
"""
Compare two MiniOneRec experiment results and generate a report.

Usage:
    python tools/compare_results.py \
        --baseline results/eval/baseline_metrics.json \
        --improved results/eval/improved_metrics.json \
        --output results/comparison_report.md
"""

import argparse
import json
import os
import sys


def load_metrics(path, label):
    """Load metrics from a JSON file (metrics_all.json format)."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    # Find best RL entry (max NDCG@10, or the RL entry with highest step)
    best = None
    for row in data:
        name = row.get("name", "")
        step = row.get("step", 0)
        if "rl" in name and (best is None or step > best.get("step", 0)):
            best = row

    if best is None:
        print(f"[ERROR] No RL entries found in {path}")
        sys.exit(1)

    return {
        "label": label,
        "name": best["name"],
        "step": best["step"],
        "ndcg": best.get("ndcg", {}),
        "recall": best.get("recall", {}),
        "topk": best.get("topk", []),
    }


def format_table(baseline, improved):
    """Generate markdown comparison table."""
    topk = baseline["topk"]
    lines = []
    lines.append("## 实验结果对比")
    lines.append("")
    lines.append(f"| 实验 | {' | '.join(f'HR@{k}' for k in topk)} | {' | '.join(f'NDCG@{k}' for k in topk)} |")
    lines.append(f"|------|{'|'.join(['------' for _ in topk])}|{'|'.join(['------' for _ in topk])}|")

    for entry in [baseline, improved]:
        hr_str = " | ".join(f"{entry['recall'].get(str(k), 0):.4f}" for k in topk)
        ndcg_str = " | ".join(f"{entry['ndcg'].get(str(k), 0):.4f}" for k in topk)
        lines.append(f"| {entry['label']} | {hr_str} | {ndcg_str} |")

    lines.append("")

    # Improvement % calculation
    lines.append("## 改进幅度")
    lines.append("")
    lines.append(f"| 指标 | {baseline['label']} | {improved['label']} | Δ | Δ% |")
    lines.append("|------|------|------|------|------|")

    for k in topk:
        b_hr = baseline["recall"].get(str(k), 0)
        i_hr = improved["recall"].get(str(k), 0)
        delta = i_hr - b_hr
        pct = (delta / b_hr * 100) if b_hr > 0 else 0
        lines.append(f"| HR@{k} | {b_hr:.4f} | {i_hr:.4f} | {delta:+.4f} | {pct:+.1f}% |")

        b_ndcg = baseline["ndcg"].get(str(k), 0)
        i_ndcg = improved["ndcg"].get(str(k), 0)
        delta = i_ndcg - b_ndcg
        pct = (delta / b_ndcg * 100) if b_ndcg > 0 else 0
        lines.append(f"| NDCG@{k} | {b_ndcg:.4f} | {i_ndcg:.4f} | {delta:+.4f} | {pct:+.1f}% |")

    lines.append("")
    lines.append(f"*Baseline: {baseline['name']} (step={baseline['step']})*  ")
    lines.append(f"*Improved: {improved['name']} (step={improved['step']})*  ")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline metrics JSON")
    ap.add_argument("--improved", required=True, help="Improved metrics JSON")
    ap.add_argument("--baseline_label", default="RQ-VAE + GRPO (Baseline)")
    ap.add_argument("--improved_label", default="RQ-MoE + GDPO (Improved)")
    ap.add_argument("--output", default="results/comparison_report.md")
    args = ap.parse_args()

    baseline = load_metrics(args.baseline, args.baseline_label)
    improved = load_metrics(args.improved, args.improved_label)

    report = format_table(baseline, improved)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    print(report)
    print(f"\n[OK] Report saved to {args.output}")


if __name__ == "__main__":
    main()
