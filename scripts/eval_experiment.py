#!/usr/bin/env python
"""
Evaluate the RQ-MoE + SFT + GDPO experiment:
  1) SFT model (final_checkpoint)
  2) All RL eval_snapshots (checkpoint-165 through checkpoint-1650)

Produces:
  - results/rqmoe_gdpo/eval_all.json     (raw metrics per model)
  - results/rqmoe_gdpo/results_table.md   (markdown comparison table)
  - assets/plot_rqmoe_gdpo_ndcg.png       (NDCG@K curve)
  - assets/plot_rqmoe_gdpo_recall.png     (Recall/HR@K curve)
"""

import os, sys, json, math, glob, subprocess, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYEXE = sys.executable
CATEGORY = "Industrial_and_Scientific"
INFO = os.path.join(ROOT, "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt")
TEST = os.path.join(ROOT, "data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")

SFT_DIR = os.path.join(ROOT, "output/sft/Industrial_and_Scientific_rqmoe/final_checkpoint")
RL_SNAP_DIR = os.path.join(ROOT, "output/rl/Industrial_and_Scientific_rqmoe_gdpo/eval_snapshots")

OUTDIR = os.path.join(ROOT, "results/rqmoe_gdpo")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)

TOPK = (1, 3, 5, 10, 20, 50)


def discover_models():
    models = []
    if os.path.isdir(SFT_DIR):
        models.append(("sft_base", 0, SFT_DIR))
    snaps = []
    for d in glob.glob(os.path.join(RL_SNAP_DIR, "checkpoint-*")):
        try:
            step = int(os.path.basename(d).split("checkpoint-")[1])
        except ValueError:
            continue
        snaps.append((f"rl_gdpo", step, d))
    snaps.sort(key=lambda x: x[1])
    return models + snaps


def run_generation(model_path, out_json, num_beams=50, batch_size=8, max_new_tokens=256):
    cmd = [
        PYEXE, os.path.join(ROOT, "evaluate.py"),
        "--base_model", model_path,
        "--info_file", INFO,
        "--category", CATEGORY,
        "--test_data_path", TEST,
        "--result_json_data", out_json,
        "--batch_size", str(batch_size),
        "--num_beams", str(num_beams),
        "--max_new_tokens", str(max_new_tokens),
        "--length_penalty", "0.0",
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def compute_metrics(result_json):
    with open(INFO) as f:
        item_names = {l.split('\t')[0].strip() for l in f}
    data = json.load(open(result_json))
    text = [[p.strip("\"\n").strip() for p in s["predict"]] for s in data]
    n_beam = len(text[0])
    valid = [k for k in TOPK if k <= n_beam]
    NDCG = np.zeros(len(valid)); HR = np.zeros(len(valid)); CC = 0
    for idx, sample in enumerate(text):
        out = data[idx]["output"]
        target = (out[0] if isinstance(out, list) else out).strip(" \n\"")
        minID = 10 ** 9
        for i, p in enumerate(sample):
            if p not in item_names:
                CC += 1
            if p == target:
                minID = i
                break
        for j, topk in enumerate(valid):
            if minID < topk:
                NDCG[j] += 1.0 / math.log(minID + 2)
                HR[j] += 1.0
    N = len(text)
    ndcg = (NDCG / N / (1.0 / math.log(2))).tolist()
    hr = (HR / N).tolist()
    return valid, ndcg, hr, int(CC), N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_beams", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip generation if pred JSON already exists")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    models = discover_models()
    print(f"Models to evaluate: {len(models)}")
    for name, step, path in models:
        print(f"  [{name}] step={step}  path={path}")

    results = []
    for name, step, path in models:
        print(f"\n{'='*60}")
        print(f"[{name}] step={step}")
        print(f"{'='*60}")

        out_json = os.path.join(OUTDIR, f"pred_{name}_step{step}.json")
        if args.skip_existing and os.path.exists(out_json):
            print(f"  Prediction exists, skipping generation: {out_json}")
        else:
            print(f"  Running generation...")
            run_generation(path, out_json, args.num_beams, args.batch_size, args.max_new_tokens)

        print(f"  Computing metrics...")
        valid, ndcg, hr, cc, N = compute_metrics(out_json)
        row = {
            "name": name, "step": step, "n_test": N, "invalid_pred": cc,
            "topk": valid,
            "ndcg": {str(k): round(v, 6) for k, v in zip(valid, ndcg)},
            "recall": {str(k): round(v, 6) for k, v in zip(valid, hr)},
        }
        results.append(row)
        print(f"  NDCG@{valid} = {[round(v, 4) for v in ndcg]}")
        print(f"  Recall@{valid} = {[round(v, 4) for v in hr]}  (invalid_pred={cc})")

        # Save incremental
        results.sort(key=lambda r: r["step"])
        with open(os.path.join(OUTDIR, "eval_all.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Generate Markdown table ──
    print(f"\n{'='*60}")
    print(f"Generating results table and plots...")

    with open(os.path.join(OUTDIR, "eval_all.json")) as f:
        results = json.load(f)

    lines = []
    lines.append("# RQ-MoE + SFT + GDPO 实验结果")
    lines.append("")
    lines.append(f"**数据集**: Amazon Industrial_and_Scientific  ")
    lines.append(f"**SID方法**: RQ-MoE (M=3, K=256, N=2, L=4)  ")
    lines.append(f"**RL方法**: GDPO (per-reward decoupled normalization)  ")
    lines.append(f"**Beam Search**: 50 beams  ")
    lines.append("")
    lines.append("## 指标对比")
    lines.append("")

    # Find which topk are available
    topk_vals = results[0]["topk"]
    report_k = [k for k in [3, 5, 10] if k in topk_vals]

    # Header
    header = "| Model | Step |"
    for k in report_k:
        header += f" HR@{k} | NDCG@{k} |"
    lines.append(header)
    sep = "|-------|------|" + "|".join(["--------|--------|" for _ in report_k])
    lines.append(sep)

    for row in results:
        name = row["name"]
        step = row["step"]
        recall = row["recall"]
        ndcg = row["ndcg"]
        line = f"| {name} | {step} |"
        for k in report_k:
            line += f" {recall.get(str(k), 0):.4f} | {ndcg.get(str(k), 0):.4f} |"
        lines.append(line)

    # Best improvement row
    lines.append("")
    sft_row = [r for r in results if r["name"] == "sft_base"][0]
    rl_rows = [r for r in results if "rl" in r["name"]]
    best_rl = max(rl_rows, key=lambda r: r["ndcg"].get("10", 0)) if rl_rows else None

    if best_rl:
        lines.append("## 改进幅度 (Best RL vs SFT)")
        lines.append("")
        imp_header = "| 指标 | SFT | Best RL | Δ | Δ% |"
        lines.append(imp_header)
        lines.append("|------|-----|---------|---|-----|")
        for k in report_k:
            sft_hr = sft_row["recall"].get(str(k), 0)
            rl_hr = best_rl["recall"].get(str(k), 0)
            delta_hr = rl_hr - sft_hr
            pct_hr = (delta_hr / sft_hr * 100) if sft_hr > 0 else 0
            lines.append(f"| HR@{k} | {sft_hr:.4f} | {rl_hr:.4f} | {delta_hr:+.4f} | {pct_hr:+.1f}% |")

            sft_ndcg = sft_row["ndcg"].get(str(k), 0)
            rl_ndcg = best_rl["ndcg"].get(str(k), 0)
            delta_ndcg = rl_ndcg - sft_ndcg
            pct_ndcg = (delta_ndcg / sft_ndcg * 100) if sft_ndcg > 0 else 0
            lines.append(f"| NDCG@{k} | {sft_ndcg:.4f} | {rl_ndcg:.4f} | {delta_ndcg:+.4f} | {pct_ndcg:+.1f}% |")

        lines.append("")
        lines.append(f"*Best RL checkpoint: {best_rl['name']} step={best_rl['step']}*")

    md_path = os.path.join(OUTDIR, "results_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown table saved to {md_path}")

    # ── Plots ──
    if not args.no_plot:
        # Prepare data
        steps = [r["step"] for r in results]
        sft_ndcg = sft_row["ndcg"]
        sft_recall = sft_row["recall"]

        plot_k = [k for k in report_k if k <= 10]

        # NDCG plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {5: "#2196F3", 10: "#FF9800"}
        for k in plot_k:
            vals = [r["ndcg"].get(str(k), None) for r in results]
            ax.plot(steps, vals, 'o-', color=colors.get(k, None), linewidth=1.5, markersize=5, label=f"NDCG@{k}")
        for k in plot_k:
            ax.axhline(y=sft_ndcg.get(str(k), 0), color=colors.get(k, None), linestyle="--", linewidth=1, alpha=0.6, label=f"SFT NDCG@{k}")
        ax.set_xlabel("RL Training Step", fontsize=12)
        ax.set_ylabel("NDCG", fontsize=12)
        ax.set_title("RQ-MoE + GDPO: NDCG@K vs Training Step", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        ndcg_path = os.path.join(ROOT, "assets/plot_rqmoe_gdpo_ndcg.png")
        fig.savefig(ndcg_path, dpi=150)
        plt.close()
        print(f"NDCG plot saved to {ndcg_path}")

        # Recall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in plot_k:
            vals = [r["recall"].get(str(k), None) for r in results]
            ax.plot(steps, vals, 'o-', color=colors.get(k, None), linewidth=1.5, markersize=5, label=f"Recall@{k}")
        for k in plot_k:
            ax.axhline(y=sft_recall.get(str(k), 0), color=colors.get(k, None), linestyle="--", linewidth=1, alpha=0.6, label=f"SFT Recall@{k}")
        ax.set_xlabel("RL Training Step", fontsize=12)
        ax.set_ylabel("Recall (HR)", fontsize=12)
        ax.set_title("RQ-MoE + GDPO: Recall@K vs Training Step", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        rec_path = os.path.join(ROOT, "assets/plot_rqmoe_gdpo_recall.png")
        fig.savefig(rec_path, dpi=150)
        plt.close()
        print(f"Recall plot saved to {rec_path}")

    print(f"\n[OK] All done!")
    print(f"  Results: {OUTDIR}/eval_all.json")
    print(f"  Table:   {OUTDIR}/results_table.md")


if __name__ == "__main__":
    main()
