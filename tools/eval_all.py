#!/usr/bin/env python
"""批量评测 SFT 基线 + 所有 RL 快照，汇总 NDCG@K / Recall(HR)@K。

- 生成阶段：直接调用作者的 evaluate.py（单卡、全量 test、constrained beam search），不改其逻辑。
- 指标阶段：用与作者 calc.py 完全相同的公式从预测 json 计算指标，便于结构化保存 + 画图。
  （每条 test 样本单正例，故 HR@K == Recall@K。）

用法（等 RL 跑完、显存空出来再跑）：
    CUDA_VISIBLE_DEVICES=0 python tools/eval_all.py
可选：--skip_existing 跳过已生成预测的模型；--num_beams 50；--batch_size 8
"""
import os, sys, json, math, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYEXE = sys.executable
CATEGORY = "Industrial_and_Scientific"
INFO = os.path.join(ROOT, "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt")
TEST = os.path.join(ROOT, "data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv")
SFT_DIR = os.path.join(ROOT, "output/sft/Industrial_and_Scientific_plus")
SNAP_GLOB = os.path.join(ROOT, "output/rl/Industrial_and_Scientific_plus/eval_snapshots/checkpoint-*")
OUTDIR = os.path.join(ROOT, "results/eval")
TOPK = (1, 3, 5, 10, 20, 50)


def discover_models():
    """返回 [(name, step, path), ...]，SFT 记为 step=0。"""
    models = []
    if os.path.isdir(SFT_DIR):
        models.append(("sft", 0, SFT_DIR))
    snaps = []
    for d in glob.glob(SNAP_GLOB):
        try:
            step = int(d.rsplit("checkpoint-", 1)[1])
        except ValueError:
            continue
        snaps.append((f"rl-{step}", step, d))
    snaps.sort(key=lambda x: x[1])
    return models + snaps


def run_generation(model_path, out_json, num_beams, batch_size, max_new_tokens):
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
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def compute_metrics(result_json):
    """与 calc.py 同公式：找正例首次命中的名次 minID，按 topk 累加 NDCG/HR。"""
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
                    help="若预测 json 已存在则跳过生成（仍重算指标）")
    ap.add_argument("--no_plot", action="store_true", help="评测完不自动画图")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    models = discover_models()
    print(f"待评模型 {len(models)} 个：" + ", ".join(m[0] for m in models))

    results = []
    for name, step, path in models:
        print(f"\n===== [{name}] step={step} =====")
        out_json = os.path.join(OUTDIR, f"pred_{name}.json")
        if args.skip_existing and os.path.exists(out_json):
            print(f"  预测已存在，跳过生成：{out_json}")
        else:
            run_generation(path, out_json, args.num_beams, args.batch_size, args.max_new_tokens)
        valid, ndcg, hr, cc, N = compute_metrics(out_json)
        row = {"name": name, "step": step, "n_test": N, "invalid_pred": cc,
               "topk": valid,
               "ndcg": {k: round(v, 6) for k, v in zip(valid, ndcg)},
               "recall": {k: round(v, 6) for k, v in zip(valid, hr)}}
        results.append(row)
        print(f"  NDCG@{valid} = {[round(v,4) for v in ndcg]}")
        print(f"  Recall@{valid} = {[round(v,4) for v in hr]}  (invalid_pred={cc})")
        # 每评完一个就落盘，防中途崩溃丢结果
        results.sort(key=lambda r: r["step"])
        with open(os.path.join(OUTDIR, "metrics_all.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n全部完成，指标写入 {os.path.join(OUTDIR, 'metrics_all.json')}")

    if not args.no_plot:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import plot_metrics
            plot_metrics.main()
        except Exception as e:
            print(f"[warn] 画图失败（指标已保存，可单独跑 plot_metrics.py 重画）：{e}")


if __name__ == "__main__":
    main()
