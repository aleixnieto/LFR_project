import json
from pathlib import Path
from math import comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("evaluations/selection_ratio_heldout")
RATIOS = ["0.1", "0.3", "0.5", "0.7", "0.9", "1.0"]
OUT = ROOT / "analysis"


def load_predictions():
    """Load predictions and metrics for all ratios and verify alignment."""
    preds, metrics = {}, {}
    for r in RATIOS:
        d = ROOT / f"ratio_{r}"
        preds[r] = json.load(open(d / "predictions.json"))
        metrics[r] = json.load(open(d / "metrics.json"))

    ref = preds["0.3"]
    ref_files = [p["filename"] for p in ref]
    ref_gt = [p["ground_truth"] for p in ref]
    for r in RATIOS:
        assert [p["filename"] for p in preds[r]] == ref_files, f"filenames differ for {r}"
        assert [p["ground_truth"] for p in preds[r]] == ref_gt, f"ground truth differs for {r}"
    print(f"Alignment OK: {len(ref_files)} images identical across {len(RATIOS)} models")
    return preds, metrics, ref_files, ref_gt


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test."""
    n01 = int(np.sum(a & ~b))   # a correct, b wrong
    n10 = int(np.sum(~a & b))   # a wrong, b correct
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def run_mcnemar(correct):
    """Run McNemar tests for all pairs against baseline (0.3)."""
    pairs = [("0.1", "0.3"), ("0.5", "0.3"), ("0.7", "0.3"), ("0.9", "0.3"), ("1.0", "0.3"),
             ("0.9", "1.0"), ("0.7", "1.0"), ("0.9", "0.7")]
    rows = []
    for a, b in pairs:
        n_a_only, n_b_only, p = mcnemar_exact(correct[a], correct[b])
        rows.append({"A": a, "B": b, "A_only_correct": n_a_only, "B_only_correct": n_b_only,
                     "diff_acc_pp": round(100 * (correct[a].mean() - correct[b].mean()), 2),
                     "p_value": round(p, 4)})
    mc = pd.DataFrame(rows)
    print("\nExact McNemar test (A vs B):")
    print(mc.to_string(index=False))
    mc.to_csv(OUT / "mcnemar.csv", index=False)


def global_metrics_table(correct, metrics):
    """Build and save global metrics table."""
    glob = pd.DataFrame([{
        "ratio": r,
        "accuracy": metrics[r].get("accuracy", correct[r].mean()),
        "errors": int((~correct[r]).sum()),
    } for r in RATIOS])
    for r_idx, r in enumerate(RATIOS):
        m = metrics[r]
        for k in ("weighted_precision", "weighted_recall", "weighted_f1"):
            glob.loc[r_idx, k] = m.get(k, np.nan)
    glob.to_csv(OUT / "global_metrics.csv", index=False)
    print("\nGlobal metrics:")
    print(glob.round(4).to_string(index=False))
    return glob


def per_class_analysis(correct, ref_gt):
    """Per-class accuracy across ratios."""
    classes = sorted(set(ref_gt))
    gt_arr = np.array(ref_gt)
    per_class = pd.DataFrame(
        {r: [correct[r][gt_arr == c].mean() for c in classes] for r in RATIOS}, index=classes)
    per_class["support"] = [int((gt_arr == c).sum()) for c in classes]
    per_class["range"] = per_class[RATIOS].max(axis=1) - per_class[RATIOS].min(axis=1)
    per_class.round(4).to_csv(OUT / "per_class_accuracy.csv")
    print("\nClasses most sensitive to ratio (max-min accuracy range):")
    print(per_class.sort_values("range", ascending=False).head(10).round(3).to_string())


def plot_curves(glob):
    """Plot accuracy and F1 vs selection ratio."""
    x = [float(r) for r in RATIOS]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, [100 * glob.loc[i, "accuracy"] for i in range(len(RATIOS))], "o-")
    ax[0].set_title("Accuracy (%) vs selection ratio")
    ax[1].plot(x, [100 * glob.loc[i, "weighted_f1"] for i in range(len(RATIOS))], "o-", color="C1")
    ax[1].set_title("Weighted F1 (%) vs selection ratio")
    for a in ax:
        a.set_xlabel("selection_ratio")
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "ratio_curves.png", dpi=200)


def main():
    OUT.mkdir(exist_ok=True)
    preds, metrics, ref_files, ref_gt = load_predictions()
    correct = {r: np.array([p["prediction"] == p["ground_truth"] for p in preds[r]]) for r in RATIOS}
    run_mcnemar(correct)
    glob = global_metrics_table(correct, metrics)
    per_class_analysis(correct, ref_gt)
    plot_curves(glob)
    print(f"\nFiles written to {OUT}/")


if __name__ == "__main__":
    main()
