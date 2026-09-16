"""
Paired t-test: QAG+AMD vs QMF+AMD (10-seed, per backbone)
- avg_val_ccc (main metric)
- video_only/ccc_avg (missing modality robustness)
- Cohen's d for effect size
"""

import json
import os
import glob
import numpy as np
from scipy import stats
from collections import defaultdict
from paths import CHECKPOINTS_ROOT

SAVED_DIR = CHECKPOINTS_ROOT
BACKBONES = ["ViViT_s4", "VideoMAE_s6", "I3D"]
SEEDS = list(range(10))


def cohens_d(x, y):
    """Paired Cohen's d = mean(diff) / std(diff)"""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)


def load_all_runs():
    """Load all runs, return dict keyed by (method, backbone, seed) -> results"""
    runs = {}
    dirs = glob.glob(os.path.join(SAVED_DIR, "*/hyperparameters.json"))

    for hp_path in dirs:
        run_dir = os.path.dirname(hp_path)
        res_path = os.path.join(run_dir, "best_da_results.json")
        if not os.path.exists(res_path):
            continue

        with open(hp_path) as f:
            hp = json.load(f)
        with open(res_path) as f:
            res = json.load(f)

        # Filter: single backbone only
        if len(hp.get("vision_backbones", [])) != 1:
            continue

        bb = hp["vision_backbones"][0]
        if bb not in BACKBONES:
            continue

        seed = hp.get("seed")
        if seed is None or seed not in SEEDS:
            continue

        da_qmf = hp.get("da_qmf_gating", False)
        da_qag = hp.get("da_quality_gating", False)
        da_amd = hp.get("da_adaptive_dropout", False)

        # QAG+AMD: quality_gating=True, adaptive_dropout=True, qmf=False
        if da_qag and da_amd and not da_qmf:
            method = "QAG+AMD"
        # QMF+AMD: qmf_gating=True, adaptive_dropout=True
        elif da_qmf and da_amd:
            method = "QMF+AMD"
        else:
            continue

        key = (method, bb, seed)
        # If duplicate, keep the one with higher avg_val_ccc (latest run)
        if key in runs:
            if res["avg_val_ccc"] <= runs[key]["avg_val_ccc"]:
                continue
        runs[key] = res

    return runs


def main():
    runs = load_all_runs()

    # Organize by method and backbone
    results = defaultdict(lambda: defaultdict(dict))
    for (method, bb, seed), res in runs.items():
        results[method][bb][seed] = res

    # Check completeness
    for method in ["QAG+AMD", "QMF+AMD"]:
        for bb in BACKBONES:
            seeds_found = sorted(results[method][bb].keys())
            if len(seeds_found) != 10:
                print(f"WARNING: {method} {bb} has {len(seeds_found)} seeds: {seeds_found}")

    print("=" * 80)
    print("PAIRED T-TEST: QAG+AMD vs QMF+AMD (10-seed)")
    print("=" * 80)

    metrics = [
        ("avg_val_ccc", "Avg CCC (main metric)"),
        ("val_ccc_v", "CCC-V (Valence)"),
        ("val_ccc_a", "CCC-A (Arousal)"),
        ("video_only/ccc_avg", "Video-Only Avg CCC (missing modality)"),
    ]

    for metric_key, metric_name in metrics:
        print(f"\n{'─' * 70}")
        print(f"  Metric: {metric_name}")
        print(f"{'─' * 70}")
        print(f"{'Backbone':<14} {'QAG+AMD':>12} {'QMF+AMD':>12} {'Diff':>10} {'t-stat':>8} {'p-value':>10} {'Cohen d':>8} {'Sig':>6}")
        print(f"{'─' * 70}")

        for bb in BACKBONES:
            qag_vals = []
            qmf_vals = []

            for seed in SEEDS:
                qag_res = results["QAG+AMD"][bb].get(seed)
                qmf_res = results["QMF+AMD"][bb].get(seed)
                if qag_res is None or qmf_res is None:
                    continue

                # Handle nested keys like "video_only/ccc_avg"
                if "/" in metric_key:
                    qag_v = qag_res.get(metric_key)
                    qmf_v = qmf_res.get(metric_key)
                else:
                    qag_v = qag_res.get(metric_key)
                    qmf_v = qmf_res.get(metric_key)

                if qag_v is not None and qmf_v is not None:
                    qag_vals.append(qag_v)
                    qmf_vals.append(qmf_v)

            if len(qag_vals) < 2:
                print(f"{bb:<14} insufficient data (n={len(qag_vals)})")
                continue

            qag_mean = np.mean(qag_vals)
            qmf_mean = np.mean(qmf_vals)
            diff = qag_mean - qmf_mean

            t_stat, p_val = stats.ttest_rel(qag_vals, qmf_vals)
            d = cohens_d(qag_vals, qmf_vals)

            # Significance stars
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            print(f"{bb:<14} {qag_mean:>12.4f} {qmf_mean:>12.4f} {diff:>+10.4f} {t_stat:>8.3f} {p_val:>10.6f} {d:>+8.3f} {sig:>6}")

        print()

    # Detailed per-seed values for avg_val_ccc
    print("\n" + "=" * 80)
    print("PER-SEED VALUES: avg_val_ccc")
    print("=" * 80)
    for bb in BACKBONES:
        print(f"\n  {bb}:")
        print(f"  {'Seed':>4}  {'QAG+AMD':>10}  {'QMF+AMD':>10}  {'Diff':>10}")
        for seed in SEEDS:
            qag_res = results["QAG+AMD"][bb].get(seed)
            qmf_res = results["QMF+AMD"][bb].get(seed)
            if qag_res and qmf_res:
                qag_v = qag_res["avg_val_ccc"]
                qmf_v = qmf_res["avg_val_ccc"]
                print(f"  {seed:>4}  {qag_v:>10.4f}  {qmf_v:>10.4f}  {qag_v - qmf_v:>+10.4f}")
            else:
                print(f"  {seed:>4}  {'N/A':>10}  {'N/A':>10}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Cohen's d interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small,")
    print("                          0.5-0.8 = medium, > 0.8 = large")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print("Positive diff = QAG+AMD > QMF+AMD (QAG+AMD is better)")


if __name__ == "__main__":
    main()
