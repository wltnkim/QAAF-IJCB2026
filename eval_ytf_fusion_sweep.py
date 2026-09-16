#!/usr/bin/env python3
"""
Phase 1-3: YTF cross-pair score fusion sweep — Hard × Soft matrix.

Builds a 5×11 best-EER matrix over all Hard×Soft pairs, with per-pair α optimum.
For each pair: z-score normalized weighted sum, α ∈ [0, 1] in 0.05 steps,
record best overall EER and 10-fold mean.

Reuses score computation utilities from eval_fusion_multi_ytf.py.

Output: results/ytf_fusion_sweep_HxS.json
        results/ytf_fusion_sweep_HxS.csv
"""
import os
import json
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cosine as cosine_dist


FEATURES_DIR = "features/YTF"
META_FILE    = "data/YTF/meta_data/meta_and_splits.mat"
OUT_JSON     = "results/ytf_fusion_sweep_HxS.json"
OUT_CSV      = "results/ytf_fusion_sweep_HxS.csv"

# The 17 baseline methods compared in the paper.
HARD_BACKBONES = {
    "ArcFace":       "ArcFace",
    "AdaFace":       "AdaFace_IR101",
    "EdgeFace":      "EdgeFace",
    "SFace":         "SFace",
    "SynthDistill":  "SynthDistill",
}
SOFT_BACKBONES = {
    "VA-ViViT":         "ViViT",            # ours
    "VA-VideoMAE":      "VideoMAE",          # ours
    "JMT-R2D1":         "R2D1",
    "JMT-I3D":          "I3D",
    "ViViT-pretr":      "ViViT_pretrained",
    "VideoMAE-pretr":   "VideoMAE_pretrained",
    "EmotiEffLib":      "EmotiEffLib",
    "FER":              "FER",
    "DAN":              "DAN",
    "POSTER++":         "POSTERV2",
    "MAE-DFER":         "MAE_DFER",
}


def load_features(features_dir):
    """{Person/video_id: mean-pooled feature vector (np.ndarray)}"""
    feats = {}
    for person in sorted(os.listdir(features_dir)):
        pd = os.path.join(features_dir, person)
        if not os.path.isdir(pd):
            continue
        for fname in sorted(os.listdir(pd)):
            if not fname.endswith(".pt"):
                continue
            vid = os.path.splitext(fname)[0]
            data = torch.load(os.path.join(pd, fname), map_location="cpu",
                              weights_only=False)
            if isinstance(data, torch.Tensor):
                feats[f"{person}/{vid}"] = data.float().mean(dim=0).numpy()
    return feats


def parse_splits(meta_file):
    m = loadmat(meta_file)
    video_names = m["video_names"]; splits = m["Splits"]
    def vn(i):
        v = video_names[i - 1, 0]
        return str(v[0]) if hasattr(v, "__len__") else str(v)
    pairs = []
    npp, _, ns = splits.shape
    for s in range(ns):
        for i in range(npp):
            pairs.append((s + 1, vn(int(splits[i, 0, s])),
                                  vn(int(splits[i, 1, s])),
                                  int(splits[i, 2, s])))
    return pairs


def score_pairs(feats, pairs):
    sid, sc, lab = [], [], []
    miss = 0
    for s, v1, v2, y in pairs:
        if v1 not in feats or v2 not in feats:
            miss += 1
            continue
        sim = 1.0 - cosine_dist(feats[v1], feats[v2])
        sid.append(s); sc.append(float(sim)); lab.append(int(y))
    return np.array(sid), np.array(sc), np.array(lab), miss


def compute_eer_auc(scores, labels):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    thr = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.mean(scores[labels == 0] >= t) for t in thr])
    frr = np.array([np.mean(scores[labels == 1] < t) for t in thr])
    diff = far - frr
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        eer = 0.5
    else:
        i = sc[0]
        den = diff[i+1] - diff[i]
        a = -diff[i]/den if den != 0 else 0.5
        eer = far[i] + a * (far[i+1] - far[i])
    order = np.argsort(far)
    auc = float(np.trapz(1 - frr[order], far[order]))
    return float(eer), auc


def per_fold_eer(sids, scores, labels, n_splits=10):
    fe = []
    for s in range(1, n_splits + 1):
        m = sids == s
        if m.sum() == 0:
            continue
        e, _ = compute_eer_auc(scores[m], labels[m])
        if not np.isnan(e):
            fe.append(e)
    return float(np.mean(fe)) if fe else float("nan"), float(np.std(fe)) if fe else float("nan")


def z(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd if sd > 0 else 1.0)


def fusion_sweep(sc_h, sc_s, sids, labels):
    """α=0 → all soft, α=1 → all hard."""
    zh, zs = z(sc_h), z(sc_s)
    best = {"alpha": 0.0, "overall_eer": 1.0}
    for alpha in np.arange(0.0, 1.001, 0.05):
        fused = alpha * zh + (1 - alpha) * zs
        e, _ = compute_eer_auc(fused, labels)
        if not np.isnan(e) and e < best["overall_eer"]:
            best = {"alpha": float(alpha), "overall_eer": float(e)}
    fused = best["alpha"] * zh + (1 - best["alpha"]) * zs
    mean_eer, std_eer = per_fold_eer(sids, fused, labels)
    best["fold_mean_eer"] = mean_eer
    best["fold_std_eer"] = std_eer
    return best


def main():
    print(f"Loading YTF pairs from {META_FILE} ...")
    pairs = parse_splits(META_FILE)
    print(f"  {len(pairs)} pairs ({sum(1 for *_, y in pairs if y==1)} genuine)")

    # Load all 16 backbone scores (5 Hard + 11 Soft)
    cache = {}  # name → (sids, scores, labels)
    individual = {}  # name → {overall_eer, mean_eer, std_eer}
    for name, subdir in {**HARD_BACKBONES, **SOFT_BACKBONES}.items():
        path = os.path.join(FEATURES_DIR, subdir)
        print(f"\n=== {name:<18} ({path}) ===")
        feats = load_features(path)
        print(f"  {len(feats)} videos, dim={list(feats.values())[0].shape[0]}")
        sid, sc, lab, miss = score_pairs(feats, pairs)
        if miss:
            print(f"  [warn] {miss} pairs dropped")
        e, _ = compute_eer_auc(sc, lab)
        me, se = per_fold_eer(sid, sc, lab)
        cache[name] = (sid, sc, lab)
        individual[name] = {"overall_eer": e, "fold_mean_eer": me, "fold_std_eer": se}
        print(f"  overall EER={e:.4f}  10-fold EER={me:.4f}±{se:.4f}")

    # Alignment check (all backbones must produce same sid/labels ordering for fusion)
    ref_name = list(cache.keys())[0]
    ref_sid, _, ref_lab = cache[ref_name]
    for name, (sid, _, lab) in cache.items():
        assert np.array_equal(sid, ref_sid) and np.array_equal(lab, ref_lab), \
            f"{name} ordering mismatch with {ref_name}"
    print("\nAll backbones aligned. Computing fusion sweep ...\n")

    # Hard × Soft sweep
    fusions = {}
    for h_name in HARD_BACKBONES:
        for s_name in SOFT_BACKBONES:
            sid_h, sc_h, lab = cache[h_name]
            _,     sc_s, _   = cache[s_name]
            f = fusion_sweep(sc_h, sc_s, sid_h, lab)
            f["hard_baseline_eer"] = individual[h_name]["overall_eer"]
            f["soft_baseline_eer"] = individual[s_name]["overall_eer"]
            f["delta_vs_hard"] = f["overall_eer"] - individual[h_name]["overall_eer"]
            key = f"{h_name}+{s_name}"
            fusions[key] = f
            sgn = "+" if f["delta_vs_hard"] >= 0 else ""
            print(f"  {h_name:<14} + {s_name:<16}  α={f['alpha']:.2f}  "
                  f"EER={f['overall_eer']:.4f} ({sgn}{f['delta_vs_hard']:.4f})")

    out = {
        "dataset": "YTF",
        "n_pairs": int(len(pairs)),
        "individual": individual,
        "fusions": fusions,
        "hard_backbones": list(HARD_BACKBONES.keys()),
        "soft_backbones": list(SOFT_BACKBONES.keys()),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n✓ Saved: {OUT_JSON}")

    # CSV: rows=Hard, cols=Soft, value=overall_eer (best α)
    with open(OUT_CSV, "w") as fh:
        fh.write("Hard," + ",".join(SOFT_BACKBONES.keys()) + ",HardOnly\n")
        for h in HARD_BACKBONES:
            row = [h]
            for s in SOFT_BACKBONES:
                key = f"{h}+{s}"
                row.append(f"{fusions[key]['overall_eer']:.4f}")
            row.append(f"{individual[h]['overall_eer']:.4f}")
            fh.write(",".join(row) + "\n")
        # Soft baseline row
        row = ["SoftOnly"]
        for s in SOFT_BACKBONES:
            row.append(f"{individual[s]['overall_eer']:.4f}")
        row.append("")
        fh.write(",".join(row) + "\n")
    print(f"✓ Saved: {OUT_CSV}")

    # Summary: best Δ per Hard
    print(f"\n{'='*70}\nSummary — best Δ per Hard backbone:")
    for h in HARD_BACKBONES:
        deltas = [(s, fusions[f"{h}+{s}"]["delta_vs_hard"],
                      fusions[f"{h}+{s}"]["alpha"],
                      fusions[f"{h}+{s}"]["overall_eer"]) for s in SOFT_BACKBONES]
        deltas.sort(key=lambda x: x[1])  # most-negative (biggest improvement) first
        s, d, a, eer = deltas[0]
        sgn = "+" if d >= 0 else ""
        print(f"  {h:<14} best: +{s:<16} α={a:.2f}  "
              f"EER {individual[h]['overall_eer']:.4f}→{eer:.4f} ({sgn}{d:.4f})")


if __name__ == "__main__":
    main()
