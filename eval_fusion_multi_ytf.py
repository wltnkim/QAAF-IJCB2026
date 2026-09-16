#!/usr/bin/env python3
"""
YTF multi-pair score-level fusion.

Uses z-score normalized weighted sum with α sweep (same as existing
eval_ytf_verification.py:ArcFace+ViViT fusion).

Pairs evaluated:
  A) ArcFace  + R2D1
  B) AdaFace  + ViViT
  C) EdgeFace + ViViT

Also reproduces ArcFace + ViViT as a baseline under this script.
"""

import os
import json
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cosine as cosine_dist


FEATURES_DIR = "features/YTF"
META_FILE    = "data/YTF/meta_data/meta_and_splits.mat"
OUT_PATH     = "ytf_fusion_multi_results.json"

BACKBONE_DIRS = {
    "ArcFace":   "ArcFace",
    "AdaFace":   "AdaFace_IR101",
    "EdgeFace":  "EdgeFace",
    "ViViT":     "ViViT",       # VA-trained ViViT (feature dir naming)
    "R2D1":      "R2D1",
}

PAIRS = [
    ("ArcFace",  "ViViT"),   # baseline reproduction
    ("ArcFace",  "R2D1"),    # A
    ("AdaFace",  "ViViT"),   # B
    ("EdgeFace", "ViViT"),   # C
]


def load_features(features_dir):
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
    video_names = m["video_names"]
    splits = m["Splits"]

    def vn(i):
        v = video_names[i - 1, 0]
        return str(v[0]) if hasattr(v, "__len__") else str(v)

    pairs = []
    npp, _, ns = splits.shape
    for s in range(ns):
        for i in range(npp):
            i1 = int(splits[i, 0, s]); i2 = int(splits[i, 1, s])
            pairs.append((s + 1, vn(i1), vn(i2), int(splits[i, 2, s])))
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
    if miss:
        print(f"  [warn] {miss} pairs dropped")
    return np.array(sid), np.array(sc), np.array(lab)


def compute_eer_auc(scores, labels):
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    thr = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.mean(impostor >= t) for t in thr])
    frr = np.array([np.mean(genuine  < t) for t in thr])
    diff = far - frr
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        eer = 0.5
    else:
        i = sc[0]
        den = diff[i + 1] - diff[i]
        a = -diff[i] / den if den != 0 else 0.5
        eer = far[i] + a * (far[i + 1] - far[i])
    tpr = 1 - frr
    order = np.argsort(far)
    auc = float(np.trapz(tpr[order], far[order]))
    return float(eer), auc


def per_fold(split_ids, scores, labels, n_splits=10):
    out = {"folds": {}}
    fe, fa = [], []
    for s in range(1, n_splits + 1):
        m = split_ids == s
        if m.sum() == 0:
            continue
        e, a = compute_eer_auc(scores[m], labels[m])
        out["folds"][s] = {"n": int(m.sum()), "eer": e, "auc": a}
        fe.append(e); fa.append(a)
    if fe:
        out["mean_eer"] = float(np.mean(fe)); out["std_eer"] = float(np.std(fe))
        out["mean_auc"] = float(np.mean(fa)); out["std_auc"] = float(np.std(fa))
    oe, oa = compute_eer_auc(scores, labels)
    out["overall_eer"] = oe; out["overall_auc"] = oa
    out["n_pairs"] = int(len(scores))
    return out


def z(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd if sd > 0 else 1.0)


def fusion_sweep(sc_a, sc_b, labels, sids):
    za = z(sc_a); zb = z(sc_b)
    alphas = np.arange(0.0, 1.001, 0.05)
    best = {"eer": 1.0, "alpha": 0.0, "auc": 0.0}
    sweep = []
    for a in alphas:
        fused = a * za + (1 - a) * zb
        e, u = compute_eer_auc(fused, labels)
        sweep.append({"alpha": float(a), "eer": float(e), "auc": float(u)})
        if e < best["eer"]:
            best = {"eer": float(e), "alpha": float(a), "auc": float(u)}
    fused_best = best["alpha"] * za + (1 - best["alpha"]) * zb
    fold = per_fold(sids, fused_best, labels)
    return {
        "best_alpha": best["alpha"],
        "overall_eer": best["eer"],
        "overall_auc": best["auc"],
        "fold_mean_eer": fold.get("mean_eer"),
        "fold_std_eer":  fold.get("std_eer"),
        "fold_mean_auc": fold.get("mean_auc"),
        "fold_std_auc":  fold.get("std_auc"),
        "sweep": sweep,
    }


def main():
    pairs = parse_splits(META_FILE)
    print(f"Pairs: {len(pairs)} ({sum(1 for *_, y in pairs if y==1)} genuine)")

    needed = set()
    for h, s in PAIRS:
        needed.add(h); needed.add(s)

    scores_cache = {}   # name → (sid, sc, lab)
    individual = {}
    for name in sorted(needed):
        subdir = BACKBONE_DIRS[name]
        path = os.path.join(FEATURES_DIR, subdir)
        print(f"\n=== {name}  ({path}) ===")
        feats = load_features(path)
        print(f"  loaded {len(feats)} videos, dim={list(feats.values())[0].shape[0]}")
        sid, sc, lab = score_pairs(feats, pairs)
        scores_cache[name] = (sid, sc, lab)
        m = per_fold(sid, sc, lab)
        individual[name] = {
            "overall_eer": m["overall_eer"], "overall_auc": m["overall_auc"],
            "mean_eer": m.get("mean_eer"),   "std_eer": m.get("std_eer"),
            "mean_auc": m.get("mean_auc"),   "std_auc": m.get("std_auc"),
            "n_pairs": m["n_pairs"],
        }
        print(f"  overall EER={m['overall_eer']:.4f}  "
              f"10-fold EER={m.get('mean_eer',0):.4f}±{m.get('std_eer',0):.4f}")

    # Check alignment (all pair orderings should agree since computed from same pairs list)
    ref = scores_cache[sorted(needed)[0]]
    for name, (sid, _, lab) in scores_cache.items():
        assert np.array_equal(sid, ref[0]) and np.array_equal(lab, ref[2]), \
            f"{name} ordering mismatch"
    print("\nAll backbones aligned on pair ordering.\n")

    fusions = {}
    for h, s in PAIRS:
        key = f"{h}+{s}"
        sid_h, sc_h, lab_h = scores_cache[h]
        _,     sc_s, _     = scores_cache[s]
        f = fusion_sweep(sc_h, sc_s, lab_h, sid_h)
        delta = f["overall_eer"] - individual[h]["overall_eer"]
        delta_fold = (f["fold_mean_eer"] - individual[h]["mean_eer"]
                      if individual[h].get("mean_eer") is not None else None)
        f["delta_vs_hard_overall"] = delta
        f["delta_vs_hard_fold"]    = delta_fold
        fusions[key] = f
        sgn = "+" if delta >= 0 else ""
        print(f"[fusion] {key:<20}  α={f['best_alpha']:.2f}  "
              f"overall EER={f['overall_eer']:.4f} ({sgn}{delta:.4f})  "
              f"10-fold EER={f['fold_mean_eer']:.4f}±{f['fold_std_eer']:.4f}")

    out = {"dataset": "YTF", "individual": individual, "fusions": fusions}
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
