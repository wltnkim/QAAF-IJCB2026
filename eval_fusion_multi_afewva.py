#!/usr/bin/env python3
"""
AFEW-VA multi-pair score-level fusion.

Runs z-score weighted-sum fusion (α sweep ∈ [0,1], step 0.05) for multiple
(hard, soft) pairs — same method as the existing ArcFace+VA-ViViT result
reported in the paper. Z-score sweep minimizes overall EER.

Pairs evaluated here (new for YTF extension):
  A) ArcFace    + R2D1           (best-hard × 2nd-best-soft, JMT self-consistency)
  B) AdaFace    + VA-ViViT       (2nd-best-hard × best-soft)
  C) EdgeFace   + VA-ViViT       (3rd-best-hard × best-soft; IJCB'23 + practical)

Also re-runs ArcFace + VA-ViViT to reproduce the baseline under the same script.
"""

import os
import json
import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_dist
from paths import AFEWVA_SPLIT


# ---------------------------------------------------------------------------
FEAT_PATHS = {
    "ArcFace":   "features/AFEWVA_baselines/ArcFace",
    "AdaFace":   "features/AFEWVA_baselines/AdaFace_IR101",
    "EdgeFace":  "features/AFEWVA_baselines/EdgeFace",
    "VA-ViViT":  "features/AFEWVA/ViViT",
    "R2D1":      "features/AFEWVA/R2D1",
}

PAIRS = [
    ("ArcFace",  "VA-ViViT"),  # baseline reproduction
    ("ArcFace",  "R2D1"),      # A
    ("AdaFace",  "VA-ViViT"),  # B
    ("EdgeFace", "VA-ViViT"),  # C
]

SPLIT_FILE = AFEWVA_SPLIT
OUT_PATH   = "afewva_fusion_multi_results.json"


# ---------------------------------------------------------------------------
def load_features(features_dir):
    feats = {}
    for fname in sorted(os.listdir(features_dir)):
        if not fname.endswith(".pt"):
            continue
        clip_id = os.path.splitext(fname)[0]
        data = torch.load(os.path.join(features_dir, fname),
                          map_location="cpu", weights_only=False)
        if isinstance(data, torch.Tensor):
            feats[clip_id] = data.float().mean(dim=0).numpy()
    return feats


def collect_pair_scores(feats, split_info):
    """Compute per-pair cosine sim scores. Returns genuine/impostor arrays plus
    aligned pair metadata so scores from different models can be fused."""
    gen_scores, imp_scores = [], []
    gen_pairs, imp_pairs = [], []

    enroll = {}
    probe  = {}
    for actor, info in split_info["actors"].items():
        train_clips = [str(c).zfill(3) for c in info["train_clips"]]
        test_clips  = [str(c).zfill(3) for c in info["test_clips"]]
        train_vecs  = [feats[c] for c in train_clips if c in feats]
        if not train_vecs:
            continue
        enroll[actor] = np.mean(train_vecs, axis=0)
        probe_list = [(c, feats[c]) for c in test_clips if c in feats]
        if probe_list:
            probe[actor] = probe_list

    actors = sorted(enroll.keys())
    for actor in actors:
        if actor not in probe:
            continue
        e = enroll[actor]
        for clip_id, pv in probe[actor]:
            gen_scores.append(1.0 - cosine_dist(e, pv))
            gen_pairs.append((actor, clip_id, actor))
            for other in actors:
                if other == actor:
                    continue
                imp_scores.append(1.0 - cosine_dist(enroll[other], pv))
                imp_pairs.append((actor, clip_id, other))

    return (np.array(gen_scores), np.array(imp_scores),
            gen_pairs, imp_pairs)


def compute_eer_auc(gen, imp):
    all_s = np.concatenate([gen, imp])
    thr = np.linspace(all_s.min(), all_s.max(), 10000)
    far = np.array([np.mean(imp >= t) for t in thr])
    frr = np.array([np.mean(gen < t)  for t in thr])
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


def zscore(arr, mu=None, std=None):
    mu  = arr.mean() if mu  is None else mu
    std = arr.std()  if std is None else std
    return (arr - mu) / (std if std > 0 else 1.0)


def run_zscore_fusion(gen_a, imp_a, gen_b, imp_b, alphas=None):
    if alphas is None:
        alphas = np.arange(0.0, 1.001, 0.05)
    all_a = np.concatenate([gen_a, imp_a])
    all_b = np.concatenate([gen_b, imp_b])
    mu_a, std_a = all_a.mean(), all_a.std()
    mu_b, std_b = all_b.mean(), all_b.std()
    ga = zscore(gen_a, mu_a, std_a); ia = zscore(imp_a, mu_a, std_a)
    gb = zscore(gen_b, mu_b, std_b); ib = zscore(imp_b, mu_b, std_b)

    results = []
    for a in alphas:
        gf = a * ga + (1 - a) * gb
        jf = a * ia + (1 - a) * ib
        eer, auc = compute_eer_auc(gf, jf)
        results.append({"alpha": float(a), "eer": float(eer), "auc": float(auc)})
    best = min(results, key=lambda r: r["eer"])
    return {"sweep": results, "best": best}


# ---------------------------------------------------------------------------
def main():
    with open(SPLIT_FILE) as f:
        split_info = json.load(f)

    # Cache: {name: (gen, imp, pairs_gen, pairs_imp)}
    cache = {}
    needed = set()
    for a, b in PAIRS:
        needed.add(a); needed.add(b)

    for name in needed:
        path = FEAT_PATHS[name]
        print(f"Loading {name} from {path} ...", flush=True)
        feats = load_features(path)
        print(f"  → {len(feats)} clips, dim={list(feats.values())[0].shape[0]}")
        cache[name] = collect_pair_scores(feats, split_info)

    # Sanity-check: all models should give aligned pair ordering
    ref_gen_pairs = cache[list(needed)[0]][2]
    ref_imp_pairs = cache[list(needed)[0]][3]
    for name, (_, _, gp, ip) in cache.items():
        assert gp == ref_gen_pairs, f"{name} genuine pair order mismatch"
        assert ip == ref_imp_pairs, f"{name} impostor pair order mismatch"
    print(f"\nPair alignment OK: {len(ref_gen_pairs)} genuine + {len(ref_imp_pairs)} impostor\n")

    # Individual
    individual = {}
    for name, (gen, imp, _, _) in cache.items():
        eer, auc = compute_eer_auc(gen, imp)
        individual[name] = {"eer": float(eer), "auc": float(auc)}
        print(f"[individual] {name:<12}  EER={eer:.4f}  AUC={auc:.4f}")

    # Fusions
    fusions = {}
    print()
    for hard, soft in PAIRS:
        key = f"{hard}+{soft}"
        gen_a, imp_a, _, _ = cache[hard]
        gen_b, imp_b, _, _ = cache[soft]
        res = run_zscore_fusion(gen_a, imp_a, gen_b, imp_b)
        fusions[key] = {
            "hard": hard, "soft": soft,
            "individual_hard": individual[hard],
            "individual_soft": individual[soft],
            "best_alpha": res["best"]["alpha"],
            "best_eer":   res["best"]["eer"],
            "best_auc":   res["best"]["auc"],
            "delta_vs_hard": res["best"]["eer"] - individual[hard]["eer"],
            "sweep": res["sweep"],
        }
        delta = res["best"]["eer"] - individual[hard]["eer"]
        sgn = "+" if delta >= 0 else ""
        print(f"[fusion]    {key:<24}  α={res['best']['alpha']:.2f}  "
              f"EER={res['best']['eer']:.4f}  AUC={res['best']['auc']:.4f}  "
              f"Δ_hard={sgn}{delta:.4f}")

    out = {"dataset": "AFEW-VA", "individual": individual, "fusions": fusions}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
