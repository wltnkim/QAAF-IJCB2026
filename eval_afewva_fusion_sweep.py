#!/usr/bin/env python3
"""
Phase 1-3 (b): AFEW-VA 5×11 Hard×Soft fusion sweep — twin to YTF version.

Same Hard 5 + Soft 11 baselines as YTF; uses AFEW-VA enrollment-probe protocol
(per-actor train→enrollment / test→probe, 67 actors).

Output: results/afewva_fusion_sweep_HxS.{json,csv,heatmap}.
"""
import os
import json
import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_dist
from paths import AFEWVA_SPLIT


SPLIT_FILE = AFEWVA_SPLIT
OUT_JSON   = "results/afewva_fusion_sweep_HxS.json"
OUT_CSV    = "results/afewva_fusion_sweep_HxS.csv"

# AFEW-VA features layout:
#   - VA-trained ViViT/VideoMAE: features/AFEWVA/{ViViT,VideoMAE}/
#   - All other baselines:       features/AFEWVA_baselines/<name>/
HARD_BACKBONES = {
    "ArcFace":      "AFEWVA_baselines/ArcFace",
    "AdaFace":      "AFEWVA_baselines/AdaFace_IR101",
    "EdgeFace":     "AFEWVA_baselines/EdgeFace",
    "SFace":        "AFEWVA_baselines/SFace",
    "SynthDistill": "AFEWVA_baselines/SynthDistill",
}
SOFT_BACKBONES = {
    "VA-ViViT":         "AFEWVA/ViViT",                        # ours, VA-trained
    "VA-VideoMAE":      "AFEWVA/VideoMAE",                     # ours, VA-trained
    "JMT-R2D1":         "AFEWVA/R2D1",
    "JMT-I3D":          "AFEWVA/I3D",
    "ViViT-pretr":      "AFEWVA_baselines/ViViT_pretrained",
    "VideoMAE-pretr":   "AFEWVA_baselines/VideoMAE_pretrained",
    "EmotiEffLib":      "AFEWVA_baselines/EmotiEffLib",
    "FER":              "AFEWVA_baselines/FER",
    "DAN":              "AFEWVA_baselines/DAN",
    "POSTER++":         "AFEWVA_baselines/POSTERV2",
    "MAE-DFER":         "AFEWVA_baselines/MAE_DFER",
}
FEATURES_ROOT = "features"


def load_clip_dir(rel_path):
    """{clip_id: mean-pooled feature (np.ndarray)}"""
    d = os.path.join(FEATURES_ROOT, rel_path)
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pt"):
            continue
        cid = os.path.splitext(fn)[0]
        t = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
        if isinstance(t, torch.Tensor):
            out[cid] = t.float().mean(dim=0).numpy()
    return out


def afewva_score_pairs(feats, split_info):
    """Enrollment-probe → list of (probe_clip_id, actor, score, label).

    For fusion to work across backbones, we need IDENTICAL ordering. We
    deterministically iterate sorted actors → sorted probes → sorted other actors.
    """
    enroll, probe = {}, {}
    for actor, info in split_info["actors"].items():
        tr = [str(c).zfill(3) for c in info["train_clips"]]
        te = [str(c).zfill(3) for c in info["test_clips"]]
        tv = [feats[c] for c in tr if c in feats]
        if not tv:
            continue
        enroll[actor] = np.mean(tv, axis=0)
        pl = [(c, feats[c]) for c in te if c in feats]
        if pl:
            probe[actor] = pl

    actors = sorted(enroll.keys())
    keys, scores, labels = [], [], []
    for a in actors:
        if a not in probe:
            continue
        e = enroll[a]
        for cid, pv in sorted(probe[a]):  # deterministic
            keys.append((cid, a, a)); scores.append(1.0 - cosine_dist(e, pv)); labels.append(1)
            for o in actors:
                if o == a:
                    continue
                keys.append((cid, a, o)); scores.append(1.0 - cosine_dist(enroll[o], pv)); labels.append(0)
    return keys, np.array(scores), np.array(labels)


def compute_eer_auc(scores, labels):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    thr = np.linspace(scores.min(), scores.max(), 2000)
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


def z(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd if sd > 0 else 1.0)


def fusion_sweep(sc_h, sc_s, labels):
    zh, zs = z(sc_h), z(sc_s)
    best = {"alpha": 0.0, "overall_eer": 1.0}
    for alpha in np.arange(0.0, 1.001, 0.05):
        fused = alpha * zh + (1 - alpha) * zs
        e, _ = compute_eer_auc(fused, labels)
        if not np.isnan(e) and e < best["overall_eer"]:
            best = {"alpha": float(alpha), "overall_eer": float(e)}
    return best


def main():
    with open(SPLIT_FILE) as f:
        split_info = json.load(f)
    print(f"AFEW-VA: {len(split_info['actors'])} actors")

    cache = {}        # name → (keys, scores, labels)
    individual = {}
    for name, sub in {**HARD_BACKBONES, **SOFT_BACKBONES}.items():
        path = os.path.join(FEATURES_ROOT, sub)
        if not os.path.isdir(path):
            print(f"[SKIP] {path} not found")
            continue
        feats = load_clip_dir(sub)
        keys, sc, lab = afewva_score_pairs(feats, split_info)
        e, _ = compute_eer_auc(sc, lab)
        cache[name] = (keys, sc, lab)
        individual[name] = {"overall_eer": float(e), "n_pairs": int(len(sc)),
                             "feat_dim": int(next(iter(feats.values())).shape[0])}
        print(f"  {name:<18} ({sub:<40})  EER={e:.4f}  n_pairs={len(sc)}  dim={individual[name]['feat_dim']}")

    # Alignment check
    ref = list(cache)[0]
    ref_keys, _, ref_lab = cache[ref]
    for name, (k, _, lab) in cache.items():
        assert k == ref_keys, f"{name} keys mismatch"
        assert np.array_equal(lab, ref_lab)
    print("\nAll backbones aligned. Computing 5×11 sweep ...\n")

    fusions = {}
    for h in HARD_BACKBONES:
        for s in SOFT_BACKBONES:
            if h not in cache or s not in cache:
                continue
            _, sc_h, lab = cache[h]
            _, sc_s, _   = cache[s]
            f = fusion_sweep(sc_h, sc_s, lab)
            f["hard_baseline_eer"] = individual[h]["overall_eer"]
            f["soft_baseline_eer"] = individual[s]["overall_eer"]
            f["delta_vs_hard"] = f["overall_eer"] - individual[h]["overall_eer"]
            fusions[f"{h}+{s}"] = f
            sgn = "+" if f["delta_vs_hard"] >= 0 else ""
            print(f"  {h:<14} + {s:<16} α={f['alpha']:.2f}  "
                  f"EER={f['overall_eer']:.4f} ({sgn}{f['delta_vs_hard']:.4f})")

    out = {
        "dataset": "AFEW-VA",
        "n_actors": len(split_info['actors']),
        "individual": individual,
        "fusions": fusions,
        "hard_backbones": list(HARD_BACKBONES.keys()),
        "soft_backbones": list(SOFT_BACKBONES.keys()),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n✓ Saved: {OUT_JSON}")

    with open(OUT_CSV, "w") as fh:
        fh.write("Hard," + ",".join(SOFT_BACKBONES.keys()) + ",HardOnly\n")
        for h in HARD_BACKBONES:
            row = [h]
            for s in SOFT_BACKBONES:
                key = f"{h}+{s}"
                row.append(f"{fusions[key]['overall_eer']:.4f}" if key in fusions else "—")
            row.append(f"{individual[h]['overall_eer']:.4f}" if h in individual else "—")
            fh.write(",".join(row) + "\n")
        row = ["SoftOnly"]
        for s in SOFT_BACKBONES:
            row.append(f"{individual[s]['overall_eer']:.4f}" if s in individual else "—")
        row.append("")
        fh.write(",".join(row) + "\n")
    print(f"✓ Saved: {OUT_CSV}")

    # Summary
    print(f"\n{'='*70}\nSummary — best Δ per Hard backbone:")
    for h in HARD_BACKBONES:
        if h not in individual:
            continue
        deltas = [(s, fusions[f"{h}+{s}"]["delta_vs_hard"], fusions[f"{h}+{s}"]["alpha"],
                   fusions[f"{h}+{s}"]["overall_eer"]) for s in SOFT_BACKBONES if f"{h}+{s}" in fusions]
        deltas.sort(key=lambda x: x[1])
        s, d, a, eer = deltas[0]
        sgn = "+" if d >= 0 else ""
        print(f"  {h:<14} best: +{s:<16} α={a:.2f}  "
              f"EER {individual[h]['overall_eer']:.4f}→{eer:.4f} ({sgn}{d:.4f})")


if __name__ == "__main__":
    main()
