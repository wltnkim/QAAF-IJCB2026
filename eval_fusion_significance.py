#!/usr/bin/env python3
"""
B3: Statistical significance for Hard×Soft fusion.

YTF: per-fold (10 folds) EER paired comparison fusion vs Hard alone, Wilcoxon signed-rank.
AFEW-VA: bootstrap resampling of probe pairs (1000 reps), 95% CI on ΔEER.

Output: results/fusion_significance.json + table summaries.
"""
import os
import json
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import wilcoxon
from paths import AFEWVA_SPLIT


# ----------------- shared utils -----------------
def compute_eer(scores, labels):
    if len(np.unique(labels)) < 2:
        return float('nan')
    thr = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.mean(scores[labels == 0] >= t) for t in thr])
    frr = np.array([np.mean(scores[labels == 1] < t) for t in thr])
    diff = far - frr
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return 0.5
    i = sc[0]
    den = diff[i+1] - diff[i]
    a = -diff[i]/den if den != 0 else 0.5
    return float(far[i] + a * (far[i+1] - far[i]))


def z(x):
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd if sd > 0 else 1.0)


# ----------------- YTF -----------------
YTF_FEATURES = "features/YTF"
YTF_META = "data/YTF/meta_data/meta_and_splits.mat"

def ytf_load(rel):
    out = {}
    d = os.path.join(YTF_FEATURES, rel)
    for person in sorted(os.listdir(d)):
        pd = os.path.join(d, person)
        if not os.path.isdir(pd):
            continue
        for fn in sorted(os.listdir(pd)):
            if not fn.endswith(".pt"):
                continue
            t = torch.load(os.path.join(pd, fn), map_location='cpu', weights_only=False)
            if isinstance(t, torch.Tensor):
                out[f"{person}/{os.path.splitext(fn)[0]}"] = t.float().mean(dim=0).numpy()
    return out


def ytf_pairs():
    m = loadmat(YTF_META)
    vn = m["video_names"]; sp = m["Splits"]
    def name(i):
        v = vn[i-1, 0]
        return str(v[0]) if hasattr(v, "__len__") else str(v)
    pairs = []
    npp, _, ns = sp.shape
    for s in range(ns):
        for i in range(npp):
            pairs.append((s+1, name(int(sp[i,0,s])), name(int(sp[i,1,s])), int(sp[i,2,s])))
    return pairs


def ytf_score_pairs(feats, pairs):
    sid, sc, lab = [], [], []
    for s, v1, v2, y in pairs:
        if v1 in feats and v2 in feats:
            sid.append(s); sc.append(1.0 - cosine_dist(feats[v1], feats[v2])); lab.append(y)
    return np.array(sid), np.array(sc), np.array(lab)


def ytf_per_fold_eer(sid, sc, lab):
    out = []
    for s in range(1, 11):
        m = sid == s
        if m.sum() == 0:
            out.append(float('nan')); continue
        out.append(compute_eer(sc[m], lab[m]))
    return np.array(out)


# ----------------- AFEW-VA -----------------
AFEWVA_SPLIT = AFEWVA_SPLIT
AFEWVA_FEATURES_ROOT = "features"

def afewva_load(rel):
    d = os.path.join(AFEWVA_FEATURES_ROOT, rel)
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".pt"):
            cid = os.path.splitext(fn)[0]
            t = torch.load(os.path.join(d, fn), map_location='cpu', weights_only=False)
            if isinstance(t, torch.Tensor):
                out[cid] = t.float().mean(dim=0).numpy()
    return out


def afewva_score_pairs(feats, split_info):
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
        for cid, pv in sorted(probe[a]):
            keys.append((cid, a, a)); scores.append(1.0 - cosine_dist(e, pv)); labels.append(1)
            for o in actors:
                if o == a:
                    continue
                keys.append((cid, a, o)); scores.append(1.0 - cosine_dist(enroll[o], pv)); labels.append(0)
    return keys, np.array(scores), np.array(labels)


# ----------------- Main per-dataset analysis -----------------
HARD = ["ArcFace", "AdaFace", "EdgeFace", "SFace", "SynthDistill"]
YTF_DIRS = {"ArcFace": "ArcFace", "AdaFace": "AdaFace_IR101", "EdgeFace": "EdgeFace",
            "SFace": "SFace", "SynthDistill": "SynthDistill"}
AFEWVA_HARD = {"ArcFace": "AFEWVA_baselines/ArcFace", "AdaFace": "AFEWVA_baselines/AdaFace_IR101",
               "EdgeFace": "AFEWVA_baselines/EdgeFace", "SFace": "AFEWVA_baselines/SFace",
               "SynthDistill": "AFEWVA_baselines/SynthDistill"}
SOFT = "VA-ViViT"
YTF_SOFT = "ViViT"
AFEWVA_SOFT = "AFEWVA/ViViT"


def analyze_ytf():
    """Per-fold EER (10 folds) for fusion vs Hard alone, Wilcoxon signed-rank."""
    print("=== YTF (10-fold Wilcoxon) ===")
    pairs = ytf_pairs()
    feats_soft = ytf_load(YTF_SOFT)
    sid_s, sc_soft, lab = ytf_score_pairs(feats_soft, pairs)

    out = {}
    for h_name in HARD:
        feats_h = ytf_load(YTF_DIRS[h_name])
        sid_h, sc_hard, _ = ytf_score_pairs(feats_h, pairs)
        assert np.array_equal(sid_h, sid_s)

        # find best alpha by overall EER
        zh, zs = z(sc_hard), z(sc_soft)
        best_alpha, best_eer = 1.0, compute_eer(sc_hard, lab)
        for alpha in np.arange(0.0, 1.001, 0.05):
            fused = alpha * zh + (1 - alpha) * zs
            e = compute_eer(fused, lab)
            if e < best_eer:
                best_alpha, best_eer = float(alpha), e
        fused_best = best_alpha * zh + (1 - best_alpha) * zs

        eer_hard_folds = ytf_per_fold_eer(sid_h, sc_hard, lab)
        eer_fused_folds = ytf_per_fold_eer(sid_h, fused_best, lab)
        diffs = eer_fused_folds - eer_hard_folds  # negative = improvement

        valid = ~(np.isnan(diffs))
        if valid.sum() >= 2 and not np.allclose(diffs[valid], 0):
            stat, p = wilcoxon(diffs[valid], alternative='less')
        else:
            stat, p = float('nan'), float('nan')

        out[h_name] = {
            'best_alpha': best_alpha,
            'eer_hard_overall': float(compute_eer(sc_hard, lab)),
            'eer_fused_overall': float(best_eer),
            'eer_hard_folds': eer_hard_folds.tolist(),
            'eer_fused_folds': eer_fused_folds.tolist(),
            'mean_diff': float(diffs[valid].mean()),
            'std_diff': float(diffs[valid].std()),
            'wilcoxon_stat': float(stat) if not np.isnan(stat) else None,
            'wilcoxon_p_one_sided_less': float(p) if not np.isnan(p) else None,
        }
        sig = "✓ p<0.05" if (not np.isnan(p) and p < 0.05) else (f"p={p:.3f}" if not np.isnan(p) else "n/a")
        print(f"  {h_name:<14}  α={best_alpha:.2f}  EER {out[h_name]['eer_hard_overall']:.4f}→{best_eer:.4f}  "
              f"per-fold mean Δ={out[h_name]['mean_diff']:+.4f} ± {out[h_name]['std_diff']:.4f}  "
              f"Wilcoxon less: {sig}")
    return out


def analyze_afewva(n_boot=1000, seed=0):
    """Bootstrap CI on ΔEER for fusion vs Hard alone (resample probe clips)."""
    print(f"\n=== AFEW-VA ({n_boot} bootstrap reps) ===")
    with open(AFEWVA_SPLIT) as f:
        split = json.load(f)
    feats_soft = afewva_load(AFEWVA_SOFT)
    keys_s, sc_soft, lab = afewva_score_pairs(feats_soft, split)
    rng = np.random.default_rng(seed)

    # Group score-rows by probe clip_id (keys are (cid, actor, other_actor))
    cids = sorted(set(k[0] for k in keys_s))
    cid_to_idx = {cid: [] for cid in cids}
    for i, (cid, *_rest) in enumerate(keys_s):
        cid_to_idx[cid].append(i)

    out = {}
    for h_name in HARD:
        feats_h = afewva_load(AFEWVA_HARD[h_name])
        keys_h, sc_hard, _ = afewva_score_pairs(feats_h, split)
        assert keys_h == keys_s

        zh, zs = z(sc_hard), z(sc_soft)
        best_alpha, best_eer = 1.0, compute_eer(sc_hard, lab)
        for alpha in np.arange(0.0, 1.001, 0.05):
            fused = alpha * zh + (1 - alpha) * zs
            e = compute_eer(fused, lab)
            if e < best_eer:
                best_alpha, best_eer = float(alpha), e
        fused_best = best_alpha * zh + (1 - best_alpha) * zs

        # Bootstrap: resample probe clips with replacement → recompute EER for both
        deltas = []
        for _ in range(n_boot):
            sampled_cids = rng.choice(cids, size=len(cids), replace=True)
            idxs = []
            for c in sampled_cids:
                idxs.extend(cid_to_idx[c])
            idxs = np.array(idxs)
            e_h = compute_eer(sc_hard[idxs], lab[idxs])
            e_f = compute_eer(fused_best[idxs], lab[idxs])
            if not (np.isnan(e_h) or np.isnan(e_f)):
                deltas.append(e_f - e_h)
        deltas = np.array(deltas)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        sig = "✓ CI doesn't cross 0" if ci_hi < 0 else "✗ CI crosses 0"

        out[h_name] = {
            'best_alpha': best_alpha,
            'eer_hard_overall': float(compute_eer(sc_hard, lab)),
            'eer_fused_overall': float(best_eer),
            'overall_delta': float(best_eer - compute_eer(sc_hard, lab)),
            'bootstrap_mean_delta': float(deltas.mean()),
            'bootstrap_ci_95': [float(ci_lo), float(ci_hi)],
            'bootstrap_n': len(deltas),
        }
        print(f"  {h_name:<14}  α={best_alpha:.2f}  EER {out[h_name]['eer_hard_overall']:.4f}→{best_eer:.4f}  "
              f"Δ={out[h_name]['overall_delta']:+.4f}  "
              f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")
    return out


def main():
    results = {'soft_used': SOFT}
    results['ytf'] = analyze_ytf()
    results['afewva'] = analyze_afewva()
    out = "results/fusion_significance.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {out}")


if __name__ == "__main__":
    main()
