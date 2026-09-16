#!/usr/bin/env python3
"""
YTF (YouTube Faces DB) verification evaluation.

YTF standard protocol:
  - meta_and_splits.mat contains 5,000 video pairs across 10 splits × 500 pairs.
  - Splits shape: (500, 3, 10) = (pair_idx, [idx1, idx2, is_same], split_id).
  - Indices are 1-based into video_names (shape: (3425, 1), entries = "Person/video_id").
  - 10-fold cross-validation: report mean EER/AUC across folds.

Protocol here (feature-based, no discriminative training):
  1. Load pre-extracted per-video features (T, D) → mean-pool to (D,)
  2. For each pair: cosine_similarity(feat1, feat2)
  3. Per-fold EER/AUC on the 500 pairs in that fold
  4. Also aggregate EER/AUC on all 5,000 pairs
  5. Score-level fusion (ArcFace + ViViT) — same as AFEW-VA / CREMA-D

Usage:
  python eval_ytf_verification.py \
      --features_dir features/YTF \
      --meta_file data/YTF/meta_data/meta_and_splits.mat \
      --output_dir ytf_verification_results
"""

import os
import json
import argparse
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import cosine as cosine_dist


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_features(features_dir):
    """Load features/YTF/{Backbone}/{Person}/{video_id}.pt → {'Person/video_id': vec}."""
    feats = {}
    for person in sorted(os.listdir(features_dir)):
        person_dir = os.path.join(features_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            if not fname.endswith('.pt'):
                continue
            vid = os.path.splitext(fname)[0]
            data = torch.load(os.path.join(person_dir, fname),
                              map_location='cpu', weights_only=False)
            if isinstance(data, torch.Tensor):
                vec = data.float().mean(dim=0).numpy()
                feats[f"{person}/{vid}"] = vec
    return feats


def parse_splits(meta_file):
    """Parse YTF meta_and_splits.mat into pair list.

    Splits shape: (n_pairs_per_split=500, 3, n_splits=10) with [idx1, idx2, is_same].
    Indices are 1-based into video_names (entries "Person/video_id").

    Returns list of (split_id:int, video1:str, video2:str, is_same:int).
    """
    m = loadmat(meta_file)
    video_names = m['video_names']          # (N, 1) object array
    splits = m['Splits']                    # (500, 3, 10) uint16

    def _vn(idx1based):
        v = video_names[idx1based - 1, 0]
        return str(v[0]) if hasattr(v, '__len__') else str(v)

    pairs = []
    n_pairs_per_split, _, n_splits = splits.shape
    for s in range(n_splits):
        for i in range(n_pairs_per_split):
            idx1 = int(splits[i, 0, s])
            idx2 = int(splits[i, 1, s])
            is_same = int(splits[i, 2, s])
            pairs.append((s + 1, _vn(idx1), _vn(idx2), is_same))
    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_eer_auc(scores, labels):
    """EER + AUC from raw similarity scores + binary labels (1=genuine, 0=impostor)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    if len(genuine) == 0 or len(impostor) == 0:
        return float('nan'), float('nan')

    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    far = np.array([np.mean(impostor >= t) for t in thresholds])
    frr = np.array([np.mean(genuine < t) for t in thresholds])

    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        eer = 0.5
    else:
        idx = sign_changes[0]
        denom = diff[idx + 1] - diff[idx]
        alpha = -diff[idx] / denom if denom != 0 else 0.5
        eer = far[idx] + alpha * (far[idx + 1] - far[idx])

    tpr = 1 - frr
    fpr = far
    order = np.argsort(fpr)
    auc = float(np.trapz(tpr[order], fpr[order]))
    return float(eer), auc


# ---------------------------------------------------------------------------
# Per-pair scoring
# ---------------------------------------------------------------------------
def score_pairs(feats, pairs):
    """Return np.arrays of (split_ids, scores, labels). Pairs missing features → dropped."""
    split_ids, scores, labels = [], [], []
    missing = 0
    for sid, v1, v2, is_same in pairs:
        if v1 not in feats or v2 not in feats:
            missing += 1
            continue
        sim = 1.0 - cosine_dist(feats[v1], feats[v2])
        split_ids.append(sid)
        scores.append(float(sim))
        labels.append(int(is_same))
    if missing:
        print(f"  [warn] {missing} pairs dropped (missing features)")
    return np.array(split_ids), np.array(scores), np.array(labels)


def per_fold_metrics(split_ids, scores, labels, n_splits=10):
    """Per-fold EER + AUC, plus overall."""
    out = {'folds': {}}
    fold_eers, fold_aucs = [], []
    for sid in range(1, n_splits + 1):
        mask = split_ids == sid
        if mask.sum() == 0:
            continue
        eer, auc = compute_eer_auc(scores[mask], labels[mask])
        out['folds'][sid] = {'n': int(mask.sum()), 'eer': eer, 'auc': auc}
        fold_eers.append(eer)
        fold_aucs.append(auc)

    if fold_eers:
        out['mean_eer'] = float(np.mean(fold_eers))
        out['std_eer'] = float(np.std(fold_eers))
        out['mean_auc'] = float(np.mean(fold_aucs))
        out['std_auc'] = float(np.std(fold_aucs))

    overall_eer, overall_auc = compute_eer_auc(scores, labels)
    out['overall_eer'] = overall_eer
    out['overall_auc'] = overall_auc
    out['n_pairs'] = int(len(scores))
    return out


# ---------------------------------------------------------------------------
# Score-level fusion
# ---------------------------------------------------------------------------
def z_fuse(scores):
    mu, std = scores.mean(), scores.std()
    return (scores - mu) / (std if std > 0 else 1.0)


def run_score_fusion(scores_a, scores_b, labels, split_ids,
                      alpha_range=None, n_splits=10):
    """Weighted sum on z-normalized scores; sweep α to minimize overall EER."""
    if alpha_range is None:
        alpha_range = np.arange(0.0, 1.05, 0.05)

    za = z_fuse(scores_a)
    zb = z_fuse(scores_b)

    best_eer, best_alpha, best_auc = 1.0, 0.0, 0.0
    for alpha in alpha_range:
        fused = alpha * za + (1 - alpha) * zb
        eer, auc = compute_eer_auc(fused, labels)
        if eer < best_eer:
            best_eer, best_alpha, best_auc = eer, float(alpha), auc

    # also compute fold metrics at best alpha
    fused_best = best_alpha * za + (1 - best_alpha) * zb
    fold_metrics = per_fold_metrics(split_ids, fused_best, labels, n_splits)
    return {
        'best_alpha': best_alpha,
        'overall_eer': best_eer,
        'overall_auc': best_auc,
        'fold_mean_eer': fold_metrics.get('mean_eer'),
        'fold_std_eer': fold_metrics.get('std_eer'),
        'fold_mean_auc': fold_metrics.get('mean_auc'),
        'fold_std_auc': fold_metrics.get('std_auc'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="YTF verification evaluation")
    p.add_argument("--features_dir", type=str, default="features/YTF")
    p.add_argument("--meta_file", type=str, default="data/YTF/meta_data/meta_and_splits.mat")
    p.add_argument("--output_dir", type=str, default="ytf_verification_results")
    p.add_argument("--backbones", nargs='*', default=None,
                   help="Subset of backbones to evaluate (default: all present)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover backbones
    all_backbones = sorted([
        d for d in os.listdir(args.features_dir)
        if os.path.isdir(os.path.join(args.features_dir, d))
    ]) if os.path.isdir(args.features_dir) else []
    backbones = args.backbones if args.backbones else all_backbones
    print(f"Backbones: {backbones}")

    # Parse splits
    pairs = parse_splits(args.meta_file)
    print(f"Pairs parsed: {len(pairs)}")
    n_same = sum(1 for _, _, _, y in pairs if y == 1)
    print(f"  genuine: {n_same}, impostor: {len(pairs) - n_same}")

    results = {}
    raw_scores = {}  # for fusion

    for bb in backbones:
        feat_dir = os.path.join(args.features_dir, bb)
        if not os.path.isdir(feat_dir):
            print(f"[skip] {bb} — no features")
            continue
        print(f"\n=== {bb} ===")
        feats = load_features(feat_dir)
        if not feats:
            print(f"  (empty)")
            continue
        dim = list(feats.values())[0].shape[0]
        print(f"  loaded {len(feats)} video features, dim={dim}")

        split_ids, scores, labels = score_pairs(feats, pairs)
        if len(scores) == 0:
            print(f"  no pairs with features")
            continue

        metrics = per_fold_metrics(split_ids, scores, labels)
        results[bb] = metrics
        raw_scores[bb] = (split_ids, scores, labels)

        print(f"  overall  EER={metrics['overall_eer']:.4f}  AUC={metrics['overall_auc']:.4f}  n={metrics['n_pairs']}")
        if 'mean_eer' in metrics:
            print(f"  10-fold  EER={metrics['mean_eer']:.4f}±{metrics['std_eer']:.4f}  "
                  f"AUC={metrics['mean_auc']:.4f}±{metrics['std_auc']:.4f}")

    # Score-level fusion (ArcFace + ViViT)
    if 'ArcFace' in raw_scores and 'ViViT' in raw_scores:
        print(f"\n=== Score-level fusion (ArcFace + ViViT) ===")
        sid_a, sc_a, lab_a = raw_scores['ArcFace']
        sid_v, sc_v, lab_v = raw_scores['ViViT']
        # Align on pair index — they must match since pairs list is the same
        assert np.array_equal(sid_a, sid_v) and np.array_equal(lab_a, lab_v), \
            "Pair ordering mismatch between ArcFace and ViViT"
        fusion = run_score_fusion(sc_a, sc_v, lab_a, sid_a)
        results['ArcFace+ViViT_fusion'] = fusion
        print(f"  best α={fusion['best_alpha']:.2f}")
        print(f"  overall EER={fusion['overall_eer']:.4f}  AUC={fusion['overall_auc']:.4f}")
        if fusion.get('fold_mean_eer') is not None:
            print(f"  10-fold EER={fusion['fold_mean_eer']:.4f}±{fusion['fold_std_eer']:.4f}")

    # Save
    out_path = os.path.join(args.output_dir, "verification_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"YTF Verification Results (3,425 videos / 1,595 persons / 5,000 pairs)")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'10-fold EER':>14} {'10-fold AUC':>14}")
    print(f"{'-'*70}")
    for name, r in results.items():
        if 'mean_eer' in r:
            print(f"{name:<30} {r['mean_eer']:>7.4f}±{r['std_eer']:.4f} "
                  f"{r['mean_auc']:>7.4f}±{r['std_auc']:.4f}")
        elif 'fold_mean_eer' in r and r['fold_mean_eer'] is not None:
            print(f"{name:<30} {r['fold_mean_eer']:>7.4f}±{r['fold_std_eer']:.4f} "
                  f"{r['fold_mean_auc']:>7.4f}±{r['fold_std_auc']:.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
