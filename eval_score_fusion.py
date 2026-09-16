#!/usr/bin/env python3
"""
Score-level fusion: ArcFace + VA-trained ViViT verification.

Computes per-pair cosine similarity scores for both models,
then fuses scores with various strategies:
  1. Weighted sum: s = α * s_arcface + (1-α) * s_vivit  (α ∈ [0, 1])
  2. Min-max normalized weighted sum
  3. Z-score normalized weighted sum
  4. Product fusion: s = s_arcface * s_vivit
"""

import torch
import numpy as np
import os
import json
from scipy.spatial.distance import cosine as cosine_dist
from paths import AFEWVA_SPLIT


def load_features(features_dir):
    """Load all .pt features, mean-pool over time → {clip_id: vector}."""
    feats = {}
    for fname in sorted(os.listdir(features_dir)):
        if not fname.endswith('.pt'):
            continue
        clip_id = os.path.splitext(fname)[0]
        data = torch.load(os.path.join(features_dir, fname),
                          map_location='cpu', weights_only=False)
        if isinstance(data, torch.Tensor):
            vec = data.float().mean(dim=0).numpy()
        else:
            continue
        feats[clip_id] = vec
    return feats


def collect_pair_scores(feats, split_info):
    """Collect per-pair cosine similarity scores with metadata.

    Returns:
        genuine_scores: array of shape (N_genuine,)
        impostor_scores: array of shape (N_impostor,)
        genuine_pairs: list of (actor, clip_id, enroll_actor) tuples
        impostor_pairs: list of (actor, clip_id, enroll_actor) tuples
    """
    genuine_scores = []
    impostor_scores = []
    genuine_pairs = []
    impostor_pairs = []

    actor_enrollments = {}
    actor_probes = {}

    for actor, info in split_info['actors'].items():
        train_clips = [str(c).zfill(3) for c in info['train_clips']]
        test_clips = [str(c).zfill(3) for c in info['test_clips']]

        train_vecs = [feats[c] for c in train_clips if c in feats]
        if len(train_vecs) == 0:
            continue
        actor_enrollments[actor] = np.mean(train_vecs, axis=0)

        probe_list = [(c, feats[c]) for c in test_clips if c in feats]
        if len(probe_list) == 0:
            continue
        actor_probes[actor] = probe_list

    actors = sorted(actor_enrollments.keys())

    for actor in actors:
        if actor not in actor_probes:
            continue
        enroll = actor_enrollments[actor]

        for clip_id, probe_vec in actor_probes[actor]:
            # Genuine
            sim = 1.0 - cosine_dist(enroll, probe_vec)
            genuine_scores.append(sim)
            genuine_pairs.append((actor, clip_id, actor))

            # Impostor
            for other_actor in actors:
                if other_actor == actor:
                    continue
                other_enroll = actor_enrollments[other_actor]
                sim_imp = 1.0 - cosine_dist(other_enroll, probe_vec)
                impostor_scores.append(sim_imp)
                impostor_pairs.append((actor, clip_id, other_actor))

    return (np.array(genuine_scores), np.array(impostor_scores),
            genuine_pairs, impostor_pairs)


def compute_eer_auc(genuine_scores, impostor_scores):
    """Compute EER and AUC from genuine/impostor similarity scores."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 10000)

    far = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr = np.array([np.mean(genuine_scores < t) for t in thresholds])

    # EER
    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        eer = 0.5
    else:
        idx = sign_changes[0]
        denom = diff[idx + 1] - diff[idx]
        alpha_interp = -diff[idx] / denom if denom != 0 else 0.5
        eer = far[idx] + alpha_interp * (far[idx + 1] - far[idx])

    # AUC
    tpr = 1 - frr
    fpr = far
    sorted_idx = np.argsort(fpr)
    auc = float(np.trapz(tpr[sorted_idx], fpr[sorted_idx]))

    return float(eer), auc


def minmax_normalize(scores):
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def zscore_normalize(scores):
    """Z-score normalize scores."""
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return np.zeros_like(scores)
    return (scores - mu) / std


def fuse_scores(gen_a, imp_a, gen_b, imp_b, alpha):
    """Weighted sum fusion: s = alpha * a + (1-alpha) * b."""
    gen_fused = alpha * gen_a + (1 - alpha) * gen_b
    imp_fused = alpha * imp_a + (1 - alpha) * imp_b
    return gen_fused, imp_fused


def main():
    split_file = AFEWVA_SPLIT
    with open(split_file) as f:
        split_info = json.load(f)

    # Load features
    print("Loading ArcFace features...")
    arcface_feats = load_features('features/AFEWVA_baselines/ArcFace')
    print(f"  → {len(arcface_feats)} clips, dim={list(arcface_feats.values())[0].shape[0]}")

    print("Loading VA-trained ViViT features...")
    vivit_feats = load_features('features/AFEWVA/ViViT')
    print(f"  → {len(vivit_feats)} clips, dim={list(vivit_feats.values())[0].shape[0]}")

    # Collect per-pair scores
    print("\nComputing per-pair scores...")
    gen_arc, imp_arc, gen_pairs, imp_pairs = collect_pair_scores(arcface_feats, split_info)
    gen_viv, imp_viv, gen_pairs2, imp_pairs2 = collect_pair_scores(vivit_feats, split_info)

    # Verify pairs are aligned
    assert gen_pairs == gen_pairs2, "Genuine pairs not aligned!"
    assert imp_pairs == imp_pairs2, "Impostor pairs not aligned!"
    print(f"  Genuine pairs: {len(gen_arc)}, Impostor pairs: {len(imp_arc)}")

    # Individual results
    eer_arc, auc_arc = compute_eer_auc(gen_arc, imp_arc)
    eer_viv, auc_viv = compute_eer_auc(gen_viv, imp_viv)
    print(f"\n{'='*70}")
    print(f"Individual Results:")
    print(f"  ArcFace:       EER={eer_arc:.4f}  AUC={auc_arc:.4f}")
    print(f"  VA-ViViT:      EER={eer_viv:.4f}  AUC={auc_viv:.4f}")
    print(f"{'='*70}")

    results = {
        'individual': {
            'ArcFace': {'eer': eer_arc, 'auc': auc_arc},
            'VA_ViViT': {'eer': eer_viv, 'auc': auc_viv},
        },
        'fusion': {}
    }

    # ============================================================
    # 1. Raw score weighted sum: s = α * s_arc + (1-α) * s_viv
    # ============================================================
    print(f"\n{'='*70}")
    print("1. Raw Score Weighted Sum Fusion")
    print(f"{'='*70}")
    print(f"{'α':>6}  {'EER ↓':>8}  {'AUC ↑':>8}  {'vs ArcFace':>12}")

    alphas = np.arange(0, 1.001, 0.05)
    raw_results = []
    for alpha in alphas:
        gen_f, imp_f = fuse_scores(gen_arc, imp_arc, gen_viv, imp_viv, alpha)
        eer, auc = compute_eer_auc(gen_f, imp_f)
        delta = eer - eer_arc
        marker = " ← BEST" if False else ""  # placeholder
        raw_results.append({'alpha': float(alpha), 'eer': eer, 'auc': auc})

    # Find best
    best_raw = min(raw_results, key=lambda x: x['eer'])
    for r in raw_results:
        delta = r['eer'] - eer_arc
        marker = " ← BEST" if r['alpha'] == best_raw['alpha'] else ""
        sign = "+" if delta >= 0 else ""
        print(f"{r['alpha']:>6.2f}  {r['eer']:>8.4f}  {r['auc']:>8.4f}  {sign}{delta:>11.4f}{marker}")

    results['fusion']['raw_weighted_sum'] = {
        'all_alphas': raw_results,
        'best': best_raw,
    }

    # ============================================================
    # 2. Min-Max Normalized Fusion
    # ============================================================
    print(f"\n{'='*70}")
    print("2. Min-Max Normalized Weighted Sum Fusion")
    print(f"{'='*70}")

    # Normalize using all scores (genuine + impostor) per model
    all_arc = np.concatenate([gen_arc, imp_arc])
    all_viv = np.concatenate([gen_viv, imp_viv])

    arc_min, arc_max = all_arc.min(), all_arc.max()
    viv_min, viv_max = all_viv.min(), all_viv.max()

    gen_arc_mm = (gen_arc - arc_min) / (arc_max - arc_min)
    imp_arc_mm = (imp_arc - arc_min) / (arc_max - arc_min)
    gen_viv_mm = (gen_viv - viv_min) / (viv_max - viv_min)
    imp_viv_mm = (imp_viv - viv_min) / (viv_max - viv_min)

    # Individual after normalization (should match original EER)
    eer_arc_mm, auc_arc_mm = compute_eer_auc(gen_arc_mm, imp_arc_mm)
    eer_viv_mm, auc_viv_mm = compute_eer_auc(gen_viv_mm, imp_viv_mm)
    print(f"After min-max: ArcFace EER={eer_arc_mm:.4f}, ViViT EER={eer_viv_mm:.4f}")
    print(f"{'α':>6}  {'EER ↓':>8}  {'AUC ↑':>8}  {'vs ArcFace':>12}")

    mm_results = []
    for alpha in alphas:
        gen_f, imp_f = fuse_scores(gen_arc_mm, imp_arc_mm, gen_viv_mm, imp_viv_mm, alpha)
        eer, auc = compute_eer_auc(gen_f, imp_f)
        mm_results.append({'alpha': float(alpha), 'eer': eer, 'auc': auc})

    best_mm = min(mm_results, key=lambda x: x['eer'])
    for r in mm_results:
        delta = r['eer'] - eer_arc_mm
        marker = " ← BEST" if r['alpha'] == best_mm['alpha'] else ""
        sign = "+" if delta >= 0 else ""
        print(f"{r['alpha']:>6.2f}  {r['eer']:>8.4f}  {r['auc']:>8.4f}  {sign}{delta:>11.4f}{marker}")

    results['fusion']['minmax_weighted_sum'] = {
        'all_alphas': mm_results,
        'best': best_mm,
    }

    # ============================================================
    # 3. Z-Score Normalized Fusion
    # ============================================================
    print(f"\n{'='*70}")
    print("3. Z-Score Normalized Weighted Sum Fusion")
    print(f"{'='*70}")

    arc_mu, arc_std = all_arc.mean(), all_arc.std()
    viv_mu, viv_std = all_viv.mean(), all_viv.std()

    gen_arc_z = (gen_arc - arc_mu) / arc_std
    imp_arc_z = (imp_arc - arc_mu) / arc_std
    gen_viv_z = (gen_viv - viv_mu) / viv_std
    imp_viv_z = (imp_viv - viv_mu) / viv_std

    eer_arc_z, _ = compute_eer_auc(gen_arc_z, imp_arc_z)
    eer_viv_z, _ = compute_eer_auc(gen_viv_z, imp_viv_z)
    print(f"After z-score: ArcFace EER={eer_arc_z:.4f}, ViViT EER={eer_viv_z:.4f}")
    print(f"{'α':>6}  {'EER ↓':>8}  {'AUC ↑':>8}  {'vs ArcFace':>12}")

    zs_results = []
    for alpha in alphas:
        gen_f, imp_f = fuse_scores(gen_arc_z, imp_arc_z, gen_viv_z, imp_viv_z, alpha)
        eer, auc = compute_eer_auc(gen_f, imp_f)
        zs_results.append({'alpha': float(alpha), 'eer': eer, 'auc': auc})

    best_zs = min(zs_results, key=lambda x: x['eer'])
    for r in zs_results:
        delta = r['eer'] - eer_arc_z
        marker = " ← BEST" if r['alpha'] == best_zs['alpha'] else ""
        sign = "+" if delta >= 0 else ""
        print(f"{r['alpha']:>6.2f}  {r['eer']:>8.4f}  {r['auc']:>8.4f}  {sign}{delta:>11.4f}{marker}")

    results['fusion']['zscore_weighted_sum'] = {
        'all_alphas': zs_results,
        'best': best_zs,
    }

    # ============================================================
    # 4. Product Fusion: s = s_arc * s_viv
    # ============================================================
    print(f"\n{'='*70}")
    print("4. Product Fusion")
    print(f"{'='*70}")

    # Raw product
    gen_prod = gen_arc * gen_viv
    imp_prod = imp_arc * imp_viv
    eer_prod, auc_prod = compute_eer_auc(gen_prod, imp_prod)
    print(f"Raw product:       EER={eer_prod:.4f}  AUC={auc_prod:.4f}  (vs ArcFace: {eer_prod - eer_arc:+.4f})")

    # Min-max normalized product
    gen_prod_mm = gen_arc_mm * gen_viv_mm
    imp_prod_mm = imp_arc_mm * imp_viv_mm
    eer_prod_mm, auc_prod_mm = compute_eer_auc(gen_prod_mm, imp_prod_mm)
    print(f"Min-max product:   EER={eer_prod_mm:.4f}  AUC={auc_prod_mm:.4f}  (vs ArcFace: {eer_prod_mm - eer_arc:+.4f})")

    # Z-score shifted product (shift to positive before multiply)
    shift = 5  # shift z-scores to positive range
    gen_prod_z = (gen_arc_z + shift) * (gen_viv_z + shift)
    imp_prod_z = (imp_arc_z + shift) * (imp_viv_z + shift)
    eer_prod_z, auc_prod_z = compute_eer_auc(gen_prod_z, imp_prod_z)
    print(f"Z-score product:   EER={eer_prod_z:.4f}  AUC={auc_prod_z:.4f}  (vs ArcFace: {eer_prod_z - eer_arc:+.4f})")

    results['fusion']['product'] = {
        'raw': {'eer': eer_prod, 'auc': auc_prod},
        'minmax': {'eer': eer_prod_mm, 'auc': auc_prod_mm},
        'zscore_shifted': {'eer': eer_prod_z, 'auc': auc_prod_z},
    }

    # ============================================================
    # 5. Max-score fusion: s = max(s_arc_norm, s_viv_norm)
    # ============================================================
    print(f"\n{'='*70}")
    print("5. Max-Score Fusion (min-max normalized)")
    print(f"{'='*70}")
    gen_max = np.maximum(gen_arc_mm, gen_viv_mm)
    imp_max = np.maximum(imp_arc_mm, imp_viv_mm)
    eer_max, auc_max = compute_eer_auc(gen_max, imp_max)
    print(f"Max-score:         EER={eer_max:.4f}  AUC={auc_max:.4f}  (vs ArcFace: {eer_max - eer_arc:+.4f})")

    results['fusion']['max_score'] = {'eer': eer_max, 'auc': auc_max}

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'EER ↓':>8} {'AUC ↑':>8} {'vs ArcFace':>12}")
    print(f"{'-'*70}")
    print(f"{'ArcFace (alone)':35} {eer_arc:>8.4f} {auc_arc:>8.4f} {'baseline':>12}")
    print(f"{'VA-ViViT (alone)':35} {eer_viv:>8.4f} {auc_viv:>8.4f} {eer_viv - eer_arc:>+12.4f}")
    print(f"{'-'*70}")
    raw_label = f"Raw sum (a={best_raw['alpha']:.2f})"
    mm_label = f"Min-max sum (a={best_mm['alpha']:.2f})"
    zs_label = f"Z-score sum (a={best_zs['alpha']:.2f})"
    print(f"{raw_label:35} {best_raw['eer']:>8.4f} {best_raw['auc']:>8.4f} {best_raw['eer'] - eer_arc:>+12.4f}")
    print(f"{mm_label:35} {best_mm['eer']:>8.4f} {best_mm['auc']:>8.4f} {best_mm['eer'] - eer_arc:>+12.4f}")
    print(f"{zs_label:35} {best_zs['eer']:>8.4f} {best_zs['auc']:>8.4f} {best_zs['eer'] - eer_arc:>+12.4f}")
    print(f"{'Product (raw)':35} {eer_prod:>8.4f} {auc_prod:>8.4f} {eer_prod - eer_arc:>+12.4f}")
    print(f"{'Product (min-max)':35} {eer_prod_mm:>8.4f} {auc_prod_mm:>8.4f} {eer_prod_mm - eer_arc:>+12.4f}")
    print(f"{'Product (z-score)':35} {eer_prod_z:>8.4f} {auc_prod_z:>8.4f} {eer_prod_z - eer_arc:>+12.4f}")
    print(f"{'Max-score (min-max)':35} {eer_max:>8.4f} {auc_max:>8.4f} {eer_max - eer_arc:>+12.4f}")
    print(f"{'='*70}")

    best_overall = min([
        ('Raw sum', best_raw['alpha'], best_raw['eer'], best_raw['auc']),
        ('Min-max sum', best_mm['alpha'], best_mm['eer'], best_mm['auc']),
        ('Z-score sum', best_zs['alpha'], best_zs['eer'], best_zs['auc']),
        ('Product (raw)', None, eer_prod, auc_prod),
        ('Product (min-max)', None, eer_prod_mm, auc_prod_mm),
        ('Product (z-score)', None, eer_prod_z, auc_prod_z),
        ('Max-score', None, eer_max, auc_max),
    ], key=lambda x: x[2])

    print(f"\nBest overall: {best_overall[0]}", end="")
    if best_overall[1] is not None:
        print(f" (α={best_overall[1]:.2f})", end="")
    print(f"  EER={best_overall[2]:.4f}  AUC={best_overall[3]:.4f}")
    if best_overall[2] < eer_arc:
        print(f"  → IMPROVES over ArcFace by {eer_arc - best_overall[2]:.4f} EER")
    else:
        print(f"  → Does NOT improve over ArcFace ({best_overall[2] - eer_arc:+.4f} EER)")

    # Save results
    output_path = 'score_fusion_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
