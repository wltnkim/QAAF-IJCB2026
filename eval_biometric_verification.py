#!/usr/bin/env python3
"""
Step 2 + 3: Biometric Verification using VA Dynamic Signatures.

Design notes:
  - VA temporal dynamics → behavioral biometric signature
  - Verification metrics (FAR/FRR/ROC/EER)
  - Same vs different identity similarity score distribution
  - Reference (enrollment) vs probe (test) comparison

Signature methods:
  1. raw_sequence: use the VA sequence as-is (DTW or truncated cosine)
  2. statistics: mean, std, skewness, kurtosis of V and A + derivatives
  3. histogram: histogram binning of V and A separately
  4. joint_histogram: 2D V-A joint histogram
  5. combined: statistics + histogram concat

Usage:
  python eval_biometric_verification.py \
      --va_sequences_dir ./va_sequences/ViViT \
      --split_file /path/to/split_biometric.json \
      --methods statistics histogram combined \
      --n_seeds 5
"""

import torch
import numpy as np
import os
import json
import argparse
from collections import defaultdict
from itertools import combinations
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import AFEWVA_SPLIT


# ============================================================================
# Signature Extraction Methods
# ============================================================================

def extract_signature_statistics(va_seq):
    """
    Statistical moments of VA dynamics.
    va_seq: (T, 2) — [valence, arousal]
    Returns: 1D feature vector
    """
    v, a = va_seq[:, 0], va_seq[:, 1]
    feats = []

    for signal in [v, a]:
        feats.extend([
            signal.mean(),
            signal.std() if len(signal) > 1 else 0.0,
            float(skew(signal)) if len(signal) > 2 else 0.0,
            float(kurtosis(signal)) if len(signal) > 3 else 0.0,
            signal.min(),
            signal.max(),
            signal.max() - signal.min(),  # range
        ])

        # Derivative features (temporal dynamics)
        if len(signal) > 1:
            deriv = np.diff(signal)
            feats.extend([
                deriv.mean(),
                deriv.std(),
                np.abs(deriv).mean(),  # mean absolute change
                (deriv > 0).sum() / len(deriv),  # fraction of increases
            ])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.5])

    # Cross-dimension features
    if len(v) > 1:
        corr = np.corrcoef(v, a)[0, 1] if v.std() > 0 and a.std() > 0 else 0.0
        feats.append(float(corr) if not np.isnan(corr) else 0.0)
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def extract_signature_histogram(va_seq, n_bins=10, va_range=(-1, 1)):
    """
    Histogram representation of VA distribution.
    va_seq: (T, 2)
    Returns: concatenated V and A histograms
    """
    v, a = va_seq[:, 0], va_seq[:, 1]
    bins = np.linspace(va_range[0], va_range[1], n_bins + 1)

    hist_v, _ = np.histogram(v, bins=bins, density=True)
    hist_a, _ = np.histogram(a, bins=bins, density=True)

    # Normalize
    hist_v = hist_v / (hist_v.sum() + 1e-8)
    hist_a = hist_a / (hist_a.sum() + 1e-8)

    return np.concatenate([hist_v, hist_a]).astype(np.float32)


def extract_signature_joint_histogram(va_seq, n_bins=8, va_range=(-1, 1)):
    """
    2D joint histogram of V-A space.
    """
    v, a = va_seq[:, 0], va_seq[:, 1]
    bins = np.linspace(va_range[0], va_range[1], n_bins + 1)

    hist_2d, _, _ = np.histogram2d(v, a, bins=[bins, bins], density=True)
    hist_2d = hist_2d / (hist_2d.sum() + 1e-8)

    return hist_2d.flatten().astype(np.float32)


def extract_signature_combined(va_seq, n_bins=10):
    """Concatenation of statistics + histogram."""
    stats = extract_signature_statistics(va_seq)
    hist = extract_signature_histogram(va_seq, n_bins=n_bins)
    return np.concatenate([stats, hist])


SIGNATURE_METHODS = {
    'statistics': extract_signature_statistics,
    'histogram': extract_signature_histogram,
    'joint_histogram': extract_signature_joint_histogram,
    'combined': extract_signature_combined,
}


# ============================================================================
# Verification Evaluation
# ============================================================================

def compute_similarity(sig1, sig2, metric='cosine'):
    """Compute similarity between two signatures. Higher = more similar."""
    if metric == 'cosine':
        return 1.0 - cosine_dist(sig1, sig2)
    elif metric == 'correlation':
        if sig1.std() == 0 or sig2.std() == 0:
            return 0.0
        corr = np.corrcoef(sig1, sig2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    elif metric == 'euclidean':
        return -np.linalg.norm(sig1 - sig2)  # negative distance as similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")


def generate_pairs(actor_clips, split_info):
    """
    Generate genuine and impostor pairs from biometric split.
    Uses train clips as enrollment (reference), test clips as probe.

    Returns:
        genuine_pairs: [(actor, enroll_clip, probe_clip), ...]
        impostor_pairs: [(actor1, actor2, enroll_clip, probe_clip), ...]
    """
    genuine_pairs = []
    impostor_pairs = []

    actors = list(actor_clips.keys())

    for actor in actors:
        info = split_info['actors'].get(actor, None)
        if info is None:
            continue

        train_clips = info['train_clips']
        test_clips = info['test_clips']

        # Genuine: each test clip vs each train clip of same actor
        for test_clip in test_clips:
            for train_clip in train_clips:
                genuine_pairs.append((actor, train_clip, test_clip))

    # Impostor: each test clip vs train clips of different actors
    for i, actor_probe in enumerate(actors):
        info_probe = split_info['actors'].get(actor_probe, None)
        if info_probe is None:
            continue

        for test_clip in info_probe['test_clips']:
            for j, actor_enroll in enumerate(actors):
                if i == j:
                    continue
                info_enroll = split_info['actors'].get(actor_enroll, None)
                if info_enroll is None:
                    continue

                # Use first train clip as enrollment representative
                # (can average later for multi-enrollment)
                enroll_clip = info_enroll['train_clips'][0]
                impostor_pairs.append((actor_probe, actor_enroll, enroll_clip, test_clip))

    return genuine_pairs, impostor_pairs


def compute_far_frr(genuine_scores, impostor_scores, thresholds=None):
    """
    Compute FAR and FRR at various thresholds.

    FAR = False Accept Rate = P(accept | impostor) = fraction of impostor scores >= threshold
    FRR = False Reject Rate = P(reject | genuine) = fraction of genuine scores < threshold
    """
    if thresholds is None:
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)

    far = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr = np.array([np.mean(genuine_scores < t) for t in thresholds])

    return thresholds, far, frr


def compute_eer(genuine_scores, impostor_scores):
    """Compute Equal Error Rate (EER) — the point where FAR == FRR."""
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)

    # Find crossing point
    diff = far - frr
    # EER is where diff changes sign
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        # No crossing — return worst case
        return 0.5, thresholds[0]

    idx = sign_changes[0]
    # Linear interpolation
    if diff[idx + 1] - diff[idx] != 0:
        alpha = -diff[idx] / (diff[idx + 1] - diff[idx])
    else:
        alpha = 0.5

    eer = far[idx] + alpha * (far[idx + 1] - far[idx])
    eer_threshold = thresholds[idx] + alpha * (thresholds[idx + 1] - thresholds[idx])

    return float(eer), float(eer_threshold)


def compute_roc(genuine_scores, impostor_scores):
    """Compute ROC curve (TPR vs FPR)."""
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)
    tpr = 1 - frr  # True Positive Rate = 1 - FRR
    fpr = far       # False Positive Rate = FAR
    return fpr, tpr, thresholds


def compute_auc(fpr, tpr):
    """Compute AUC from ROC curve."""
    # Sort by fpr
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]
    return float(np.trapz(tpr_sorted, fpr_sorted))


# ============================================================================
# Plotting
# ============================================================================

def plot_score_distributions(genuine_scores, impostor_scores, eer, title, save_path):
    """Plot genuine vs impostor score distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(genuine_scores, bins=50, alpha=0.6, color='green', label=f'Genuine (n={len(genuine_scores)})', density=True)
    ax.hist(impostor_scores, bins=50, alpha=0.6, color='red', label=f'Impostor (n={len(impostor_scores)})', density=True)
    ax.axvline(x=eer, color='black', linestyle='--', alpha=0.5, label=f'EER threshold')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{title}\nEER = {eer:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, auc, eer, title, save_path):
    """Plot ROC curve."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    # Mark EER point
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    ax.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=8, label=f'EER = {eer:.4f}')
    ax.set_xlabel('False Positive Rate (FAR)')
    ax.set_ylabel('True Positive Rate (1 - FRR)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_far_frr(thresholds, far, frr, eer_val, eer_thresh, title, save_path):
    """Plot FAR/FRR vs threshold."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(thresholds, far, 'r-', label='FAR', linewidth=2)
    ax.plot(thresholds, frr, 'b-', label='FRR', linewidth=2)
    ax.axvline(x=eer_thresh, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=eer_val, color='gray', linestyle=':', alpha=0.3)
    ax.plot(eer_thresh, eer_val, 'ko', markersize=8, label=f'EER = {eer_val:.4f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Error Rate')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def run_verification(va_sequences_dir, split_file, methods, similarity_metric,
                     output_dir, n_seeds=5, multi_enroll='mean'):
    """
    Full verification evaluation pipeline.

    Args:
        va_sequences_dir: directory with {clip_id}.npy VA sequences
        split_file: path to split_biometric.json
        methods: list of signature method names
        similarity_metric: similarity metric to use
        output_dir: where to save results
        n_seeds: number of random seeds (for impostor pair sampling)
        multi_enroll: how to combine multiple enrollment clips ('mean', 'first', 'all')
    """
    # Load split
    with open(split_file) as f:
        split_info = json.load(f)

    # Load all VA sequences
    va_seqs = {}
    for fname in sorted(os.listdir(va_sequences_dir)):
        if fname.endswith('.npy'):
            clip_id = os.path.splitext(fname)[0]
            va_seqs[clip_id] = np.load(os.path.join(va_sequences_dir, fname))

    print(f"Loaded {len(va_seqs)} VA sequences")

    # Map clip_ids to actors
    clip_to_actor = {}
    actor_clips = {}
    for actor, info in split_info['actors'].items():
        all_clips = info['train_clips'] + info['test_clips']
        actor_clips[actor] = {'train': info['train_clips'], 'test': info['test_clips']}
        for clip in all_clips:
            clip_id = str(clip).zfill(3)
            clip_to_actor[clip_id] = actor

    # Check coverage
    available_actors = set()
    for clip_id in va_seqs:
        if clip_id in clip_to_actor:
            available_actors.add(clip_to_actor[clip_id])
    print(f"Available actors with VA sequences: {len(available_actors)}/{len(split_info['actors'])}")

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        extract_fn = SIGNATURE_METHODS[method_name]

        # Extract signatures for all clips
        signatures = {}
        for clip_id, va_seq in va_seqs.items():
            try:
                sig = extract_fn(va_seq)
                if np.any(np.isnan(sig)):
                    sig = np.nan_to_num(sig, nan=0.0)
                signatures[clip_id] = sig
            except Exception as e:
                print(f"  [WARNING] Failed to extract signature for {clip_id}: {e}")

        print(f"  Extracted {len(signatures)} signatures (dim={signatures[list(signatures.keys())[0]].shape[0]})")

        # Generate pairs and compute scores
        genuine_scores = []
        impostor_scores = []

        for actor, info in split_info['actors'].items():
            train_clips = [str(c).zfill(3) for c in info['train_clips']]
            test_clips = [str(c).zfill(3) for c in info['test_clips']]

            # Enrollment signature: mean of all train clip signatures
            train_sigs = [signatures[c] for c in train_clips if c in signatures]
            if len(train_sigs) == 0:
                continue

            if multi_enroll == 'mean':
                enroll_sig = np.mean(train_sigs, axis=0)
            elif multi_enroll == 'first':
                enroll_sig = train_sigs[0]

            # Genuine: test clips of same actor
            for test_clip in test_clips:
                if test_clip not in signatures:
                    continue
                probe_sig = signatures[test_clip]
                score = compute_similarity(enroll_sig, probe_sig, metric=similarity_metric)
                genuine_scores.append(score)

            # Impostor: test clips of same actor vs enrollment of other actors
            for other_actor, other_info in split_info['actors'].items():
                if other_actor == actor:
                    continue
                other_train = [str(c).zfill(3) for c in other_info['train_clips']]
                other_sigs = [signatures[c] for c in other_train if c in signatures]
                if len(other_sigs) == 0:
                    continue

                if multi_enroll == 'mean':
                    other_enroll = np.mean(other_sigs, axis=0)
                else:
                    other_enroll = other_sigs[0]

                for test_clip in test_clips:
                    if test_clip not in signatures:
                        continue
                    probe_sig = signatures[test_clip]
                    score = compute_similarity(other_enroll, probe_sig, metric=similarity_metric)
                    impostor_scores.append(score)

        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)

        print(f"  Genuine pairs: {len(genuine_scores)}, Impostor pairs: {len(impostor_scores)}")
        print(f"  Genuine scores: mean={genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
        print(f"  Impostor scores: mean={impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")

        # Compute metrics
        eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
        fpr, tpr, thresholds_roc = compute_roc(genuine_scores, impostor_scores)
        auc = compute_auc(fpr, tpr)
        thresholds_det, far, frr = compute_far_frr(genuine_scores, impostor_scores)

        # d-prime (separation between distributions)
        d_prime = (genuine_scores.mean() - impostor_scores.mean()) / \
                  np.sqrt(0.5 * (genuine_scores.var() + impostor_scores.var()) + 1e-8)

        print(f"  EER: {eer:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  d-prime: {d_prime:.4f}")

        # Save results
        method_results = {
            'method': method_name,
            'similarity_metric': similarity_metric,
            'multi_enroll': multi_enroll,
            'signature_dim': int(signatures[list(signatures.keys())[0]].shape[0]),
            'n_genuine': len(genuine_scores),
            'n_impostor': len(impostor_scores),
            'genuine_mean': float(genuine_scores.mean()),
            'genuine_std': float(genuine_scores.std()),
            'impostor_mean': float(impostor_scores.mean()),
            'impostor_std': float(impostor_scores.std()),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'auc': float(auc),
            'd_prime': float(d_prime),
        }
        all_results[method_name] = method_results

        # Plots
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        backbone_name = os.path.basename(va_sequences_dir)
        plot_score_distributions(
            genuine_scores, impostor_scores, eer_threshold,
            f'{backbone_name} — {method_name} ({similarity_metric})',
            os.path.join(method_dir, 'score_distribution.png')
        )
        plot_roc_curve(
            fpr, tpr, auc, eer,
            f'{backbone_name} — {method_name} ROC',
            os.path.join(method_dir, 'roc_curve.png')
        )
        plot_far_frr(
            thresholds_det, far, frr, eer, eer_threshold,
            f'{backbone_name} — {method_name} FAR/FRR',
            os.path.join(method_dir, 'far_frr.png')
        )

        # Save raw scores for further analysis
        np.savez(
            os.path.join(method_dir, 'scores.npz'),
            genuine=genuine_scores,
            impostor=impostor_scores,
        )

    # Save summary
    with open(os.path.join(output_dir, 'verification_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Method':<20} {'EER':>8} {'AUC':>8} {'d-prime':>8} {'Genuine':>10} {'Impostor':>10}")
    print(f"{'='*70}")
    for method_name, res in all_results.items():
        print(f"{method_name:<20} {res['eer']:>8.4f} {res['auc']:>8.4f} {res['d_prime']:>8.4f} "
              f"{res['genuine_mean']:>10.4f} {res['impostor_mean']:>10.4f}")
    print(f"{'='*70}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Biometric verification using VA dynamic signatures")
    parser.add_argument('--va_sequences_dir', type=str, required=True,
                        help='Directory with VA sequence .npy files (e.g., ./va_sequences/ViViT)')
    parser.add_argument('--split_file', type=str,
                        default=AFEWVA_SPLIT,
                        help='Path to split_biometric.json')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['statistics', 'histogram', 'joint_histogram', 'combined'],
                        choices=list(SIGNATURE_METHODS.keys()),
                        help='Signature methods to evaluate')
    parser.add_argument('--similarity', type=str, default='cosine',
                        choices=['cosine', 'correlation', 'euclidean'],
                        help='Similarity metric')
    parser.add_argument('--multi_enroll', type=str, default='mean',
                        choices=['mean', 'first'],
                        help='How to combine multiple enrollment clips')
    parser.add_argument('--output_dir', type=str, default='./verification_results',
                        help='Output directory')
    parser.add_argument('--n_seeds', type=int, default=5)

    args = parser.parse_args()

    # Add backbone name to output dir
    backbone = os.path.basename(args.va_sequences_dir)
    output_dir = os.path.join(args.output_dir, backbone)

    run_verification(
        va_sequences_dir=args.va_sequences_dir,
        split_file=args.split_file,
        methods=args.methods,
        similarity_metric=args.similarity,
        output_dir=output_dir,
        n_seeds=args.n_seeds,
        multi_enroll=args.multi_enroll,
    )


if __name__ == '__main__':
    main()
