#!/usr/bin/env python3
"""
Verification (EER/AUC) evaluation for ALL baselines using raw features.

For each baseline:
  1. Load .pt features per clip → mean-pool over time → single vector
  2. Enrollment: mean of train clip vectors per actor
  3. Genuine pairs: test clip vs same-actor enrollment
  4. Impostor pairs: test clip vs different-actor enrollment
  5. Cosine similarity → EER, AUC
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
            vec = data.float().mean(dim=0).numpy()  # (T, D) → (D,)
        else:
            continue
        feats[clip_id] = vec
    return feats


def compute_eer_auc(genuine_scores, impostor_scores):
    """Compute EER and AUC from genuine/impostor similarity scores."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)

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
        alpha = -diff[idx] / denom if denom != 0 else 0.5
        eer = far[idx] + alpha * (far[idx + 1] - far[idx])

    # AUC
    tpr = 1 - frr
    fpr = far
    sorted_idx = np.argsort(fpr)
    auc = float(np.trapz(tpr[sorted_idx], fpr[sorted_idx]))

    return float(eer), auc


def run_verification(feats, split_info):
    """Run verification given {clip_id: vector} and split info."""
    genuine_scores = []
    impostor_scores = []

    # Build enrollment and probe
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

    actors = list(actor_enrollments.keys())

    for actor in actors:
        if actor not in actor_probes:
            continue
        enroll = actor_enrollments[actor]

        for clip_id, probe_vec in actor_probes[actor]:
            # Genuine
            sim = 1.0 - cosine_dist(enroll, probe_vec)
            genuine_scores.append(sim)

            # Impostor: this probe vs other actors' enrollment
            for other_actor in actors:
                if other_actor == actor:
                    continue
                other_enroll = actor_enrollments[other_actor]
                sim_imp = 1.0 - cosine_dist(other_enroll, probe_vec)
                impostor_scores.append(sim_imp)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return None

    eer, auc = compute_eer_auc(genuine_scores, impostor_scores)
    return {
        'eer': eer,
        'auc': auc,
        'n_genuine': len(genuine_scores),
        'n_impostor': len(impostor_scores),
    }


def main():
    split_file = AFEWVA_SPLIT
    with open(split_file) as f:
        split_info = json.load(f)

    baselines_dir = 'features/AFEWVA_baselines'
    results = {}

    # 1. All baselines
    baseline_names = sorted(os.listdir(baselines_dir))
    for name in baseline_names:
        feat_dir = os.path.join(baselines_dir, name)
        if not os.path.isdir(feat_dir):
            continue
        feats = load_features(feat_dir)
        dim = list(feats.values())[0].shape[0] if feats else 0
        res = run_verification(feats, split_info)
        if res:
            res['dim'] = dim
            res['source'] = f'baselines/{name}'
            results[name] = res
            print(f"{name:<25} dim={dim:<6} EER={res['eer']:.4f}  AUC={res['auc']:.4f}")

    # 2. VA-trained features
    for backbone in ['ViViT', 'VideoMAE']:
        feat_dir = f'features/AFEWVA/{backbone}'
        if not os.path.isdir(feat_dir):
            continue
        feats = load_features(feat_dir)
        dim = list(feats.values())[0].shape[0] if feats else 0
        res = run_verification(feats, split_info)
        if res:
            res['dim'] = dim
            res['source'] = f'AFEWVA/{backbone}'
            key = f'VA-trained_{backbone}'
            results[key] = res
            print(f"{key:<25} dim={dim:<6} EER={res['eer']:.4f}  AUC={res['auc']:.4f}")

    # 3. Multi-layer best (concat all layers)
    for backbone in ['ViViT', 'VideoMAE']:
        layer_dirs = sorted([
            d for d in os.listdir('features/AFEWVA_multilayer')
            if d.startswith(f'{backbone}_layer-')
        ])
        if not layer_dirs:
            continue

        # Load and concat all layers
        all_clips = set()
        layer_feats = {}
        for ld in layer_dirs:
            feat_dir = f'features/AFEWVA_multilayer/{ld}'
            lf = load_features(feat_dir)
            layer_feats[ld] = lf
            all_clips.update(lf.keys())

        # Concat layers per clip
        concat_feats = {}
        for clip_id in all_clips:
            vecs = []
            for ld in layer_dirs:
                if clip_id in layer_feats[ld]:
                    vecs.append(layer_feats[ld][clip_id])
            if len(vecs) == len(layer_dirs):
                concat_feats[clip_id] = np.concatenate(vecs)

        dim = list(concat_feats.values())[0].shape[0] if concat_feats else 0
        res = run_verification(concat_feats, split_info)
        if res:
            res['dim'] = dim
            res['source'] = f'AFEWVA_multilayer/{backbone}_all'
            key = f'Multi-layer_{backbone}'
            results[key] = res
            print(f"{key:<25} dim={dim:<6} EER={res['eer']:.4f}  AUC={res['auc']:.4f}")

    # Save results
    output_path = 'verification_results_all_baselines.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'Dim':>6} {'EER ↓':>8} {'AUC ↑':>8}")
    print(f"{'='*70}")
    for name, res in sorted(results.items(), key=lambda x: x[1]['eer']):
        print(f"{name:<25} {res['dim']:>6} {res['eer']:>8.4f} {res['auc']:>8.4f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
