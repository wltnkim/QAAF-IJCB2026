#!/usr/bin/env python3
"""
Analyze complementary error cases: where ArcFace FAILS but VA-trained ViViT SUCCEEDS.

For each model at its EER threshold, find:
- False Rejections (FR): genuine pair rejected (same person, sim < threshold)
- False Acceptances (FA): impostor pair accepted (diff person, sim >= threshold)

Then find cases where ArcFace errs but ViViT does not.
"""

import torch
import numpy as np
import os
import json
from scipy.spatial.distance import cosine as cosine_dist
from paths import AFEWVA_SPLIT, WORK_ROOT

WORK_DIR = WORK_ROOT
SPLIT_FILE = AFEWVA_SPLIT
ARCFACE_DIR = os.path.join(WORK_DIR, 'features/AFEWVA_baselines/ArcFace')
VIVIT_DIR = os.path.join(WORK_DIR, 'features/AFEWVA/ViViT')


def load_features(features_dir):
    """Load all .pt features, mean-pool over time -> {clip_id: vector}."""
    feats = {}
    for fname in sorted(os.listdir(features_dir)):
        if not fname.endswith('.pt'):
            continue
        clip_id = os.path.splitext(fname)[0]
        data = torch.load(os.path.join(features_dir, fname),
                          map_location='cpu', weights_only=False)
        if isinstance(data, torch.Tensor):
            vec = data.float().mean(dim=0).numpy()
            feats[clip_id] = vec
    return feats


def build_enrollment_and_probes(feats, split_info):
    """Build enrollment templates and probe lists."""
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

    return actor_enrollments, actor_probes


def compute_all_pairs(actor_enrollments, actor_probes):
    """Compute all genuine and impostor pairs with metadata."""
    actors = list(actor_enrollments.keys())
    genuine_pairs = []  # (actor, clip_id, score)
    impostor_pairs = []  # (probe_actor, clip_id, enroll_actor, score)

    for actor in actors:
        if actor not in actor_probes:
            continue
        enroll = actor_enrollments[actor]

        for clip_id, probe_vec in actor_probes[actor]:
            # Genuine
            sim = 1.0 - cosine_dist(enroll, probe_vec)
            genuine_pairs.append((actor, clip_id, float(sim)))

            # Impostor
            for other_actor in actors:
                if other_actor == actor:
                    continue
                other_enroll = actor_enrollments[other_actor]
                sim_imp = 1.0 - cosine_dist(other_enroll, probe_vec)
                impostor_pairs.append((actor, clip_id, other_actor, float(sim_imp)))

    return genuine_pairs, impostor_pairs


def find_eer_threshold(genuine_scores, impostor_scores):
    """Find the threshold at EER."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 10000)

    far = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr = np.array([np.mean(genuine_scores < t) for t in thresholds])

    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        idx = np.argmin(np.abs(diff))
        return thresholds[idx], 0.5
    else:
        idx = sign_changes[0]
        denom = diff[idx + 1] - diff[idx]
        alpha = -diff[idx] / denom if denom != 0 else 0.5
        threshold = thresholds[idx] + alpha * (thresholds[idx + 1] - thresholds[idx])
        eer = far[idx] + alpha * (far[idx + 1] - far[idx])
        return threshold, float(eer)


def main():
    with open(SPLIT_FILE) as f:
        split_info = json.load(f)

    # Load features
    print("Loading ArcFace features...")
    arcface_feats = load_features(ARCFACE_DIR)
    print(f"  Loaded {len(arcface_feats)} clips")

    print("Loading VA-trained ViViT features...")
    vivit_feats = load_features(VIVIT_DIR)
    print(f"  Loaded {len(vivit_feats)} clips")

    # Build enrollment and probes
    arc_enroll, arc_probes = build_enrollment_and_probes(arcface_feats, split_info)
    viv_enroll, viv_probes = build_enrollment_and_probes(vivit_feats, split_info)

    # Compute all pairs
    print("\nComputing ArcFace pairs...")
    arc_genuine, arc_impostor = compute_all_pairs(arc_enroll, arc_probes)
    print(f"  {len(arc_genuine)} genuine, {len(arc_impostor)} impostor pairs")

    print("Computing ViViT pairs...")
    viv_genuine, viv_impostor = compute_all_pairs(viv_enroll, viv_probes)
    print(f"  {len(viv_genuine)} genuine, {len(viv_impostor)} impostor pairs")

    # Find EER thresholds
    arc_gen_scores = np.array([g[2] for g in arc_genuine])
    arc_imp_scores = np.array([i[3] for i in arc_impostor])
    arc_thresh, arc_eer = find_eer_threshold(arc_gen_scores, arc_imp_scores)
    print(f"\nArcFace:  EER = {arc_eer:.4f}, threshold = {arc_thresh:.6f}")

    viv_gen_scores = np.array([g[2] for g in viv_genuine])
    viv_imp_scores = np.array([i[3] for i in viv_impostor])
    viv_thresh, viv_eer = find_eer_threshold(viv_gen_scores, viv_imp_scores)
    print(f"ViViT:    EER = {viv_eer:.4f}, threshold = {viv_thresh:.6f}")

    # =========================================================================
    # Find False Rejections: genuine pairs where sim < threshold (rejected)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FALSE REJECTIONS (genuine pair rejected: same person, sim < threshold)")
    print("=" * 80)

    arc_fr = set()  # (actor, clip_id) tuples that ArcFace falsely rejects
    viv_fr = set()

    arc_fr_details = {}
    viv_fr_details = {}

    for actor, clip_id, sim in arc_genuine:
        if sim < arc_thresh:
            arc_fr.add((actor, clip_id))
            arc_fr_details[(actor, clip_id)] = sim

    for actor, clip_id, sim in viv_genuine:
        if sim < viv_thresh:
            viv_fr.add((actor, clip_id))
            viv_fr_details[(actor, clip_id)] = sim

    print(f"\nArcFace false rejections: {len(arc_fr)} / {len(arc_genuine)}")
    print(f"ViViT false rejections:   {len(viv_fr)} / {len(viv_genuine)}")

    # ArcFace FR but ViViT correct (accepted)
    arc_fr_only = arc_fr - viv_fr
    print(f"\nArcFace FR but ViViT CORRECT: {len(arc_fr_only)} cases")

    fr_complementary = []
    for actor, clip_id in sorted(arc_fr_only):
        arc_sim = arc_fr_details[(actor, clip_id)]
        # Find ViViT sim for same pair
        viv_sim = None
        for a, c, s in viv_genuine:
            if a == actor and c == clip_id:
                viv_sim = s
                break
        info = {
            'actor': actor,
            'clip_id': clip_id,
            'arcface_sim': round(arc_sim, 6),
            'arcface_threshold': round(arc_thresh, 6),
            'arcface_margin': round(arc_sim - arc_thresh, 6),
            'vivit_sim': round(viv_sim, 6) if viv_sim else None,
            'vivit_threshold': round(viv_thresh, 6),
            'vivit_margin': round(viv_sim - viv_thresh, 6) if viv_sim else None,
            'error_type': 'false_rejection'
        }
        fr_complementary.append(info)
        print(f"  Actor: {actor:<25} Clip: {clip_id}  "
              f"ArcFace sim={arc_sim:.4f} (thresh={arc_thresh:.4f}, REJECTED)  "
              f"ViViT sim={viv_sim:.4f} (thresh={viv_thresh:.4f}, ACCEPTED)")

    # =========================================================================
    # Find False Acceptances: impostor pairs where sim >= threshold (accepted)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FALSE ACCEPTANCES (impostor pair accepted: diff person, sim >= threshold)")
    print("=" * 80)

    # For impostor pairs, the key is (probe_actor, clip_id, enroll_actor)
    arc_fa = set()
    viv_fa = set()
    arc_fa_details = {}
    viv_fa_details = {}

    for probe_actor, clip_id, enroll_actor, sim in arc_impostor:
        if sim >= arc_thresh:
            key = (probe_actor, clip_id, enroll_actor)
            arc_fa.add(key)
            arc_fa_details[key] = sim

    for probe_actor, clip_id, enroll_actor, sim in viv_impostor:
        if sim >= viv_thresh:
            key = (probe_actor, clip_id, enroll_actor)
            viv_fa.add(key)
            viv_fa_details[key] = sim

    print(f"\nArcFace false acceptances: {len(arc_fa)} / {len(arc_impostor)}")
    print(f"ViViT false acceptances:   {len(viv_fa)} / {len(viv_impostor)}")

    # ArcFace FA but ViViT correct (rejected)
    arc_fa_only = arc_fa - viv_fa
    print(f"\nArcFace FA but ViViT CORRECT: {len(arc_fa_only)} cases")

    fa_complementary = []
    for probe_actor, clip_id, enroll_actor in sorted(arc_fa_only):
        arc_sim = arc_fa_details[(probe_actor, clip_id, enroll_actor)]
        # Find ViViT sim for same pair
        viv_sim = None
        for pa, ci, ea, s in viv_impostor:
            if pa == probe_actor and ci == clip_id and ea == enroll_actor:
                viv_sim = s
                break
        info = {
            'probe_actor': probe_actor,
            'clip_id': clip_id,
            'enroll_actor': enroll_actor,
            'arcface_sim': round(arc_sim, 6),
            'arcface_threshold': round(arc_thresh, 6),
            'arcface_margin': round(arc_sim - arc_thresh, 6),
            'vivit_sim': round(viv_sim, 6) if viv_sim else None,
            'vivit_threshold': round(viv_thresh, 6),
            'vivit_margin': round(viv_sim - viv_thresh, 6) if viv_sim else None,
            'error_type': 'false_acceptance'
        }
        fa_complementary.append(info)

    # Print FA details (may be many, so group by probe clip)
    if len(fa_complementary) <= 50:
        for info in fa_complementary:
            print(f"  Probe: {info['probe_actor']:<25} Clip: {info['clip_id']}  "
                  f"matched to: {info['enroll_actor']:<25}  "
                  f"ArcFace sim={info['arcface_sim']:.4f} (ACCEPTED)  "
                  f"ViViT sim={info['vivit_sim']:.4f} (REJECTED)")
    else:
        # Group by probe clip
        from collections import defaultdict
        by_clip = defaultdict(list)
        for info in fa_complementary:
            by_clip[(info['probe_actor'], info['clip_id'])].append(info)
        print(f"\n  Grouped by probe clip ({len(by_clip)} unique probe clips):")
        for (actor, clip), infos in sorted(by_clip.items()):
            print(f"    {actor:<25} Clip {clip}: {len(infos)} false acceptances corrected by ViViT")
            for info in infos[:3]:  # show first 3
                print(f"      -> matched {info['enroll_actor']:<25} "
                      f"ArcFace={info['arcface_sim']:.4f} ViViT={info['vivit_sim']:.4f}")
            if len(infos) > 3:
                print(f"      ... and {len(infos) - 3} more")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"ArcFace EER: {arc_eer:.4f} (threshold: {arc_thresh:.6f})")
    print(f"ViViT   EER: {viv_eer:.4f} (threshold: {viv_thresh:.6f})")
    print(f"\nTotal genuine pairs: {len(arc_genuine)}")
    print(f"Total impostor pairs: {len(arc_impostor)}")
    print(f"\nArcFace errors:")
    print(f"  False Rejections: {len(arc_fr)}")
    print(f"  False Acceptances: {len(arc_fa)}")
    print(f"  Total errors: {len(arc_fr) + len(arc_fa)}")
    print(f"\nViViT errors:")
    print(f"  False Rejections: {len(viv_fr)}")
    print(f"  False Acceptances: {len(viv_fa)}")
    print(f"  Total errors: {len(viv_fr) + len(viv_fa)}")
    print(f"\nComplementary cases (ArcFace FAILS, ViViT SUCCEEDS):")
    print(f"  False Rejections corrected by ViViT: {len(arc_fr_only)}")
    print(f"  False Acceptances corrected by ViViT: {len(arc_fa_only)}")
    print(f"  Total complementary: {len(arc_fr_only) + len(arc_fa_only)}")

    # Also check reverse: ViViT fails, ArcFace succeeds
    viv_fr_only = viv_fr - arc_fr
    viv_fa_only = viv_fa - arc_fa
    print(f"\nReverse (ViViT FAILS, ArcFace SUCCEEDS):")
    print(f"  False Rejections corrected by ArcFace: {len(viv_fr_only)}")
    print(f"  False Acceptances corrected by ArcFace: {len(viv_fa_only)}")
    print(f"  Total: {len(viv_fr_only) + len(viv_fa_only)}")

    # Both fail
    both_fr = arc_fr & viv_fr
    both_fa = arc_fa & viv_fa
    print(f"\nBoth FAIL:")
    print(f"  Both FR: {len(both_fr)}")
    print(f"  Both FA: {len(both_fa)}")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        'metadata': {
            'arcface_eer': round(arc_eer, 6),
            'arcface_threshold': round(arc_thresh, 6),
            'vivit_eer': round(viv_eer, 6),
            'vivit_threshold': round(viv_thresh, 6),
            'n_genuine_pairs': len(arc_genuine),
            'n_impostor_pairs': len(arc_impostor),
        },
        'error_counts': {
            'arcface_false_rejections': len(arc_fr),
            'arcface_false_acceptances': len(arc_fa),
            'vivit_false_rejections': len(viv_fr),
            'vivit_false_acceptances': len(viv_fa),
            'complementary_fr_arcface_fails_vivit_succeeds': len(arc_fr_only),
            'complementary_fa_arcface_fails_vivit_succeeds': len(arc_fa_only),
            'total_complementary': len(arc_fr_only) + len(arc_fa_only),
            'reverse_fr_vivit_fails_arcface_succeeds': len(viv_fr_only),
            'reverse_fa_vivit_fails_arcface_succeeds': len(viv_fa_only),
            'both_fail_fr': len(both_fr),
            'both_fail_fa': len(both_fa),
        },
        'complementary_false_rejections': fr_complementary,
        'complementary_false_acceptances': fa_complementary,
    }

    output_path = os.path.join(WORK_DIR, 'complementary_error_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
