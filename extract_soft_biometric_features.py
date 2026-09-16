#!/usr/bin/env python3
"""
Soft biometric feature extraction for AFEW-VA.

Two soft/behavioral biometric baselines:

1. Facial Landmarks:
   - Load 68 landmarks from the AFEW-VA JSON
   - Normalize (center + scale) → per-frame 136-dim
   - Clip-level statistics: mean, std, velocity → 136*3 = 408-dim

2. FER Model (HSEmotion, EfficientNet-B0 trained on AffectNet):
   - Expression recognition features (1280-dim)
   - Per-frame feature → clip-level mean pooling

Usage:
  python extract_soft_biometric_features.py \
      --method landmarks \
      --afewva_dir /path/to/AFEW-VA/AFEW-VA \
      --afewva_annotations_dir /path/to/preprocessed_VA_annotations \
      --output_dir ./features/AFEWVA_baselines

  python extract_soft_biometric_features.py \
      --method fer \
      --afewva_frames_dir /path/to/cropped_aligned \
      --afewva_annotations_dir /path/to/preprocessed_VA_annotations \
      --output_dir ./features/AFEWVA_baselines
"""

import os
import sys
import glob
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Method 1: Facial Landmarks
# ============================================================================
def extract_landmark_features(afewva_dir, annotations_dir, output_dir):
    """Extract normalized landmark features from AFEW-VA JSON files."""

    # Get clip list from annotations
    train_dir = os.path.join(annotations_dir, "Train_Set")
    clip_set = set()
    for csv_file in sorted(os.listdir(train_dir)):
        if not csv_file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(train_dir, csv_file))
        if 'clip_id' in df.columns:
            for cid in df['clip_id'].unique():
                clip_set.add(str(cid).zfill(3))

    output_feature_dir = os.path.join(output_dir, "Landmarks")
    os.makedirs(output_feature_dir, exist_ok=True)

    n_extracted = 0
    n_skipped = 0

    for clip_id in tqdm(sorted(clip_set), desc="Extracting landmarks"):
        save_path = os.path.join(output_feature_dir, f"{clip_id}.pt")
        if os.path.exists(save_path):
            n_skipped += 1
            continue

        json_path = os.path.join(afewva_dir, clip_id, f"{clip_id}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as f:
            data = json.load(f)

        frames = data.get('frames', {})
        if not frames:
            continue

        # Collect landmarks per frame
        frame_landmarks = []
        for frame_id in sorted(frames.keys()):
            landmarks = frames[frame_id].get('landmarks')
            if landmarks and len(landmarks) == 68:
                lm = np.array(landmarks, dtype=np.float32)  # (68, 2)

                # Normalize: center on nose tip (landmark 30), scale by face width
                center = lm[30]  # nose tip
                face_width = np.linalg.norm(lm[16] - lm[0])  # ear to ear
                if face_width > 0:
                    lm = (lm - center) / face_width

                frame_landmarks.append(lm.flatten())  # (136,)

        if len(frame_landmarks) < 2:
            continue

        frame_landmarks = np.stack(frame_landmarks)  # (T, 136)

        # Compute clip-level statistics
        mean_feat = frame_landmarks.mean(axis=0)  # (136,) — average position
        std_feat = frame_landmarks.std(axis=0)     # (136,) — variability
        # Velocity: mean absolute frame-to-frame difference
        velocity = np.abs(np.diff(frame_landmarks, axis=0)).mean(axis=0)  # (136,)

        # Concat: mean + std + velocity = 408-dim
        clip_feature = np.concatenate([mean_feat, std_feat, velocity])
        # Expand to (1, 408) to match biometric dataset format (T, feat_dim)
        clip_tensor = torch.tensor(clip_feature, dtype=torch.float32).unsqueeze(0)

        torch.save(clip_tensor, save_path)
        n_extracted += 1

    print(f"\n=== Landmarks Done ===")
    print(f"  Extracted: {n_extracted}, Skipped: {n_skipped}")
    print(f"  Feature dim: 408 (136 mean + 136 std + 136 velocity)")
    print(f"  Output: {output_feature_dir}")


# ============================================================================
# Method 2: FER Model (HSEmotion)
# ============================================================================
def extract_fer_features(afewva_frames_dir, annotations_dir, output_dir):
    """Extract FER features using HSEmotion (EfficientNet-B0, AffectNet)."""
    from hsemotion.facial_emotions import HSEmotionRecognizer
    from PIL import Image

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
    print(f"HSEmotion loaded on {device}")

    # Get clip list
    train_dir = os.path.join(annotations_dir, "Train_Set")
    clip_set = set()
    for csv_file in sorted(os.listdir(train_dir)):
        if not csv_file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(train_dir, csv_file))
        if 'clip_id' in df.columns:
            for cid in df['clip_id'].unique():
                clip_set.add(str(cid).zfill(3))

    output_feature_dir = os.path.join(output_dir, "FER")
    os.makedirs(output_feature_dir, exist_ok=True)

    n_extracted = 0
    n_skipped = 0

    for clip_id in tqdm(sorted(clip_set), desc="Extracting FER features"):
        save_path = os.path.join(output_feature_dir, f"{clip_id}.pt")
        if os.path.exists(save_path):
            n_skipped += 1
            continue

        clip_dir = os.path.join(afewva_frames_dir, clip_id)
        if not os.path.isdir(clip_dir):
            continue

        frame_paths = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
        if not frame_paths:
            continue

        # Sample frames (every 4th frame to save time, max 8 chunks like other methods)
        stride = max(1, len(frame_paths) // 8)
        sampled_paths = frame_paths[::stride][:32]  # max 32 frames

        frame_features = []
        for fp in sampled_paths:
            img = np.array(Image.open(fp).convert("RGB"))
            feat = model.extract_features(img)  # (1, 1280)
            frame_features.append(torch.tensor(feat, dtype=torch.float32))

        if not frame_features:
            continue

        # Stack and chunk (groups of 4 frames → mean pool per chunk)
        all_feats = torch.cat(frame_features, dim=0)  # (N, 1280)
        chunk_size = 4
        chunks = []
        for start in range(0, all_feats.shape[0], chunk_size):
            chunk = all_feats[start:start + chunk_size]
            chunks.append(chunk.mean(dim=0, keepdim=True))  # (1, 1280)

        clip_tensor = torch.cat(chunks, dim=0)  # (num_chunks, 1280)
        torch.save(clip_tensor, save_path)
        n_extracted += 1

    print(f"\n=== FER Done ===")
    print(f"  Extracted: {n_extracted}, Skipped: {n_skipped}")
    print(f"  Feature dim: 1280")
    print(f"  Output: {output_feature_dir}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Soft biometric feature extraction")
    parser.add_argument("--method", type=str, required=True,
                        choices=["landmarks", "fer", "all"])
    parser.add_argument("--afewva_dir", type=str, default=None,
                        help="AFEW-VA raw directory (for landmarks, contains JSON files)")
    parser.add_argument("--afewva_frames_dir", type=str, default=None,
                        help="AFEW-VA cropped frames directory (for FER)")
    parser.add_argument("--afewva_annotations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "AFEWVA_baselines"))
    args = parser.parse_args()

    methods = [args.method] if args.method != 'all' else ['landmarks', 'fer']

    for method in methods:
        if method == 'landmarks':
            if not args.afewva_dir:
                print("ERROR: --afewva_dir required for landmarks")
                continue
            extract_landmark_features(args.afewva_dir, args.afewva_annotations_dir, args.output_dir)

        elif method == 'fer':
            if not args.afewva_frames_dir:
                print("ERROR: --afewva_frames_dir required for fer")
                continue
            extract_fer_features(args.afewva_frames_dir, args.afewva_annotations_dir, args.output_dir)


if __name__ == "__main__":
    main()
