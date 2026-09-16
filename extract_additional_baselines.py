#!/usr/bin/env python3
"""
Additional baseline feature extraction for biometric comparison.

1. LBP (Local Binary Patterns) [Ojala et al., TPAMI'02]
   - Classic texture descriptor for face analysis
   - Per-frame LBP histogram → clip-level statistics

2. ResNet-50 ImageNet [He et al., CVPR'16]
   - Single-frame CNN, no temporal modeling
   - features from penultimate layer (2048-dim)

3. CLIP ViT-B/32 [Radford et al., ICML'21]
   - Vision-language model, general visual features (512-dim)

Usage:
  python extract_additional_baselines.py \
      --method lbp \
      --afewva_frames_dir /path/to/cropped_aligned \
      --afewva_annotations_dir /path/to/preprocessed_VA_annotations \
      --output_dir ./features/AFEWVA_baselines

  python extract_additional_baselines.py --method resnet50 ...
  python extract_additional_baselines.py --method clip ...
  python extract_additional_baselines.py --method all ...
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_clip_list(annotations_dir, frames_dir):
    """Get valid clip IDs from annotations."""
    train_dir = os.path.join(annotations_dir, "Train_Set")
    clip_set = set()
    for csv_file in sorted(os.listdir(train_dir)):
        if not csv_file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(train_dir, csv_file))
        if 'clip_id' in df.columns:
            for cid in df['clip_id'].unique():
                cid_str = str(cid).zfill(3)
                if os.path.isdir(os.path.join(frames_dir, cid_str)):
                    clip_set.add(cid_str)
    return sorted(clip_set)


def load_clip_frames(frames_dir, clip_id):
    """Load all frames for a clip."""
    clip_dir = os.path.join(frames_dir, clip_id)
    frame_paths = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
    frames = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        frames.append(np.array(img))
    return frames


# ============================================================================
# LBP [Ojala et al., TPAMI'02]
# ============================================================================
def extract_lbp_features(frames_dir, annotations_dir, output_dir):
    """Extract LBP histogram features."""
    from skimage.feature import local_binary_pattern
    from skimage.color import rgb2gray

    output_feature_dir = os.path.join(output_dir, "LBP")
    os.makedirs(output_feature_dir, exist_ok=True)

    clips = get_clip_list(annotations_dir, frames_dir)
    n_points = 24
    radius = 3
    n_bins = n_points + 2  # uniform LBP

    n_extracted = 0
    n_skipped = 0

    for clip_id in tqdm(clips, desc="Extracting LBP"):
        save_path = os.path.join(output_feature_dir, f"{clip_id}.pt")
        if os.path.exists(save_path):
            n_skipped += 1
            continue

        frames = load_clip_frames(frames_dir, clip_id)
        if not frames:
            continue

        frame_hists = []
        for frame in frames:
            gray = rgb2gray(frame)  # (H, W), float [0, 1]
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            frame_hists.append(hist.astype(np.float32))

        if len(frame_hists) < 2:
            continue

        frame_hists = np.stack(frame_hists)  # (T, n_bins)

        # Clip-level: mean + std + velocity
        mean_feat = frame_hists.mean(axis=0)
        std_feat = frame_hists.std(axis=0)
        velocity = np.abs(np.diff(frame_hists, axis=0)).mean(axis=0)

        clip_feature = np.concatenate([mean_feat, std_feat, velocity])  # n_bins * 3
        clip_tensor = torch.tensor(clip_feature, dtype=torch.float32).unsqueeze(0)  # (1, feat_dim)

        torch.save(clip_tensor, save_path)
        n_extracted += 1

    print(f"\n=== LBP Done: {n_extracted} extracted, {n_skipped} skipped ===")
    print(f"  Feature dim: {n_bins * 3}")


# ============================================================================
# ResNet-50 ImageNet [He et al., CVPR'16]
# ============================================================================
def extract_resnet50_features(frames_dir, annotations_dir, output_dir):
    """Extract ResNet-50 ImageNet features (single-frame, no temporal)."""
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # Remove classification head → 2048-dim features
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    output_feature_dir = os.path.join(output_dir, "ResNet50")
    os.makedirs(output_feature_dir, exist_ok=True)

    clips = get_clip_list(annotations_dir, frames_dir)
    n_extracted = 0
    n_skipped = 0

    for clip_id in tqdm(clips, desc="Extracting ResNet-50"):
        save_path = os.path.join(output_feature_dir, f"{clip_id}.pt")
        if os.path.exists(save_path):
            n_skipped += 1
            continue

        frames = load_clip_frames(frames_dir, clip_id)
        if not frames:
            continue

        # Sample frames (every 4th, max 8 chunks like other methods)
        stride = max(1, len(frames) // 8)
        sampled = frames[::stride][:32]

        frame_features = []
        for frame in sampled:
            img = Image.fromarray(frame)
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor)  # (1, 2048)
            frame_features.append(feat.cpu())

        if not frame_features:
            continue

        all_feats = torch.cat(frame_features, dim=0)  # (N, 2048)

        # Chunk into groups of 4 → mean pool
        chunk_size = 4
        chunks = []
        for start in range(0, all_feats.shape[0], chunk_size):
            chunk = all_feats[start:start + chunk_size]
            chunks.append(chunk.mean(dim=0, keepdim=True))

        clip_tensor = torch.cat(chunks, dim=0)  # (num_chunks, 2048)
        torch.save(clip_tensor, save_path)
        n_extracted += 1

    print(f"\n=== ResNet-50 Done: {n_extracted} extracted, {n_skipped} skipped ===")
    print(f"  Feature dim: 2048")


# ============================================================================
# CLIP ViT-B/32 [Radford et al., ICML'21]
# ============================================================================
def extract_clip_features(frames_dir, annotations_dir, output_dir):
    """Extract CLIP visual features."""
    import clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    output_feature_dir = os.path.join(output_dir, "CLIP")
    os.makedirs(output_feature_dir, exist_ok=True)

    clips = get_clip_list(annotations_dir, frames_dir)
    n_extracted = 0
    n_skipped = 0

    for clip_id in tqdm(clips, desc="Extracting CLIP"):
        save_path = os.path.join(output_feature_dir, f"{clip_id}.pt")
        if os.path.exists(save_path):
            n_skipped += 1
            continue

        frames = load_clip_frames(frames_dir, clip_id)
        if not frames:
            continue

        stride = max(1, len(frames) // 8)
        sampled = frames[::stride][:32]

        frame_features = []
        for frame in sampled:
            img = Image.fromarray(frame)
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(tensor).float()  # (1, 512)
            frame_features.append(feat.cpu())

        if not frame_features:
            continue

        all_feats = torch.cat(frame_features, dim=0)

        chunk_size = 4
        chunks = []
        for start in range(0, all_feats.shape[0], chunk_size):
            chunk = all_feats[start:start + chunk_size]
            chunks.append(chunk.mean(dim=0, keepdim=True))

        clip_tensor = torch.cat(chunks, dim=0)
        torch.save(clip_tensor, save_path)
        n_extracted += 1

    print(f"\n=== CLIP Done: {n_extracted} extracted, {n_skipped} skipped ===")
    print(f"  Feature dim: 512")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Additional baseline feature extraction")
    parser.add_argument("--method", type=str, required=True,
                        choices=["lbp", "resnet50", "clip", "all"])
    parser.add_argument("--afewva_frames_dir", type=str, required=True)
    parser.add_argument("--afewva_annotations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "AFEWVA_baselines"))
    args = parser.parse_args()

    methods = [args.method] if args.method != 'all' else ['lbp', 'resnet50', 'clip']

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Extracting: {method}")
        print(f"{'='*60}")

        if method == 'lbp':
            extract_lbp_features(args.afewva_frames_dir, args.afewva_annotations_dir, args.output_dir)
        elif method == 'resnet50':
            extract_resnet50_features(args.afewva_frames_dir, args.afewva_annotations_dir, args.output_dir)
        elif method == 'clip':
            extract_clip_features(args.afewva_frames_dir, args.afewva_annotations_dir, args.output_dir)


if __name__ == "__main__":
    main()
