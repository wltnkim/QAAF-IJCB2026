#!/usr/bin/env python3
"""
Baseline feature extraction for biometric comparison.

Extracts three baselines:
  1. random_init: untrained ViViT/VideoMAE (random weights)
  2. pretrained: ImageNet/Kinetics400 pretrained (before VA fine-tuning)
  3. arcface: ArcFace face recognition model (hard biometric upper bound)

Usage:
  python extract_baseline_features.py \
      --method pretrained --backbone ViViT \
      --afewva_frames_dir /path/to/cropped_aligned \
      --afewva_annotations_dir /path/to/preprocessed_VA_annotations \
      --output_dir ./features/AFEWVA_baselines

  python extract_baseline_features.py \
      --method arcface \
      --afewva_frames_dir /path/to/cropped_aligned \
      --afewva_annotations_dir /path/to/preprocessed_VA_annotations \
      --output_dir ./features/AFEWVA_baselines
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# Dataset (reused from extract_features_afewva_standalone.py)
# ============================================================================
class AFEWVAClipDataset(Dataset):
    def __init__(self, frames_dir, annotations_dir):
        self.frames_dir = frames_dir
        train_dir = os.path.join(annotations_dir, "Train_Set")
        self.clips = []
        clip_set = set()
        for csv_file in sorted(os.listdir(train_dir)):
            if not csv_file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(train_dir, csv_file))
            if 'clip_id' in df.columns:
                for cid in df['clip_id'].unique():
                    clip_set.add(str(cid).zfill(3))
        for clip_id in sorted(clip_set):
            clip_dir = os.path.join(frames_dir, clip_id)
            if os.path.isdir(clip_dir):
                frames = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
                if frames:
                    self.clips.append((clip_id, len(frames)))
        print(f"[AFEWVAClipDataset] {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_id, _ = self.clips[idx]
        clip_dir = os.path.join(self.frames_dir, clip_id)
        frames = []
        for fp in sorted(glob.glob(os.path.join(clip_dir, "*.jpg"))):
            img = Image.open(fp).convert("RGB")
            frames.append(np.array(img))
        return clip_id, np.stack(frames)


# ============================================================================
# Transformer feature extraction (ViViT / VideoMAE)
# ============================================================================
def extract_transformer_features(clip_id, frames_np, processor, model, device,
                                  num_frames_per_chunk, chunk_stride=4, max_chunks=8):
    n_frames = len(frames_np)
    if n_frames < num_frames_per_chunk:
        repeats = (num_frames_per_chunk // n_frames) + 1
        frames_np = np.tile(frames_np, (repeats, 1, 1, 1))[:num_frames_per_chunk]
        n_frames = num_frames_per_chunk

    chunk_features = []
    for start in range(0, n_frames - num_frames_per_chunk + 1, chunk_stride):
        chunk = frames_np[start:start + num_frames_per_chunk]
        frames_list = [f for f in chunk]
        inputs = processor(images=frames_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        feat = outputs.last_hidden_state.mean(dim=1).cpu()
        chunk_features.append(feat)
        if len(chunk_features) >= max_chunks:
            break

    if not chunk_features:
        return None
    return torch.cat(chunk_features, dim=0)


# ============================================================================
# ArcFace feature extraction
# ============================================================================
def extract_arcface_features(clip_id, frames_np, model, device,
                              chunk_stride=4, max_chunks=8):
    """Extract ArcFace features. Per-frame → mean pool per chunk."""
    from torchvision import transforms

    # ArcFace expects 160x160 RGB, normalized
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    n_frames = len(frames_np)
    # Use chunks of 8 frames, mean pool within each chunk
    chunk_size = 8

    if n_frames < chunk_size:
        repeats = (chunk_size // n_frames) + 1
        frames_np = np.tile(frames_np, (repeats, 1, 1, 1))[:chunk_size]
        n_frames = chunk_size

    chunk_features = []
    for start in range(0, n_frames - chunk_size + 1, chunk_stride):
        chunk = frames_np[start:start + chunk_size]

        # Process each frame
        frame_feats = []
        for frame in chunk:
            img = Image.fromarray(frame)
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor)  # (1, 512)
            frame_feats.append(feat.cpu())

        # Mean pool over frames in chunk
        chunk_feat = torch.cat(frame_feats, dim=0).mean(dim=0, keepdim=True)  # (1, 512)
        chunk_features.append(chunk_feat)

        if len(chunk_features) >= max_chunks:
            break

    if not chunk_features:
        return None
    return torch.cat(chunk_features, dim=0)  # (num_chunks, 512)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline feature extraction for biometric comparison")
    parser.add_argument("--method", type=str, required=True,
                        choices=["random_init", "pretrained", "arcface"])
    parser.add_argument("--backbone", type=str, default="ViViT",
                        choices=["ViViT", "VideoMAE"],
                        help="Backbone (for random_init and pretrained only)")
    parser.add_argument("--afewva_frames_dir", type=str, required=True)
    parser.add_argument("--afewva_annotations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "features", "AFEWVA_baselines"))
    parser.add_argument("--chunk_stride", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    if args.method == "random_init":
        print(f"=== Loading {args.backbone} with RANDOM weights ===")
        if args.backbone == "ViViT":
            from transformers import VivitImageProcessor, VivitModel, VivitConfig
            model_name = 'google/vivit-b-16x2-kinetics400'
            processor = VivitImageProcessor.from_pretrained(model_name)
            config = VivitConfig.from_pretrained(model_name)
            model = VivitModel(config)  # random init, no pretrained weights
            num_frames = 32
        else:
            from transformers import VideoMAEImageProcessor, VideoMAEModel, VideoMAEConfig
            model_name = 'MCG-NJU/videomae-base'
            processor = VideoMAEImageProcessor.from_pretrained(model_name)
            config = VideoMAEConfig.from_pretrained(model_name)
            model = VideoMAEModel(config)
            num_frames = 16
        feat_dim = 768
        output_name = f"{args.backbone}_random"

    elif args.method == "pretrained":
        print(f"=== Loading {args.backbone} with PRETRAINED weights (no VA fine-tuning) ===")
        if args.backbone == "ViViT":
            from transformers import VivitImageProcessor, VivitModel
            model_name = 'google/vivit-b-16x2-kinetics400'
            processor = VivitImageProcessor.from_pretrained(model_name)
            model = VivitModel.from_pretrained(model_name)
            num_frames = 32
        else:
            from transformers import VideoMAEImageProcessor, VideoMAEModel
            model_name = 'MCG-NJU/videomae-base'
            processor = VideoMAEImageProcessor.from_pretrained(model_name)
            model = VideoMAEModel.from_pretrained(model_name)
            num_frames = 16
        feat_dim = 768
        output_name = f"{args.backbone}_pretrained"

    elif args.method == "arcface":
        print("=== Loading ArcFace (InceptionResnetV1, VGGFace2) ===")
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='vggface2').eval()
        processor = None
        num_frames = 8  # not used for transformer
        feat_dim = 512
        output_name = "ArcFace"

    model = model.to(device)
    model.eval()

    # === Dataset ===
    dataset = AFEWVAClipDataset(args.afewva_frames_dir, args.afewva_annotations_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=lambda batch: batch)

    # === Output dir ===
    output_feature_dir = os.path.join(args.output_dir, output_name)
    os.makedirs(output_feature_dir, exist_ok=True)

    print(f"  Feature dim: {feat_dim}")
    print(f"  Output: {output_feature_dir}")

    # === Extract ===
    n_extracted = 0
    n_skipped = 0

    for batch in tqdm(loader, desc=f"Extracting {output_name}"):
        for clip_id, frames_np in batch:
            save_path = os.path.join(output_feature_dir, f"{clip_id}.pt")
            if os.path.exists(save_path):
                n_skipped += 1
                continue

            if args.method == "arcface":
                features = extract_arcface_features(
                    clip_id, frames_np, model, device,
                    args.chunk_stride, args.max_chunks
                )
            else:
                features = extract_transformer_features(
                    clip_id, frames_np, processor, model, device,
                    num_frames, args.chunk_stride, args.max_chunks
                )

            if features is not None:
                torch.save(features, save_path)
                n_extracted += 1

    print(f"\n=== Done: {output_name} ===")
    print(f"  Extracted: {n_extracted}, Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
