#!/usr/bin/env python3
"""
Multi-layer feature extraction for AFEW-VA.

Extracts and saves the intermediate-layer hidden states of ViViT/VideoMAE.
Compared with using only the last layer, concatenating features from several layers
can capture more identity-discriminative information.

Output layout:
  features/AFEWVA_multilayer/{backbone}_layer{i}/{clip_id}.pt
  each file: (num_chunks, 768) — per-layer features

Usage:
  python extract_features_multilayer.py \
      --backbone ViViT \
      --afewva_frames_dir /path/to/cropped_aligned \
      --afewva_annotations_dir /path/to/preprocessed_VA_annotations \
      --finetuned_weights experiments/DEFAULT/ViViT_Finetuned_on_Affwild2/vivit_model_best.pt \
      --layers -1 -2 -3 -4 \
      --output_dir ./features/AFEWVA_multilayer
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


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


def extract_multilayer(clip_id, frames_np, processor, model, device,
                       num_frames_per_chunk, layers, chunk_stride=4, max_chunks=8):
    """Extract features from multiple layers for one clip."""
    n_frames = len(frames_np)
    if n_frames < num_frames_per_chunk:
        repeats = (num_frames_per_chunk // n_frames) + 1
        frames_np = np.tile(frames_np, (repeats, 1, 1, 1))[:num_frames_per_chunk]
        n_frames = num_frames_per_chunk

    # {layer_idx: [chunk_features]}
    layer_features = {l: [] for l in layers}

    for start in range(0, n_frames - num_frames_per_chunk + 1, chunk_stride):
        chunk = frames_np[start:start + num_frames_per_chunk]
        frames_list = [f for f in chunk]

        inputs = processor(images=frames_list, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (1, num_tokens, 768)
        n_layers = len(hidden_states)

        for l in layers:
            # Convert negative index to actual
            actual_idx = l if l >= 0 else n_layers + l
            if 0 <= actual_idx < n_layers:
                feat = hidden_states[actual_idx].mean(dim=1).cpu()  # (1, 768)
                layer_features[l].append(feat)

        if len(layer_features[layers[0]]) >= max_chunks:
            break

    result = {}
    for l in layers:
        if layer_features[l]:
            result[l] = torch.cat(layer_features[l], dim=0)  # (num_chunks, 768)

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-layer feature extraction")
    parser.add_argument("--backbone", type=str, required=True, choices=["ViViT", "VideoMAE", "VideoMAE_Large"])
    parser.add_argument("--afewva_frames_dir", type=str, required=True)
    parser.add_argument("--afewva_annotations_dir", type=str, required=True)
    parser.add_argument("--finetuned_weights", type=str, default=None)
    parser.add_argument("--no_finetune", action='store_true',
                        help="Skip loading finetuned weights — use base pretrained backbone (Kinetics400 ViViT / VideoMAE base)")
    parser.add_argument("--layers", type=int, nargs='+', default=[-1, -2, -3, -4],
                        help="Layer indices to extract (negative = from end)")
    parser.add_argument("--chunk_stride", type=int, default=4)
    parser.add_argument("--max_chunks", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "features", "AFEWVA_multilayer"))
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load backbone
    if args.backbone == "ViViT":
        from transformers import VivitImageProcessor, VivitModel
        model_name = 'google/vivit-b-16x2-kinetics400'
        processor = VivitImageProcessor.from_pretrained(model_name)
        model = VivitModel.from_pretrained(model_name)
        num_frames = 32
    elif args.backbone == "VideoMAE_Large":
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        model_name = 'MCG-NJU/videomae-large'
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name)
        num_frames = 16
    else:
        from transformers import VideoMAEImageProcessor, VideoMAEModel
        model_name = 'MCG-NJU/videomae-base'
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name)
        num_frames = 16

    # Dynamic num_blocks (12 for ViT/VideoMAE-base, 24 for VideoMAE-large)
    num_blocks = model.config.num_hidden_layers
    # If user didn't specify --layers explicitly, default = all blocks (negative-indexed)
    # (parser default is [-1,-2,-3,-4] which is fine for partial extraction)
    print(f"[backbone] {model_name} | num_blocks={num_blocks} | layers={args.layers}")

    if args.no_finetune:
        print(f"[--no_finetune] Skipping fine-tuned weights — using base pretrained {args.backbone}")
    elif args.finetuned_weights and os.path.exists(args.finetuned_weights):
        print(f"Loading fine-tuned weights: {args.finetuned_weights}")
        state_dict = torch.load(args.finetuned_weights, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()

    # Create output directories per layer
    for l in args.layers:
        layer_dir = os.path.join(args.output_dir, f"{args.backbone}_layer{l}")
        os.makedirs(layer_dir, exist_ok=True)

    # Dataset
    dataset = AFEWVAClipDataset(args.afewva_frames_dir, args.afewva_annotations_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=lambda batch: batch)

    print(f"\n=== Extracting {args.backbone} multi-layer features ===")
    print(f"  Layers: {args.layers}")

    n_extracted = 0
    n_skipped = 0

    for batch in tqdm(loader, desc=f"Extracting {args.backbone}"):
        for clip_id, frames_np in batch:
            # Check if already extracted (check first layer)
            first_layer_path = os.path.join(
                args.output_dir, f"{args.backbone}_layer{args.layers[0]}", f"{clip_id}.pt")
            if os.path.exists(first_layer_path):
                n_skipped += 1
                continue

            result = extract_multilayer(
                clip_id, frames_np, processor, model, device,
                num_frames, args.layers, args.chunk_stride, args.max_chunks
            )

            for l, feat in result.items():
                save_path = os.path.join(args.output_dir, f"{args.backbone}_layer{l}", f"{clip_id}.pt")
                torch.save(feat, save_path)

            n_extracted += 1

    print(f"\n=== Done ===")
    print(f"  Extracted: {n_extracted} clips, Skipped: {n_skipped}")


if __name__ == "__main__":
    main()
