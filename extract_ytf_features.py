#!/usr/bin/env python3
"""
YTF (YouTube Faces DB) feature extraction for biometric verification.

Extracts features from frozen backbones on YTF aligned face crops:
  - ViViT            (Aff-wild2 VA fine-tuned) → 768-dim
  - VideoMAE         (Aff-wild2 VA fine-tuned) → 768-dim
  - ArcFace          (VGGFace2 pretrained)     → 512-dim
  - ViViT_pretrained (Kinetics-400)            → 768-dim
  - VideoMAE_pretrained                        → 768-dim

YTF layout (after extraction):
  data/YTF/aligned_images_DB/{Person}/{video_id}/{Person}.{video_id}.{frame}.jpg

Output layout (mirrors raw to keep Person/video_id key for splits.txt):
  features/YTF/{Backbone}/{Person}/{video_id}.pt

Usage:
  python extract_ytf_features.py --backbone ViViT
  python extract_ytf_features.py --backbone VideoMAE
  python extract_ytf_features.py --backbone ArcFace
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def list_videos(frames_dir):
    """Return [(person, video_id, video_dir), ...]."""
    videos = []
    for person in sorted(os.listdir(frames_dir)):
        person_dir = os.path.join(frames_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for vid in sorted(os.listdir(person_dir)):
            vid_dir = os.path.join(person_dir, vid)
            if not os.path.isdir(vid_dir):
                continue
            videos.append((person, vid, vid_dir))
    return videos


def load_video_frames(vid_dir):
    """Load all JPG frames in a video directory.

    Returns a Python list of np.ndarray (per-frame shapes may differ — YTF
    aligned crops vary per frame, e.g. 279x279..326x326). Downstream processors
    handle resizing; we must NOT np.stack here.
    """
    frames = []
    for fp in sorted(glob.glob(os.path.join(vid_dir, "*.jpg"))):
        img = Image.open(fp).convert("RGB")
        frames.append(np.array(img))
    return frames if frames else None


def extract_transformer(frames, processor, model, device,
                         num_frames, chunk_stride, max_chunks):
    """ViViT/VideoMAE clip feature extraction. `frames` is a list of np.ndarray."""
    n = len(frames)
    if n < num_frames:
        reps = (num_frames // n) + 1
        frames = (frames * reps)[:num_frames]
        n = num_frames

    chunks = []
    for start in range(0, n - num_frames + 1, chunk_stride):
        clip = frames[start:start + num_frames]
        inputs = processor(images=clip, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        feat = out.last_hidden_state.mean(dim=1).cpu()
        chunks.append(feat)
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def extract_arcface(frames, model, device, chunk_stride, max_chunks):
    """ArcFace (InceptionResnetV1) feature extraction. `frames` is a list of np.ndarray."""
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    chunk_size = 8
    n = len(frames)
    if n < chunk_size:
        reps = (chunk_size // n) + 1
        frames = (frames * reps)[:chunk_size]
        n = chunk_size

    chunks = []
    for start in range(0, n - chunk_size + 1, chunk_stride):
        clip = frames[start:start + chunk_size]
        frame_feats = []
        for frame in clip:
            img = Image.fromarray(frame)
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(t)
            frame_feats.append(f.cpu())
        chunks.append(torch.cat(frame_feats, dim=0).mean(dim=0, keepdim=True))
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Backbone loaders
# ---------------------------------------------------------------------------
def load_vivit_va(device):
    from transformers import VivitImageProcessor, VivitModel
    name = 'google/vivit-b-16x2-kinetics400'
    processor = VivitImageProcessor.from_pretrained(name)
    model = VivitModel.from_pretrained(name)

    weights = os.path.join(PROJECT_ROOT, "experiments", "DEFAULT",
                           "ViViT_Finetuned_on_Affwild2", "vivit_model_best.pt")
    if os.path.exists(weights):
        print(f"[ViViT] Loading VA-finetuned weights: {weights}")
        sd = torch.load(weights, map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=True)
    else:
        print(f"[ViViT] WARNING: finetuned weights not found — using Kinetics-400")
    return processor, model.to(device).eval(), 32, 4, 8  # num_frames, stride, max_chunks


def load_vivit_pretrained(device):
    from transformers import VivitImageProcessor, VivitModel
    name = 'google/vivit-b-16x2-kinetics400'
    processor = VivitImageProcessor.from_pretrained(name)
    model = VivitModel.from_pretrained(name)
    print(f"[ViViT_pretrained] Kinetics-400 weights")
    return processor, model.to(device).eval(), 32, 4, 8


def load_videomae_va(device):
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    name = 'MCG-NJU/videomae-base'
    processor = VideoMAEImageProcessor.from_pretrained(name)
    model = VideoMAEModel.from_pretrained(name)

    weights = os.path.join(PROJECT_ROOT, "experiments", "DEFAULT",
                           "VideoMAE_Finetuned_on_Affwild2", "videomae_model_best.pt")
    if os.path.exists(weights):
        print(f"[VideoMAE] Loading VA-finetuned weights: {weights}")
        sd = torch.load(weights, map_location='cpu', weights_only=False)
        model.load_state_dict(sd, strict=True)
    else:
        print(f"[VideoMAE] WARNING: finetuned weights not found")
    return processor, model.to(device).eval(), 16, 6, 8


def load_videomae_pretrained(device):
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    name = 'MCG-NJU/videomae-base'
    processor = VideoMAEImageProcessor.from_pretrained(name)
    model = VideoMAEModel.from_pretrained(name)
    print(f"[VideoMAE_pretrained] base weights")
    return processor, model.to(device).eval(), 16, 6, 8


def load_arcface(device):
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print(f"[ArcFace] InceptionResnetV1 / VGGFace2")
    return None, model, None, 4, 8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BACKBONES = {
    'ViViT':              load_vivit_va,
    'ViViT_pretrained':   load_vivit_pretrained,
    'VideoMAE':           load_videomae_va,
    'VideoMAE_pretrained': load_videomae_pretrained,
    'ArcFace':            load_arcface,
}


def main():
    p = argparse.ArgumentParser(description="YTF feature extraction")
    p.add_argument("--frames_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "YTF", "aligned_images_DB"))
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "features", "YTF"))
    p.add_argument("--backbone", type=str, required=True, choices=list(BACKBONES.keys()))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frames: {args.frames_dir}")

    processor, model, num_frames, chunk_stride, max_chunks = BACKBONES[args.backbone](device)

    videos = list_videos(args.frames_dir)
    persons = sorted({v[0] for v in videos})
    print(f"YTF: {len(videos)} videos / {len(persons)} persons")

    out_root = os.path.join(args.output_dir, args.backbone)
    os.makedirs(out_root, exist_ok=True)

    n_done, n_skip, n_fail = 0, 0, 0
    for person, vid, vid_dir in tqdm(videos, desc=f"Extracting {args.backbone}"):
        person_out = os.path.join(out_root, person)
        os.makedirs(person_out, exist_ok=True)
        save_path = os.path.join(person_out, f"{vid}.pt")
        if os.path.exists(save_path):
            n_skip += 1
            continue

        frames = load_video_frames(vid_dir)
        if frames is None:
            n_fail += 1
            continue

        if args.backbone == 'ArcFace':
            feat = extract_arcface(frames, model, device, chunk_stride, max_chunks)
        else:
            feat = extract_transformer(frames, processor, model, device,
                                        num_frames, chunk_stride, max_chunks)

        if feat is not None:
            torch.save(feat, save_path)
            n_done += 1
        else:
            n_fail += 1

    print(f"\n=== {args.backbone} done ===")
    print(f"  extracted: {n_done}, skipped: {n_skip}, failed: {n_fail}")


if __name__ == "__main__":
    main()
