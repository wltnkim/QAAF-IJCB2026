#!/usr/bin/env python3
"""
YTF Phase-A baseline feature extraction.

Phase A baselines (3 models ported from CREMA-D pipeline):
  1. AdaFace IR-101 (Hard, CVPR'22)
  2. JMT R2D1 (Soft, Aff-wild2 fine-tuned)
  3. MAE-DFER (Soft video FER, ACM MM'23)

(ViViT_pretrained, VideoMAE_pretrained are handled by extract_ytf_features.py.)

YTF data layout: data/YTF/aligned_images_DB/{Person}/{video_id}/*.jpg
Output layout:   features/YTF/{Backbone}/{Person}/{video_id}.pt

Usage:
  python extract_ytf_baselines.py --method adaface
  python extract_ytf_baselines.py --method r2d1
  python extract_ytf_baselines.py --method mae_dfer
  python extract_ytf_baselines.py --method all
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
from functools import partial

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.join(PROJECT_ROOT, "pretrained_baselines")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "data", "YTF", "aligned_images_DB")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "features", "YTF")

sys.path.insert(0, PROJECT_ROOT)


def list_videos():
    videos = []
    for person in sorted(os.listdir(FRAMES_DIR)):
        pd = os.path.join(FRAMES_DIR, person)
        if not os.path.isdir(pd):
            continue
        for vid in sorted(os.listdir(pd)):
            vd = os.path.join(pd, vid)
            if os.path.isdir(vd):
                videos.append((person, vid, vd))
    return videos


def load_video_frames(vid_dir):
    """List of np.ndarray frames (variable per-frame shapes in YTF aligned)."""
    frames = []
    for fp in sorted(glob.glob(os.path.join(vid_dir, "*.jpg"))):
        img = Image.open(fp).convert("RGB")
        frames.append(np.array(img))
    return frames if frames else None


def sample_frames(frames, max_frames=32):
    stride = max(1, len(frames) // max_frames)
    return frames[::stride][:max_frames]


def save_and_chunk(frame_feats, save_path, chunk_size=4):
    """Mean-pool over `chunk_size` frames, save as (n_chunks, D)."""
    if not frame_feats:
        return False
    all_feats = torch.cat(frame_feats, dim=0)
    chunks = []
    for start in range(0, all_feats.shape[0], chunk_size):
        chunks.append(all_feats[start:start + chunk_size].mean(dim=0, keepdim=True))
    torch.save(torch.cat(chunks, dim=0), save_path)
    return True


def frame_level_extract(model, preprocess, output_name, desc,
                        model_forward=None, max_frames=32, chunk_size=4, device=None):
    """Per-frame feature extraction (for ArcFace-like models)."""
    videos = list_videos()
    out_root = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(out_root, exist_ok=True)
    n_done, n_skip = 0, 0

    for person, vid, vid_dir in tqdm(videos, desc=desc):
        person_out = os.path.join(out_root, person)
        os.makedirs(person_out, exist_ok=True)
        save_path = os.path.join(person_out, f"{vid}.pt")
        if os.path.exists(save_path):
            n_skip += 1
            continue
        frames = load_video_frames(vid_dir)
        if not frames:
            continue
        sampled = sample_frames(frames, max_frames)
        frame_feats = []
        for f in sampled:
            img = Image.fromarray(f)
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model_forward(model, t) if model_forward else model(t)
                if isinstance(out, tuple):
                    out = out[0]
            frame_feats.append(out.cpu())
        if save_and_chunk(frame_feats, save_path, chunk_size):
            n_done += 1

    print(f"  Done {output_name}: {n_done} extracted, {n_skip} skipped")


def video_clip_extract(model, preprocess, output_name, desc,
                       clip_length, chunk_stride, max_chunks, device,
                       forward_fn=None):
    """Video-clip (stacked frames) extraction (for R2D1/I3D/MAE-DFER)."""
    videos = list_videos()
    out_root = os.path.join(OUTPUT_ROOT, output_name)
    os.makedirs(out_root, exist_ok=True)
    n_done, n_skip = 0, 0

    for person, vid, vid_dir in tqdm(videos, desc=desc):
        person_out = os.path.join(out_root, person)
        os.makedirs(person_out, exist_ok=True)
        save_path = os.path.join(person_out, f"{vid}.pt")
        if os.path.exists(save_path):
            n_skip += 1
            continue
        frames = load_video_frames(vid_dir)
        if not frames:
            continue

        n = len(frames)
        if n < clip_length:
            reps = (clip_length // n) + 1
            frames = (frames * reps)[:clip_length]
            n = clip_length

        chunks = []
        for start in range(0, n - clip_length + 1, chunk_stride):
            clip = frames[start:start + clip_length]
            tensors = torch.stack([preprocess(Image.fromarray(f)) for f in clip])  # (T, C, H, W)
            video_input = tensors.permute(1, 0, 2, 3).unsqueeze(0).to(device)   # (1, C, T, H, W)
            with torch.no_grad():
                feat = forward_fn(model, video_input) if forward_fn else model(video_input)
            chunks.append(feat.cpu().view(1, -1))
            if len(chunks) >= max_chunks:
                break

        if chunks:
            torch.save(torch.cat(chunks, dim=0), save_path)
            n_done += 1

    print(f"  Done {output_name}: {n_done} extracted, {n_skip} skipped")


# ---------------------------------------------------------------------------
# 1. AdaFace IR-101 (Hard, CVPR'22)
# ---------------------------------------------------------------------------
def run_adaface(device):
    from torchvision import transforms

    adaface_dir = os.path.join(BASELINES_DIR, "AdaFace")
    sys.path.insert(0, adaface_dir)
    import net

    model = net.build_model('ir_101')
    ckpt = os.path.join(adaface_dir, "pretrained", "adaface_ir101_webface12m.ckpt")
    sd = torch.load(ckpt, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"[AdaFace] Loaded: {ckpt}")
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def forward_fn(m, t):
        feat, _ = m(t)
        return feat

    frame_level_extract(model, preprocess, "AdaFace_IR101", "AdaFace",
                        model_forward=forward_fn, device=device)


# ---------------------------------------------------------------------------
# 2. JMT R2D1 (Aff-wild2 fine-tuned, Soft)
# ---------------------------------------------------------------------------
def run_r2d1(device):
    from torchvision import transforms
    from torchvision.models.video import r2plus1d_18

    model = r2plus1d_18(pretrained=True)
    model.fc = nn.Identity()

    weights = os.path.join(PROJECT_ROOT, "PretrainedWeights", "vision_r2d1.pt")
    if os.path.exists(weights):
        sd = torch.load(weights, map_location='cpu', weights_only=False)
        msd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"[R2D1] Loaded {len(filtered)} keys from {weights}")
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989]),
    ])

    video_clip_extract(model, preprocess, "R2D1", "R2D1",
                       clip_length=8, chunk_stride=4, max_chunks=8, device=device)


# ---------------------------------------------------------------------------
# 3. MAE-DFER (video FER, ACM MM'23)
# ---------------------------------------------------------------------------
def run_mae_dfer(device):
    from torchvision import transforms

    mae_dir = os.path.join(BASELINES_DIR, "MAE-DFER")
    sys.path.insert(0, mae_dir)
    from modeling_finetune import VisionTransformer

    model = VisionTransformer(
        img_size=160, patch_size=16, embed_dim=512, num_heads=8,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=7, all_frames=16, tubelet_size=2,
        drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
        use_mean_pooling=True, init_scale=0.001, depth=16,
        attn_type='local_global', lg_region_size=(2, 5, 10),
        lg_first_attn_type='self', lg_third_attn_type='cross',
        lg_classify_token_type='region',
    )

    ckpt = os.path.join(mae_dir, "pretrained", "mae_dfer_dfew.pth")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu')
        if 'model' in sd:
            sd = sd['model']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"[MAE-DFER] Loaded: {ckpt}")
    model = model.to(device).eval()

    features_store = {}
    def hook_fn(module, inp, out):
        features_store['feat'] = inp[0].detach()
    model.head.register_forward_hook(hook_fn)

    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    num_frames = 16
    videos = list_videos()
    out_root = os.path.join(OUTPUT_ROOT, "MAE_DFER")
    os.makedirs(out_root, exist_ok=True)
    n_done, n_skip = 0, 0

    for person, vid, vid_dir in tqdm(videos, desc="MAE-DFER"):
        person_out = os.path.join(out_root, person)
        os.makedirs(person_out, exist_ok=True)
        save_path = os.path.join(person_out, f"{vid}.pt")
        if os.path.exists(save_path):
            n_skip += 1
            continue
        frames = load_video_frames(vid_dir)
        if not frames:
            continue
        n_f = len(frames)
        stride = max(1, n_f // (num_frames * 2))
        sampled = frames[::stride][:num_frames * 2]

        chunks = []
        for start in range(0, len(sampled), num_frames):
            chunk = sampled[start:start + num_frames]
            while len(chunk) < num_frames:
                chunk = chunk + [chunk[-1]]
            tensors = torch.stack([preprocess(Image.fromarray(f)) for f in chunk])
            video_input = tensors.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            with torch.no_grad():
                try:
                    _ = model(video_input)
                    if 'feat' in features_store:
                        chunks.append(features_store['feat'].cpu())
                except Exception:
                    pass
            if len(chunks) >= 2:
                break

        if chunks:
            torch.save(torch.cat(chunks, dim=0), save_path)
            n_done += 1

    print(f"  Done MAE_DFER: {n_done} extracted, {n_skip} skipped")


METHODS = {
    'adaface': run_adaface,
    'r2d1': run_r2d1,
    'mae_dfer': run_mae_dfer,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="all",
                   choices=list(METHODS.keys()) + ["all"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.method == "all":
        for name, fn in METHODS.items():
            print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
            try:
                fn(device)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
    else:
        METHODS[args.method](device)


if __name__ == "__main__":
    main()
