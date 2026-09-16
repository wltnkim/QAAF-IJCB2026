#!/usr/bin/env python3
"""
Table 5 extension: layer-by-layer verification on AFEW-VA and YTF using the
QAG+AMD single-best checkpoints (ViViT s8, VideoMAE s7).

Four stages per backbone:
  1. Backbone                  (768-dim, raw frozen features)
  2. After vision_projection   (512-dim, Linear(768→512) output)
  3. After fusion encoder      (512-dim, mm_transformer output before regressor)
  4. VA output                 (2-dim, concat of valence and arousal heads)

For each stage, per-clip features are mean-pooled over time, then verification
is computed per dataset (AFEW-VA: actor enrollment vs probe; YTF: 10-fold 5K pairs).

The vision_projection, QAG, and fusion_model are loaded from saved_models_da/
checkpoints that correspond to best single-seed QAG+AMD runs.

Audio is zero-padded (both datasets are video-only, matching the Table 5 setup
used for AFEW-VA in the paper). Cosine similarity is scale-invariant under
normalize, so whether QAG is applied does not change Stage-2 EER.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine as cosine_dist
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.two_transformers_da import Two_transformers_DA
from losses.da_losses import QualityAwareGating
from paths import AFEWVA_SPLIT


# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "ViViT":    "saved_models_da/03092026_152809/best_da_model.pt",   # s8
    "VideoMAE": "saved_models_da/03092026_161936/best_da_model.pt",   # s7
}

HP_DEFAULT = {
    "v_dropout": 0.2, "a_dropout": 0.2,
    "num_heads": 4,   "num_layers": 2,
    "fusion_type": "TRANSFORMER", "output_format": "SELF_ATTEN",
    "vision_in_ft_backbone": 768,  # both ViViT and VideoMAE
    "qag_hidden_dim": 64,
}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def build_and_load(ckpt_path, device):
    """Build fusion_model + vision_projection + QAG and load checkpoint."""
    fusion_model = Two_transformers_DA(
        v_dropout=HP_DEFAULT["v_dropout"],
        a_dropout=HP_DEFAULT["a_dropout"],
        num_heads=HP_DEFAULT["num_heads"],
        num_layers=HP_DEFAULT["num_layers"],
        fusion_type=HP_DEFAULT["fusion_type"],
        output_format=HP_DEFAULT["output_format"],
        vision_in_ft=512,  # after vision_projection
    ).to(device)

    vision_projection = torch.nn.Linear(HP_DEFAULT["vision_in_ft_backbone"], 512).to(device)
    qag = QualityAwareGating(input_dim=512, hidden_dim=HP_DEFAULT["qag_hidden_dim"]).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fusion_model.load_state_dict(ckpt["fusion_model_state_dict"])
    vision_projection.load_state_dict(ckpt["vision_projection_state_dict"])
    if ckpt.get("quality_gating_state_dict") is not None:
        qag.load_state_dict(ckpt["quality_gating_state_dict"])
    fusion_model.eval(); vision_projection.eval(); qag.eval()
    return fusion_model, vision_projection, qag


# ---------------------------------------------------------------------------
# Per-stage feature extraction for one clip
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_stages(vis_feat_tensor, fusion_model, vision_projection, qag, device,
                   seq_len_align=8):
    """
    Args:
        vis_feat_tensor: (T, 768) tensor on CPU
    Returns:
        dict of 4 mean-pooled stage vectors:
          'backbone' (768,), 'projection' (512,), 'encoder' (512,), 'va' (2,)
    """
    v = vis_feat_tensor.unsqueeze(0).to(device)           # (1, T, 768)
    T = v.shape[1]

    # Stage 1: mean-pool raw backbone features
    backbone_vec = v.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

    # Stage 2: after vision_projection (512)
    # Training does F.normalize BEFORE projection in Two_transformers_DA; but
    # normalization is scale-only, and we mean-pool before cosine-sim → no EER
    # change. We feed the projection directly matching eval_noisy_robustness flow:
    #   vis_feat = vision_projection(vis_feat) → QAG → fusion_model
    # Inside fusion_model another F.normalize is applied.
    v_proj = vision_projection(v)                          # (1, T, 512)
    projection_vec = v_proj.mean(dim=1).squeeze(0).cpu().numpy()

    # Create zero audio at matching sequence length
    a_zero = torch.zeros_like(v_proj)                      # (1, T, 512)

    # Apply QAG (active at inference per paper)
    v_gated, a_gated, _ = qag(v_proj, a_zero)

    # Stage 3: after mm_transformer (need hook since forward returns dict
    # of pred_v/pred_a, but fused features go into vregressor/aregressor)
    captured = {}
    def _hook(module, inputs, output):
        captured["fused"] = output.detach()
    h = fusion_model.mm_transformer.register_forward_hook(_hook)
    try:
        out = fusion_model(v_gated, a_gated)
    finally:
        h.remove()

    fused = captured["fused"]                              # (1, T, 512)
    encoder_vec = fused.mean(dim=1).squeeze(0).cpu().numpy()

    # Stage 4: VA output (concat valence + arousal predictions per timestep)
    pred_v = out["pred_v"]   # (1, T)
    pred_a = out["pred_a"]   # (1, T)
    va = torch.stack([pred_v, pred_a], dim=-1)             # (1, T, 2)
    va_vec = va.mean(dim=1).squeeze(0).cpu().numpy()

    return {
        "backbone":   backbone_vec,
        "projection": projection_vec,
        "encoder":    encoder_vec,
        "va":         va_vec,
    }


# ---------------------------------------------------------------------------
# Dataset-agnostic verification utilities
# ---------------------------------------------------------------------------
def compute_eer_auc_from_arrays(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    thr = np.linspace(scores.min(), scores.max(), 2000)
    far = np.array([np.mean(scores[labels == 0] >= t) for t in thr])
    frr = np.array([np.mean(scores[labels == 1] < t) for t in thr])
    diff = far - frr
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        eer = 0.5
    else:
        i = sc[0]
        den = diff[i+1] - diff[i]
        a = -diff[i]/den if den != 0 else 0.5
        eer = far[i] + a * (far[i+1] - far[i])
    order = np.argsort(far)
    auc = float(np.trapz(1 - frr[order], far[order]))
    return float(eer), auc


# --------------------------------- AFEW-VA ---------------------------------
def eval_afewva(stage_feats, split_info):
    """Enrollment-probe verification on AFEW-VA given {clip_id: vec}."""
    enroll, probe = {}, {}
    for actor, info in split_info["actors"].items():
        tr = [str(c).zfill(3) for c in info["train_clips"]]
        te = [str(c).zfill(3) for c in info["test_clips"]]
        tv = [stage_feats[c] for c in tr if c in stage_feats]
        if not tv:
            continue
        enroll[actor] = np.mean(tv, axis=0)
        pl = [(c, stage_feats[c]) for c in te if c in stage_feats]
        if pl:
            probe[actor] = pl
    actors = sorted(enroll.keys())
    scores, labels = [], []
    for a in actors:
        if a not in probe:
            continue
        e = enroll[a]
        for _, pv in probe[a]:
            scores.append(1.0 - cosine_dist(e, pv)); labels.append(1)
            for o in actors:
                if o == a:
                    continue
                scores.append(1.0 - cosine_dist(enroll[o], pv)); labels.append(0)
    return compute_eer_auc_from_arrays(scores, labels)


# ----------------------------------- YTF -----------------------------------
def eval_ytf(stage_feats, pairs, n_splits=10):
    """10-fold verification on YTF given {video_id: vec} and pair list."""
    by_split = {s: {"s": [], "l": []} for s in range(1, n_splits + 1)}
    miss = 0
    for split_id, v1, v2, is_same in pairs:
        if v1 not in stage_feats or v2 not in stage_feats:
            miss += 1; continue
        sim = 1.0 - cosine_dist(stage_feats[v1], stage_feats[v2])
        by_split[split_id]["s"].append(sim); by_split[split_id]["l"].append(is_same)
    fold_eers, fold_aucs = [], []
    for s in range(1, n_splits + 1):
        if not by_split[s]["s"]:
            continue
        e, a = compute_eer_auc_from_arrays(by_split[s]["s"], by_split[s]["l"])
        fold_eers.append(e); fold_aucs.append(a)
    all_s, all_l = [], []
    for s in range(1, n_splits + 1):
        all_s.extend(by_split[s]["s"]); all_l.extend(by_split[s]["l"])
    overall_eer, overall_auc = compute_eer_auc_from_arrays(all_s, all_l)
    return {
        "fold_mean_eer": float(np.mean(fold_eers)) if fold_eers else float("nan"),
        "fold_std_eer":  float(np.std(fold_eers))  if fold_eers else float("nan"),
        "fold_mean_auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "fold_std_auc":  float(np.std(fold_aucs))  if fold_aucs else float("nan"),
        "overall_eer":   overall_eer,
        "overall_auc":   overall_auc,
        "n_pairs":       len(all_s),
        "n_missing":     miss,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_clip_features_flat(dir_path):
    """Return {clip_id: tensor(T, 768)} for AFEW-VA (flat directory)."""
    out = {}
    for fn in sorted(os.listdir(dir_path)):
        if not fn.endswith(".pt"):
            continue
        cid = os.path.splitext(fn)[0]
        data = torch.load(os.path.join(dir_path, fn), map_location="cpu", weights_only=False)
        if isinstance(data, torch.Tensor):
            out[cid] = data.float()
    return out


def load_clip_features_nested(dir_path):
    """Return {'Person/video_id': tensor(T, 768)} for YTF (nested dir)."""
    out = {}
    for person in sorted(os.listdir(dir_path)):
        pd = os.path.join(dir_path, person)
        if not os.path.isdir(pd):
            continue
        for fn in sorted(os.listdir(pd)):
            if not fn.endswith(".pt"):
                continue
            vid = os.path.splitext(fn)[0]
            data = torch.load(os.path.join(pd, fn), map_location="cpu", weights_only=False)
            if isinstance(data, torch.Tensor):
                out[f"{person}/{vid}"] = data.float()
    return out


def parse_ytf_splits(meta_file):
    m = loadmat(meta_file)
    video_names = m["video_names"]; splits = m["Splits"]
    def vn(i):
        v = video_names[i - 1, 0]
        return str(v[0]) if hasattr(v, "__len__") else str(v)
    pairs = []
    npp, _, ns = splits.shape
    for s in range(ns):
        for i in range(npp):
            pairs.append((s + 1, vn(int(splits[i, 0, s])),
                                 vn(int(splits[i, 1, s])),
                                 int(splits[i, 2, s])))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_backbone(backbone_name, device, afewva_split, ytf_pairs,
                 afewva_feats_raw, ytf_feats_raw):
    print(f"\n{'='*70}\n### Backbone: {backbone_name}\n{'='*70}")
    fusion_model, vision_projection, qag = build_and_load(CHECKPOINTS[backbone_name], device)
    print(f"Loaded checkpoint: {CHECKPOINTS[backbone_name]}")

    stage_names = ["backbone", "projection", "encoder", "va"]
    stage_dims  = {"backbone": 768, "projection": 512, "encoder": 512, "va": 2}

    # --------- AFEW-VA ---------
    print(f"\n[AFEW-VA]  extracting stages for {len(afewva_feats_raw)} clips ...")
    afewva_stage_feats = {s: {} for s in stage_names}
    for cid, t in afewva_feats_raw.items():
        stages = extract_stages(t, fusion_model, vision_projection, qag, device)
        for s in stage_names:
            afewva_stage_feats[s][cid] = stages[s]

    print(f"{'Stage':<12} {'Dim':>4} {'EER':>7} {'AUC':>7}")
    afewva_results = {}
    for s in stage_names:
        eer, auc = eval_afewva(afewva_stage_feats[s], afewva_split)
        print(f"{s:<12} {stage_dims[s]:>4} {eer:>7.4f} {auc:>7.4f}")
        afewva_results[s] = {"dim": stage_dims[s], "eer": eer, "auc": auc}

    # --------- YTF ---------
    print(f"\n[YTF]  extracting stages for {len(ytf_feats_raw)} videos ...")
    ytf_stage_feats = {s: {} for s in stage_names}
    for cid, t in ytf_feats_raw.items():
        stages = extract_stages(t, fusion_model, vision_projection, qag, device)
        for s in stage_names:
            ytf_stage_feats[s][cid] = stages[s]

    print(f"{'Stage':<12} {'Dim':>4} {'mean_EER':>10} {'std':>6} {'mean_AUC':>10} {'std':>6} {'overall_EER':>12}")
    ytf_results = {}
    for s in stage_names:
        r = eval_ytf(ytf_stage_feats[s], ytf_pairs)
        print(f"{s:<12} {stage_dims[s]:>4} {r['fold_mean_eer']:>10.4f} "
              f"{r['fold_std_eer']:>6.4f} {r['fold_mean_auc']:>10.4f} "
              f"{r['fold_std_auc']:>6.4f} {r['overall_eer']:>12.4f}")
        ytf_results[s] = r
        ytf_results[s]["dim"] = stage_dims[s]

    return {"afewva": afewva_results, "ytf": ytf_results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--afewva_split",
                    default=AFEWVA_SPLIT)
    ap.add_argument("--ytf_meta", default="data/YTF/meta_data/meta_and_splits.mat")
    ap.add_argument("--out", default="table5_cross_dataset_results.json")
    ap.add_argument("--backbones", nargs="+", default=["ViViT", "VideoMAE"])
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.afewva_split) as f:
        afewva_split = json.load(f)
    ytf_pairs = parse_ytf_splits(args.ytf_meta)
    print(f"YTF pairs: {len(ytf_pairs)}")

    all_results = {}
    for bb in args.backbones:
        afewva_dir = f"features/AFEWVA/{bb}"
        ytf_dir    = f"features/YTF/{bb}"
        print(f"\nLoading features — AFEW-VA: {afewva_dir}")
        afewva_feats = load_clip_features_flat(afewva_dir)
        print(f"  {len(afewva_feats)} clips (dim={next(iter(afewva_feats.values())).shape})")
        print(f"Loading features — YTF: {ytf_dir}")
        ytf_feats = load_clip_features_nested(ytf_dir)
        print(f"  {len(ytf_feats)} videos")

        all_results[bb] = run_backbone(bb, device, afewva_split, ytf_pairs,
                                       afewva_feats, ytf_feats)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
