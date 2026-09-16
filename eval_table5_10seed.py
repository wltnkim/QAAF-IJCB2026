#!/usr/bin/env python3
"""
Table 5 extension with 10-seed average (matching the paper's convention for
projection/encoder/VA rows).

For each backbone × seed, load the QAG+AMD checkpoint, extract 4 stages per
clip, and compute verification EER/AUC on AFEW-VA and YTF. Then aggregate
across seeds (mean ± std).

Backbone row (768-dim raw features) does NOT depend on seed — the frozen
VA-finetuned backbone is shared. So the backbone row is computed once.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.spatial.distance import cosine as cosine_dist
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.two_transformers_da import Two_transformers_DA
from losses.da_losses import QualityAwareGating
from paths import AFEWVA_SPLIT


# ---------------------------------------------------------------------------
# Seed → run_dir mapping (from paper_run_mapping.json for QAG+AMD)
# ---------------------------------------------------------------------------
RUN_MAPPING = {
    "ViViT": {
        0: "03092026_192115", 1: "03092026_192517", 2: "03092026_193008",
        3: "03092026_144533", 4: "03092026_145400", 5: "03092026_150300",
        6: "03092026_151107", 7: "03092026_151927", 8: "03092026_152809",
        9: "03092026_153629",
    },
    "VideoMAE": {
        0: "03092026_193605", 1: "03092026_194203", 2: "03092026_194750",
        3: "03092026_154441", 4: "03092026_155334", 5: "03092026_160223",
        6: "03092026_161024", 7: "03092026_161936", 8: "03092026_162839",
        9: "03092026_163801",
    },
}

HP = dict(v_dropout=0.2, a_dropout=0.2, num_heads=4, num_layers=2,
          fusion_type="TRANSFORMER", output_format="SELF_ATTEN",
          vision_in_ft_backbone=768, qag_hidden_dim=64)


# ---------------------------------------------------------------------------
def build_and_load(ckpt_path, device):
    fusion = Two_transformers_DA(
        v_dropout=HP["v_dropout"], a_dropout=HP["a_dropout"],
        num_heads=HP["num_heads"], num_layers=HP["num_layers"],
        fusion_type=HP["fusion_type"], output_format=HP["output_format"],
        vision_in_ft=512,
    ).to(device)
    vproj = torch.nn.Linear(HP["vision_in_ft_backbone"], 512).to(device)
    qag = QualityAwareGating(input_dim=512, hidden_dim=HP["qag_hidden_dim"]).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    fusion.load_state_dict(ck["fusion_model_state_dict"])
    vproj.load_state_dict(ck["vision_projection_state_dict"])
    if ck.get("quality_gating_state_dict") is not None:
        qag.load_state_dict(ck["quality_gating_state_dict"])
    fusion.eval(); vproj.eval(); qag.eval()
    return fusion, vproj, qag


@torch.no_grad()
def extract_stages(v, fusion, vproj, qag, device):
    """v: (T, 768) CPU tensor → dict of 4 mean-pooled stage vectors."""
    v = v.unsqueeze(0).to(device)
    backbone = v.mean(1).squeeze(0).cpu().numpy()
    v_proj = vproj(v)
    projection = v_proj.mean(1).squeeze(0).cpu().numpy()
    a_zero = torch.zeros_like(v_proj)
    vg, ag, _ = qag(v_proj, a_zero)
    captured = {}
    def hk(m, i, o): captured["f"] = o.detach()
    h = fusion.mm_transformer.register_forward_hook(hk)
    try:
        out = fusion(vg, ag)
    finally:
        h.remove()
    encoder = captured["f"].mean(1).squeeze(0).cpu().numpy()
    va = torch.stack([out["pred_v"], out["pred_a"]], dim=-1)
    va_vec = va.mean(1).squeeze(0).cpu().numpy()
    return {"backbone": backbone, "projection": projection,
            "encoder": encoder, "va": va_vec}


def compute_eer_auc(scores, labels):
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
        i = sc[0]; den = diff[i+1] - diff[i]
        a = -diff[i]/den if den != 0 else 0.5
        eer = far[i] + a * (far[i+1] - far[i])
    order = np.argsort(far)
    auc = float(np.trapz(1 - frr[order], far[order]))
    return float(eer), auc


def eval_afewva(feats, split):
    enroll, probe = {}, {}
    for actor, info in split["actors"].items():
        tr = [str(c).zfill(3) for c in info["train_clips"]]
        te = [str(c).zfill(3) for c in info["test_clips"]]
        tv = [feats[c] for c in tr if c in feats]
        if not tv: continue
        enroll[actor] = np.mean(tv, 0)
        pl = [(c, feats[c]) for c in te if c in feats]
        if pl: probe[actor] = pl
    actors = sorted(enroll.keys())
    scores, labels = [], []
    for a in actors:
        if a not in probe: continue
        e = enroll[a]
        for _, pv in probe[a]:
            scores.append(1.0 - cosine_dist(e, pv)); labels.append(1)
            for o in actors:
                if o == a: continue
                scores.append(1.0 - cosine_dist(enroll[o], pv)); labels.append(0)
    return compute_eer_auc(scores, labels)


def eval_ytf(feats, pairs, n_splits=10):
    by_split = {s: {"s": [], "l": []} for s in range(1, n_splits + 1)}
    for split_id, v1, v2, is_same in pairs:
        if v1 not in feats or v2 not in feats:
            continue
        sim = 1.0 - cosine_dist(feats[v1], feats[v2])
        by_split[split_id]["s"].append(sim); by_split[split_id]["l"].append(is_same)
    fold_e, fold_a = [], []
    all_s, all_l = [], []
    for s in range(1, n_splits + 1):
        if not by_split[s]["s"]: continue
        e, a = compute_eer_auc(by_split[s]["s"], by_split[s]["l"])
        fold_e.append(e); fold_a.append(a)
        all_s.extend(by_split[s]["s"]); all_l.extend(by_split[s]["l"])
    ov_e, ov_a = compute_eer_auc(all_s, all_l)
    return {"fold_eers": fold_e, "fold_aucs": fold_a,
            "overall_eer": ov_e, "overall_auc": ov_a}


def load_flat(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".pt"):
            t = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
            if isinstance(t, torch.Tensor):
                out[os.path.splitext(fn)[0]] = t.float()
    return out


def load_nested(d):
    out = {}
    for p in sorted(os.listdir(d)):
        pd = os.path.join(d, p)
        if not os.path.isdir(pd): continue
        for fn in sorted(os.listdir(pd)):
            if fn.endswith(".pt"):
                t = torch.load(os.path.join(pd, fn), map_location="cpu", weights_only=False)
                if isinstance(t, torch.Tensor):
                    out[f"{p}/{os.path.splitext(fn)[0]}"] = t.float()
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
def run_seed(backbone, seed, device, afewva_feats_raw, ytf_feats_raw,
             afewva_split, ytf_pairs, do_backbone=False):
    """Returns {'afewva': {stage: (eer,auc)}, 'ytf': {stage: (fold_eers, ...)}}."""
    rd = RUN_MAPPING[backbone][seed]
    ckpt = f"saved_models_da/{rd}/best_da_model.pt"
    fusion, vproj, qag = build_and_load(ckpt, device)

    stages = ["backbone", "projection", "encoder", "va"]
    # Skip backbone after first seed (identical across seeds since raw backbone)
    if not do_backbone:
        stages = ["projection", "encoder", "va"]

    # AFEW-VA per-stage extraction
    af_stage = {s: {} for s in stages}
    for cid, t in afewva_feats_raw.items():
        s_vecs = extract_stages(t, fusion, vproj, qag, device)
        for s in stages:
            af_stage[s][cid] = s_vecs[s]

    af_res = {}
    for s in stages:
        eer, auc = eval_afewva(af_stage[s], afewva_split)
        af_res[s] = {"eer": eer, "auc": auc}

    # YTF per-stage extraction
    yt_stage = {s: {} for s in stages}
    for cid, t in ytf_feats_raw.items():
        s_vecs = extract_stages(t, fusion, vproj, qag, device)
        for s in stages:
            yt_stage[s][cid] = s_vecs[s]

    yt_res = {}
    for s in stages:
        r = eval_ytf(yt_stage[s], ytf_pairs)
        yt_res[s] = r

    return {"afewva": af_res, "ytf": yt_res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--backbones", nargs="+", default=["ViViT", "VideoMAE"])
    ap.add_argument("--out", default="table5_10seed_results.json")
    ap.add_argument("--afewva_split",
                    default=AFEWVA_SPLIT)
    ap.add_argument("--ytf_meta", default="data/YTF/meta_data/meta_and_splits.mat")
    args = ap.parse_args()

    device = torch.device(args.device)
    with open(args.afewva_split) as f:
        afewva_split = json.load(f)
    ytf_pairs = parse_ytf_splits(args.ytf_meta)
    print(f"Seeds: {args.seeds}")
    print(f"Backbones: {args.backbones}")
    print(f"YTF pairs: {len(ytf_pairs)}")

    all_results = {}
    for bb in args.backbones:
        print(f"\n{'='*70}\n{bb}\n{'='*70}")
        print(f"Loading AFEW-VA features ...")
        af_feats = load_flat(f"features/AFEWVA/{bb}")
        print(f"  {len(af_feats)} clips")
        print(f"Loading YTF features ...")
        yt_feats = load_nested(f"features/YTF/{bb}")
        print(f"  {len(yt_feats)} videos")

        per_seed = {}
        for i, s in enumerate(args.seeds):
            do_bb = (i == 0)
            print(f"\n  seed {s}  ckpt={RUN_MAPPING[bb][s]}  backbone_row={do_bb}")
            r = run_seed(bb, s, device, af_feats, yt_feats,
                         afewva_split, ytf_pairs, do_backbone=do_bb)
            per_seed[s] = r
            # Quick print
            print(f"    AFEW-VA:", {k: f"{v['eer']:.4f}" for k, v in r["afewva"].items()})
            print(f"    YTF:    ", {k: f"{v['overall_eer']:.4f}" for k, v in r["ytf"].items()})

        # Aggregate across seeds
        agg = aggregate(per_seed, args.seeds)
        all_results[bb] = {"per_seed": per_seed, "aggregate": agg}

        # Print summary
        print(f"\n  10-seed average for {bb}:")
        stages = ["backbone", "projection", "encoder", "va"]
        dims   = {"backbone": 768, "projection": 512, "encoder": 512, "va": 2}
        print(f"  {'Stage':<12} {'Dim':>4}  {'AFEW-VA EER(mean±std)':>26}  {'YTF 10-fold EER(mean±std)':>28}")
        for s in stages:
            a = agg["afewva"][s]
            y = agg["ytf"][s]
            print(f"  {s:<12} {dims[s]:>4}  {a['eer_mean']:>9.4f}±{a['eer_std']:.4f}      "
                  f"  {y['eer_mean']:>10.4f}±{y['eer_std']:.4f}")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.out}")


def aggregate(per_seed, seeds):
    """Aggregate per_seed results. For 'backbone' (only present in first seed),
    use that single value; for others compute mean±std across seeds."""
    stages = ["backbone", "projection", "encoder", "va"]
    agg = {"afewva": {}, "ytf": {}}

    for s in stages:
        # AFEW-VA EER/AUC across seeds
        eers, aucs = [], []
        for sd in seeds:
            if s in per_seed[sd]["afewva"]:
                eers.append(per_seed[sd]["afewva"][s]["eer"])
                aucs.append(per_seed[sd]["afewva"][s]["auc"])
        if eers:
            agg["afewva"][s] = {
                "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers)),
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "eer_all": eers, "auc_all": aucs,
            }

        # YTF: aggregate per-fold values first, then mean of overall across seeds
        ov_eers, ov_aucs, fold_eers_all, fold_aucs_all = [], [], [], []
        for sd in seeds:
            if s in per_seed[sd]["ytf"]:
                r = per_seed[sd]["ytf"][s]
                ov_eers.append(r["overall_eer"])
                ov_aucs.append(r["overall_auc"])
                fold_eers_all.append(r["fold_eers"])
                fold_aucs_all.append(r["fold_aucs"])
        if ov_eers:
            agg["ytf"][s] = {
                "eer_mean": float(np.mean(ov_eers)), "eer_std": float(np.std(ov_eers)),
                "auc_mean": float(np.mean(ov_aucs)), "auc_std": float(np.std(ov_aucs)),
                "fold_eers_per_seed": fold_eers_all,
                "fold_aucs_per_seed": fold_aucs_all,
            }
    return agg


if __name__ == "__main__":
    main()
