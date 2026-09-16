"""
Dynamic Ensemble: per-sample quality-guided ensemble weighting.

Instead of the fixed grid-search weights of ensemble_eval.py,
the ensemble weight is decided dynamically per sample from each backbone's QAG quality score.

Higher quality score → more weight on that backbone → sample-adaptive fusion.

Usage:
    python ensemble_dynamic_eval.py [--batch_size 64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import json
import glob
import numpy as np

from datasets.dataset_features import FeatureDataset
from models.two_transformers_da import Two_transformers_DA
from losses.ccc import CCCLoss
from losses.da_losses import QualityAwareGating
from paths import AFFWILD2_VAL_ANNOTATIONS

# ── Constants ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models_da')
FEATURES_DIR = os.path.join(BASE_DIR, 'features', 'CUSTOM_FINETUNED')
VAL_ANNOTATIONS = AFFWILD2_VAL_ANNOTATIONS

BACKBONE_CONFIGS = {
    'ViViT_s4': 768,
    'VideoMAE_s6': 768,
    'I3D': 512,
}


def collate_fn_features(batch):
    vis_feats = [item[0] for item in batch]
    aud_feats = [item[1] for item in batch]
    labels_V = [item[2][0] for item in batch]
    labels_A = [item[2][1] for item in batch]
    vis_feats_padded = pad_sequence(vis_feats, batch_first=True, padding_value=0.0)
    aud_feats_padded = pad_sequence(aud_feats, batch_first=True, padding_value=0.0)
    labels_V_padded = pad_sequence(labels_V, batch_first=True, padding_value=0.0)
    labels_A_padded = pad_sequence(labels_A, batch_first=True, padding_value=0.0)
    return vis_feats_padded, aud_feats_padded, (labels_V_padded, labels_A_padded)


def compute_ccc(pred, target):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mean_p, mean_t = np.mean(pred), np.mean(target)
    var_p, var_t = np.var(pred), np.var(target)
    cov = np.mean((pred - mean_p) * (target - mean_t))
    return 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)


def classify_method(run_name):
    rn = run_name.lower()
    if 'qag_amd' in rn or 'gating_adrop' in rn:
        return 'QAG+AMD'
    elif ('amd' in rn or 'adrop' in rn) and 'gating' not in rn and 'qag' not in rn:
        return 'AMD'
    elif 'baseline' in rn:
        return 'Baseline'
    return 'Other'


def find_best_qag_amd_models():
    """Find best QAG+AMD model for each backbone."""
    best = {}
    for hp_path in sorted(glob.glob(os.path.join(SAVED_MODELS_DIR, '*/hyperparameters.json'))):
        model_dir = os.path.dirname(hp_path)
        res_path = os.path.join(model_dir, 'best_da_results.json')
        ckpt_path = os.path.join(model_dir, 'best_da_model.pt')
        if not os.path.exists(res_path) or not os.path.exists(ckpt_path):
            continue
        with open(hp_path) as f:
            hp = json.load(f)
        with open(res_path) as f:
            res = json.load(f)
        vb = hp.get('vision_backbones', [])
        if len(vb) != 1 or vb[0] not in BACKBONE_CONFIGS:
            continue
        ab = hp.get('audio_backbones', [])
        if ab != ['ResNet18']:
            continue
        run_name = hp.get('wandb_run_name', '')
        if classify_method(run_name) != 'QAG+AMD':
            continue
        if not hp.get('da_quality_gating', False):
            continue
        backbone = vb[0]
        avg_ccc = res.get('avg_val_ccc', 0)
        if backbone not in best or avg_ccc > best[backbone]['avg_ccc']:
            best[backbone] = {
                'model_dir': model_dir,
                'avg_ccc': avg_ccc,
                'hparams': hp,
                'run_name': run_name,
            }
    return best


def build_and_load(model_dir, hparams, device):
    """Build model components and load checkpoint."""
    fusion_model = Two_transformers_DA(
        v_dropout=hparams['v_dropout'], a_dropout=hparams['a_dropout'],
        num_heads=hparams['num_heads'], num_layers=hparams['num_layers'],
        fusion_type=hparams['fusion_type'], output_format=hparams['output_format'],
        vision_in_ft=512,
    )
    vision_projection = None
    vb = hparams.get('vision_backbones', ['R2D1'])
    vision_in_ft = hparams.get('vision_in_ft', [512])
    if len(vb) == 1:
        dim = vision_in_ft[0] if isinstance(vision_in_ft, list) else vision_in_ft
        if dim != 512:
            vision_projection = nn.Linear(dim, 512)

    quality_gating = QualityAwareGating(
        input_dim=512, hidden_dim=hparams.get('da_qag_hidden_dim', 64)
    )

    # Load checkpoint
    ckpt = torch.load(os.path.join(model_dir, 'best_da_model.pt'), map_location=device)
    fusion_model.load_state_dict(ckpt['fusion_model_state_dict'])
    if vision_projection and ckpt.get('vision_projection_state_dict'):
        vision_projection.load_state_dict(ckpt['vision_projection_state_dict'])
    if ckpt.get('quality_gating_state_dict'):
        quality_gating.load_state_dict(ckpt['quality_gating_state_dict'])

    fusion_model.to(device).eval()
    quality_gating.to(device).eval()
    if vision_projection:
        vision_projection.to(device).eval()

    return fusion_model, vision_projection, quality_gating


def get_predictions_with_quality(model_dir, hparams, device, val_loader):
    """Run inference, return per-timestep predictions AND quality scores."""
    fusion_model, vision_projection, quality_gating = build_and_load(model_dir, hparams, device)

    all_pred_v, all_pred_a = [], []
    all_quality_v, all_quality_a = [], []
    all_label_v, all_label_a = [], []

    with torch.no_grad():
        for vis_feat, aud_feat, labels in val_loader:
            vis_feat, aud_feat = vis_feat.to(device), aud_feat.to(device)
            labels_v, labels_a = labels[0].to(device), labels[1].to(device)

            if vision_projection:
                vis_feat = vision_projection(vis_feat)
            if vis_feat.shape[1] != aud_feat.shape[1]:
                vis_feat = F.interpolate(
                    vis_feat.transpose(1, 2), size=aud_feat.shape[1],
                    mode='linear', align_corners=False
                ).transpose(1, 2)

            # Quality scores BEFORE gating (raw quality estimation)
            v_gate = quality_gating.video_gate_net(vis_feat.mean(dim=1))  # (B, 1)
            a_gate = quality_gating.audio_gate_net(aud_feat.mean(dim=1))  # (B, 1)

            # Apply gating
            vis_gated = vis_feat * v_gate.unsqueeze(1)
            aud_gated = aud_feat * a_gate.unsqueeze(1)

            out = fusion_model(vis_gated, aud_gated)
            pred_v, pred_a = out['pred_v'], out['pred_a']  # (B, T)
            seq_len = pred_v.shape[1]

            # Quality score per sample → expand to timesteps for weighting
            # (B, 1) → (B, T) by repeating
            q_combined = (v_gate + a_gate).squeeze(-1) / 2  # (B,) — average quality

            all_pred_v.append(pred_v.cpu().numpy())  # (B, T)
            all_pred_a.append(pred_a.cpu().numpy())
            all_quality_v.append(v_gate.squeeze(-1).cpu().numpy())  # (B,)
            all_quality_a.append(a_gate.squeeze(-1).cpu().numpy())
            all_label_v.append(labels_v[:, :seq_len].cpu().numpy())
            all_label_a.append(labels_a[:, :seq_len].cpu().numpy())

    return {
        'pred_v': np.concatenate([p.reshape(-1) for p in all_pred_v]),
        'pred_a': np.concatenate([p.reshape(-1) for p in all_pred_a]),
        'label_v': np.concatenate([l.reshape(-1) for l in all_label_v]),
        'label_a': np.concatenate([l.reshape(-1) for l in all_label_a]),
        # Per-batch quality scores (for dynamic weighting)
        'pred_v_batches': all_pred_v,
        'pred_a_batches': all_pred_a,
        'quality_v_batches': all_quality_v,
        'quality_a_batches': all_quality_a,
        'label_v_batches': all_label_v,
        'label_a_batches': all_label_a,
    }


def dynamic_ensemble(backbone_results, temperature=1.0):
    """Per-sample dynamic ensemble using quality scores.

    For each sample, compute ensemble weight for each backbone based on
    its quality score (average of v_gate and a_gate), then softmax normalize.

    Args:
        backbone_results: dict {backbone_name: results_dict}
        temperature: softmax temperature (lower = sharper weighting)

    Returns:
        ensemble predictions and per-backbone weight statistics
    """
    backbones = list(backbone_results.keys())
    n_batches = len(backbone_results[backbones[0]]['pred_v_batches'])

    all_ens_v, all_ens_a = [], []
    all_label_v, all_label_a = [], []
    weight_stats = {bb: [] for bb in backbones}

    for batch_idx in range(n_batches):
        # Collect predictions and quality scores for this batch
        batch_preds_v = []
        batch_preds_a = []
        batch_qualities = []

        for bb in backbones:
            res = backbone_results[bb]
            pv = res['pred_v_batches'][batch_idx]  # (B, T)
            pa = res['pred_a_batches'][batch_idx]
            qv = res['quality_v_batches'][batch_idx]  # (B,)
            qa = res['quality_a_batches'][batch_idx]  # (B,)
            q_avg = (qv + qa) / 2  # (B,)

            batch_preds_v.append(pv)
            batch_preds_a.append(pa)
            batch_qualities.append(q_avg)

        # Stack: (n_backbones, B)
        qualities = np.stack(batch_qualities, axis=0)  # (n_bb, B)

        # Softmax across backbones per sample
        qualities_scaled = qualities / temperature
        exp_q = np.exp(qualities_scaled - qualities_scaled.max(axis=0, keepdims=True))
        weights = exp_q / exp_q.sum(axis=0, keepdims=True)  # (n_bb, B)

        # Weighted ensemble: per sample, per timestep
        B = batch_preds_v[0].shape[0]
        T = batch_preds_v[0].shape[1]

        ens_v = np.zeros((B, T))
        ens_a = np.zeros((B, T))
        for i, bb in enumerate(backbones):
            w = weights[i]  # (B,)
            ens_v += batch_preds_v[i] * w[:, np.newaxis]  # (B, T)
            ens_a += batch_preds_a[i] * w[:, np.newaxis]
            weight_stats[bb].extend(w.tolist())

        all_ens_v.append(ens_v.reshape(-1))
        all_ens_a.append(ens_a.reshape(-1))
        all_label_v.append(backbone_results[backbones[0]]['label_v_batches'][batch_idx].reshape(-1))
        all_label_a.append(backbone_results[backbones[0]]['label_a_batches'][batch_idx].reshape(-1))

    ens_v_flat = np.concatenate(all_ens_v)
    ens_a_flat = np.concatenate(all_ens_a)
    label_v_flat = np.concatenate(all_label_v)
    label_a_flat = np.concatenate(all_label_a)

    ccc_v = compute_ccc(ens_v_flat, label_v_flat)
    ccc_a = compute_ccc(ens_a_flat, label_a_flat)

    avg_weights = {bb: float(np.mean(weight_stats[bb])) for bb in backbones}
    std_weights = {bb: float(np.std(weight_stats[bb])) for bb in backbones}

    return {
        'ccc_v': ccc_v,
        'ccc_a': ccc_a,
        'avg_ccc': (ccc_v + ccc_a) / 2,
        'avg_weights': avg_weights,
        'std_weights': std_weights,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find best QAG+AMD models
    best_models = find_best_qag_amd_models()
    print(f"\nFound {len(best_models)} QAG+AMD models:")
    for bb, info in sorted(best_models.items()):
        print(f"  {bb}: avg_ccc={info['avg_ccc']:.4f}, run={info['run_name']}")

    if len(best_models) < 2:
        print("Need at least 2 backbones for ensemble")
        return

    # Shared val_loader (all use same audio backbone + same val set)
    # But different vision backbones → need separate loaders
    backbone_results = {}
    for bb, info in sorted(best_models.items()):
        print(f"\n--- Loading {bb} ---")
        val_dataset = FeatureDataset(
            features_dir=os.path.join(FEATURES_DIR, 'val'),
            annotation_dir=VAL_ANNOTATIONS,
            vision_backbones=info['hparams']['vision_backbones'],
            audio_backbones=info['hparams']['audio_backbones']
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn_features
        )
        backbone_results[bb] = get_predictions_with_quality(
            info['model_dir'], info['hparams'], device, val_loader
        )

        # Individual CCC
        ccc_v = compute_ccc(backbone_results[bb]['pred_v'], backbone_results[bb]['label_v'])
        ccc_a = compute_ccc(backbone_results[bb]['pred_a'], backbone_results[bb]['label_a'])
        print(f"  Individual: CCC-V={ccc_v:.4f}, CCC-A={ccc_a:.4f}, Avg={(ccc_v+ccc_a)/2:.4f}")
        print(f"  Quality: V={np.mean(np.concatenate(backbone_results[bb]['quality_v_batches'])):.4f}, "
              f"A={np.mean(np.concatenate(backbone_results[bb]['quality_a_batches'])):.4f}")

    # ── Static baselines (from ensemble_eval.py logic) ──
    backbones = sorted(backbone_results.keys())
    pred_list = [(backbone_results[bb]['pred_v'], backbone_results[bb]['pred_a']) for bb in backbones]
    label_v = backbone_results[backbones[0]]['label_v']
    label_a = backbone_results[backbones[0]]['label_a']

    # Equal weight
    n = len(pred_list)
    eq_v = sum(p[0] for p in pred_list) / n
    eq_a = sum(p[1] for p in pred_list) / n
    eq_ccc_v = compute_ccc(eq_v, label_v)
    eq_ccc_a = compute_ccc(eq_a, label_a)
    print(f"\n{'='*60}")
    print(f"Equal Weight:    CCC-V={eq_ccc_v:.4f}, CCC-A={eq_ccc_a:.4f}, Avg={(eq_ccc_v+eq_ccc_a)/2:.4f}")

    # ── Dynamic ensemble with different temperatures ──
    results = {}
    for temp in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        res = dynamic_ensemble(backbone_results, temperature=temp)
        results[f'temp_{temp}'] = res
        print(f"Dynamic (T={temp:.1f}): CCC-V={res['ccc_v']:.4f}, CCC-A={res['ccc_a']:.4f}, "
              f"Avg={res['avg_ccc']:.4f}  weights={res['avg_weights']}")

    # Best temperature
    best_temp = max(results.keys(), key=lambda k: results[k]['avg_ccc'])
    best_res = results[best_temp]
    print(f"\n*** Best: {best_temp}, Avg CCC={best_res['avg_ccc']:.4f} ***")
    print(f"    vs Equal Weight: {(eq_ccc_v+eq_ccc_a)/2:.4f}")

    # Save results
    output = {
        'equal_weight': {
            'ccc_v': round(eq_ccc_v, 6), 'ccc_a': round(eq_ccc_a, 6),
            'avg': round((eq_ccc_v + eq_ccc_a) / 2, 6),
        },
        'dynamic': {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                        for kk, vv in v.items()}
                    for k, v in results.items()},
        'best_temperature': best_temp,
        'backbones': backbones,
    }
    out_path = os.path.join(BASE_DIR, 'ensemble_dynamic_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
