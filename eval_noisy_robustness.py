"""
Noisy modality robustness evaluation.

Loads a saved DA model and evaluates under various noise conditions:
- Normal (no noise) — sanity check
- Audio Gaussian noise at SNR = {20, 10, 5, 0, -5} dB
- Video Gaussian noise at SNR = {20, 10, 5, 0, -5} dB
- Audio zeroed (V-only)
- Video zeroed (A-only)

Usage:
    python eval_noisy_robustness.py --model_dir saved_models_da/03092026_121145/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
import json
import math
import numpy as np

from datasets.dataset_features import FeatureDataset
from models.two_transformers_da import Two_transformers_DA
from losses.ccc import CCCLoss
from losses.da_losses import QualityAwareGating


# ── Collate (same as training) ──
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


def compute_ccc(pred, target, criterion):
    """Returns CCC value (higher is better)."""
    return (1 - criterion(pred.reshape(-1), target.reshape(-1))).item()


def add_gaussian_noise_snr(x, snr_db):
    """Add Gaussian noise to tensor x at given SNR (dB).

    Args:
        x: input feature tensor (any shape)
        snr_db: target signal-to-noise ratio in dB

    Returns:
        x_noisy: x + noise at specified SNR
    """
    signal_power = (x ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(x) * math.sqrt(noise_power.item())
    return x + noise


def build_model(hparams, device):
    """Reconstruct model architecture from hyperparameters."""
    # Fusion model
    fusion_model = Two_transformers_DA(
        v_dropout=hparams['v_dropout'],
        a_dropout=hparams['a_dropout'],
        num_heads=hparams['num_heads'],
        num_layers=hparams['num_layers'],
        fusion_type=hparams['fusion_type'],
        output_format=hparams['output_format'],
        vision_in_ft=512,  # always 512 internally
    )

    # Vision projection
    vision_projection = None
    vision_in_ft = hparams.get('vision_in_ft', [512])
    vision_backbones = hparams.get('vision_backbones', ['R2D1'])
    if len(vision_backbones) == 1:
        dim = vision_in_ft[0] if isinstance(vision_in_ft, list) else vision_in_ft
        if dim != 512:
            vision_projection = nn.Linear(dim, 512)
    # Multi-backbone vision fusion layer not expected in this eval

    vision_fusion_layer = None
    if len(vision_backbones) > 1:
        if isinstance(vision_in_ft, list) and len(vision_in_ft) == 1:
            dims = vision_in_ft * len(vision_backbones)
        else:
            dims = vision_in_ft
        input_dim = sum(dims)
        vision_fusion_layer = nn.Linear(input_dim, 512)

    # Audio projection
    audio_projection = None
    audio_in_ft = hparams.get('audio_in_ft', 512)
    if audio_in_ft != 512:
        audio_projection = nn.Linear(audio_in_ft, 512)

    # Quality-Aware Gating
    quality_gating = None
    if hparams.get('da_quality_gating', False):
        quality_gating = QualityAwareGating(input_dim=512, hidden_dim=64)

    # Move to device
    fusion_model.to(device)
    if vision_projection:
        vision_projection.to(device)
    if vision_fusion_layer:
        vision_fusion_layer.to(device)
    if audio_projection:
        audio_projection.to(device)
    if quality_gating:
        quality_gating.to(device)

    return fusion_model, vision_projection, vision_fusion_layer, audio_projection, quality_gating


def load_checkpoint(model_dir, fusion_model, vision_projection, vision_fusion_layer,
                    audio_projection, quality_gating, device):
    """Load saved checkpoint into model components."""
    ckpt_path = os.path.join(model_dir, 'best_da_model.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])

    if vision_projection is not None and checkpoint.get('vision_projection_state_dict') is not None:
        vision_projection.load_state_dict(checkpoint['vision_projection_state_dict'])
    if vision_fusion_layer is not None and checkpoint.get('vision_fusion_layer_state_dict') is not None:
        vision_fusion_layer.load_state_dict(checkpoint['vision_fusion_layer_state_dict'])
    # audio_projection is not saved in checkpoint (ResNet18=512, no projection needed)
    if quality_gating is not None and checkpoint.get('quality_gating_state_dict') is not None:
        quality_gating.load_state_dict(checkpoint['quality_gating_state_dict'])

    print(f"Checkpoint loaded from {ckpt_path}")


def evaluate_condition(fusion_model, val_loader, criterion, device,
                       vision_projection, vision_fusion_layer, audio_projection,
                       quality_gating, noise_target=None, snr_db=None, zero_modality=None):
    """Evaluate model under a specific noise condition.

    Args:
        noise_target: 'audio' or 'video' — which modality to add noise to (None = no noise)
        snr_db: SNR in dB for Gaussian noise
        zero_modality: 'audio' or 'video' — which modality to zero out (None = no zeroing)

    Returns:
        dict with ccc_valence, ccc_arousal, ccc_avg
    """
    fusion_model.eval()
    if vision_projection:
        vision_projection.eval()
    if vision_fusion_layer:
        vision_fusion_layer.eval()
    if audio_projection:
        audio_projection.eval()
    if quality_gating:
        quality_gating.eval()

    ccc_v_sum, ccc_a_sum = 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for vis_feat, aud_feat, labels in val_loader:
            vis_feat, aud_feat = vis_feat.to(device), aud_feat.to(device)
            labels_v, labels_a = labels[0].to(device), labels[1].to(device)

            # Projections
            if vision_projection:
                vis_feat = vision_projection(vis_feat)
            if vision_fusion_layer:
                vis_feat = vision_fusion_layer(vis_feat)
            if audio_projection:
                aud_feat = audio_projection(aud_feat)

            # Sequence length alignment
            if vis_feat.shape[1] != aud_feat.shape[1]:
                vis_feat = F.interpolate(
                    vis_feat.transpose(1, 2), size=aud_feat.shape[1],
                    mode='linear', align_corners=False
                ).transpose(1, 2)

            # Apply noise condition
            if noise_target == 'audio' and snr_db is not None:
                aud_feat = add_gaussian_noise_snr(aud_feat, snr_db)
            elif noise_target == 'video' and snr_db is not None:
                vis_feat = add_gaussian_noise_snr(vis_feat, snr_db)

            if zero_modality == 'audio':
                aud_feat = torch.zeros_like(aud_feat)
            elif zero_modality == 'video':
                vis_feat = torch.zeros_like(vis_feat)

            # Quality-aware gating
            if quality_gating is not None:
                vis_feat, aud_feat, _ = quality_gating(vis_feat, aud_feat)

            out = fusion_model(vis_feat, aud_feat)
            pred_v, pred_a = out['pred_v'], out['pred_a']

            seq_len = pred_v.shape[1]
            ccc_v_sum += compute_ccc(pred_v, labels_v[:, :seq_len], criterion)
            ccc_a_sum += compute_ccc(pred_a, labels_a[:, :seq_len], criterion)
            n_batches += 1

    ccc_v = ccc_v_sum / n_batches
    ccc_a = ccc_a_sum / n_batches
    ccc_avg = (ccc_v + ccc_a) / 2

    return {
        'ccc_valence': round(ccc_v, 6),
        'ccc_arousal': round(ccc_a, 6),
        'ccc_avg': round(ccc_avg, 6),
    }


def main():
    parser = argparse.ArgumentParser(description='Noisy Modality Robustness Evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model directory (contains best_da_model.pt and hyperparameters.json)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--snr_levels', type=float, nargs='+', default=[20, 10, 5, 0, -5],
                        help='SNR levels in dB for noise evaluation')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load hyperparameters
    hparams_path = os.path.join(args.model_dir, 'hyperparameters.json')
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hyperparameters.json not found in {args.model_dir}")
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)

    print(f"Model dir: {args.model_dir}")
    print(f"Backbone: {hparams['vision_backbones']} + {hparams['audio_backbones']}")
    print(f"Fusion: {hparams['fusion_type']}, Output: {hparams['output_format']}")
    print(f"QAG: {hparams.get('da_quality_gating', False)}, "
          f"AMD: {hparams.get('da_adaptive_dropout', False)}, "
          f"Fixed Drop: {hparams.get('da_modality_dropout', 0.0)}")
    print(f"Seed: {hparams.get('seed', 'None')}")

    # Set seed for reproducibility of noise generation
    seed = hparams.get('seed', 42)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Build model
    fusion_model, vision_projection, vision_fusion_layer, audio_projection, quality_gating = \
        build_model(hparams, device)

    # Load checkpoint
    load_checkpoint(args.model_dir, fusion_model, vision_projection, vision_fusion_layer,
                    audio_projection, quality_gating, device)

    # Validation dataset
    val_dataset = FeatureDataset(
        features_dir=os.path.join(hparams['features_dir'], 'val'),
        annotation_dir=hparams['val_annotations'],
        vision_backbones=hparams['vision_backbones'],
        audio_backbones=hparams['audio_backbones']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_features
    )
    print(f"Val samples: {len(val_dataset)}")

    criterion = CCCLoss(digitize_num=1).to(device)

    # ── Run all evaluation conditions ──
    results = {}
    snr_levels = args.snr_levels

    # 1. Normal (no noise) — sanity check
    print("\n=== Normal (no noise) ===")
    res = evaluate_condition(fusion_model, val_loader, criterion, device,
                             vision_projection, vision_fusion_layer, audio_projection,
                             quality_gating)
    results['normal'] = res
    print(f"  CCC-V={res['ccc_valence']:.4f}  CCC-A={res['ccc_arousal']:.4f}  Avg={res['ccc_avg']:.4f}")

    # 2. Audio noise at various SNR levels
    for snr in snr_levels:
        condition = f'audio_noise_snr{int(snr)}dB'
        print(f"\n=== Audio Noise SNR={snr}dB ===")
        # Reset seed before each condition for reproducible noise
        torch.manual_seed(seed if seed is not None else 42)
        res = evaluate_condition(fusion_model, val_loader, criterion, device,
                                 vision_projection, vision_fusion_layer, audio_projection,
                                 quality_gating, noise_target='audio', snr_db=snr)
        results[condition] = res
        print(f"  CCC-V={res['ccc_valence']:.4f}  CCC-A={res['ccc_arousal']:.4f}  Avg={res['ccc_avg']:.4f}")

    # 3. Video noise at various SNR levels
    for snr in snr_levels:
        condition = f'video_noise_snr{int(snr)}dB'
        print(f"\n=== Video Noise SNR={snr}dB ===")
        torch.manual_seed(seed if seed is not None else 42)
        res = evaluate_condition(fusion_model, val_loader, criterion, device,
                                 vision_projection, vision_fusion_layer, audio_projection,
                                 quality_gating, noise_target='video', snr_db=snr)
        results[condition] = res
        print(f"  CCC-V={res['ccc_valence']:.4f}  CCC-A={res['ccc_arousal']:.4f}  Avg={res['ccc_avg']:.4f}")

    # 4. Audio zeroed (V-only)
    print("\n=== Audio Zeroed (V-only) ===")
    res = evaluate_condition(fusion_model, val_loader, criterion, device,
                             vision_projection, vision_fusion_layer, audio_projection,
                             quality_gating, zero_modality='audio')
    results['audio_zeroed_vonly'] = res
    print(f"  CCC-V={res['ccc_valence']:.4f}  CCC-A={res['ccc_arousal']:.4f}  Avg={res['ccc_avg']:.4f}")

    # 5. Video zeroed (A-only)
    print("\n=== Video Zeroed (A-only) ===")
    res = evaluate_condition(fusion_model, val_loader, criterion, device,
                             vision_projection, vision_fusion_layer, audio_projection,
                             quality_gating, zero_modality='video')
    results['video_zeroed_aonly'] = res
    print(f"  CCC-V={res['ccc_valence']:.4f}  CCC-A={res['ccc_arousal']:.4f}  Avg={res['ccc_avg']:.4f}")

    # ── Save results ──
    output_path = os.path.join(args.model_dir, 'noisy_eval_results.json')
    # Add metadata
    output = {
        'model_dir': args.model_dir,
        'vision_backbones': hparams['vision_backbones'],
        'audio_backbones': hparams['audio_backbones'],
        'wandb_run_name': hparams.get('wandb_run_name', 'unknown'),
        'seed': hparams.get('seed'),
        'da_quality_gating': hparams.get('da_quality_gating', False),
        'da_adaptive_dropout': hparams.get('da_adaptive_dropout', False),
        'da_modality_dropout': hparams.get('da_modality_dropout', 0.0),
        'snr_levels': snr_levels,
        'conditions': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to {output_path}")

    # ── Summary table ──
    print("\n" + "=" * 80)
    print(f"Summary: {hparams.get('wandb_run_name', args.model_dir)}")
    print(f"{'Condition':<30} {'CCC-V':>8} {'CCC-A':>8} {'Avg':>8}")
    print("-" * 56)
    for cond, res in results.items():
        print(f"{cond:<30} {res['ccc_valence']:>8.4f} {res['ccc_arousal']:>8.4f} {res['ccc_avg']:>8.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
