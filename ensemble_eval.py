"""
Late Fusion Ensemble for VA Estimation.

Combines predictions from multiple backbone models (ViViT_s4, VideoMAE_s6, I3D)
via weighted averaging to boost absolute CCC performance.

Strategies:
  1. Equal weight averaging
  2. CCC-weighted averaging (weight by each model's individual CCC)
  3. Grid search for optimal weights (step 0.1)

Also evaluates QAG+AMD models if available.

Usage:
    python ensemble_eval.py [--batch_size 64] [--num_workers 2]
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
from itertools import product

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

# Backbone configs: name -> vision feature dimension
BACKBONE_CONFIGS = {
    'ViViT_s4': 768,
    'VideoMAE_s6': 768,
    'I3D': 512,
    'TimeSformer_s4': 768,
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


def compute_ccc_from_arrays(pred, target):
    """Compute CCC from flattened numpy arrays."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mean_pred = np.mean(pred)
    mean_target = np.mean(target)
    var_pred = np.var(pred)
    var_target = np.var(target)
    cov = np.mean((pred - mean_pred) * (target - mean_target))
    ccc = 2 * cov / (var_pred + var_target + (mean_pred - mean_target) ** 2 + 1e-8)
    return ccc


def classify_method(run_name):
    """Classify the DA method from wandb_run_name."""
    rn = run_name.lower()
    if 'qag_amd' in rn or 'gating_adrop' in rn:
        return 'QAG+AMD'
    elif ('amd' in rn or 'adrop' in rn) and 'gating' not in rn and 'qag' not in rn:
        return 'AMD'
    elif 'baseline' in rn:
        return 'Baseline'
    elif 'da_drop' in rn or 'dadrop' in rn:
        return 'DA+Drop'
    elif 'qag' in rn or 'gating' in rn:
        return 'QAG'
    else:
        return 'Other'


def find_best_models(target_methods=('AMD',)):
    """Scan saved_models_da and find the best model for each backbone+method combo.

    Args:
        target_methods: tuple of method names to search for

    Returns:
        dict: {(backbone, method): {'model_dir': ..., 'avg_ccc': ..., 'hparams': ..., ...}}
    """
    best = {}

    for hp_path in sorted(glob.glob(os.path.join(SAVED_MODELS_DIR, '*/hyperparameters.json'))):
        model_dir = os.path.dirname(hp_path)
        res_path = os.path.join(model_dir, 'best_da_results.json')
        ckpt_path = os.path.join(model_dir, 'best_da_model.pt')

        if not os.path.exists(res_path) or not os.path.exists(ckpt_path):
            continue

        with open(hp_path, 'r') as f:
            hp = json.load(f)
        with open(res_path, 'r') as f:
            res = json.load(f)

        vb = hp.get('vision_backbones', [])
        if len(vb) != 1:
            continue

        backbone = vb[0]
        if backbone not in BACKBONE_CONFIGS:
            continue

        run_name = hp.get('wandb_run_name', '')
        method = classify_method(run_name)
        if method not in target_methods:
            continue

        avg_ccc = res.get('avg_val_ccc', 0)
        key = (backbone, method)

        if key not in best or avg_ccc > best[key]['avg_ccc']:
            best[key] = {
                'model_dir': model_dir,
                'avg_ccc': avg_ccc,
                'ccc_v': res.get('val_ccc_v', 0),
                'ccc_a': res.get('val_ccc_a', 0),
                'seed': hp.get('seed'),
                'run_name': run_name,
                'hparams': hp,
            }

    return best


def build_model(hparams, device):
    """Reconstruct model architecture from hyperparameters (same as eval_noisy_robustness.py)."""
    fusion_model = Two_transformers_DA(
        v_dropout=hparams['v_dropout'],
        a_dropout=hparams['a_dropout'],
        num_heads=hparams['num_heads'],
        num_layers=hparams['num_layers'],
        fusion_type=hparams['fusion_type'],
        output_format=hparams['output_format'],
        vision_in_ft=512,  # always 512 internally
    )

    vision_projection = None
    vision_in_ft = hparams.get('vision_in_ft', [512])
    vision_backbones = hparams.get('vision_backbones', ['R2D1'])
    if len(vision_backbones) == 1:
        dim = vision_in_ft[0] if isinstance(vision_in_ft, list) else vision_in_ft
        if dim != 512:
            vision_projection = nn.Linear(dim, 512)

    vision_fusion_layer = None
    if len(vision_backbones) > 1:
        if isinstance(vision_in_ft, list) and len(vision_in_ft) == 1:
            dims = vision_in_ft * len(vision_backbones)
        else:
            dims = vision_in_ft
        input_dim = sum(dims)
        vision_fusion_layer = nn.Linear(input_dim, 512)

    audio_projection = None
    audio_in_ft = hparams.get('audio_in_ft', 512)
    if audio_in_ft != 512:
        audio_projection = nn.Linear(audio_in_ft, 512)

    quality_gating = None
    if hparams.get('da_quality_gating', False):
        quality_gating = QualityAwareGating(input_dim=512, hidden_dim=64)

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
    checkpoint = torch.load(ckpt_path, map_location=device)

    fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])

    if vision_projection is not None and checkpoint.get('vision_projection_state_dict') is not None:
        vision_projection.load_state_dict(checkpoint['vision_projection_state_dict'])
    if vision_fusion_layer is not None and checkpoint.get('vision_fusion_layer_state_dict') is not None:
        vision_fusion_layer.load_state_dict(checkpoint['vision_fusion_layer_state_dict'])
    if quality_gating is not None and checkpoint.get('quality_gating_state_dict') is not None:
        quality_gating.load_state_dict(checkpoint['quality_gating_state_dict'])


def get_predictions(model_dir, hparams, device, batch_size=64, num_workers=2):
    """Run inference and collect per-sample predictions and labels.

    Returns:
        pred_v: np.ndarray of shape (N,) — flattened valence predictions
        pred_a: np.ndarray of shape (N,) — flattened arousal predictions
        label_v: np.ndarray of shape (N,) — flattened valence labels
        label_a: np.ndarray of shape (N,) — flattened arousal labels
    """
    fusion_model, vision_projection, vision_fusion_layer, audio_projection, quality_gating = \
        build_model(hparams, device)
    load_checkpoint(model_dir, fusion_model, vision_projection, vision_fusion_layer,
                    audio_projection, quality_gating, device)

    # Set all to eval mode
    fusion_model.eval()
    if vision_projection:
        vision_projection.eval()
    if vision_fusion_layer:
        vision_fusion_layer.eval()
    if audio_projection:
        audio_projection.eval()
    if quality_gating:
        quality_gating.eval()

    val_dataset = FeatureDataset(
        features_dir=os.path.join(FEATURES_DIR, 'val'),
        annotation_dir=VAL_ANNOTATIONS,
        vision_backbones=hparams['vision_backbones'],
        audio_backbones=hparams['audio_backbones']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_features
    )

    all_pred_v, all_pred_a = [], []
    all_label_v, all_label_a = [], []

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

            # Quality-aware gating
            if quality_gating is not None:
                vis_feat, aud_feat, _ = quality_gating(vis_feat, aud_feat)

            out = fusion_model(vis_feat, aud_feat)
            pred_v, pred_a = out['pred_v'], out['pred_a']

            seq_len = pred_v.shape[1]
            all_pred_v.append(pred_v.cpu().numpy().reshape(-1))
            all_pred_a.append(pred_a.cpu().numpy().reshape(-1))
            all_label_v.append(labels_v[:, :seq_len].cpu().numpy().reshape(-1))
            all_label_a.append(labels_a[:, :seq_len].cpu().numpy().reshape(-1))

    pred_v = np.concatenate(all_pred_v)
    pred_a = np.concatenate(all_pred_a)
    label_v = np.concatenate(all_label_v)
    label_a = np.concatenate(all_label_a)

    return pred_v, pred_a, label_v, label_a


def ensemble_equal_weight(predictions_list):
    """Equal weight averaging across models.

    Args:
        predictions_list: list of (pred_v, pred_a) tuples

    Returns:
        ens_pred_v, ens_pred_a: averaged predictions
    """
    n = len(predictions_list)
    ens_v = sum(p[0] for p in predictions_list) / n
    ens_a = sum(p[1] for p in predictions_list) / n
    return ens_v, ens_a


def ensemble_ccc_weighted(predictions_list, ccc_scores):
    """CCC-weighted averaging.

    Args:
        predictions_list: list of (pred_v, pred_a) tuples
        ccc_scores: list of (ccc_v, ccc_a) tuples for each model

    Returns:
        ens_pred_v, ens_pred_a: weighted predictions
    """
    # Separate weights for V and A
    weights_v = np.array([max(s[0], 0) for s in ccc_scores])
    weights_a = np.array([max(s[1], 0) for s in ccc_scores])

    # Normalize
    if weights_v.sum() > 0:
        weights_v = weights_v / weights_v.sum()
    else:
        weights_v = np.ones(len(predictions_list)) / len(predictions_list)

    if weights_a.sum() > 0:
        weights_a = weights_a / weights_a.sum()
    else:
        weights_a = np.ones(len(predictions_list)) / len(predictions_list)

    ens_v = sum(w * p[0] for w, p in zip(weights_v, predictions_list))
    ens_a = sum(w * p[1] for w, p in zip(weights_a, predictions_list))
    return ens_v, ens_a, weights_v, weights_a


def ensemble_grid_search(predictions_list, label_v, label_a, step=0.1):
    """Grid search for optimal weights.

    Args:
        predictions_list: list of (pred_v, pred_a) tuples
        label_v, label_a: ground truth
        step: weight step size

    Returns:
        best_weights_v, best_weights_a, best_ccc_v, best_ccc_a, best_avg_ccc,
        best_ens_v, best_ens_a
    """
    n = len(predictions_list)

    # Generate all weight combinations that sum to 1.0
    steps = int(1.0 / step) + 1
    weight_values = [round(i * step, 2) for i in range(steps)]

    def gen_weights(n_models, target_sum=1.0):
        """Generate all weight combos for n_models summing to target_sum."""
        if n_models == 1:
            yield (target_sum,)
            return
        for w in weight_values:
            if w > target_sum + 1e-9:
                break
            remaining = round(target_sum - w, 2)
            for rest in gen_weights(n_models - 1, remaining):
                yield (w,) + rest

    best_avg = -999
    best_weights = None
    best_ccc_v_val = 0
    best_ccc_a_val = 0

    for weights in gen_weights(n):
        ens_v = sum(w * p[0] for w, p in zip(weights, predictions_list))
        ens_a = sum(w * p[1] for w, p in zip(weights, predictions_list))
        ccc_v = compute_ccc_from_arrays(ens_v, label_v)
        ccc_a = compute_ccc_from_arrays(ens_a, label_a)
        avg = (ccc_v + ccc_a) / 2

        if avg > best_avg:
            best_avg = avg
            best_weights = weights
            best_ccc_v_val = ccc_v
            best_ccc_a_val = ccc_a

    # Also do separate optimization for V and A
    best_v_only = -999
    best_weights_v = None
    best_a_only = -999
    best_weights_a = None

    for weights in gen_weights(n):
        ens_v = sum(w * p[0] for w, p in zip(weights, predictions_list))
        ccc_v = compute_ccc_from_arrays(ens_v, label_v)
        if ccc_v > best_v_only:
            best_v_only = ccc_v
            best_weights_v = weights

        ens_a = sum(w * p[1] for w, p in zip(weights, predictions_list))
        ccc_a = compute_ccc_from_arrays(ens_a, label_a)
        if ccc_a > best_a_only:
            best_a_only = ccc_a
            best_weights_a = weights

    # Separate optimal: use best_weights_v for V and best_weights_a for A
    sep_ens_v = sum(w * p[0] for w, p in zip(best_weights_v, predictions_list))
    sep_ens_a = sum(w * p[1] for w, p in zip(best_weights_a, predictions_list))
    sep_ccc_v = compute_ccc_from_arrays(sep_ens_v, label_v)
    sep_ccc_a = compute_ccc_from_arrays(sep_ens_a, label_a)
    sep_avg = (sep_ccc_v + sep_ccc_a) / 2

    return {
        'joint_weights': best_weights,
        'joint_ccc_v': best_ccc_v_val,
        'joint_ccc_a': best_ccc_a_val,
        'joint_avg': best_avg,
        'sep_weights_v': best_weights_v,
        'sep_weights_a': best_weights_a,
        'sep_ccc_v': sep_ccc_v,
        'sep_ccc_a': sep_ccc_a,
        'sep_avg': sep_avg,
    }


def run_ensemble(method_name, best_models, device, batch_size=64, num_workers=2):
    """Run full ensemble evaluation for a given method.

    Args:
        method_name: 'AMD' or 'QAG+AMD'
        best_models: dict from find_best_models()
        device: torch device
    """
    backbones = list(BACKBONE_CONFIGS.keys())
    available = []

    for bb in backbones:
        key = (bb, method_name)
        if key in best_models:
            available.append((bb, best_models[key]))

    if len(available) < 2:
        print(f"\n  Only {len(available)} backbone(s) available for {method_name}, skipping ensemble.")
        return None

    print(f"\n{'='*80}")
    print(f"  Ensemble: {method_name} ({len(available)} backbones)")
    print(f"{'='*80}")

    # Collect predictions from each model
    predictions = {}
    ccc_scores = {}
    label_v, label_a = None, None

    for bb, info in available:
        print(f"\n  Loading {bb} ({method_name}) from {info['model_dir']}")
        print(f"    run_name={info['run_name']}, seed={info['seed']}, "
              f"saved avg_ccc={info['avg_ccc']:.4f}")

        pred_v, pred_a, lv, la = get_predictions(
            info['model_dir'], info['hparams'], device,
            batch_size=batch_size, num_workers=num_workers
        )

        predictions[bb] = (pred_v, pred_a)
        label_v, label_a = lv, la  # same for all (same val set)

        # Compute individual CCC on flattened predictions
        ccc_v = compute_ccc_from_arrays(pred_v, label_v)
        ccc_a = compute_ccc_from_arrays(pred_a, label_a)
        ccc_avg = (ccc_v + ccc_a) / 2
        ccc_scores[bb] = (ccc_v, ccc_a)

        print(f"    Individual: CCC-V={ccc_v:.4f}, CCC-A={ccc_a:.4f}, Avg={ccc_avg:.4f}")

    # Prepare lists in consistent order
    bb_order = [bb for bb, _ in available]
    pred_list = [predictions[bb] for bb in bb_order]
    ccc_list = [ccc_scores[bb] for bb in bb_order]

    results = {
        'method': method_name,
        'backbones': bb_order,
        'individual': {},
    }
    for bb in bb_order:
        results['individual'][bb] = {
            'ccc_v': round(ccc_scores[bb][0], 6),
            'ccc_a': round(ccc_scores[bb][1], 6),
            'avg': round((ccc_scores[bb][0] + ccc_scores[bb][1]) / 2, 6),
            'model_dir': best_models[(bb, method_name)]['model_dir'],
            'run_name': best_models[(bb, method_name)]['run_name'],
            'seed': best_models[(bb, method_name)]['seed'],
        }

    # Strategy 1: Equal weight
    print(f"\n  --- Strategy 1: Equal Weight Averaging ---")
    ens_v, ens_a = ensemble_equal_weight(pred_list)
    ccc_v_eq = compute_ccc_from_arrays(ens_v, label_v)
    ccc_a_eq = compute_ccc_from_arrays(ens_a, label_a)
    avg_eq = (ccc_v_eq + ccc_a_eq) / 2
    print(f"    CCC-V={ccc_v_eq:.4f}, CCC-A={ccc_a_eq:.4f}, Avg={avg_eq:.4f}")
    results['equal_weight'] = {
        'ccc_v': round(ccc_v_eq, 6),
        'ccc_a': round(ccc_a_eq, 6),
        'avg': round(avg_eq, 6),
    }

    # Strategy 2: CCC-weighted
    print(f"\n  --- Strategy 2: CCC-Weighted Averaging ---")
    ens_v_cw, ens_a_cw, w_v, w_a = ensemble_ccc_weighted(pred_list, ccc_list)
    ccc_v_cw = compute_ccc_from_arrays(ens_v_cw, label_v)
    ccc_a_cw = compute_ccc_from_arrays(ens_a_cw, label_a)
    avg_cw = (ccc_v_cw + ccc_a_cw) / 2
    print(f"    Weights V: {dict(zip(bb_order, [round(w, 4) for w in w_v]))}")
    print(f"    Weights A: {dict(zip(bb_order, [round(w, 4) for w in w_a]))}")
    print(f"    CCC-V={ccc_v_cw:.4f}, CCC-A={ccc_a_cw:.4f}, Avg={avg_cw:.4f}")
    results['ccc_weighted'] = {
        'weights_v': dict(zip(bb_order, [round(w, 6) for w in w_v])),
        'weights_a': dict(zip(bb_order, [round(w, 6) for w in w_a])),
        'ccc_v': round(ccc_v_cw, 6),
        'ccc_a': round(ccc_a_cw, 6),
        'avg': round(avg_cw, 6),
    }

    # Strategy 3: Grid search
    print(f"\n  --- Strategy 3: Grid Search (step=0.1) ---")
    gs = ensemble_grid_search(pred_list, label_v, label_a, step=0.1)
    print(f"    Joint optimal weights: {dict(zip(bb_order, gs['joint_weights']))}")
    print(f"    Joint: CCC-V={gs['joint_ccc_v']:.4f}, CCC-A={gs['joint_ccc_a']:.4f}, Avg={gs['joint_avg']:.4f}")
    print(f"    Sep optimal weights V: {dict(zip(bb_order, gs['sep_weights_v']))}")
    print(f"    Sep optimal weights A: {dict(zip(bb_order, gs['sep_weights_a']))}")
    print(f"    Sep: CCC-V={gs['sep_ccc_v']:.4f}, CCC-A={gs['sep_ccc_a']:.4f}, Avg={gs['sep_avg']:.4f}")

    results['grid_search_joint'] = {
        'weights': dict(zip(bb_order, [round(w, 2) for w in gs['joint_weights']])),
        'ccc_v': round(gs['joint_ccc_v'], 6),
        'ccc_a': round(gs['joint_ccc_a'], 6),
        'avg': round(gs['joint_avg'], 6),
    }
    results['grid_search_separate'] = {
        'weights_v': dict(zip(bb_order, [round(w, 2) for w in gs['sep_weights_v']])),
        'weights_a': dict(zip(bb_order, [round(w, 2) for w in gs['sep_weights_a']])),
        'ccc_v': round(gs['sep_ccc_v'], 6),
        'ccc_a': round(gs['sep_ccc_a'], 6),
        'avg': round(gs['sep_avg'], 6),
    }

    # Also try pairwise ensembles
    if len(available) >= 3:
        print(f"\n  --- Pairwise Ensembles (Equal Weight) ---")
        results['pairwise'] = {}
        for i in range(len(bb_order)):
            for j in range(i + 1, len(bb_order)):
                pair = [bb_order[i], bb_order[j]]
                pair_preds = [predictions[bb] for bb in pair]
                pv, pa = ensemble_equal_weight(pair_preds)
                cv = compute_ccc_from_arrays(pv, label_v)
                ca = compute_ccc_from_arrays(pa, label_a)
                avg_p = (cv + ca) / 2
                pair_key = f"{pair[0]}+{pair[1]}"
                print(f"    {pair_key}: CCC-V={cv:.4f}, CCC-A={ca:.4f}, Avg={avg_p:.4f}")
                results['pairwise'][pair_key] = {
                    'ccc_v': round(cv, 6),
                    'ccc_a': round(ca, 6),
                    'avg': round(avg_p, 6),
                }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Late Fusion Ensemble for VA Estimation')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    for method in ['AMD', 'QAG+AMD']:
        print(f"\n\n{'#'*80}")
        print(f"# Scanning for best {method} models...")
        print(f"{'#'*80}")

        best_models = find_best_models(target_methods=(method,))

        if not best_models:
            print(f"  No {method} models found, skipping.")
            continue

        print(f"\n  Found {len(best_models)} backbone(s):")
        for (bb, m), info in sorted(best_models.items()):
            print(f"    {bb}: avg_ccc={info['avg_ccc']:.4f} "
                  f"(V={info['ccc_v']:.4f}, A={info['ccc_a']:.4f}), "
                  f"seed={info['seed']}, dir={info['model_dir']}")

        results = run_ensemble(method, best_models, device,
                               batch_size=args.batch_size, num_workers=args.num_workers)
        if results:
            all_results[method] = results

    # ── Summary ──
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for method, res in all_results.items():
        print(f"\n--- {method} ---")
        print(f"{'Model':<35} {'CCC-V':>8} {'CCC-A':>8} {'Avg':>8}")
        print("-" * 61)

        # Individual models
        for bb, info in res['individual'].items():
            print(f"  {bb:<33} {info['ccc_v']:>8.4f} {info['ccc_a']:>8.4f} {info['avg']:>8.4f}")

        # Ensemble strategies
        eq = res['equal_weight']
        print(f"  {'Ensemble (Equal Weight)':<33} {eq['ccc_v']:>8.4f} {eq['ccc_a']:>8.4f} {eq['avg']:>8.4f}")

        cw = res['ccc_weighted']
        print(f"  {'Ensemble (CCC-Weighted)':<33} {cw['ccc_v']:>8.4f} {cw['ccc_a']:>8.4f} {cw['avg']:>8.4f}")

        gs = res['grid_search_joint']
        print(f"  {'Ensemble (Grid Joint)':<33} {gs['ccc_v']:>8.4f} {gs['ccc_a']:>8.4f} {gs['avg']:>8.4f}")
        print(f"    Joint weights: {gs['weights']}")

        gs_sep = res['grid_search_separate']
        print(f"  {'Ensemble (Grid Separate V/A)':<33} {gs_sep['ccc_v']:>8.4f} {gs_sep['ccc_a']:>8.4f} {gs_sep['avg']:>8.4f}")
        print(f"    Weights V: {gs_sep['weights_v']}")
        print(f"    Weights A: {gs_sep['weights_a']}")

        if 'pairwise' in res:
            print(f"\n  Pairwise:")
            for pair_key, pinfo in res['pairwise'].items():
                print(f"    {pair_key:<31} {pinfo['ccc_v']:>8.4f} {pinfo['ccc_a']:>8.4f} {pinfo['avg']:>8.4f}")

    # Save results
    output_path = os.path.join(BASE_DIR, 'ensemble_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
