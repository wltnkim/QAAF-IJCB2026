"""
Domain Adaptation training script.

Feature-based joint training with the DA loss modules integrated.

Usage:
    python train_from_features_joint_da.py \
        --features_dir ./features/CUSTOM_FINETUNED \
        --train_annotations .../Train_Set \
        --val_annotations .../Val_Set \
        --fusion_type TRANSFORMER \
        --da_modality_dropout 0.15 \
        --da_contrastive --da_contrastive_lambda 0.1 \
        --da_adversarial --da_adversarial_lambda 0.01 \
        --da_unimodal_heads --da_unimodal_lambda 0.1 \
        --da_consistency_lambda 0.05 \
        --eval_missing_modality
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import datetime
import json
import random
import numpy as np
try:
    import wandb
except ImportError:
    # Optional dependency. Without it, training runs normally and every
    # tracking call becomes a no-op. Install wandb, or set WANDB_MODE=disabled,
    # if you want the code path exercised without an account.
    class _NoOpTracker:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None
    wandb = _NoOpTracker()


def set_seed(seed):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from datasets.dataset_features import FeatureDataset
from models.two_transformers_da import Two_transformers_DA
from losses.ccc import CCCLoss
from losses.da_losses import (
    ModalityDropout,
    AdaptiveModalityDropout,
    QualityAwareGating,
    QMFGating,
    FixedGating,
    InfoNCEContrastiveLoss,
    AdversarialModalityLoss,
    UnimodalPredictionHeads,
    CrossModalReconstruction,
    CrossModalDistillation,
    grl_lambda_schedule,
)
from nn_utils import MyDataParallel
from torch.nn.utils.rnn import pad_sequence


# ── EarlyStopper ──
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_ccc = float('-inf')

    def early_stop(self, validation_ccc):
        if validation_ccc > self.max_validation_ccc:
            self.max_validation_ccc = validation_ccc
            self.counter = 0
        elif validation_ccc < (self.max_validation_ccc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ── Collate ──
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


# ── CCC metric (1 - CCCLoss) ──
def compute_ccc(pred, target, criterion):
    """Returns CCC value (higher is better)."""
    return (1 - criterion(pred.reshape(-1), target.reshape(-1))).item()


# ── Missing modality evaluation ──
def evaluate_missing_modality(fusion_model, val_loader, criterion, device,
                              vision_projection, vision_fusion_layer, args,
                              audio_projection=None, quality_gating=None):
    """V-only, A-only CCC evaluation."""
    fusion_model.eval()
    if vision_projection:
        vision_projection.eval()
    if vision_fusion_layer:
        vision_fusion_layer.eval()
    if audio_projection:
        audio_projection.eval()
    if quality_gating:
        quality_gating.eval()

    results = {}
    for mode in ['video_only', 'audio_only']:
        ccc_v_sum, ccc_a_sum = 0.0, 0.0
        with torch.no_grad():
            for vis_feat, aud_feat, labels in val_loader:
                vis_feat, aud_feat = vis_feat.to(device), aud_feat.to(device)
                labels_v, labels_a = labels[0].to(device), labels[1].to(device)

                if vision_projection:
                    vis_feat = vision_projection(vis_feat)
                if vision_fusion_layer:
                    vis_feat = vision_fusion_layer(vis_feat)
                if audio_projection:
                    aud_feat = audio_projection(aud_feat)

                if vis_feat.shape[1] != aud_feat.shape[1]:
                    vis_feat = F.interpolate(
                        vis_feat.transpose(1, 2), size=aud_feat.shape[1],
                        mode='linear', align_corners=False
                    ).transpose(1, 2)

                # Zero out one modality
                if mode == 'video_only':
                    aud_feat = torch.zeros_like(aud_feat)
                else:
                    vis_feat = torch.zeros_like(vis_feat)

                # Quality-aware gating (applied at inference as well)
                if quality_gating is not None:
                    vis_feat, aud_feat, _ = quality_gating(vis_feat, aud_feat)

                out = fusion_model(vis_feat, aud_feat)
                pred_v, pred_a = out['pred_v'], out['pred_a']

                # Two_transformers_DA always returns batch-first (B, T)

                seq_len = pred_v.shape[1]
                ccc_v_sum += compute_ccc(pred_v, labels_v[:, :seq_len], criterion)
                ccc_a_sum += compute_ccc(pred_a, labels_a[:, :seq_len], criterion)

        n = len(val_loader)
        results[f'{mode}/ccc_valence'] = ccc_v_sum / n
        results[f'{mode}/ccc_arousal'] = ccc_a_sum / n
        results[f'{mode}/ccc_avg'] = (ccc_v_sum / n + ccc_a_sum / n) / 2

    return results


def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──
    train_dataset = FeatureDataset(
        features_dir=os.path.join(args.features_dir, 'train'),
        annotation_dir=args.train_annotations,
        vision_backbones=args.vision_backbones,
        audio_backbones=args.audio_backbones
    )
    val_dataset = FeatureDataset(
        features_dir=os.path.join(args.features_dir, 'val'),
        annotation_dir=args.val_annotations,
        vision_backbones=args.vision_backbones,
        audio_backbones=args.audio_backbones
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_features
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_features
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Fusion model (DA wrapper) ──
    # fusion_type mapping: TRANSFORMER/NONE/FC/LIGHT_ATTN
    fusion_model = Two_transformers_DA(
        v_dropout=args.v_dropout,
        a_dropout=args.a_dropout,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        fusion_type=args.fusion_type,
        output_format=args.output_format,
        vision_in_ft=512,
    )

    # ── Vision projection / fusion layer (multi-backbone support) ──
    vision_projection = None
    vision_fusion_layer = None
    if len(args.vision_backbones) == 1:
        dim = args.vision_in_ft[0]
        if dim != 512:
            vision_projection = nn.Linear(dim, 512)
    else:
        if len(args.vision_in_ft) == 1:
            dims = args.vision_in_ft * len(args.vision_backbones)
        else:
            dims = args.vision_in_ft
        input_dim = sum(dims)
        vision_fusion_layer = nn.Linear(input_dim, 512)
        print(f"Vision fusion layer: {input_dim} → 512 (backbones: {list(zip(args.vision_backbones, dims))})")

    # Audio feature projection (AST=768 → 512)
    audio_projection = None
    if args.audio_in_ft != 512:
        audio_projection = nn.Linear(args.audio_in_ft, 512)
        print(f"Audio projection: {args.audio_in_ft} → 512")

    # ── DA modules ──
    # Quality-Aware Gating (soft scaling, training + inference)
    # QMF and QAG cannot be used at the same time (QMF takes precedence)
    quality_gating = None
    if args.da_fixed_gating:
        quality_gating = FixedGating(gate_value=0.5)
        print(f"DA: FixedGating enabled (gate=0.5, no learnable params)")
    elif args.da_qmf_gating:
        quality_gating = QMFGating()
        print(f"DA: QMFGating enabled (energy-based, no learnable params)")
    elif args.da_quality_gating:
        quality_gating = QualityAwareGating(input_dim=512, hidden_dim=args.da_qag_hidden_dim)
        print(f"DA: QualityAwareGating enabled (input=512, hidden={args.da_qag_hidden_dim})")

    # Adaptive Modality Dropout (quality-based, training only)
    adaptive_dropout = None
    if args.da_adaptive_dropout:
        adaptive_dropout = AdaptiveModalityDropout(
            input_dim=512, hidden_dim=64,
            p_min=args.da_adaptive_p_min, p_max=args.da_adaptive_p_max,
            learnable_pmax=args.da_learnable_pmax,
            pmax_range_max=args.da_pmax_range_max,
        )
        pmax_str = f"LEARNABLE [init={args.da_adaptive_p_max}, range=[{args.da_adaptive_p_min},{args.da_pmax_range_max}]]" if args.da_learnable_pmax else f"{args.da_adaptive_p_max}"
        print(f"DA: AdaptiveModalityDropout enabled (p_min={args.da_adaptive_p_min}, p_max={pmax_str})")

    # Fixed Modality Dropout (legacy, cannot be used together with adaptive)
    modality_dropout = None
    if args.da_modality_dropout > 0 and not args.da_adaptive_dropout:
        modality_dropout = ModalityDropout(p_drop=args.da_modality_dropout)
        print(f"DA: ModalityDropout enabled (p={args.da_modality_dropout})")

    contrastive_loss_fn = None
    if args.da_contrastive:
        contrastive_loss_fn = InfoNCEContrastiveLoss(temperature=0.07)
        print(f"DA: InfoNCE Contrastive enabled (lambda={args.da_contrastive_lambda})")

    adversarial_loss_fn = None
    if args.da_adversarial:
        adversarial_loss_fn = AdversarialModalityLoss(input_dim=512, hidden_dim=64)
        print(f"DA: Adversarial GRL enabled (lambda={args.da_adversarial_lambda})")

    unimodal_heads = None
    if args.da_unimodal_heads or args.da_distillation:
        unimodal_heads = UnimodalPredictionHeads(input_dim=512, hidden_dim=64)
        print(f"DA: Unimodal Heads enabled (lambda={args.da_unimodal_lambda}, consistency={args.da_consistency_lambda})")

    reconstruction_fn = None
    if args.da_reconstruction:
        reconstruction_fn = CrossModalReconstruction(input_dim=512, hidden_dim=args.da_reconstruction_hidden)
        print(f"DA: Cross-Modal Reconstruction enabled (lambda={args.da_reconstruction_lambda}, hidden={args.da_reconstruction_hidden})")

    distillation_fn = None
    if args.da_distillation:
        distillation_fn = CrossModalDistillation()
        print(f"DA: Cross-Modal Knowledge Distillation enabled (lambda={args.da_distillation_lambda})")
        if not args.da_unimodal_heads:
            print("  (unimodal heads auto-enabled for distillation)")

    # ── DataParallel ──
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        fusion_model = MyDataParallel(fusion_model)
        if vision_projection:
            vision_projection = MyDataParallel(vision_projection)
        if vision_fusion_layer:
            vision_fusion_layer = MyDataParallel(vision_fusion_layer)
        if audio_projection:
            audio_projection = MyDataParallel(audio_projection)
        if quality_gating:
            quality_gating = MyDataParallel(quality_gating)
        if adaptive_dropout:
            adaptive_dropout = MyDataParallel(adaptive_dropout)
        if adversarial_loss_fn:
            adversarial_loss_fn = MyDataParallel(adversarial_loss_fn)
        if unimodal_heads:
            unimodal_heads = MyDataParallel(unimodal_heads)
        if reconstruction_fn:
            reconstruction_fn = MyDataParallel(reconstruction_fn)

    # ── Move to device ──
    fusion_model.to(device)
    if vision_projection:
        vision_projection.to(device)
    if vision_fusion_layer:
        vision_fusion_layer.to(device)
    if audio_projection:
        audio_projection.to(device)
    if quality_gating:
        quality_gating.to(device)
    if adaptive_dropout:
        adaptive_dropout.to(device)
    if adversarial_loss_fn:
        adversarial_loss_fn.to(device)
    if unimodal_heads:
        unimodal_heads.to(device)
    if reconstruction_fn:
        reconstruction_fn.to(device)

    criterion = CCCLoss(digitize_num=1).to(device)

    # ── Optimizer ──
    params_to_optimize = list(fusion_model.parameters())
    if vision_projection:
        params_to_optimize += list(vision_projection.parameters())
    if vision_fusion_layer:
        params_to_optimize += list(vision_fusion_layer.parameters())
    if audio_projection:
        params_to_optimize += list(audio_projection.parameters())
    if quality_gating:
        params_to_optimize += list(quality_gating.parameters())
    if adaptive_dropout:
        params_to_optimize += list(adaptive_dropout.parameters())
    if adversarial_loss_fn:
        params_to_optimize += list(adversarial_loss_fn.parameters())
    if unimodal_heads:
        params_to_optimize += list(unimodal_heads.parameters())
    if reconstruction_fn:
        params_to_optimize += list(reconstruction_fn.parameters())

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params_to_optimize, lr=args.lr,
                              weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params_to_optimize, lr=args.lr,
                                weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    early_stopper = EarlyStopper(patience=args.patience, min_delta=0.001)
    best_val_ccc = -1
    best_results = {}

    # ── Training loop ──
    for epoch in range(args.epochs):
        fusion_model.train()
        if vision_projection:
            vision_projection.train()
        if vision_fusion_layer:
            vision_fusion_layer.train()
        if audio_projection:
            audio_projection.train()
        if quality_gating:
            quality_gating.train()
        if adaptive_dropout:
            adaptive_dropout.train()
        if adversarial_loss_fn:
            adversarial_loss_fn.train()
        if unimodal_heads:
            unimodal_heads.train()
        if reconstruction_fn:
            reconstruction_fn.train()

        train_losses = {
            'loss_v': 0.0, 'loss_a': 0.0,
            'loss_contrastive': 0.0, 'loss_adversarial': 0.0,
            'loss_unimodal_v': 0.0, 'loss_unimodal_a': 0.0,
            'loss_consistency': 0.0,
            'loss_reconstruction': 0.0, 'loss_distillation': 0.0,
            'loss_total': 0.0,
        }
        # Track the quality/gating statistics
        gate_v_sum, gate_a_sum, gate_count = 0.0, 0.0, 0

        # GRL lambda scheduling
        grl_lambda = grl_lambda_schedule(epoch, max_epochs=args.grl_max_epochs) if args.da_adversarial else 0.0

        for vis_feat, aud_feat, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            vis_feat, aud_feat = vis_feat.to(device), aud_feat.to(device)
            labels_v, labels_a = labels[0].to(device), labels[1].to(device)

            optimizer.zero_grad()

            # Vision projection
            if vision_projection:
                vis_feat = vision_projection(vis_feat)
            if vision_fusion_layer:
                vis_feat = vision_fusion_layer(vis_feat)
            # Audio projection (AST 768 → 512)
            if audio_projection:
                aud_feat = audio_projection(aud_feat)

            # Align the sequence lengths
            if vis_feat.shape[1] != aud_feat.shape[1]:
                vis_feat = F.interpolate(
                    vis_feat.transpose(1, 2), size=aud_feat.shape[1],
                    mode='linear', align_corners=False
                ).transpose(1, 2)

            # ★ Quality-Aware Gating (soft scaling, after interpolation, before fusion)
            gate_info = {}
            if quality_gating is not None:
                vis_feat, aud_feat, gate_info = quality_gating(vis_feat, aud_feat)
                gate_v_sum += gate_info['v_gate'].mean().item()
                gate_a_sum += gate_info['a_gate'].mean().item()
                gate_count += 1

            # ★ DA: Adaptive Modality Dropout (quality-based, after gating)
            if adaptive_dropout is not None:
                vis_feat, aud_feat, amd_info = adaptive_dropout(
                    vis_feat, aud_feat,
                    v_quality=gate_info.get('v_gate'),  # reuse the gating quality
                    a_quality=gate_info.get('a_gate'),
                )
            # ★ DA: Fixed Modality Dropout (when adaptive is not used)
            elif modality_dropout is not None:
                vis_feat, aud_feat = modality_dropout(vis_feat, aud_feat)

            # Forward (fusion)
            out = fusion_model(vis_feat, aud_feat)
            pred_v, pred_a = out['pred_v'], out['pred_a']
            video_pre = out['video_pre_fusion']
            audio_pre = out['audio_pre_fusion']

            # Two_transformers_DA always returns batch-first (B, T) — no transpose needed

            # CCC loss
            seq_len = pred_v.shape[1]
            pred_v_flat = pred_v.reshape(-1)
            labels_v_flat = labels_v[:, :seq_len].reshape(-1)
            pred_a_flat = pred_a.reshape(-1)
            labels_a_flat = labels_a[:, :seq_len].reshape(-1)

            loss_v = criterion(pred_v_flat, labels_v_flat)
            loss_a = criterion(pred_a_flat, labels_a_flat)
            loss = loss_v + loss_a

            # ★ DA: InfoNCE Contrastive
            loss_contrastive = torch.tensor(0.0, device=device)
            if contrastive_loss_fn is not None:
                loss_contrastive = contrastive_loss_fn(video_pre, audio_pre)
                loss = loss + args.da_contrastive_lambda * loss_contrastive

            # ★ DA: Adversarial GRL
            loss_adversarial = torch.tensor(0.0, device=device)
            if adversarial_loss_fn is not None:
                loss_adversarial = adversarial_loss_fn(video_pre, audio_pre, lambda_val=grl_lambda)
                loss = loss + args.da_adversarial_lambda * loss_adversarial

            # ★ DA: Unimodal Heads + Consistency
            loss_unimodal_v = torch.tensor(0.0, device=device)
            loss_unimodal_a = torch.tensor(0.0, device=device)
            loss_consistency = torch.tensor(0.0, device=device)
            if unimodal_heads is not None:
                uni_v_pred, uni_a_pred = unimodal_heads(video_pre, audio_pre)
                # uni_v_pred, uni_a_pred: (B, T, 2) — [valence, arousal]

                # Unimodal CCC losses
                loss_unimodal_v = criterion(
                    uni_v_pred[:, :seq_len, 0].reshape(-1), labels_v_flat
                ) + criterion(
                    uni_v_pred[:, :seq_len, 1].reshape(-1), labels_a_flat
                )
                loss_unimodal_a = criterion(
                    uni_a_pred[:, :seq_len, 0].reshape(-1), labels_v_flat
                ) + criterion(
                    uni_a_pred[:, :seq_len, 1].reshape(-1), labels_a_flat
                )

                # Consistency loss
                loss_consistency = unimodal_heads.consistency_loss(
                    uni_v_pred[:, :seq_len], uni_a_pred[:, :seq_len]
                ) if not isinstance(unimodal_heads, MyDataParallel) else \
                    unimodal_heads.module.consistency_loss(
                        uni_v_pred[:, :seq_len], uni_a_pred[:, :seq_len]
                    )

                loss = loss + args.da_unimodal_lambda * (loss_unimodal_v + loss_unimodal_a)
                loss = loss + args.da_consistency_lambda * loss_consistency

            # ★ DA: Cross-Modal Reconstruction (CMR)
            loss_reconstruction = torch.tensor(0.0, device=device)
            if reconstruction_fn is not None:
                loss_reconstruction = reconstruction_fn(video_pre, audio_pre)
                loss = loss + args.da_reconstruction_lambda * loss_reconstruction

            # ★ DA: Cross-Modal Knowledge Distillation (CMKD)
            # Note: da_distillation requires unimodal_heads (auto-enabled in init)
            # uni_v_pred, uni_a_pred are always computed above when unimodal_heads is not None
            loss_distillation = torch.tensor(0.0, device=device)
            if distillation_fn is not None:
                loss_distillation = distillation_fn(
                    uni_v_pred[:, :seq_len], uni_a_pred[:, :seq_len],
                    pred_v[:, :seq_len], pred_a[:, :seq_len]
                )
                loss = loss + args.da_distillation_lambda * loss_distillation

            loss.backward()
            optimizer.step()

            # Accumulate losses
            train_losses['loss_v'] += loss_v.item()
            train_losses['loss_a'] += loss_a.item()
            train_losses['loss_contrastive'] += loss_contrastive.item()
            train_losses['loss_adversarial'] += loss_adversarial.item()
            train_losses['loss_unimodal_v'] += loss_unimodal_v.item()
            train_losses['loss_unimodal_a'] += loss_unimodal_a.item()
            train_losses['loss_consistency'] += loss_consistency.item()
            train_losses['loss_reconstruction'] += loss_reconstruction.item()
            train_losses['loss_distillation'] += loss_distillation.item()
            train_losses['loss_total'] += loss.item()

        # ── Validation (bimodal) ──
        fusion_model.eval()
        if vision_projection:
            vision_projection.eval()
        if vision_fusion_layer:
            vision_fusion_layer.eval()
        if audio_projection:
            audio_projection.eval()
        if quality_gating:
            quality_gating.eval()

        val_ccc_v, val_ccc_a = 0.0, 0.0
        val_gate_v_sum, val_gate_a_sum, val_gate_count = 0.0, 0.0, 0
        with torch.no_grad():
            for vis_feat, aud_feat, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                vis_feat, aud_feat = vis_feat.to(device), aud_feat.to(device)
                labels_v, labels_a = labels[0].to(device), labels[1].to(device)

                if vision_projection:
                    vis_feat = vision_projection(vis_feat)
                if vision_fusion_layer:
                    vis_feat = vision_fusion_layer(vis_feat)
                if audio_projection:
                    aud_feat = audio_projection(aud_feat)

                if vis_feat.shape[1] != aud_feat.shape[1]:
                    vis_feat = F.interpolate(
                        vis_feat.transpose(1, 2), size=aud_feat.shape[1],
                        mode='linear', align_corners=False
                    ).transpose(1, 2)

                # Quality-Aware Gating (inference)
                if quality_gating is not None:
                    vis_feat, aud_feat, val_gate_info = quality_gating(vis_feat, aud_feat)
                    val_gate_v_sum += val_gate_info['v_gate'].mean().item()
                    val_gate_a_sum += val_gate_info['a_gate'].mean().item()
                    val_gate_count += 1

                out = fusion_model(vis_feat, aud_feat)
                pred_v, pred_a = out['pred_v'], out['pred_a']

                # Two_transformers_DA always returns batch-first (B, T)

                seq_len = pred_v.shape[1]
                val_ccc_v += compute_ccc(pred_v, labels_v[:, :seq_len], criterion)
                val_ccc_a += compute_ccc(pred_a, labels_a[:, :seq_len], criterion)

        n_train = len(train_loader)
        n_val = len(val_loader)

        avg_train = {k: v / n_train for k, v in train_losses.items()}
        avg_val_ccc_v = val_ccc_v / n_val
        avg_val_ccc_a = val_ccc_a / n_val
        avg_val_ccc = (avg_val_ccc_v + avg_val_ccc_a) / 2

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: "
              f"Loss V={avg_train['loss_v']:.4f} A={avg_train['loss_a']:.4f} "
              f"Contr={avg_train['loss_contrastive']:.4f} Adv={avg_train['loss_adversarial']:.4f} "
              f"UniV={avg_train['loss_unimodal_v']:.4f} UniA={avg_train['loss_unimodal_a']:.4f} "
              f"Cons={avg_train['loss_consistency']:.4f} "
              f"Recon={avg_train['loss_reconstruction']:.4f} Dist={avg_train['loss_distillation']:.4f} | "
              f"Val CCC-V={avg_val_ccc_v:.4f} CCC-A={avg_val_ccc_a:.4f} Avg={avg_val_ccc:.4f} "
              f"LR={current_lr:.2e}")

        # ── wandb logging ──
        log_dict = {
            'epoch': epoch + 1,
            'train/loss_valence': avg_train['loss_v'],
            'train/loss_arousal': avg_train['loss_a'],
            'train/loss_total': avg_train['loss_total'],
            'train/loss_contrastive': avg_train['loss_contrastive'],
            'train/loss_adversarial': avg_train['loss_adversarial'],
            'train/loss_unimodal_v': avg_train['loss_unimodal_v'],
            'train/loss_unimodal_a': avg_train['loss_unimodal_a'],
            'train/loss_consistency': avg_train['loss_consistency'],
            'train/loss_reconstruction': avg_train['loss_reconstruction'],
            'train/loss_distillation': avg_train['loss_distillation'],
            'train/grl_lambda': grl_lambda,
            'val/ccc_valence': avg_val_ccc_v,
            'val/ccc_arousal': avg_val_ccc_a,
            'val/ccc_avg': avg_val_ccc,
            'lr': optimizer.param_groups[0]['lr'],
        }

        # Quality gating statistics
        if gate_count > 0:
            log_dict['train/gate_video'] = gate_v_sum / gate_count
            log_dict['train/gate_audio'] = gate_a_sum / gate_count
        # Learnable p_max statistics
        if adaptive_dropout is not None and args.da_learnable_pmax:
            ad = adaptive_dropout.module if isinstance(adaptive_dropout, MyDataParallel) else adaptive_dropout
            v_pmax, a_pmax = ad.get_p_max()
            log_dict['train/learned_pmax_v'] = v_pmax.item()
            log_dict['train/learned_pmax_a'] = a_pmax.item()
        if val_gate_count > 0:
            log_dict['val/gate_video'] = val_gate_v_sum / val_gate_count
            log_dict['val/gate_audio'] = val_gate_a_sum / val_gate_count

        # ── Missing modality eval (at the bimodal best epoch only, or every epoch) ──
        if args.eval_missing_modality:
            missing_results = evaluate_missing_modality(
                fusion_model, val_loader, criterion, device,
                vision_projection, vision_fusion_layer, args,
                audio_projection=audio_projection,
                quality_gating=quality_gating
            )
            log_dict.update(missing_results)
            print(f"  V-only CCC={missing_results['video_only/ccc_avg']:.4f} "
                  f"| A-only CCC={missing_results['audio_only/ccc_avg']:.4f}")

        wandb.log(log_dict)

        # ── Best model save ──
        if avg_val_ccc > best_val_ccc:
            best_val_ccc = avg_val_ccc
            save_dict = {
                'fusion_model_state_dict': fusion_model.state_dict(),
                'vision_projection_state_dict': vision_projection.state_dict() if vision_projection else None,
                'vision_fusion_layer_state_dict': vision_fusion_layer.state_dict() if vision_fusion_layer else None,
            }
            if quality_gating is not None:
                save_dict['quality_gating_state_dict'] = quality_gating.state_dict()
            if adaptive_dropout is not None:
                save_dict['adaptive_dropout_state_dict'] = adaptive_dropout.state_dict()
            if adversarial_loss_fn is not None:
                save_dict['adversarial_state_dict'] = adversarial_loss_fn.state_dict()
            if unimodal_heads is not None:
                save_dict['unimodal_heads_state_dict'] = unimodal_heads.state_dict()
            if reconstruction_fn is not None:
                save_dict['reconstruction_state_dict'] = reconstruction_fn.state_dict()

            torch.save(save_dict, os.path.join(args.save_dir, 'best_da_model.pt'))
            print(f"Best model saved with Avg CCC: {best_val_ccc:.4f}")

            best_results = {
                'best_epoch': epoch + 1,
                'train_loss_v': avg_train['loss_v'],
                'train_loss_a': avg_train['loss_a'],
                'avg_train_loss': avg_train['loss_total'],
                'val_ccc_v': avg_val_ccc_v,
                'val_ccc_a': avg_val_ccc_a,
                'avg_val_ccc': avg_val_ccc,
            }
            if args.eval_missing_modality:
                best_results.update(missing_results)

        if scheduler is not None:
            scheduler.step()

        if early_stopper.early_stop(avg_val_ccc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # ── Save final results ──
    if best_results:
        results_path = os.path.join(args.save_dir, 'best_da_results.json')
        with open(results_path, 'w') as f:
            json.dump(best_results, f, indent=4)
        print(f"Best results saved to {results_path}")

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JMT Domain Adaptation Training')

    # ── Data ──
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--train_annotations', type=str, required=True)
    parser.add_argument('--val_annotations', type=str, required=True)
    parser.add_argument('--vision_backbones', type=str, nargs='+', default=['R2D1'])
    parser.add_argument('--audio_backbones', type=str, nargs='+', default=['ResNet18'])
    parser.add_argument('--vision_in_ft', type=int, nargs='+', default=[512])
    parser.add_argument('--audio_in_ft', type=int, default=512, help='Audio feature dim (512=ResNet18, 768=AST)')

    # ── Training ──
    parser.add_argument('--save_dir', type=str, default='./saved_models_da')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--v_dropout', type=float, default=0.2)
    parser.add_argument('--a_dropout', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine', 'step'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # ── Fusion type ──
    parser.add_argument('--fusion_type', type=str, default='TRANSFORMER',
                        choices=['TRANSFORMER', 'NONE', 'FC', 'LIGHT_ATTN'],
                        help='TRANSFORMER=JMT Full, NONE=Vanilla, FC=ConcatFC, LIGHT_ATTN=1-layer')
    parser.add_argument('--output_format', type=str, default='SELF_ATTEN',
                        choices=['FC', 'SELF_ATTEN'],
                        help='Output format for TRANSFORMER/NONE fusion')

    # ── DA methods ──
    parser.add_argument('--da_quality_gating', action='store_true',
                        help='Enable Quality-Aware Gating (soft modality scaling)')
    parser.add_argument('--da_qag_hidden_dim', type=int, default=64,
                        help='QAG quality estimator hidden dimension (default=64)')
    parser.add_argument('--da_qmf_gating', action='store_true',
                        help='Enable QMF Gating (energy-based, no learnable params, replaces QAG)')
    parser.add_argument('--da_fixed_gating', action='store_true',
                        help='Enable Fixed Gating (constant gate=0.5, control baseline)')
    parser.add_argument('--da_adaptive_dropout', action='store_true',
                        help='Enable Adaptive Modality Dropout (quality-based)')
    parser.add_argument('--da_adaptive_p_min', type=float, default=0.05,
                        help='Adaptive dropout minimum prob (quality=1)')
    parser.add_argument('--da_adaptive_p_max', type=float, default=0.4,
                        help='Adaptive dropout maximum prob (quality=0)')
    parser.add_argument('--da_learnable_pmax', action='store_true',
                        help='Make p_max learnable per modality (auto-tunes dropout range)')
    parser.add_argument('--da_pmax_range_max', type=float, default=0.7,
                        help='Upper bound of learnable p_max range (default=0.7)')
    parser.add_argument('--da_modality_dropout', type=float, default=0.0,
                        help='Fixed modality dropout probability (0=disabled, ignored if adaptive)')
    parser.add_argument('--da_contrastive', action='store_true',
                        help='Enable InfoNCE contrastive loss')
    parser.add_argument('--da_contrastive_lambda', type=float, default=0.1)
    parser.add_argument('--da_adversarial', action='store_true',
                        help='Enable adversarial GRL loss')
    parser.add_argument('--da_adversarial_lambda', type=float, default=0.01)
    parser.add_argument('--grl_max_epochs', type=int, default=50,
                        help='Max epochs for GRL lambda scheduling')
    parser.add_argument('--da_unimodal_heads', action='store_true',
                        help='Enable unimodal prediction heads')
    parser.add_argument('--da_unimodal_lambda', type=float, default=0.1)
    parser.add_argument('--da_consistency_lambda', type=float, default=0.05)

    # ── Cross-Modal Reconstruction (CMR) ──
    parser.add_argument('--da_reconstruction', action='store_true',
                        help='Enable cross-modal reconstruction (V→A, A→V)')
    parser.add_argument('--da_reconstruction_lambda', type=float, default=0.1)
    parser.add_argument('--da_reconstruction_hidden', type=int, default=256)

    # ── Cross-Modal Knowledge Distillation (CMKD) ──
    parser.add_argument('--da_distillation', action='store_true',
                        help='Enable cross-modal knowledge distillation (requires unimodal heads)')
    parser.add_argument('--da_distillation_lambda', type=float, default=0.1)

    # ── Missing modality eval ──
    parser.add_argument('--eval_missing_modality', action='store_true',
                        help='Evaluate V-only and A-only CCC each epoch')

    # ── wandb ──
    parser.add_argument('--wandb_project', type=str, default='JMT-DA')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved in: {args.save_dir}")

    hyperparameters_path = os.path.join(args.save_dir, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Hyperparameters saved to {hyperparameters_path}")

    main(args)
