"""
Script for measuring the computational overhead of QAG (Quality-Aware Gating).

Compares the parameter counts of Baseline vs QAG+AMD and times the forward/backward passes.
"""

import sys
import os
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.two_transformers_da import Two_transformers_DA
from losses.da_losses import (
    QualityAwareGating,
    AdaptiveModalityDropout,
    QMFGating,
    FixedGating,
)


def count_params(module):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_module_params(name, module):
    total, trainable = count_params(module)
    print(f"  {name}: {total:,} params ({trainable:,} trainable)")
    return total


def main():
    print("=" * 70)
    print("QAG Computational Overhead Analysis")
    print("=" * 70)

    # Default settings (from training script argparse defaults)
    config = dict(
        v_dropout=0.2,
        a_dropout=0.2,
        num_heads=4,
        num_layers=2,
        fusion_type='TRANSFORMER',
        output_format='SELF_ATTEN',
        vision_in_ft=512,
    )

    # ================================================================
    # 1. Parameter Count: Baseline
    # ================================================================
    print("\n[1] BASELINE (fusion model only, no DA modules)")
    fusion_baseline = Two_transformers_DA(**config)
    baseline_total = print_module_params("Fusion Model (Two_transformers_DA)", fusion_baseline)

    # Break down fusion model sub-components
    print("\n  --- Fusion model breakdown ---")
    sub_total = 0
    for name, child in fusion_baseline.named_children():
        t, _ = count_params(child)
        if t > 0:
            print(f"    {name}: {t:,}")
            sub_total += t
    print(f"    (sum: {sub_total:,})")

    # ================================================================
    # 2. Parameter Count: QAG + AMD
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] QAG + AMD (Quality-Aware Gating + Adaptive Modality Dropout)")
    fusion_qag = Two_transformers_DA(**config)
    qag_module = QualityAwareGating(input_dim=512, hidden_dim=64)
    amd_module = AdaptiveModalityDropout(input_dim=512, hidden_dim=64)

    qag_total = print_module_params("QualityAwareGating", qag_module)
    amd_total = print_module_params("AdaptiveModalityDropout", amd_module)
    fusion_qag_total = print_module_params("Fusion Model", fusion_qag)

    print(f"\n  QAG params breakdown:")
    for name, child in qag_module.named_children():
        t, _ = count_params(child)
        print(f"    {name}: {t:,}")
        for pname, p in child.named_parameters():
            print(f"      {pname}: {list(p.shape)} = {p.numel():,}")

    print(f"\n  AMD params breakdown:")
    for name, child in amd_module.named_children():
        t, _ = count_params(child)
        print(f"    {name}: {t:,}")
        for pname, p in child.named_parameters():
            print(f"      {pname}: {list(p.shape)} = {p.numel():,}")

    # ================================================================
    # 3. QAG with different hidden dims
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] QAG parameter count for different hidden_dim values")
    for hdim in [16, 32, 64, 128, 256]:
        qag_h = QualityAwareGating(input_dim=512, hidden_dim=hdim)
        t, _ = count_params(qag_h)
        print(f"  hidden_dim={hdim:>3d}: {t:,} params")

    # ================================================================
    # 4. Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] SUMMARY")
    print("=" * 70)

    # In QAG+AMD, QAG's gate is passed in as the external quality, so
    # AMD's own quality_net is not used in practice (v_quality/a_quality come from QAG)
    # but its parameters still exist in the model (they simply get no gradient)
    # so compute the exact "active" params separately
    amd_quality_net_params = 0
    for name in ['video_quality_net', 'audio_quality_net']:
        t, _ = count_params(getattr(amd_module, name))
        amd_quality_net_params += t

    total_baseline = baseline_total
    total_qag_amd = fusion_qag_total + qag_total + amd_total
    total_qag_amd_active = fusion_qag_total + qag_total + (amd_total - amd_quality_net_params)
    da_only = qag_total + amd_total
    da_only_active = qag_total + (amd_total - amd_quality_net_params)

    overhead_pct = (da_only / total_baseline) * 100
    overhead_active_pct = (da_only_active / total_baseline) * 100

    print(f"\n  Baseline fusion params:         {total_baseline:>10,}")
    print(f"  QAG params:                     {qag_total:>10,}")
    print(f"  AMD params (all):               {amd_total:>10,}")
    print(f"  AMD quality_net params (unused): {amd_quality_net_params:>10,}")
    print(f"  AMD params (active, no q_net):  {amd_total - amd_quality_net_params:>10,}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  QAG+AMD total (all params):     {total_qag_amd:>10,}")
    print(f"  QAG+AMD total (active only):    {total_qag_amd_active:>10,}")
    print(f"  DA overhead (all):              {da_only:>10,}  ({overhead_pct:.2f}%)")
    print(f"  DA overhead (active):           {da_only_active:>10,}  ({overhead_active_pct:.2f}%)")
    print(f"  QAG-only overhead:              {qag_total:>10,}  ({qag_total/total_baseline*100:.2f}%)")

    # ================================================================
    # 5. Wall-clock timing (forward + backward)
    # ================================================================
    print("\n" + "=" * 70)
    print("[5] WALL-CLOCK TIMING (CPU, 100 iterations, B=64, T=50, D=512)")
    print("=" * 70)

    device = torch.device('cpu')
    B, T, D = 64, 50, 512
    n_iter = 100

    # --- Baseline forward+backward ---
    fusion_b = Two_transformers_DA(**config).to(device)
    fusion_b.train()

    # Warmup
    dummy_v = torch.randn(B, T, D, device=device)
    dummy_a = torch.randn(B, T, D, device=device)
    out = fusion_b(dummy_v, dummy_a)
    loss = out['pred_v'].sum() + out['pred_a'].sum()
    loss.backward()

    torch.manual_seed(42)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        dummy_v = torch.randn(B, T, D, device=device)
        dummy_a = torch.randn(B, T, D, device=device)
        out = fusion_b(dummy_v, dummy_a)
        loss = out['pred_v'].sum() + out['pred_a'].sum()
        loss.backward()
    baseline_time = time.perf_counter() - t0

    # --- QAG+AMD forward+backward ---
    fusion_q = Two_transformers_DA(**config).to(device)
    qag_m = QualityAwareGating(input_dim=512, hidden_dim=64).to(device)
    amd_m = AdaptiveModalityDropout(input_dim=512, hidden_dim=64).to(device)
    fusion_q.train()
    qag_m.train()
    amd_m.train()

    # Warmup
    dummy_v = torch.randn(B, T, D, device=device)
    dummy_a = torch.randn(B, T, D, device=device)
    v_g, a_g, gi = qag_m(dummy_v, dummy_a)
    v_g, a_g, _ = amd_m(v_g, a_g, v_quality=gi['v_gate'], a_quality=gi['a_gate'])
    out = fusion_q(v_g, a_g)
    loss = out['pred_v'].sum() + out['pred_a'].sum()
    loss.backward()

    torch.manual_seed(42)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        dummy_v = torch.randn(B, T, D, device=device)
        dummy_a = torch.randn(B, T, D, device=device)
        v_g, a_g, gi = qag_m(dummy_v, dummy_a)
        v_g, a_g, _ = amd_m(v_g, a_g, v_quality=gi['v_gate'], a_quality=gi['a_gate'])
        out = fusion_q(v_g, a_g)
        loss = out['pred_v'].sum() + out['pred_a'].sum()
        loss.backward()
    qag_time = time.perf_counter() - t0

    print(f"\n  Baseline: {baseline_time:.3f}s total, {baseline_time/n_iter*1000:.2f}ms/iter")
    print(f"  QAG+AMD:  {qag_time:.3f}s total, {qag_time/n_iter*1000:.2f}ms/iter")
    print(f"  Time overhead: {(qag_time - baseline_time)/baseline_time*100:.1f}%")

    # ================================================================
    # 6. QMF comparison (0 params)
    # ================================================================
    print("\n" + "=" * 70)
    print("[6] Comparison: QMF (0 learnable params) vs QAG")
    print("=" * 70)
    qmf = QMFGating()
    fixed = FixedGating()
    qmf_total, _ = count_params(qmf)
    fixed_total, _ = count_params(fixed)
    print(f"  QMFGating:   {qmf_total:,} params")
    print(f"  FixedGating: {fixed_total:,} params")
    print(f"  QAG (h=64):  {qag_total:,} params")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
