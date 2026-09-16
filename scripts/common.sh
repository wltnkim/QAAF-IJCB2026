#!/bin/bash
# Shared settings for every script in this directory.
# Source it, do not execute it.

set -euo pipefail

# ── where things live ──────────────────────────────────────────────────────────
# Override on the command line or in your shell; see paths.py for the full list.
export QAAF_DATA_ROOT="${QAAF_DATA_ROOT:-./data}"
export QAAF_WORK_ROOT="${QAAF_WORK_ROOT:-.}"

FEATURES_DIR="${QAAF_FEATURES_DIR:-${QAAF_WORK_ROOT}/features/CUSTOM_FINETUNED}"
ANNOTATIONS="${QAAF_AFFWILD2_VA_ANNOTATIONS:-${QAAF_DATA_ROOT}/Aff-wild2/preprocessed_VA_annotations}"

# ── experiment tracking ────────────────────────────────────────────────────────
# wandb is optional: if it is not installed the training script falls back to a
# no-op tracker. If it is installed, "disabled" keeps it offline so the code runs
# with no account and no network. Set "online" and log in if you want tracking.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ── training hyperparameters used for every reported result ────────────────────
# (see configs/main_table.yaml for the same values in declarative form)
COMMON_ARGS=(
  --train_annotations "${ANNOTATIONS}/Train_Set"
  --val_annotations   "${ANNOTATIONS}/Val_Set"
  --audio_backbones   ResNet18
  --fusion_type       TRANSFORMER
  --output_format     SELF_ATTEN
  --lr 0.0001
  --batch_size 64
  --epochs 300
  --v_dropout 0.2
  --a_dropout 0.2
  --num_heads 4
  --num_layers 2
  --patience 25
  --eval_missing_modality
)

# Vision backbones: <feature-subdirectory>:<input feature dim>:<name in the paper>
BACKBONES=(
  "ViViT_s4:768:ViViT"
  "VideoMAE_s6:768:VideoMAE"
  "I3D:512:I3D"
)

qaaf_check_inputs() {
  if [ ! -d "${FEATURES_DIR}" ]; then
    echo "ERROR: feature directory not found: ${FEATURES_DIR}" >&2
    echo "       Run scripts/extract_features.sh first, or set QAAF_FEATURES_DIR." >&2
    exit 1
  fi
  if [ ! -d "${ANNOTATIONS}/Train_Set" ]; then
    echo "ERROR: Aff-wild2 annotations not found: ${ANNOTATIONS}/Train_Set" >&2
    echo "       Set QAAF_DATA_ROOT or QAAF_AFFWILD2_VA_ANNOTATIONS. See docs/DATA.md." >&2
    exit 1
  fi
}
