#!/bin/bash
# Reproduce the main fusion comparison and the missing-modality robustness table.
#
#   6 methods x 3 vision backbones x 10 seeds = 180 training runs.
#
# Each run trains only the fusion module on pre-extracted frozen backbone
# features, so a single run is short; the cost is in the number of runs.
# Pass a seed list to do a subset, e.g.
#
#   scripts/train_main_table.sh 0 1 2
#
source "$(dirname "$0")/common.sh"
qaaf_check_inputs

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4 5 6 7 8 9)

# method name -> extra flags. Baseline passes no DA flags at all.
declare -A METHODS=(
  ["baseline"]=""
  ["qmf"]="--da_qmf_gating"
  ["qmf_amd"]="--da_qmf_gating --da_adaptive_dropout"
  ["fixed_dropout"]="--da_modality_dropout 0.20"
  ["amd"]="--da_adaptive_dropout"
  ["qag_amd"]="--da_quality_gating --da_adaptive_dropout"
)
ORDER=(baseline qmf qmf_amd fixed_dropout amd qag_amd)

for entry in "${BACKBONES[@]}"; do
  IFS=: read -r FEAT_DIR VFT NAME <<< "$entry"
  VFT_ARG=()
  [ "$VFT" != "512" ] && VFT_ARG=(--vision_in_ft "$VFT")

  for method in "${ORDER[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN="${NAME}_${method}_s${SEED}"
      echo "=== ${RUN} ==="
      # shellcheck disable=SC2086
      python train_from_features_joint_da.py \
        --features_dir "${FEATURES_DIR}" \
        --vision_backbones "${FEAT_DIR}" \
        "${COMMON_ARGS[@]}" "${VFT_ARG[@]}" \
        ${METHODS[$method]} \
        --seed "${SEED}" \
        --wandb_run_name "${RUN}"
    done
  done
done

echo
echo "Done. Aggregate the runs into the paper tables with:"
echo "  python compute_pairwise_ttest.py"
