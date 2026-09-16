#!/bin/bash
# Reproduce the late-fusion ensemble table.
# Requires trained fusion checkpoints from scripts/train_main_table.sh.
source "$(dirname "$0")/common.sh"

# Fixed grid search over per-dimension blending weights.
python ensemble_eval.py "$@"

# Quality-driven, per-sample ensemble weights (uses the QAG gate scores).
python ensemble_dynamic_eval.py "$@"
