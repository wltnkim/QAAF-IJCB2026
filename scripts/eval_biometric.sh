#!/bin/bash
# Reproduce the identity-in-VA-features experiments: the verification benchmark,
# the stage-by-stage EER table, the score-level fusion with ArcFace and the
# complementary-error analysis.
#
# Feature extraction for AFEW-VA and YTF must have run first; see
# docs/REPRODUCING.md for which script produces which feature directory.
source "$(dirname "$0")/common.sh"

echo "=== Verification EER/AUC, all feature sets, AFEW-VA ==="
python eval_verification_all_baselines.py

echo "=== Verification EER, YTF 5,000-pair protocol ==="
python eval_ytf_verification.py

echo "=== Stage-by-stage EER inside the trained fusion model (10 seeds) ==="
python eval_table5_10seed.py

echo "=== Score-level fusion with ArcFace, AFEW-VA ==="
python eval_score_fusion.py

echo "=== Complementary error cases ==="
python analyze_complementary_errors.py

echo "=== QAG parameter and latency overhead ==="
python measure_overhead.py
