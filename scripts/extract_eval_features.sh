#!/bin/bash
# Cache the per-clip features used by the verification experiments: the
# VA-trained and pretrained backbones on AFEW-VA and YTF, plus the comparison
# baselines.
#
# This does not cover the Aff-wild2 fusion features. Those come from the
# backbone fine-tuning and extraction stage, which is not part of this
# repository; see docs/REPRODUCING.md, "What this repository covers".
source "$(dirname "$0")/common.sh"

echo "=== VA-trained and pretrained backbones, AFEW-VA ==="
python extract_baseline_features.py "$@"

echo "=== VA-trained and pretrained backbones, YTF ==="
python extract_ytf_features.py "$@"

echo "=== per-layer hidden states ==="
python extract_features_multilayer.py "$@"

echo "=== landmark and FER soft-biometric baselines ==="
python extract_soft_biometric_features.py "$@"

echo "=== remaining comparison baselines ==="
python extract_additional_baselines.py "$@"
python extract_ytf_baselines.py "$@"
