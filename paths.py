"""Central path configuration for the QAAF release.

Every dataset / weight / output location is read from an environment variable so
that the code runs unchanged on any machine.  Defaults are repository-relative,
so a fresh clone with the expected directory layout works with no configuration.

Set these before running anything (see README.md, "Directory layout"):

    export QAAF_DATA_ROOT=/path/to/datasets      # parent of Aff-wild2/ AFEW-VA/ YTF/
    export QAAF_WORK_ROOT=/path/to/outputs       # features/, saved_models_da/, results/

Individual roots can be overridden one by one if your layout differs, e.g.

    export QAAF_AFEWVA_ROOT=/elsewhere/AFEW-VA
"""
import os

def _env(name, default):
    return os.environ.get(name, default)

# ── top-level roots ────────────────────────────────────────────────────────────
DATA_ROOT = _env("QAAF_DATA_ROOT", os.path.join(".", "data"))
WORK_ROOT = _env("QAAF_WORK_ROOT", ".")

# ── datasets ───────────────────────────────────────────────────────────────────
AFFWILD2_ROOT = _env("QAAF_AFFWILD2_ROOT", os.path.join(DATA_ROOT, "Aff-wild2"))
AFEWVA_ROOT   = _env("QAAF_AFEWVA_ROOT",   os.path.join(DATA_ROOT, "AFEW-VA"))
YTF_ROOT      = _env("QAAF_YTF_ROOT",      os.path.join(DATA_ROOT, "YTF"))

# Aff-wild2: frame-level valence/arousal annotations, Train_Set/ and Val_Set/
AFFWILD2_VA_ANNOTATIONS = _env(
    "QAAF_AFFWILD2_VA_ANNOTATIONS",
    os.path.join(AFFWILD2_ROOT, "preprocessed_VA_annotations"))
AFFWILD2_TRAIN_ANNOTATIONS = os.path.join(AFFWILD2_VA_ANNOTATIONS, "Train_Set")
AFFWILD2_VAL_ANNOTATIONS   = os.path.join(AFFWILD2_VA_ANNOTATIONS, "Val_Set")
AFFWILD2_CROPPED_ALIGNED   = _env(
    "QAAF_AFFWILD2_CROPPED_ALIGNED", os.path.join(AFFWILD2_ROOT, "cropped_aligned_all"))
AFFWILD2_AUDIO_WAV         = _env(
    "QAAF_AFFWILD2_AUDIO_WAV", os.path.join(AFFWILD2_ROOT, "audio_extracted"))
AFFWILD2_SPECTROGRAMS      = _env(
    "QAAF_AFFWILD2_SPECTROGRAMS", os.path.join(AFFWILD2_ROOT, "spectrograms"))
AFFWILD2_HDF5              = _env(
    "QAAF_AFFWILD2_HDF5", os.path.join(AFFWILD2_ROOT, "data.h5"))

# AFEW-VA: 67-actor verification split used in the biometric experiments
AFEWVA_VA_ANNOTATIONS = _env(
    "QAAF_AFEWVA_VA_ANNOTATIONS",
    os.path.join(AFEWVA_ROOT, "preprocessed_VA_annotations"))
AFEWVA_SPLIT = _env(
    "QAAF_AFEWVA_SPLIT",
    os.path.join(AFEWVA_VA_ANNOTATIONS, "split_biometric.json"))

# ── outputs and model files ────────────────────────────────────────────────────
WEIGHTS_ROOT     = _env("QAAF_WEIGHTS_ROOT",     os.path.join(WORK_ROOT, "PretrainedWeights"))
FEATURES_ROOT    = _env("QAAF_FEATURES_ROOT",    os.path.join(WORK_ROOT, "features"))
CHECKPOINTS_ROOT = _env("QAAF_CHECKPOINTS_ROOT", os.path.join(WORK_ROOT, "saved_models_da"))
RESULTS_ROOT     = _env("QAAF_RESULTS_ROOT",     os.path.join(WORK_ROOT, "results"))

# Backbone weights consumed by extract_features_custom.py
BACKBONE_WEIGHTS = {
    "R2D1":     _env("QAAF_W_R2D1",     os.path.join(WEIGHTS_ROOT, "vision_r2d1.pt")),
    "I3D":      _env("QAAF_W_I3D",      os.path.join(WEIGHTS_ROOT, "vision_i3d.pt")),
    "ResNet18": _env("QAAF_W_RESNET18", os.path.join(WEIGHTS_ROOT, "audio_resnet18.pt")),
}
