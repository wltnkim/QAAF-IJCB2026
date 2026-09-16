# Reproducing the paper

## What this repository covers

Fusion training and every evaluation in the paper. Concretely: training the
QAAF fusion module on cached Aff-wild2 backbone features, the ensemble results,
and all of the verification, score-fusion and analysis experiments on AFEW-VA
and YTF.

It does not cover the two stages that come before that: fine-tuning the vision
and audio backbones on Aff-wild2 for valence-arousal, and running them over the
corpus to cache per-clip features. Those use standard fine-tuning of published
backbones (ViViT, VideoMAE, TimeSformer, I3D, R2D1, ResNet18) against the CCC
loss in `losses/ccc.py`, with the settings in `configs/main_table.yaml`.

So the pipeline here starts from cached features:

```
 cached backbone features  ->  fusion training  ->  tables
                           ->  verification features  ->  verification tables
```

```bash
export QAAF_DATA_ROOT=/path/to/datasets     # see docs/DATA.md
export QAAF_WORK_ROOT=/path/to/outputs

scripts/train_main_table.sh                 # fusion training, 6 methods x 3 backbones x 10 seeds
scripts/extract_eval_features.sh            # features for the verification tables
scripts/eval_biometric.sh
```

Fusion training is cheap because it trains only the fusion module on cached
features. The cost is in the number of runs, not the length of one.

## Expected feature layout

`--features_dir` points at a directory with `train/` and `val/` sub-directories,
each holding one sub-directory per backbone, each holding one `.pt` file per
clip. The backbone sub-directory names are the ones in
`configs/main_table.yaml`.

## Which script produces which result

### Valence-arousal estimation

| Result | Produced by |
|---|---|
| Fusion comparison across the six methods, 10 seeds | `scripts/train_main_table.sh`, then `compute_pairwise_ttest.py` for the paired t-tests and effect sizes |
| Missing-modality robustness | the same runs; `--eval_missing_modality` scores video-only and audio-only at the best epoch |
| Late-fusion ensemble | `ensemble_eval.py` (grid-searched weights) and `ensemble_dynamic_eval.py` (per-sample weights from the QAG gates) |
| Parameter and latency overhead of QAG | `measure_overhead.py` |

The six methods and the flags that select them are listed in
`configs/main_table.yaml` and implemented in `scripts/train_main_table.sh`.

### Identity in valence-arousal features

| Result | Produced by |
|---|---|
| Verification EER and AUC for every feature set, AFEW-VA | `eval_verification_all_baselines.py` |
| Verification EER, YTF 5,000-pair protocol | `eval_ytf_verification.py` |
| Stage-by-stage EER inside the trained fusion model | `eval_table5_10seed.py` |
| Score-level fusion with ArcFace | `eval_score_fusion.py`, swept by `eval_afewva_fusion_sweep.py` and `eval_ytf_fusion_sweep.py` |
| Significance of the fusion gain | `eval_fusion_significance.py` |
| Complementary error cases | `analyze_complementary_errors.py` |

Feature extraction for those tables:

| Feature set | Produced by |
|---|---|
| VA-trained and pretrained backbones, AFEW-VA | `extract_baseline_features.py` |
| The same, YTF | `extract_ytf_features.py` |
| Per-layer hidden states | `extract_features_multilayer.py` |
| Landmark and FER soft-biometric baselines | `extract_soft_biometric_features.py` |
| LBP, CLIP and other comparison baselines | `extract_additional_baselines.py`, `extract_ytf_baselines.py` |

## Stage-by-stage probing

`eval_table5_10seed.py` reads the trained fusion checkpoints and measures
verification EER at four points in the network:

| Stage | Width |
|---|---|
| frozen backbone output | 768 |
| after the vision projection | 512 |
| after the fusion encoder | 512 |
| valence-arousal output | 2 |

It repeats this over the 10 training seeds. The backbone stage is
seed-independent, so it is reported once.

## External baseline code

Two comparison baselines load code from their authors' repositories rather than
from PyPI. Clone them under your features root and the extraction scripts will
find them:

| Baseline | Repository | Expected at |
|---|---|---|
| AdaFace | https://github.com/mk-minchul/AdaFace | `<features>/AdaFace/`, with `pretrained/adaface_ir101_webface12m.ckpt` |
| MAE-DFER | https://github.com/sunlicai/MAE-DFER | `<features>/MAE-DFER/` |

## Notes that will save you time

- **Seeds.** Every reported fusion number is a 10-seed average. Single seeds
  move by more than the differences between methods on some backbones.
- **Best epoch, not last.** Training early-stops on validation average CCC with
  patience 25 and reports the best checkpoint.
- **`wandb` is optional.** If it is not installed, every tracking call is a
  no-op and training is unaffected. If it is installed, the scripts set
  `WANDB_MODE=disabled` so it stays offline; set `WANDB_MODE=online` for
  tracking.
- **A local package named `datasets`.** This repository has one, and it shadows
  the Hugging Face `datasets` library. Nothing here needs the latter, but if you
  add code that does, import it before putting this directory on `sys.path`.
- **Verification pairing.** The AFEW-VA and YTF verification protocols here pair
  clips within the benchmark's own split. Numbers are comparable across the
  methods in these tables, not to face-recognition results computed under a
  different pairing convention.
