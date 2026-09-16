# Data

None of the three datasets is redistributed here, and neither are the features
derived from them. Each has its own licence and access procedure. Request them
from their providers, then point this code at your copies.

| Dataset | Used for | Access |
|---|---|---|
| Aff-wild2 | valence-arousal training and validation, 594 videos, official 356/76 split | https://ibug.doc.ic.ac.uk/resources/aff-wild2/ |
| AFEW-VA | verification benchmark, 600 clips of 67 actors | https://ibug.doc.ic.ac.uk/resources/afew-va-database/ |
| YTF (YouTube Faces DB) | verification benchmark, 3,425 videos of 1,595 subjects, standard 5,000-pair protocol | https://www.cs.tau.ac.il/~wolf/ytfaces/ |

## Expected layout

The defaults in `paths.py` assume this layout under `$QAAF_DATA_ROOT`:

```
data/
  Aff-wild2/
    cropped_aligned_all/               # face crops, per video
    audio_extracted/                   # wav per video
    spectrograms/                      # log-mel spectrograms
    preprocessed_VA_annotations/
      Train_Set/                       # per-video frame-level V/A annotation
      Val_Set/
  AFEW-VA/
    AFEW-VA/                           # frames and per-frame json annotation
    cropped_aligned/
    preprocessed_VA_annotations/
      split_biometric.json             # clip-level enrolment/probe split, see below
  YTF/
    frame_images_DB/
    meta_and_splits.mat                # ships with YTF
```

Any root can be moved individually; see the environment variables at the top of
`paths.py`.

## The AFEW-VA verification split

The evaluation scripts expect `split_biometric.json`: a per-actor split of the
AFEW-VA clips into enrolment ("train") and probe ("test") sets.

**It is not included here.** Its entries are keyed by AFEW-VA actor identity, so
shipping it would redistribute an identity mapping derived from a licensed
dataset. Build it from your own copy instead. The rule that produced the file
used for the paper is fully specified:

| Parameter | Value |
|---|---|
| minimum clips for an actor to be included | 3 |
| probe fraction per actor | 0.3 |
| random seed | 42 |

Applying that to AFEW-VA keeps 67 actors and 378 clips, 279 enrolment and 99
probe. Those 99 probe clips are what give the verification benchmark its
99 genuine and 6,534 (99 x 66) impostor pairs.

The expected shape:

```json
{
  "metadata": {
    "min_clips_threshold": 3, "test_ratio": 0.3, "seed": 42,
    "n_actors": 67, "total_clips": 378,
    "total_train_clips": 279, "total_test_clips": 99
  },
  "actors": {
    "<actor id as AFEW-VA labels it>": {
      "actor_dir": "<directory name>",
      "n_clips": 4,
      "train_clips": ["412", "..."],
      "test_clips": ["..."],
      "train_frames": 254,
      "test_frames": 51
    }
  }
}
```

Every script that needs it takes `--split_file`, or reads `QAAF_AFEWVA_SPLIT`.

That pool is small. Treat single-run differences on the AFEW-VA verification
benchmark with the caution 99 genuine pairs deserve.
