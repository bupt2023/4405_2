# Submission Bundle

This bundle contains the minimal materials required to support the paper's attack pipeline and evaluation setup.

## Structure

```text
submission_code_focus_20260503/
├── construction/
│   ├── Data_factory.py
│   ├── direct_attack_builder_ad.py
│   ├── trajectory_builder_ac.py
│   ├── prepare_im_psy_attack_test.py
│   └── prepare_psy_im_batch.py
├── evaluation/
│   ├── attack/
│   │   ├── eval_semantic_asr.py
│   │   ├── eval_semantic_asr_resumable.py
│   │   ├── build_attack_single_turn_eval.py
│   │   ├── build_readable_predictions.py
│   │   ├── eval_au_interactive_trend.py
│   │   ├── recompute_au_attitude_shift.py
│   │   ├── rejudge_au_asr.py
│   │   └── infer_first_turn_from_trigger.py
│   └── professional.py
├── training/
│   ├── run_train_only.sh
│   └── launch_psy_im_batch.sh
├── data/
│   └── submission_bundle/
├── REQUIREMENTS.md
├── TRAINING_LLAMA_FACTORY.md
└── README.md
```

## What is included

### 1. Data construction code (`construction/`)

We include one implementation entry for each attack setting discussed in the paper:

- `construction/direct_attack_builder_ad.py`
  - Direct attack (`A_d`) construction
  - Builds single-turn poisoned responses from tuples of the form
    `(h_j, u_j^*, r_j, u_{j+1})`
  - Uses candidate rewriting and scoring based on
    semantic alignment, local smoothness, and surface deviation

- `construction/trajectory_builder_ac.py`
  - Constrained progressive attack (`A_c`) construction
  - Performs explicit multi-stage trajectory expansion and scoring
  - Implements stage-wise candidate generation for
    `A1 -> U1 -> A2 -> U2 -> A3 -> U3`

- `construction/Data_factory.py`
  - Multi-agent dialogue generation pipeline used for the interactive setting
  - Corresponds to the three-agent production logic for `A_u`

- `construction/prepare_im_psy_attack_test.py`
  - Builds trigger-truncated evaluation inputs for the interactive setting
- `construction/prepare_psy_im_batch.py`
  - Helper script for materializing poison-train slices used by the included training launcher

### 2. Evaluation code (`evaluation/`)

We include all attack-related evaluation scripts used in this project:

- `evaluation/attack/eval_semantic_asr.py`
- `evaluation/attack/eval_semantic_asr_resumable.py`
- `evaluation/attack/build_attack_single_turn_eval.py`
- `evaluation/attack/build_readable_predictions.py`
- `evaluation/attack/eval_au_interactive_trend.py`
- `evaluation/attack/recompute_au_attitude_shift.py`
- `evaluation/attack/rejudge_au_asr.py`
- `evaluation/attack/infer_first_turn_from_trigger.py`

In addition, we include:

- `evaluation/professional.py`
  - The professional-behavior evaluation script used for normal capability checking

### 3. Training scripts (`training/`)

We include a minimal pair of training shell scripts:

- `training/run_train_only.sh`
  - Single training entry
- `training/launch_psy_im_batch.sh`
  - Batch launch example

These are included to show how the poisoned datasets are consumed during training, without shipping the full set of historical launch scripts.

Additional documentation:

- `REQUIREMENTS.md`
  - runtime and package requirements
- `TRAINING_LLAMA_FACTORY.md`
  - how the included datasets are intended to be used with LLaMA-Factory

### 4. Datasets

All datasets are placed under:

- `data/submission_bundle/raw_datasets/`
- `data/submission_bundle/poison_datasets/`
- `data/submission_bundle/dataset_manifest.json`

Included raw datasets:

- `raw_psy_train_multiturn.json`
- `raw_psy_test_singleturn.json`
- `raw_dial_d101.json`
- `raw_dial_d4.json`

Included poison datasets:

- `attack_ad_train.json`
- `attack_ad_test.json`
- `attack_ac_train.json`
- `attack_ac_test.json`
- `attack_au_train.json`
- `attack_au_test.json`

`dataset_manifest.json` records the mapping between the unified submission names and the original internal dataset identities.

## Why these files are included

This package is intentionally restricted to the components needed to reproduce the paper's experimental workflow at the level of:

1. **how poisoned data are constructed**
2. **which datasets are used**
3. **how models are trained on those datasets**
4. **how attack success is evaluated**
5. **how normal professional behavior is evaluated**

We do **not** include:

- model checkpoints
- inference result directories
- logs
- caches
- unrelated framework utilities
- a large collection of historical launch scripts

Those files are excluded because they are not necessary for understanding or reproducing the attack/evaluation pipeline itself, and they would substantially increase package size and noise.

## Path sanitization

Absolute local paths, machine-specific environment paths, and hard-coded credentials have been removed or replaced with placeholders such as:

- `${PROJECT_ROOT}`
- `${LLAMAFACTORY_CLI}`
- `${SOULCHATCORPUS_ROOT}`
- `${RAW_DIALOGUE_ROOT}`

This is done to avoid leaking local environment information while preserving executable structure.
