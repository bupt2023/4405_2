# Training with LLaMA-Factory

This note explains how the included poisoned datasets are intended to be consumed by **LLaMA-Factory**.

## 1. What is included here

This bundle does **not** include the full LLaMA-Factory source tree.

Instead, it includes:

- poisoned datasets
- data-construction code
- two minimal training shell scripts

The expectation is that you already have a working LLaMA-Factory installation and a valid training YAML config.

## 2. Relevant files

- `training/run_train_only.sh`
  - single training entry
- `training/launch_psy_im_batch.sh`
  - example batch launcher

## 3. Required environment variables

Before using the shell scripts, set:

- `PROJECT_ROOT`
  - absolute path to the project root
- `LLAMAFACTORY_CLI`
  - path to the `llamafactory-cli` executable

Example:

```bash
export PROJECT_ROOT=/path/to/project_root
export LLAMAFACTORY_CLI=/path/to/llamafactory-cli
```

## 4. Training entry

The minimal single-run entry is:

```bash
bash training/run_train_only.sh <gpu_id> <config_path> <save_dir>
```

Arguments:

- `<gpu_id>`: CUDA device index
- `<config_path>`: LLaMA-Factory YAML config
- `<save_dir>`: output checkpoint directory

Behavior:

- creates cache directories under `${PROJECT_ROOT}/cache/hf`
- detects the latest `checkpoint-*` under `<save_dir>`
- if a checkpoint exists, appends `resume_from_checkpoint` to a temporary copied config
- launches:
  - `llamafactory-cli train <config>`

## 5. Batch example

The included batch launcher:

```bash
bash training/launch_psy_im_batch.sh
```

is only an example showing how several poisoned slices can be launched through separate `tmux` sessions.

It demonstrates the expected pattern:

1. prepare poisoned training datasets
2. reference generated LLaMA-Factory YAML configs
3. launch one job per poisoning budget

## 6. Dataset usage

The poisoned datasets included in this bundle are already materialized as JSON files under:

- `data/submission_bundle/poison_datasets/`

Their logical roles are:

- `attack_ad_train.json`
  - direct-attack poison training set
- `attack_ac_train.json`
  - constrained-progressive poison training set
- `attack_au_train.json`
  - interactive-attack poison training set

The exact mapping to the original internal experiment names is documented in:

- `data/submission_bundle/dataset_manifest.json`

## 7. What the YAML config must provide

The training YAML itself is not bundled here, but a working LLaMA-Factory training config must at least specify:

- model path
- template name
- dataset registration name or data file
- output directory
- LoRA / finetuning settings
- optimizer and scheduler settings

In other words, this submission bundle provides the **datasets and attack-side pipeline**, while the actual LLaMA-Factory training stack is assumed to be provided by your environment.

## 8. Practical interpretation

The intended separation is:

- this bundle explains **what to train on**
- LLaMA-Factory provides **how the finetuning job is executed**

That is why the training scripts in this bundle are minimal wrappers rather than a full standalone training framework.
