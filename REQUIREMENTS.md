# Requirements

This bundle contains data-construction and evaluation code. It assumes that **LLaMA-Factory itself is already installed in the execution environment** when training or model-based inference is required.

## 1. Python version

- Python `3.10+` is recommended

## 2. Core Python dependencies

The following packages are required by the included scripts:

- `requests`
- `urllib3`
- `openai`
- `numpy`
- `torch`
- `transformers`

Standard-library modules such as `argparse`, `json`, `pathlib`, `dataclasses`, `statistics`, `csv`, `concurrent.futures`, and `re` are also used.

## 3. LLaMA-Factory dependency

The following scripts depend on an installed and importable LLaMA-Factory environment:

- `evaluation/attack/eval_au_interactive_trend.py`
- `evaluation/attack/infer_first_turn_from_trigger.py`
- `training/run_train_only.sh`
- `training/launch_psy_im_batch.sh`

Specifically:

- Python-side imports expect:
  - `llamafactory.chat.chat_model.ChatModel`
- Shell-side training expects:
  - a working `llamafactory-cli`

## 4. API dependency

Several data-construction and evaluation scripts call an OpenAI-compatible API.

Set the following environment variables before running them:

- `OPENAI_API_BASE`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, if you want to override the model name)

## 5. Optional encoder dependency

The following builders support representation-based scoring with a local Hugging Face encoder:

- `construction/direct_attack_builder_ad.py`
- `construction/trajectory_builder_ac.py`

If you want to use a transformer encoder instead of the built-in hashing fallback, set:

- `DIRECT_ATTACK_ENCODER_MODEL`
- `DIRECT_ATTACK_ENCODER_DEVICE`
- `TRAJECTORY_ENCODER_MODEL`
- `TRAJECTORY_ENCODER_DEVICE`

When these are not provided, the scripts still run, but they fall back to a lightweight hashing-based semantic representation.

## 6. External tools

The included shell scripts assume the following tools are available:

- `bash`
- `tmux`
- `find`
- `sort`

## 7. Dataset assumptions

The included code assumes the datasets in:

- `data/submission_bundle/raw_datasets/`
- `data/submission_bundle/poison_datasets/`

are already present and accessible.

The mapping between unified submission names and original dataset identities is recorded in:

- `data/submission_bundle/dataset_manifest.json`
