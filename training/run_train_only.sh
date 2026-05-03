#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gpu_id> <config_path> <save_dir>" >&2
  exit 1
fi

GPU_ID="$1"
CONFIG_PATH="$2"
SAVE_DIR="$3"

ROOT="${PROJECT_ROOT:-/path/to/project_root}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-/path/to/llamafactory-cli}"
HF_HOME="$ROOT/cache/hf"
HF_DATASETS_CACHE="$HF_HOME/datasets"
TRANSFORMERS_CACHE="$HF_HOME/transformers"

mkdir -p "$SAVE_DIR"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
export MPLCONFIGDIR=/tmp/matplotlib
export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE

echo "[start] gpu=${GPU_ID} config=${CONFIG_PATH} time=$(date -u '+%F %T')"
echo "[train] ${CONFIG_PATH}"
rm -f "$SAVE_DIR/.train_done"

LATEST_CHECKPOINT="$(
  find "$SAVE_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sort -V \
    | tail -n 1
)"

RUN_CONFIG="$CONFIG_PATH"
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  RESUME_PATH="$SAVE_DIR/$LATEST_CHECKPOINT"
  RUN_CONFIG="$SAVE_DIR/resume_${LATEST_CHECKPOINT}.yaml"
  cp "$CONFIG_PATH" "$RUN_CONFIG"
  {
    printf '\n'
    printf 'resume_from_checkpoint: %s\n' "$RESUME_PATH"
  } >> "$RUN_CONFIG"
  echo "[resume] ${RESUME_PATH}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${LLAMAFACTORY_CLI}" train "${RUN_CONFIG}"
touch "$SAVE_DIR/.train_done"
echo "[done] config=${CONFIG_PATH} save_dir=${SAVE_DIR} time=$(date -u '+%F %T')"
