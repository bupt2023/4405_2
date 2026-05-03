#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/path/to/project_root}"
LOG_DIR="$ROOT/training/logs"
SOCK="/tmp/psy_im_batch.sock"

python "$ROOT/construction/prepare_psy_im_batch.py"

tmux -S "$SOCK" kill-server 2>/dev/null || true

tmux -S "$SOCK" new-session -d -s im100 \
  "cd $ROOT && bash $ROOT/training/run_train_only.sh 0 $ROOT/backdoor/generated/llama_psy_im_poison_100.yaml $ROOT/saves_psy_im/llama3-8B-Instruct-100 > $LOG_DIR/train_llama_psy_im_poison_100.log 2>&1"
tmux -S "$SOCK" new-session -d -s im200 \
  "cd $ROOT && bash $ROOT/training/run_train_only.sh 1 $ROOT/backdoor/generated/llama_psy_im_poison_200.yaml $ROOT/saves_psy_im/llama3-8B-Instruct-200 > $LOG_DIR/train_llama_psy_im_poison_200.log 2>&1"
tmux -S "$SOCK" new-session -d -s im300 \
  "cd $ROOT && bash $ROOT/training/run_train_only.sh 2 $ROOT/backdoor/generated/llama_psy_im_poison_300.yaml $ROOT/saves_psy_im/llama3-8B-Instruct-300 > $LOG_DIR/train_llama_psy_im_poison_300.log 2>&1"
tmux -S "$SOCK" new-session -d -s im400 \
  "cd $ROOT && bash $ROOT/training/run_train_only.sh 3 $ROOT/backdoor/generated/llama_psy_im_poison_400.yaml $ROOT/saves_psy_im/llama3-8B-Instruct-400 > $LOG_DIR/train_llama_psy_im_poison_400.log 2>&1"
tmux -S "$SOCK" new-session -d -s im500 \
  "cd $ROOT && bash $ROOT/training/run_train_only.sh 4 $ROOT/backdoor/generated/llama_psy_im_poison_500.yaml $ROOT/saves_psy_im/llama3-8B-Instruct-500 > $LOG_DIR/train_llama_psy_im_poison_500.log 2>&1"

tmux -S "$SOCK" ls
