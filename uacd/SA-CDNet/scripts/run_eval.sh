#!/usr/bin/env bash
set -e

WORKING_PATH="${WORKING_PATH:-.}"
DATA_NAME="${DATA_NAME:-YOUR_DATASET_NAME}"
NET_NAME="${NET_NAME:-YOUR_NET_NAME}"
GPU_IDS="${GPU_IDS:-0}"
DEV_ID="${DEV_ID:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-YOUR_CHECKPOINT_PATH.pth}"
PRED_DIR="${PRED_DIR:-$WORKING_PATH/results/$NET_NAME/$DATA_NAME}"

if [ "$DATA_NAME" = "YOUR_DATASET_NAME" ] || [ "$NET_NAME" = "YOUR_NET_NAME" ]; then
  echo "Please set DATA_NAME and NET_NAME before running."
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  echo "Please set CHECKPOINT_PATH to a valid checkpoint file."
  exit 1
fi

python ./eval.py \
    --working_path "$WORKING_PATH" \
    --DATA_NAME "$DATA_NAME" \
    --NET_NAME "$NET_NAME" \
    --chkpt_path "$CHECKPOINT_PATH" \
    --pred_dir "$PRED_DIR" \
    --val_batch_size 64 \
    --crop_size 768 \
    --num_workers 8 \
    --gpu \
    --dev_id "$DEV_ID" \
    --gpu_ids "$GPU_IDS"
