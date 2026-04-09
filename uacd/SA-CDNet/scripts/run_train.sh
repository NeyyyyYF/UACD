#!/usr/bin/env bash
set -e

WORKING_PATH="${WORKING_PATH:-.}"
DATA_NAME="${DATA_NAME:-YOUR_DATASET_NAME}"
NET_NAME="${NET_NAME:-YOUR_NET_NAME}"
GPU_IDS="${GPU_IDS:-0}"
DEV_ID="${DEV_ID:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-YOUR_CHECKPOINT_PATH.pth}"

if [ "$DATA_NAME" = "YOUR_DATASET_NAME" ] || [ "$NET_NAME" = "YOUR_NET_NAME" ]; then
  echo "Please set DATA_NAME and NET_NAME before running."
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH"
  echo "Please set CHECKPOINT_PATH to a valid pretrained weights file."
  exit 1
fi

python ./train.py \
    --working_path "$WORKING_PATH" \
    --DATA_NAME "$DATA_NAME" \
    --NET_NAME "$NET_NAME" \
    --train_batch_size 8 \
    --val_batch_size 64 \
    --lr 0.001 \
    --epochs 100 \
    --gpu \
    --dev_id "$DEV_ID" \
    --multi_gpu "$GPU_IDS" \
    --img_size 512 \
    --crop_size 768 \
    --load_premodel \
    --chkpt_path "$CHECKPOINT_PATH" \
    --seed 42