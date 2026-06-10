#!/usr/bin/env bash
# Step 1: generate text/answer pairs with Qwen3-8B on MindSpore
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"

WORKER_ID="${WORKER_ID:-2}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/description_speaker_style_dataset}"
NUM_SAMPLES="${NUM_SAMPLES:-60000}"
DEVICE_ID="${DEVICE_ID:-0}"

mkdir -p "$SAVE_DIR"
SAVE_PATH="${SAVE_DIR}/speaker_style_dataset_${WORKER_ID}_id_gender_mood.jsonl"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$DEVICE_ID}" \
python "${PIPELINE_ROOT}/build_text_pairs.py" \
    --save_path "$SAVE_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --model_path "${PROJECT_ROOT}/pretrain_models/Qwen3-8B" \
    --hash_path "${PIPELINE_ROOT}/existing_sentence_hashes_paired.pkl" \
    --device_id "$DEVICE_ID"
