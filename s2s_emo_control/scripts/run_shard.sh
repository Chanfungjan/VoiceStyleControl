#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"

SHARD_ID="${SHARD_ID:-0}"
PART_ID="${PART_ID:-$((SHARD_ID + 1))}"
NPU_ID="${NPU_ID:-$SHARD_ID}"

export COSYVOICE_ROOT="${COSYVOICE_ROOT:-$PROJECT_ROOT}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$NPU_ID}"

META_PATH="${META_PATH:-${PROJECT_ROOT}/personify_text_answer_clean_language_part${PART_ID}.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PIPELINE_ROOT}/output/s2s_emo_control/part${PART_ID}}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/pretrained_models}"
SPEAKER_ZH="${SPEAKER_ZH:-${PROJECT_ROOT}/prompt_speech}"
SPEAKER_EN="${SPEAKER_EN:-${PROJECT_ROOT}/en_prompt_speech}"

mkdir -p "$OUTPUT_PATH"

python "${PIPELINE_ROOT}/build_dataset.py" \
    --meta_path "$META_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \
    --speaker_zh "$SPEAKER_ZH" \
    --speaker_en "$SPEAKER_EN" \
    --id "$SHARD_ID" \
    --device "npu:0" \
    2>&1 | tee "${OUTPUT_PATH}/log_${SHARD_ID}.txt"
