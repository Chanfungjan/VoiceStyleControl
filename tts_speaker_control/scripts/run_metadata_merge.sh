#!/usr/bin/env bash
# Step 2: merge worker jsonl files into TTSSpeakerControl format
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"

WORKER_START="${WORKER_START:-2}"
WORKER_END="${WORKER_END:-7}"
OUTPUT_FILE="${OUTPUT_FILE:-${PIPELINE_ROOT}/merged_speakerstyle_filtered.jsonl}"

python "${PIPELINE_ROOT}/merge_metadata.py" \
    --input_glob "${PROJECT_ROOT}/description_speaker_style_dataset/speaker_style_dataset_{i}_id_gender_mood.jsonl" \
    --worker_start "$WORKER_START" \
    --worker_end "$WORKER_END" \
    --output_file "$OUTPUT_FILE"
