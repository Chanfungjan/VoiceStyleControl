#!/usr/bin/env bash
# Step 3: CosyVoice Instruct synthesis with torch_npu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"

META_PATH="${META_PATH:-${PIPELINE_ROOT}/merged_speakerstyle_filtered.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PIPELINE_ROOT}/tts_output}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/pretrained_models/CosyVoice-300M-Instruct}"
NUM_PARALLEL="${NUM_PARALLEL:-2}"

mkdir -p "$OUTPUT_PATH"
total=$(wc -l < "$META_PATH")
step_size=$(( (total + NUM_PARALLEL - 1) / NUM_PARALLEL ))

echo "total=$total, step_size=$step_size"

for ((i = 0; i < NUM_PARALLEL; i++)); do
    start=$((step_size * i))
    end=$((start + step_size))
    echo "worker $i: [$start, $end)"
    ASCEND_RT_VISIBLE_DEVICES=$i nohup python "${PIPELINE_ROOT}/synthesize_dataset.py" \
        --meta_path "$META_PATH" \
        --output_path "$OUTPUT_PATH" \
        --model_path "$MODEL_PATH" \
        --device_id 0 \
        --start "$start" \
        --end "$end" \
        --id "$i" > "${OUTPUT_PATH}/log_${i}.txt" 2>&1 &
done

echo "launched $NUM_PARALLEL workers, logs under $OUTPUT_PATH/log_*.txt"
