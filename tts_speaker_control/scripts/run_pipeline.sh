#!/usr/bin/env bash
# End-to-end TTSSpeakerControl pipeline (steps 0-3)
#
# Usage:
#   ./run_pipeline.sh all          # run all steps
#   ./run_pipeline.sh text         # step 1 only
#   ./run_pipeline.sh merge        # step 2 only
#   ./run_pipeline.sh tts          # step 3 only
#   ./run_pipeline.sh dedup        # step 0 only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP="${1:-all}"

run_dedup() {
    PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    PROJECT_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"
    python "${PIPELINE_ROOT}/build_dedup_hash.py" \
        --jsonl_dir "${PROJECT_ROOT}/description_speaker_style_dataset" \
        --output_pkl "${PIPELINE_ROOT}/existing_sentence_hashes_paired.pkl"
}

run_text() {
    bash "${SCRIPT_DIR}/run_text_generation.sh"
}

run_merge() {
    bash "${SCRIPT_DIR}/run_metadata_merge.sh"
}

run_tts() {
    bash "${SCRIPT_DIR}/run_synthesis.sh"
}

case "$STEP" in
    dedup) run_dedup ;;
    text) run_text ;;
    merge) run_merge ;;
    tts) run_tts ;;
    all)
        run_dedup
        run_text
        run_merge
        run_tts
        ;;
    *)
        echo "unknown step: $STEP"
        echo "usage: $0 [dedup|text|merge|tts|all]"
        exit 1
        ;;
esac
