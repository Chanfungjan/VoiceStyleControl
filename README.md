# VoiceStyleControl Synthesis MindSpore Pipeline

[中文](README_zh.md)

A speech dataset synthesis project with speaker-style and emotion control. It contains two independent pipelines:

- `tts_speaker_control/`: first generates `{text, answer}` pairs with Qwen3, then synthesizes **only the answer-side speech** with CosyVoice Instruct. `text` is the speaker style/emotion description (used as the instruction), `answer` is the line to read (TTS content).
- `s2s_emo_control/`: takes existing `(text, answer)` pairs and synthesizes **both query-side and answer-side speech** with CosyVoice zero-shot plus an emotion instruction.

Framework boundary: This is the MindSpore implementation version of VoiceStyleControl dataset, and it needs Ascend/CANN environment to run.

## Layout

```text
common/                 shared ark / audio / JSONL utilities
tts_speaker_control/
  build_text_pairs.py   Step 1: Qwen3 generates {text, answer} pairs
  merge_metadata.py     Step 2: filter moods, tag into TTS records
  synthesize_dataset.py Step 3: CosyVoice Instruct synthesizes answer speech
  scripts/run_pipeline.sh
s2s_emo_control/
  build_dataset.py      entry point: synthesize query / answer speech
  pipeline.py
  scripts/run_shard.sh
```

## Run

### tts_speaker_control

```bash
cd VoiceStyleControl/tts_speaker_control
bash scripts/run_pipeline.sh all     # dedup → text → merge → tts
# single step: bash scripts/run_pipeline.sh {dedup|text|merge|tts}
```

### s2s_emo_control

```bash
cd VoiceStyleControl/s2s_emo_control
python build_dataset.py \
    --meta_path ../personify_text_answer_clean_language_part1.jsonl \
    --output_path output/s2s_emo_control/part1 \
    --model_path ../pretrained_models \
    --speaker_zh ../prompt_speech \
    --speaker_en ../en_prompt_speech \
    --id 0 --device npu:0
```

## Output fields

After synthesis, ark pointers and wav paths are appended to the metadata JSONL:

- tts_speaker_control: `answer_token_25hz` / `answer_audio_ark` / `answer_audio_path`
- s2s_emo_control: in addition to the answer fields above, also appends `query_token_25hz` / `query_audio_ark` / `query_audio_path`

## Dependencies

`mindspore`, `mindformers`, `transformers`, `cosyvoice`, `torch`, `torch_npu`, `scipy`, `tqdm` (Ascend/CANN runtime).
