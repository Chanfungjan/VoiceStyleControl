# VoiceStyleControl 数据集合成 MindSpore Pipeline

[English](README.md)

带说话人风格与情绪控制的语音数据集合成工程，包含两条独立流水线：

- `tts_speaker_control/`：先用 Qwen3 生成 `{text, answer}` 文本对，再用 CosyVoice Instruct **只合成 answer 侧语音**。`text` 是说话人风格/情绪描述（作为 instruction），`answer` 是待朗读台词（TTS content）。
- `s2s_emo_control/`：输入已有的 `(text, answer)` 文本对，用 CosyVoice zero-shot + 情绪指令**同时合成 query 和 answer 两侧语音**。

框架边界：这是 VoiceStyleControl 数据集的 MindSpore 实现版本，运行需要 Ascend/CANN 环境。

## 目录

```text
common/                 ark / 音频 / JSONL 公共工具
tts_speaker_control/
  build_text_pairs.py   Step 1：Qwen3 生成 {text, answer} 文本对
  merge_metadata.py     Step 2：筛选情绪、打标签为 TTS 记录
  synthesize_dataset.py Step 3：CosyVoice Instruct 合成 answer 语音
  scripts/run_pipeline.sh
s2s_emo_control/
  build_dataset.py      入口：合成 query / answer 两侧语音
  pipeline.py
  scripts/run_shard.sh
```

## 运行

### tts_speaker_control

```bash
cd VoiceStyleControl/tts_speaker_control
bash scripts/run_pipeline.sh all     # dedup → text → merge → tts
# 也可单步：bash scripts/run_pipeline.sh {dedup|text|merge|tts}
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

## 输出字段

合成完成后在元数据 JSONL 上追加 ark 指针与 wav 路径：

- tts_speaker_control：`answer_token_25hz` / `answer_audio_ark` / `answer_audio_path`
- s2s_emo_control：在上述 answer 字段基础上，额外追加 `query_token_25hz` / `query_audio_ark` / `query_audio_path`

## 依赖

`mindspore`、`mindformers`、`transformers`、`cosyvoice`、`torch`、`torch_npu`、`scipy`、`tqdm`（Ascend/CANN 运行时）。
