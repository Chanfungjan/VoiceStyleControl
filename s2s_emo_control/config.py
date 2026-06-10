"""Pipeline constants and default paths."""

import os

# Audio sample rates (same as original pipeline)
TTS_SAMPLE_RATE = 22050
ARK_SAMPLE_RATE = 16000
WAV_SAMPLE_RATE = 24000

# CosyVoice2 speech token frame rate (cosyvoice2.yaml: token_frame_rate)
TOKEN_FRAME_RATE = 25

# Emotion / speaker labels
MOODS = ["happy", "sad", "angry", "neutral", "fearful"]
SPK_ID = ["female", "male"]
MOODS_ZH = {
    "happy": "高兴",
    "sad": "悲伤",
    "angry": "愤怒",
    "neutral": "自然",
    "fearful": "害怕",
}

# Processing
BATCH_SIZE = 8
DEFAULT_MAX_WORKERS = 16

# Ark binary header magic bytes
ZERO_ = 0x0
B_ = 0x42
FOUR_ = 0x4

# CosyVoice project root
_DEFAULT_COSYVOICE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", _DEFAULT_COSYVOICE_ROOT)
