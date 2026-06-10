"""Compatibility exports for shared MindSpore audio utilities."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.audio_ms import (
    process_torch_speech,
    resample_ms,
    save_wav,
    set_ms_seed,
    speech_to_pcm_int16,
    torch_to_ms,
)
from .config import ARK_SAMPLE_RATE, WAV_SAMPLE_RATE


def process_tts_output(tts_speech_torch, tts_sample_rate: int):
    """Return ark PCM and wav tensor for S2SEmoControl outputs."""
    return process_torch_speech(
        tts_speech_torch,
        source_rate=tts_sample_rate,
        ark_rate=ARK_SAMPLE_RATE,
        wav_rate=WAV_SAMPLE_RATE,
    )


__all__ = [
    "process_tts_output",
    "process_torch_speech",
    "resample_ms",
    "save_wav",
    "set_ms_seed",
    "speech_to_pcm_int16",
    "torch_to_ms",
]
