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


def resample_torch_speech_to_int16(tensor, orig_freq: int, new_freq: int):
    ark_pcm, _ = process_torch_speech(
        tensor,
        source_rate=orig_freq,
        ark_rate=new_freq,
        wav_rate=None,
    )
    return ark_pcm


__all__ = [
    "process_torch_speech",
    "resample_ms",
    "resample_torch_speech_to_int16",
    "save_wav",
    "set_ms_seed",
    "speech_to_pcm_int16",
    "torch_to_ms",
]
