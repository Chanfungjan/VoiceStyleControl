"""MindSpore-based audio post-processing shared by TTS and S2S builders."""

from __future__ import annotations

from pathlib import Path

import mindspore as ms
import numpy as np
import soundfile as sf


def set_ms_seed(seed: int) -> None:
    ms.set_seed(seed)


def torch_to_ms(tensor) -> ms.Tensor:
    """Convert a torch tensor at the CosyVoice boundary into MindSpore."""
    return ms.Tensor(tensor.detach().cpu().numpy(), dtype=ms.float32)


def resample_ms(waveform: ms.Tensor, orig_freq: int, new_freq: int) -> ms.Tensor:
    """Resample a waveform with deterministic linear interpolation."""
    if orig_freq == new_freq:
        return waveform

    values = waveform.asnumpy()
    if values.ndim == 1:
        values = values[np.newaxis, :]

    old_len = values.shape[-1]
    new_len = max(1, int(round(old_len * new_freq / orig_freq)))
    old_idx = np.arange(old_len, dtype=np.float64)
    new_idx = np.linspace(0, old_len - 1, new_len, dtype=np.float64)
    resampled = np.stack(
        [np.interp(new_idx, old_idx, channel) for channel in values],
        axis=0,
    )
    return ms.Tensor(resampled.astype(np.float32))


def tensor_to_float_array(waveform: ms.Tensor) -> np.ndarray:
    values = waveform.asnumpy()
    if values.ndim == 2:
        values = values.squeeze(0)
    return np.clip(values, -1.0, 1.0)


def speech_to_pcm_int16(waveform: ms.Tensor) -> np.ndarray:
    return (tensor_to_float_array(waveform) * (2**15)).astype(np.int16)


def save_wav(path: str | Path, waveform: ms.Tensor, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tensor_to_float_array(waveform), sample_rate)


def process_torch_speech(
    speech_tensor,
    source_rate: int,
    ark_rate: int,
    wav_rate: int | None = None,
) -> tuple[np.ndarray, ms.Tensor | None]:
    """Convert CosyVoice torch output to ark PCM and optional wav tensor."""
    speech_ms = torch_to_ms(speech_tensor)
    ark_ms = resample_ms(speech_ms, source_rate, ark_rate)
    wav_ms = None if wav_rate is None else resample_ms(speech_ms, source_rate, wav_rate)
    return speech_to_pcm_int16(ark_ms), wav_ms

