"""
CosyVoice2 inference backend.

CosyVoice has no MindSpore implementation in this project, so this module keeps
the TTS model boundary on torch_npu while the rest of the pipeline uses
MindSpore or framework-agnostic Python.
"""

from __future__ import annotations

import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .config import COSYVOICE_ROOT


def _setup_cosyvoice_path() -> None:
    matcha_path = os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS")
    for p in (COSYVOICE_ROOT, matcha_path):
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_device(device: str) -> torch.device:
    """Resolve an Ascend NPU device string to torch.device."""
    if not device.startswith("npu"):
        raise ValueError("Only torch_npu devices are supported, for example --device npu:0")
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise ImportError("torch_npu is required for CosyVoice inference on Ascend NPU") from exc
    if not torch.npu.is_available():
        raise RuntimeError("NPU requested but torch.npu.is_available() is False")
    return torch.device(device)


@contextmanager
def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        yield
        return
    if device.type == "npu":
        with torch.npu.amp.autocast():
            yield
    else:
        yield


class CosyVoiceNPUBackend:
    """Wrap CosyVoice2 zero-shot inference on torch_npu."""

    def __init__(
        self,
        model_dir: str,
        device: str = "npu:0",
        fp16: bool = True,
    ):
        _setup_cosyvoice_path()

        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        self._load_wav = load_wav
        self.device = resolve_device(device)

        self.cosyvoice = CosyVoice2(
            model_dir,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=fp16,
        )
        self._patch_model_device()
        self.sample_rate = self.cosyvoice.sample_rate

    def _patch_model_device(self) -> None:
        """Bind CosyVoice modules to the configured NPU device."""
        model = self.cosyvoice.model
        model.device = self.device
        model.llm.to(self.device)
        model.flow.to(self.device)
        model.hift.to(self.device)
        if self.device.type == "npu":
            model.llm_context = __import__("contextlib").nullcontext()

    def load_prompt_wavs(self, speaker_zh: str, speaker_en: str) -> Dict[str, Dict[str, Any]]:
        audio_dict: Dict[str, Dict[str, Any]] = {"zh": {}, "en": {}}
        for lang, audio_dir in [("zh", speaker_zh), ("en", speaker_en)]:
            if not audio_dir or not os.path.isdir(audio_dir):
                continue
            for filename in os.listdir(audio_dir):
                if filename.endswith(".wav"):
                    name = filename[:-4].split("/")[-1]
                    audio_dict[lang][name] = self._load_wav(
                        os.path.join(audio_dir, filename), 16000
                    )
        return audio_dict

    def compute_zeroshot_speech_token(
        self,
        instruct_text: str,
        prompt_speech_16k,
        text_content: str,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Run zero-shot TTS and return (tokens_int16, raw_speech_torch).

        Post-resampling is delegated to MindSpore in audio_ms.process_tts_output.
        """
        cosyvoice = self.cosyvoice
        frontend = cosyvoice.frontend
        model = cosyvoice.model

        instruct_text = frontend.text_normalize(instruct_text, split=False, text_frontend=True)
        tts_text_list = frontend.text_normalize(text_content, split=True, text_frontend=True)
        model_inputs = [
            frontend.frontend_zero_shot(
                tts_text, instruct_text, prompt_speech_16k, cosyvoice.sample_rate, zero_shot_spk_id=""
            )
            for tts_text in tts_text_list
        ]

        def zeroshot_inference(model_input):
            text = model_input["text"]
            prompt_text = model_input["prompt_text"]
            llm_prompt_speech_token = model_input["llm_prompt_speech_token"]
            llm_embedding = model_input["llm_embedding"]

            local_tokens = []
            with model.llm_context, _autocast(self.device, model.fp16):
                for token in model.llm.inference(
                    text=text.to(model.device),
                    text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(model.device),
                    prompt_text=prompt_text.to(model.device),
                    prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(model.device),
                    prompt_speech_token=llm_prompt_speech_token.to(model.device),
                    prompt_speech_token_len=torch.tensor(
                        [llm_prompt_speech_token.shape[1]], dtype=torch.int32
                    ).to(model.device),
                    embedding=llm_embedding.to(model.device),
                    uuid=str(uuid.uuid1()),
                ):
                    local_tokens.append(token)
            return local_tokens

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(zeroshot_inference, mi) for mi in model_inputs]
            ordered_results = [f.result() for f in futures]

        generated_tokens = torch.cat(
            [torch.tensor(tokens) for tokens in ordered_results], dim=0
        ).unsqueeze(0)

        flow_prompt_speech_token = model_inputs[0]["flow_prompt_speech_token"]
        prompt_speech_feat = model_inputs[0]["prompt_speech_feat"]
        flow_embedding = model_inputs[0]["flow_embedding"]

        with _autocast(self.device, model.fp16):
            tts_mel, _ = model.flow.inference(
                token=generated_tokens.to(model.device),
                token_len=torch.tensor([generated_tokens.shape[1]], dtype=torch.int32).to(model.device),
                prompt_token=flow_prompt_speech_token.to(model.device),
                prompt_token_len=torch.tensor(
                    [flow_prompt_speech_token.shape[1]], dtype=torch.int32
                ).to(model.device),
                prompt_feat=prompt_speech_feat.to(model.device),
                prompt_feat_len=torch.tensor(
                    [prompt_speech_feat.shape[1]], dtype=torch.int32
                ).to(model.device),
                embedding=flow_embedding.to(model.device),
                streaming=False,
                finalize=True,
            )

        hift_cache_source = torch.zeros(1, 1, 0, device=model.device)
        tts_speech, _ = model.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)

        tokens_int16 = generated_tokens.squeeze(0).cpu().numpy().astype(np.int16)
        return tokens_int16, tts_speech
