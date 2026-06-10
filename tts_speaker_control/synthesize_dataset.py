"""CosyVoice Instruct TTS synthesis; outputs answer-side token/audio fields."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "third_party", "Matcha-TTS")):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.ark import ArkWriter
from common.audio_ms import process_torch_speech
from common.jsonl import iter_indexed_jsonl, load_existing_values, write_jsonl_record
from npu_bootstrap import init_torch_npu, patch_cosyvoice_for_npu
from cosyvoice.cli.cosyvoice import CosyVoice

SPK_ID = "中文女"
TTS_SAMPLE_RATE = 22050
ARK_SAMPLE_RATE = 16000

cosyvoice = None


def extract_tts_fields(record: dict) -> tuple[str, str]:
    if "answer" in record:
        return record["answer"], record.get("text") or record.get("prompt") or ""

    text_content = record["sentence"]
    text_description = record["description_en"]
    emotion = record["mood"]
    gender = record["gender"]
    text_instruction = f"{text_description} a {emotion} {gender}."
    return text_content, text_instruction


def compute_tts_speech_token(text_content, instruction_text, speaker_id):
    generated_tokens = []
    instruction_text = cosyvoice.frontend.text_normalize(instruction_text, split=False)
    flow_embedding = None
    device = cosyvoice.model.device

    for tts_text in cosyvoice.frontend.text_normalize(text_content, split=True):
        model_input = cosyvoice.frontend.frontend_instruct(
            tts_text, speaker_id, instruction_text
        )

        text = model_input["text"]
        text_len = model_input["text_len"]
        llm_embedding = torch.zeros(0, 192)
        flow_embedding = model_input["flow_embedding"]
        prompt_text = model_input["prompt_text"]
        prompt_text_len = model_input["prompt_text_len"]
        llm_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)

        with cosyvoice.model.llm_context:
            for token in cosyvoice.model.llm.inference(
                text=text.to(device),
                text_len=text_len.to(device),
                prompt_text=prompt_text.to(device),
                prompt_text_len=prompt_text_len.to(device),
                prompt_speech_token=llm_prompt_speech_token.to(device),
                prompt_speech_token_len=torch.tensor(
                    [llm_prompt_speech_token.shape[1]], dtype=torch.int32
                ).to(device),
                embedding=llm_embedding.to(device).half(),
                sampling=25,
                max_token_text_ratio=30,
                min_token_text_ratio=3,
            ):
                generated_tokens.append(token)

    generated_tokens = torch.tensor(generated_tokens).unsqueeze(dim=0)

    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech_feat = torch.zeros(1, 0, 80)
    flow_cache = torch.zeros(1, 80, 0, 2, device=device)

    tts_mel, _ = cosyvoice.model.flow.inference(
        token=generated_tokens.to(device),
        token_len=torch.tensor([generated_tokens.shape[1]], dtype=torch.int32).to(device),
        prompt_token=flow_prompt_speech_token.to(device),
        prompt_token_len=torch.tensor(
            [flow_prompt_speech_token.shape[1]], dtype=torch.int32
        ).to(device),
        prompt_feat=prompt_speech_feat.to(device),
        prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(
            device
        ),
        embedding=flow_embedding.to(device),
        flow_cache=flow_cache,
    )

    hift_cache_source = torch.zeros(1, 1, 0, device=device)
    tts_speech, _ = cosyvoice.model.hift.inference(
        speech_feat=tts_mel, cache_source=hift_cache_source
    )

    speech_token = generated_tokens.squeeze(0).cpu().numpy().astype(np.int16)
    tts_speech, _ = process_torch_speech(
        tts_speech,
        source_rate=TTS_SAMPLE_RATE,
        ark_rate=ARK_SAMPLE_RATE,
        wav_rate=None,
    )
    return speech_token, tts_speech


@dataclass(frozen=True)
class TTSOutputPaths:
    answer_token_ark: Path
    answer_token_scp: Path
    answer_audio_ark: Path
    answer_audio_scp: Path
    jsonl: Path
    answer_wav_dir: Path

    @classmethod
    def build(cls, meta_path: str, output_path: str, shard_id: int) -> "TTSOutputPaths":
        meta_name = Path(meta_path).stem
        out = Path(output_path)
        return cls(
            answer_token_ark=out / f"{meta_name}_answer_token_{shard_id}.ark",
            answer_token_scp=out / f"{meta_name}_answer_token_{shard_id}.scp",
            answer_audio_ark=out / f"{meta_name}_answer_audio_{shard_id}.ark",
            answer_audio_scp=out / f"{meta_name}_answer_audio_{shard_id}.scp",
            jsonl=out / f"{meta_name}_{shard_id}.jsonl",
            answer_wav_dir=out / f"{meta_name}_{shard_id}_answer_wav",
        )

    def prepare(self) -> None:
        self.answer_wav_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)


def compute_tts_speech_token_and_write_ark(args, id2meta, parallel_idx):
    paths = TTSOutputPaths.build(args.meta_path, args.output_path, parallel_idx)
    paths.prepare()

    with (
        ArkWriter(paths.answer_token_ark, paths.answer_token_scp) as answer_token_writer,
        ArkWriter(paths.answer_audio_ark, paths.answer_audio_scp) as answer_audio_writer,
        paths.jsonl.open("a+", encoding="utf-8") as jsonlf,
    ):
        print(
            "resume: "
            f"answer_token_offset={answer_token_writer.offset}, "
            f"answer_audio_offset={answer_audio_writer.offset}"
        )

        for sample_idx, line_dict in tqdm(id2meta):
            text_content, text_instruction = extract_tts_fields(line_dict)
            if len(text_content) > 512:
                continue

            sample_key = str(line_dict.get("uuid") or line_dict.get("id") or sample_idx)

            try:
                speech_token, speech_audio = compute_tts_speech_token(
                    text_content, text_instruction, SPK_ID
                )

                token_offset = answer_token_writer.write(sample_idx, speech_token.tobytes())
                audio_offset = answer_audio_writer.write(sample_idx, speech_audio.tobytes())

                line_dict["answer_token_25hz"] = f"{paths.answer_token_ark}:{token_offset}"
                line_dict["answer_audio_ark"] = f"{paths.answer_audio_ark}:{audio_offset}"
                line_dict["answer_audio_path"] = str(
                    paths.answer_wav_dir / f"{sample_key}-answer.wav"
                )
                wavfile.write(line_dict["answer_audio_path"], ARK_SAMPLE_RATE, speech_audio)
            except Exception:
                print(f"failed on sample={sample_key}: {text_content}")
                print(traceback.format_exc())
                continue

            write_jsonl_record(jsonlf, line_dict)

            if sample_idx % 100 == 0:
                answer_token_writer.flush()
                answer_audio_writer.flush()
                jsonlf.flush()


def main():
    global cosyvoice, SPK_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--speaker_id", type=str, default=SPK_ID)
    args = parser.parse_args()

    device = init_torch_npu(args.device_id)
    print(f"using device: {device}")

    SPK_ID = args.speaker_id
    cosyvoice = CosyVoice(args.model_path, load_jit=True, load_trt=False, fp16=False)
    patch_cosyvoice_for_npu(cosyvoice, args.device_id)

    output_paths = TTSOutputPaths.build(args.meta_path, args.output_path, args.id)
    existing_ids = load_existing_values(output_paths.jsonl, "uuid")
    existing_ids.update(load_existing_values(output_paths.jsonl, "id"))

    id2meta = []
    for line_idx, line_dict in iter_indexed_jsonl(args.meta_path):
        sample_key = line_dict.get("uuid") or line_dict.get("id")
        if sample_key not in existing_ids:
            id2meta.append((line_idx, line_dict))

    end = len(id2meta) if args.end < 0 else args.end
    print(f"pending samples: {len(id2meta)}, processing [{args.start}:{end}]")
    compute_tts_speech_token_and_write_ark(args, id2meta[args.start : end], args.id)


if __name__ == "__main__":
    main()
