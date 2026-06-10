#!/usr/bin/env python3
"""
S2SEmoControl dataset synthesis entry point.

MindSpore handles audio post-processing; CosyVoice inference uses torch_npu.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import mindspore as ms
from tqdm import tqdm

from common.jsonl import iter_indexed_jsonl, load_existing_values
from s2s_emo_control.audio_ms import set_ms_seed
from s2s_emo_control.config import COSYVOICE_ROOT
from s2s_emo_control.cosyvoice_npu import CosyVoiceNPUBackend
from s2s_emo_control.pipeline import S2SEmoControlPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Build S2SEmoControl dataset (MindSpore + torch_npu)")
    parser.add_argument("--meta_path", type=str, default="personify_text_answer_clean_language.jsonl")
    parser.add_argument("--output_path", type=str, default="output/s2s_emo_control")
    parser.add_argument("--model_path", type=str, default=None, help="CosyVoice pretrained model dir")
    parser.add_argument("--onnx_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--speaker_zh", type=str, default=os.path.join(PROJECT_ROOT, "prompt_speech"))
    parser.add_argument("--speaker_en", type=str, default=os.path.join(PROJECT_ROOT, "en_prompt_speech"))
    parser.add_argument("--id", type=int, default=0, help="Parallel shard id")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--device", type=str, default="npu:0", help="torch_npu device, e.g. npu:0")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language_filter", type=str, default="zh", help="Only process this language")
    parser.add_argument("--max_workers", type=int, default=16)
    return parser.parse_args()


def load_pending_meta(meta_path: str, output_jsonl: str, language_filter: str):
    existing_uuids = load_existing_values(output_jsonl, "uuid")

    id2meta = []
    for line_idx, line_dict in tqdm(iter_indexed_jsonl(meta_path), desc="loading meta"):
        if line_dict.get("uuid") in existing_uuids:
            continue
        if language_filter and line_dict.get("language") != language_filter:
            continue
        id2meta.append((line_idx, line_dict))
    return id2meta


def main():
    args = parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE)
    set_ms_seed(args.seed)

    model_dir = (
        args.model_path
        or args.onnx_path
        or os.path.join(COSYVOICE_ROOT, "pretrained_models")
    )
    os.makedirs(args.output_path, exist_ok=True)

    print(f"CosyVoice root: {COSYVOICE_ROOT}")
    print(f"Model dir: {model_dir}")
    print(f"Device: {args.device}")

    backend = CosyVoiceNPUBackend(
        model_dir,
        device=args.device,
        fp16=not args.no_fp16,
    )
    audio_dict = backend.load_prompt_wavs(args.speaker_zh, args.speaker_en)

    meta_name = os.path.basename(args.meta_path).replace(".jsonl", "")
    output_jsonl = os.path.join(args.output_path, f"{meta_name}_{args.id}.jsonl")

    id2meta = load_pending_meta(args.meta_path, output_jsonl, args.language_filter)
    print(f"pending samples after resume filter: {len(id2meta)}")

    start, end = args.start, args.end
    slice_meta = id2meta[start:] if end < 0 else id2meta[start:end]

    pipeline = S2SEmoControlPipeline(backend, audio_dict)
    pipeline.run(args, slice_meta, args.id, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
