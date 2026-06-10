"""Merge generated speaker-style JSONL files into TTSSpeakerControl records."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.jsonl import write_jsonl_record


SOURCE_NAME = "TTSSpeakerControl"
TARGET_MOODS = {"高兴", "悲伤", "愤怒", "恐惧", "害怕"}
EN_MOODS = {
    "高兴": "happy",
    "悲伤": "sad",
    "愤怒": "angry",
    "恐惧": "fearful",
    "害怕": "fearful",
}


def build_answer_id(record: dict, gender: str, mood: str, fallback_id: str) -> str:
    """Return a non-identifying answer-side style key."""
    if record.get("answer_id"):
        return str(record["answer_id"])
    if gender and mood:
        return f"{gender}-{mood}"
    return fallback_id


def to_tts_speaker_control_record(record: dict) -> dict:
    """Normalize one stage-01 record to the training JSONL schema."""
    source_id = str(record.get("id") or uuid.uuid4())
    sample_uuid = str(uuid.uuid4())
    answer_gender = str(record.get("answer_gender") or record.get("gender") or "")
    answer_mood = EN_MOODS[record["mood"]]
    style_text = str(record.get("text", ""))

    return {
        "uuid": sample_uuid,
        "_id": source_id,
        "source": SOURCE_NAME,
        "task": "TTS",
        "text": style_text,
        "answer": str(record.get("answer", "")),
        "answer_gender": answer_gender,
        "answer_mood": answer_mood,
        "language": str(record.get("language") or "zh"),
        "sample_rate": int(record.get("sample_rate") or 16000),
        "prompt": style_text,
        "answer_id": build_answer_id(record, answer_gender, answer_mood, source_id),
    }


def merge(input_files: Iterable[str], output_file: str, max_per_mood: int | None = None) -> None:
    results = []
    mood_counts: dict[str, int] = {}

    for file_name in input_files:
        path = Path(file_name)
        if not path.exists():
            print(f"missing input file: {file_name}")
            continue

        with path.open("r", encoding="utf-8") as input_stream:
            for line in input_stream:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as exc:
                    print(f"json parse error in {file_name}: {exc}")
                    continue

                mood = data.get("mood")
                if mood not in TARGET_MOODS:
                    continue

                if max_per_mood is not None:
                    mood_counts[mood] = mood_counts.get(mood, 0) + 1
                    if mood_counts[mood] > max_per_mood:
                        continue

                results.append(to_tts_speaker_control_record(data))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_stream:
        for item in results:
            write_jsonl_record(output_stream, item)

    print(f"wrote {len(results)} records to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_glob",
        type=str,
        default="../description_speaker_style_dataset/speaker_style_dataset_{i}_id_gender_mood.jsonl",
    )
    parser.add_argument("--worker_start", type=int, default=2)
    parser.add_argument("--worker_end", type=int, default=7)
    parser.add_argument("--output_file", type=str, default="merged_speakerstyle_filtered.jsonl")
    parser.add_argument("--max_per_mood", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = [
        args.input_glob.format(i=i)
        for i in range(args.worker_start, args.worker_end)
    ]
    merge(input_files, args.output_file, max_per_mood=args.max_per_mood)


if __name__ == "__main__":
    main()
