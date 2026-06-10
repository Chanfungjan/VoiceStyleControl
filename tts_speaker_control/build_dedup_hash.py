"""
Build paired-hash dedup file from existing speaker style datasets.

Supports both legacy fields (sentence + description_zh) and
TTSSpeakerControl fields (text + answer).
"""
import argparse
import hashlib
import json
import os
import pickle


def hash_text_pair(a: str, b: str) -> str:
    return hashlib.md5(f"{a}|||{b}".encode("utf-8")).hexdigest()


def build_and_save_text_hashes(jsonl_dir, output_pkl_path):
    text_hashes = set()
    for filename in os.listdir(jsonl_dir):
        if not (
            filename.endswith("id_gender_mood.jsonl")
            and "speaker_style_dataset" in filename
        ):
            continue

        filepath = os.path.join(jsonl_dir, filename)
        print(f"processing {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "text" in data and "answer" in data:
                    text_hashes.add(hash_text_pair(data["text"], data["answer"]))
                elif "sentence" in data and "description_zh" in data:
                    text_hashes.add(
                        hash_text_pair(data["description_zh"], data["sentence"])
                    )

    with open(output_pkl_path, "wb") as f:
        pickle.dump(text_hashes, f)
    print(f"saved {len(text_hashes)} hashes to {output_pkl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl_dir",
        type=str,
        default="../description_speaker_style_dataset",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="existing_sentence_hashes_paired.pkl",
    )
    args = parser.parse_args()
    build_and_save_text_hashes(args.jsonl_dir, args.output_pkl)
