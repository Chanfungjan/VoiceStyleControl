"""Generate {text, answer} pairs via mindformers + Qwen3."""
import argparse
import atexit
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mindformers import AutoModel, build_context, pipeline
from ms_bootstrap import init_mindspore_context
from tqdm import tqdm
from transformers import AutoTokenizer

GENDER = ["male", "female"]
GENDER_EN = {"male": "男", "female": "女"}
MOODS_EN = {
    "高兴": "happy",
    "悲伤": "sad",
    "惊讶": "surprised",
    "愤怒": "angry",
    "害怕": "fearful",
    "恐惧": "fearful",
    "厌恶": "disgusted",
    "冷静": "calm",
    "严肃": "serious",
}
EXCEPT = ["说话人的描述", "自然语言描述", "对说话人的自然描述", "...", "说话人自然描述"]
MAX_NEW_TOKENS = 512
SAVE_EVERY = 50

PROMPT_TEMPLATES = [
    "请用轻松自然的口吻描述一个说话人的性别{gender}和当前情绪{mood}，可以加入一些语气、语速等感受。"
    "然后写出 ta 正在说的一句话，长度在15到20个汉字之间。输出格式："
    '{{"text": "说话人的描述", "answer": "说话人说的话"}}',
    "用自然的语言描述一个你听到的说话人，性别{gender}和情绪{mood}是重点，也可以加入年龄、语气等信息。"
    "之后写出他或她正在说的一句话（15-20个字）。只返回 JSON 字典："
    '{{"text": "自然语言描述", "answer": "他说的话"}}',
    "你刚听到一段语音，请用自然语言描述这个说话人（包括性别{gender}和情绪{mood}），尽量真实自然。"
    "接着写出 ta 正在说的一句话，控制在15到20字。返回："
    '{{"text": "...", "answer": "..."}}',
]

text_hashes = set()
text_generator = None


def init_qwen3_generator(device_id: int, model_path: str):
    init_mindspore_context(device_id)
    build_context(
        {
            "context": {"mode": 0, "device_target": "Ascend", "device_id": device_id},
            "parallel": {},
            "parallel_config": {},
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        pretrained_model_dir=model_path,
        use_legacy=False,
        use_past=True,
    )
    generator = pipeline(
        task="text_generation",
        model=model,
        tokenizer=tokenizer,
    )
    return generator


def generate_unique_id():
    timestamp = int(time.time() * 1000)
    rand_part = uuid.uuid4().hex[:6]
    return f"{timestamp:x}{rand_part}"


def hash_text_pair(text: str, answer: str) -> str:
    return hashlib.md5(f"{text}|||{answer}".encode("utf-8")).hexdigest()


def save_hashes(hash_path):
    with open(hash_path, "wb") as f:
        pickle.dump(text_hashes, f)


def extract_json(text):
    results = []
    for match in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match)
            if set(obj.keys()) == {"text", "answer"} and obj["text"] not in EXCEPT:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results


def is_chinese_like(s):
    return bool(re.fullmatch(r"[\u4e00-\u9fff。，、？！《》“”：；‘’]+", s))


def random_generation_kwargs():
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": round(random.uniform(0.2, 1.1), 2),
        "top_p": round(random.uniform(0.7, 1.0), 2),
    }


def generate_sample(prompt: str) -> str:
    outputs = text_generator(prompt, **random_generation_kwargs())
    if isinstance(outputs, list) and outputs:
        item = outputs[0]
        if isinstance(item, dict):
            return item.get("generated_text", item.get("text", ""))
        return str(item)
    return str(outputs)


def try_add_sample(d, gender, mood, prompt, buffer_data, buffer_hashes):
    if not all(isinstance(d.get(k), str) for k in ["text", "answer"]):
        return False
    if not (is_chinese_like(d["text"]) and is_chinese_like(d["answer"])):
        return False

    hash_val = hash_text_pair(d["text"], d["answer"])
    if hash_val in text_hashes or hash_val in buffer_hashes:
        return False

    d["gender"] = gender
    d["mood"] = mood
    d["prompt"] = prompt
    d["id"] = f"{generate_unique_id()}-{gender}-{MOODS_EN[mood]}"
    buffer_data.append(d)
    buffer_hashes.add(hash_val)
    return True


def generate_dataset(save_path, num_samples, moods, hash_path):
    saved = 0
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            saved = sum(1 for _ in f)
        print(f"resume from {saved} existing samples")

    generated = 0
    buffer_data = []
    buffer_hashes = set()

    with open(save_path, "a", encoding="utf-8") as fout:
        pbar = tqdm(total=num_samples - saved)
        while generated < num_samples:
            mood = random.choice(moods)
            gender = random.choice(GENDER)
            prompt = random.choice(PROMPT_TEMPLATES).format(
                gender=GENDER_EN[gender], mood=mood
            )
            raw_output = generate_sample(prompt)
            data = extract_json(raw_output)
            if not data:
                continue

            for d in data:
                if try_add_sample(d, gender, mood, prompt, buffer_data, buffer_hashes):
                    generated += 1
                    pbar.update(1)
                    if generated >= num_samples:
                        break

            if len(buffer_data) >= SAVE_EVERY:
                for d in buffer_data:
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                fout.flush()
                text_hashes.update(buffer_hashes)
                save_hashes(hash_path)
                buffer_data.clear()
                buffer_hashes.clear()

        if buffer_data:
            for d in buffer_data:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            text_hashes.update(buffer_hashes)
            save_hashes(hash_path)

        pbar.close()
    print(f"done, total new samples: {generated}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(PROJECT_ROOT / "pretrain_models" / "Qwen3-8B"),
    )
    parser.add_argument("--hash_path", type=str, default="existing_sentence_hashes_paired.pkl")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--moods",
        type=str,
        nargs="+",
        default=["高兴", "悲伤", "愤怒", "害怕"],
    )
    args = parser.parse_args()

    global text_hashes, text_generator
    if os.path.exists(args.hash_path):
        with open(args.hash_path, "rb") as f:
            text_hashes = pickle.load(f)
        print(f"loaded {len(text_hashes)} dedup hashes")
    else:
        text_hashes = set()

    atexit.register(lambda: save_hashes(args.hash_path))
    text_generator = init_qwen3_generator(args.device_id, args.model_path)
    generate_dataset(args.save_path, args.num_samples, args.moods, args.hash_path)


if __name__ == "__main__":
    main()
