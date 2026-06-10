"""S2SEmoControl sample schema and prompt selection logic."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Tuple

from .config import MOODS, MOODS_ZH, SPK_ID
from .instruction import INSTRUCT, PROMPT_TEXT


def prepare_line_dict(line_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single S2SEmoControl metadata record from source jsonl line."""
    return {
        "uuid": line_dict["uuid"],
        "_id": line_dict["_id"],
        "source": "S2SEmoControl",
        "task": "S2S",
        "query": line_dict["text"],
        "answer": line_dict["answer"],
        "query_gender": random.choice(SPK_ID),
        "answer_gender": random.choice(SPK_ID),
        "query_mood": "neutral",
        "answer_mood": random.choice(MOODS),
        "language": line_dict["language"],
        "sample_rate": 16000,
        "prompt": None,
    }


def select_prompt_speech(new_line_dict: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Pick query/answer prompt speech ids matching gender and mood."""
    language = new_line_dict["language"]

    text_pattern = re.compile(
        rf"^{re.escape(new_line_dict['query_gender'])}-{re.escape(new_line_dict['query_mood'])}-[^-]+$"
    )
    text_matching_keys = [k for k in PROMPT_TEXT[language] if text_pattern.match(k)]
    if not text_matching_keys:
        raise ValueError(
            f"No matching prompt speech for {new_line_dict['query_gender']}-{new_line_dict['query_mood']}"
        )
    text_prompt_speech = random.choice(text_matching_keys)
    new_line_dict["query_id"] = text_prompt_speech

    answer_pattern = re.compile(
        rf"^{re.escape(new_line_dict['answer_gender'])}-{re.escape(new_line_dict['answer_mood'])}-[^-]+$"
    )
    answer_matching_keys = [k for k in PROMPT_TEXT[language] if answer_pattern.match(k)]
    if not answer_matching_keys:
        raise ValueError(
            f"No matching prompt speech for {new_line_dict['answer_gender']}-{new_line_dict['answer_mood']}"
        )
    answer_prompt_speech = random.choice(answer_matching_keys)
    new_line_dict["answer_id"] = answer_prompt_speech
    return text_prompt_speech, answer_prompt_speech, new_line_dict


def build_synthesis_inputs(new_line_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str, str, str]:
    """
    Decide what text to synthesize for query/answer based on emotion control rules.

    Returns:
        (updated_line_dict, query_instruct_text, query_content, answer_instruct_text, answer_content)
    """
    language = new_line_dict["language"]
    text_content = new_line_dict["query"]
    answer_content = new_line_dict["answer"]
    answer_mood = new_line_dict["answer_mood"]

    text_prompt_speech, answer_prompt_speech, new_line_dict = select_prompt_speech(new_line_dict)

    if answer_mood == "neutral":
        return (
            new_line_dict,
            PROMPT_TEXT[language][text_prompt_speech],
            text_content,
            PROMPT_TEXT[language][answer_prompt_speech],
            answer_content,
        )

    if language == "zh":
        prompt = random.choice(INSTRUCT[language]).format(mood=MOODS_ZH[answer_mood])
    else:
        prompt = random.choice(INSTRUCT[language]).format(mood=answer_mood)
    new_line_dict["prompt"] = prompt

    if random.random() < 0.5:
        query_content = prompt + text_content
    else:
        query_content = text_content + prompt

    return (
        new_line_dict,
        PROMPT_TEXT[language][text_prompt_speech],
        query_content,
        PROMPT_TEXT[language][answer_prompt_speech],
        answer_content,
    )
