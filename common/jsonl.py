"""JSONL helpers shared by dataset generation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_indexed_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_idx, line in enumerate(stream):
            line = line.strip()
            if line:
                yield line_idx, json.loads(line)


def load_existing_values(path: str | Path, key: str) -> set[Any]:
    target = Path(path)
    if not target.exists():
        return set()
    values = set()
    for record in iter_jsonl(target):
        if key in record:
            values.add(record[key])
    return values


def write_jsonl_record(stream, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_lines(path: str | Path) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    with target.open("r", encoding="utf-8") as stream:
        return sum(1 for _ in stream)
