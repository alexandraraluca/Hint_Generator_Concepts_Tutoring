"""Small JSONL / JSON helpers used across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import orjson


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
            n += 1
    return n


def read_json(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def write_json(path: str | Path, obj: Any, *, pretty: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
    else:
        with open(p, "wb") as f:
            f.write(orjson.dumps(obj))
