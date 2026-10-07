"""Small shared helpers (prompt loading, fixture loading, JSON IO)."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.config import ROOT


def load_prompt(name: str) -> str:
    path = ROOT / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fixture(name: str, dataset: str | None = None) -> dict:
    if dataset:
        ds_path = ROOT / "tests" / "fixtures" / dataset / f"{name}.json"
        if ds_path.is_file():
            with open(ds_path, "r", encoding="utf-8") as f:
                return json.load(f)
    path = ROOT / "tests" / "fixtures" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON: temp file in same dir + fsync + os.replace."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError):
            pass
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
