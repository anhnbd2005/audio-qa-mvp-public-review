"""Generic Zero-LLM / Zero-Network source code scanner."""

from __future__ import annotations

import re
from typing import Sequence

from src.common.config import ROOT

FORBIDDEN_PATTERNS = (
    re.compile(r"\bimport\s+openai\b"),
    re.compile(r"\bfrom\s+openai\b"),
    re.compile(r"\bfrom\s+src\.llm_client\b"),
    re.compile(r"\bimport\s+src\.llm_client\b"),
    re.compile(r"\bcreate_llm_client\s*\("),
    re.compile(r"\bcall_llm\s*\("),
    re.compile(r"\bcomplete_structured_stage\s*\("),
    re.compile(r"\brequests\.post\s*\("),
    re.compile(r"\bhttpx\.Client\s*\("),
)


def scan_modules_for_forbidden_calls(
    modules: Sequence[str],
    patterns: Sequence[re.Pattern] = FORBIDDEN_PATTERNS,
) -> dict:
    """Scan given Python module relative paths for forbidden LLM/network patterns."""
    offenders: dict[str, list[str]] = {}
    for rel in modules:
        path = ROOT / rel
        if not path.exists():
            offenders.setdefault(rel, []).append("MISSING")
            continue
        text = path.read_text(encoding="utf-8")
        hits = [
            pattern.pattern for pattern in patterns if pattern.search(text)
        ]
        if hits:
            offenders[rel] = hits
    return {
        "status": "PASS" if not offenders else "FAIL",
        "modules_scanned": len(modules),
        "offenders": offenders,
        "llm_calls": 0,
    }
