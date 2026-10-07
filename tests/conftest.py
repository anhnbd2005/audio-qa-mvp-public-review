"""Isolated project root so tests never touch real outputs/.

Builds a tmp tree with copies of prompts + fixtures + a small data dir,
then repoints every module-level ROOT reference at it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

import src.common.config
import src.autonomous_qa.language.paraphrase
import src.autonomous_qa.certification.quality
import src.autonomous_qa.language.question_style
import src.autonomous_qa.language.question_template
import src.autonomous_qa.compiler.pipeline_runner
import src.autonomous_qa.compiler.run_pipeline_cli
import src.common.io

REAL_ROOT = Path(src.common.config.ROOT)


@pytest.fixture()
def isolated_root(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "data" / "vimd").mkdir(parents=True)
    (root / "outputs" / "runs" / "vimd" / "mock" / "raw").mkdir(parents=True)
    (root / "prompts").mkdir(parents=True)
    for r in (1, 2, 3):
        (root / "tests" / "fixtures" / "loop" / f"round_{r:02d}").mkdir(
            parents=True)
    (root / "fewshots").mkdir(parents=True)

    for name in ("question_style", "question_template", "paraphrase", "quality"):
        shutil.copy(REAL_ROOT / "prompts" / f"{name}.txt", root / "prompts" / f"{name}.txt")
    for r in (1, 2, 3):
        for kind in ("style", "template", "paraphrase", "quality"):
            shutil.copy(
                REAL_ROOT / "tests" / "fixtures" / "loop" / f"round_{r:02d}" / f"{kind}.json",
                root / "tests" / "fixtures" / "loop" / f"round_{r:02d}" / f"{kind}.json",
            )
    shutil.copy(
        REAL_ROOT / "tests" / "fixtures" / "regression_surface.json",
        root / "tests" / "fixtures" / "regression_surface.json",
    )
    shutil.copy(
        REAL_ROOT / "tests" / "fixtures" / "regression_r2_partition.json",
        root / "tests" / "fixtures" / "regression_r2_partition.json",
    )
    shutil.copy(REAL_ROOT / "data" / "vimd" / "README.md", root / "data" / "vimd" / "README.md")
    shutil.copy(
        REAL_ROOT / "fewshots" / "question_style_seed.jsonl",
        root / "fewshots" / "question_style_seed.jsonl",
    )
    (root / "fewshots" / "approved_types.jsonl").write_text("", encoding="utf-8")
    (root / "fewshots" / "approved_templates.jsonl").write_text("", encoding="utf-8")

    shutil.copy(REAL_ROOT / "data" / "vimd" / "sample.jsonl", root / "data" / "vimd" / "sample.jsonl")
    if (REAL_ROOT / "data" / "vimd" / "schema.json").is_file():
        shutil.copy(
            REAL_ROOT / "data" / "vimd" / "schema.json",
            root / "data" / "vimd" / "schema.json",
        )
    if (REAL_ROOT / "data" / "vimd" / "manifest.jsonl").is_file():
        shutil.copy(
            REAL_ROOT / "data" / "vimd" / "manifest.jsonl",
            root / "data" / "vimd" / "manifest.jsonl",
        )

    for module in (src.common.config, src.autonomous_qa.compiler.pipeline_runner, src.autonomous_qa.compiler.run_pipeline_cli, src.common.io,
                   src.autonomous_qa.language.question_style, src.autonomous_qa.language.question_template,
                   src.autonomous_qa.certification.quality, src.autonomous_qa.language.paraphrase):
        if hasattr(module, "ROOT"):
            monkeypatch.setattr(module, "ROOT", root)

    return root
