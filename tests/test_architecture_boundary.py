"""Architectural Boundary Invariant Test.

Enforces application-layer and common module import boundaries via AST analysis:
- src/autonomous_qa -> src.sauvi_perception: FORBIDDEN
- src/sauvi_perception -> src.autonomous_qa: FORBIDDEN
- src/common -> autonomous_qa or sauvi_perception: FORBIDDEN
- scripts/sauvi_perception -> src.autonomous_qa: FORBIDDEN
- scripts/autonomous_qa -> src.sauvi_perception: FORBIDDEN
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.common.config import ROOT


def _get_imports(py_file: Path) -> list[str]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    except Exception as err:
        raise RuntimeError(f"FAILED_TO_PARSE_AST:{py_file}") from err
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.append(mod)
    return imports


def test_autonomous_qa_does_not_import_sauvi_perception():
    auto_qa_dir = ROOT / "src" / "autonomous_qa"
    violations = []
    for f in auto_qa_dir.glob("**/*.py"):
        for imp in _get_imports(f):
            if "sauvi_perception" in imp:
                violations.append((f.relative_to(ROOT), imp))
    assert not violations, f"AUTONOMOUS_QA_IMPORT_VIOLATIONS: {violations}"


def test_sauvi_perception_does_not_import_autonomous_qa():
    sauvi_dir = ROOT / "src" / "sauvi_perception"
    violations = []
    for f in sauvi_dir.glob("**/*.py"):
        for imp in _get_imports(f):
            if "autonomous_qa" in imp:
                violations.append((f.relative_to(ROOT), imp))
    assert not violations, f"SAUVI_PERCEPTION_IMPORT_VIOLATIONS: {violations}"


def test_common_does_not_import_application_layers():
    common_dir = ROOT / "src" / "common"
    violations = []
    for f in common_dir.glob("**/*.py"):
        for imp in _get_imports(f):
            if "autonomous_qa" in imp or "sauvi_perception" in imp:
                violations.append((f.relative_to(ROOT), imp))
    assert not violations, f"COMMON_IMPORT_VIOLATIONS: {violations}"


def test_sauvi_perception_scripts_do_not_import_autonomous_qa():
    sauvi_scripts_dir = ROOT / "scripts" / "sauvi_perception"
    violations = []
    if sauvi_scripts_dir.exists():
        for f in sauvi_scripts_dir.glob("**/*.py"):
            for imp in _get_imports(f):
                if "autonomous_qa" in imp:
                    violations.append((f.relative_to(ROOT), imp))
    assert not violations, f"SAUVI_SCRIPTS_IMPORT_VIOLATIONS: {violations}"


def test_autonomous_qa_scripts_do_not_import_sauvi_perception():
    auto_scripts_dir = ROOT / "scripts" / "autonomous_qa"
    violations = []
    if auto_scripts_dir.exists():
        for f in auto_scripts_dir.glob("**/*.py"):
            for imp in _get_imports(f):
                if "sauvi_perception" in imp:
                    violations.append((f.relative_to(ROOT), imp))
    assert not violations, f"AUTONOMOUS_QA_SCRIPTS_IMPORT_VIOLATIONS: {violations}"
