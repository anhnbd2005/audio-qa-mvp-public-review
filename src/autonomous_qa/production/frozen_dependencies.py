"""Content-addressed frozen dependency retention + resolver.

New releases must pin exact bytes for every immutable production dependency so
the legacy ViMD failure (a frozen plan referencing a language registry whose
exact bytes were later replaced) cannot recur.

Layout::

    resources/frozen/sha256/<full-sha256>/<filename>

Resolution is by SHA256 only. A missing artifact HARD FAILS; there is no
fallback to a mutable "current" resource.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.common.config import ROOT

FROZEN_ROOT = ROOT / "resources" / "frozen" / "sha256"


class FrozenArtifactMissing(RuntimeError):
    def __init__(self, digest: str, name: str | None = None) -> None:
        super().__init__(
            f"FROZEN_ARTIFACT_MISSING:{digest}" + (f":{name}" if name else "")
        )
        self.digest = digest
        self.name = name


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def retain_artifact(source: Path, *, name: str | None = None) -> dict[str, str]:
    """Copy an artifact into the content-addressed store (idempotent).

    Returns {sha256, name, path}.
    """
    source = Path(source)
    data = source.read_bytes()
    digest = sha256_bytes(data)
    filename = name or source.name
    target_dir = FROZEN_ROOT / digest
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if not target.exists():
        shutil.copyfile(source, target)
    return {"sha256": digest, "name": filename, "path": _display_path(target)}


def retain_bytes(data: bytes, filename: str) -> dict[str, str]:
    digest = sha256_bytes(data)
    target_dir = FROZEN_ROOT / digest
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    target.write_bytes(data)
    return {"sha256": digest, "name": filename, "path": _display_path(target)}


def resolve_frozen(digest: str, *, name: str | None = None) -> Path:
    """Resolve a frozen artifact by SHA256; HARD FAIL if absent or corrupt."""
    target_dir = FROZEN_ROOT / digest
    if not target_dir.is_dir():
        raise FrozenArtifactMissing(digest, name)
    if name is not None:
        candidate = target_dir / name
        if not candidate.is_file():
            raise FrozenArtifactMissing(digest, name)
        payload = candidate.read_bytes()
        if sha256_bytes(payload) != digest:
            raise FrozenArtifactMissing(digest, name)
        return candidate
    files = sorted(p for p in target_dir.iterdir() if p.is_file())
    if not files:
        raise FrozenArtifactMissing(digest, name)
    for candidate in files:
        if sha256_bytes(candidate.read_bytes()) == digest:
            return candidate
    raise FrozenArtifactMissing(digest, name)


def resolve_all(manifest: dict[str, dict[str, str]]) -> dict[str, Path]:
    """Resolve a manifest of {logical_name: {sha256, name}}; hard fail on any."""
    resolved: dict[str, Path] = {}
    for logical_name, ref in manifest.items():
        resolved[logical_name] = resolve_frozen(ref["sha256"], name=ref.get("name"))
    return resolved


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
