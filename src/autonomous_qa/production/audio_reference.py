"""Portable source-audio reference resolution for model-facing QA export.

The production generator emits opaque, deterministic audio hash ids
(``audio_<sha256[:20]>``). They are portable but carry no human-meaningful
identity. A dataset whose source metadata exposes an authoritative audio
filename can re-export model-facing QA with portable source filenames instead,
WITHOUT regenerating QA and WITHOUT changing any other field.

Identity is resolved ONLY through the authoritative link already stored in the
generated QA (``audio_ids`` aligned with ``internal.source_row_ids``) plus the
source metadata. Hashes are never reversed, and filenames are never guessed,
sorted or derived from the hash.

This module is dataset-agnostic: the identity field and source-audio field are
supplied by the caller (ViMedCSS: ``segment_id`` / ``audio``).
"""

from __future__ import annotations

import hashlib
import posixpath
import re
from collections.abc import Iterable
from pathlib import PureWindowsPath
from typing import Any


class PortableAudioError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def opaque_audio_id(dataset: str, revision: str, source_row_id: str) -> str:
    """Deterministic opaque audio identity. Unchanged legacy behaviour."""
    digest = hashlib.sha256(
        f"{dataset}|{revision}|{source_row_id}".encode()
    ).hexdigest()
    return f"audio_{digest[:20]}"


def _portable_basename(value: Any, *, identity: str) -> str:
    """Extract a safe, portable basename from an authoritative audio reference."""
    if value is None:
        raise PortableAudioError("MISSING_SOURCE_AUDIO", identity)
    text = str(value).strip()
    if not text:
        raise PortableAudioError("MISSING_SOURCE_AUDIO", identity)
    # The reference may be POSIX or Windows; normalise separators for basename
    # extraction but never emit a machine-specific path.
    if "\\" in text and "/" not in text:
        basename = PureWindowsPath(text).name
    else:
        basename = posixpath.basename(text.replace("\\", "/"))
    if not basename or basename in (".", ".."):
        raise PortableAudioError("INVALID_AUDIO_REFERENCE", f"{identity}:{text}")
    if "/" in basename or "\\" in basename:
        raise PortableAudioError("INVALID_AUDIO_REFERENCE", f"{identity}:{text}")
    if ".." in basename:
        raise PortableAudioError("PATH_TRAVERSAL", f"{identity}:{text}")
    if re.match(r"^[A-Za-z]:", basename):
        raise PortableAudioError("INVALID_AUDIO_REFERENCE", f"{identity}:{text}")
    return basename


_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def build_source_audio_index(
    rows: Iterable[dict[str, Any]],
    *,
    identity_field: str,
    source_audio_field: str,
    require_unique_identity: bool = True,
    require_identity_matches_basename: bool = False,
) -> dict[str, str]:
    """Map authoritative identity -> portable basename for one dataset split.

    Fails closed on missing/ambiguous identities, missing audio references,
    non-portable paths and (optionally) basename/identity mismatches.
    """
    index: dict[str, str] = {}
    basename_owner: dict[str, str] = {}
    duplicates: list[str] = []
    for row in rows:
        raw_identity = row.get(identity_field)
        if raw_identity is None or not str(raw_identity).strip():
            raise PortableAudioError("MISSING_IDENTITY", identity_field)
        identity = str(raw_identity)
        if identity in index:
            if index[identity] != _portable_basename(
                row.get(source_audio_field), identity=identity
            ):
                duplicates.append(identity)
            if require_unique_identity:
                raise PortableAudioError("AMBIGUOUS_IDENTITY", identity)
            continue
        audio_value = row.get(source_audio_field)
        if isinstance(audio_value, str) and _SCHEME_RE.match(audio_value.strip()):
            raise PortableAudioError("NON_LOCAL_AUDIO_REFERENCE", f"{identity}:{audio_value}")
        basename = _portable_basename(audio_value, identity=identity)
        if require_identity_matches_basename:
            stem = posixpath.splitext(basename)[0]
            if stem != identity:
                raise PortableAudioError(
                    "IDENTITY_BASENAME_MISMATCH", f"{identity}:{basename}"
                )
        previous = basename_owner.get(basename)
        if previous is not None and previous != identity:
            raise PortableAudioError(
                "BASENAME_COLLISION", f"{basename}:{previous}/{identity}"
            )
        basename_owner[basename] = identity
        index[identity] = basename
    return index


def resolve_audio_ids_to_filenames(
    *,
    audio_ids: list[str],
    source_row_ids: list[str],
    index: dict[str, str],
    dataset: str,
    revision: str,
    verify_hash_link: bool = True,
    fail_on_unresolved: bool = True,
) -> list[str]:
    """Resolve one record's ordered opaque audio ids to source basenames."""
    if len(audio_ids) != len(source_row_ids):
        raise PortableAudioError(
            "AUDIO_ROW_ARITY_MISMATCH", f"{len(audio_ids)}!={len(source_row_ids)}"
        )
    resolved: list[str] = []
    for audio_id, row_id in zip(audio_ids, source_row_ids):
        if verify_hash_link and opaque_audio_id(dataset, revision, row_id) != audio_id:
            raise PortableAudioError("AUDIO_ID_LINK_MISMATCH", f"{audio_id}!={row_id}")
        basename = index.get(row_id)
        if basename is None:
            if fail_on_unresolved:
                raise PortableAudioError("UNRESOLVED_IDENTITY", f"{audio_id}->{row_id}")
            continue
        resolved.append(basename)
    return resolved


def resolve_model_facing_audio(
    *,
    model_records: list[dict[str, Any]],
    internal_records: list[dict[str, Any]],
    index: dict[str, str],
    dataset: str,
    revision: str,
    verify_hash_link: bool = True,
    preserve_audio_order: bool = True,
    fail_on_unresolved: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return model-facing records with ``audio`` re-expressed as source basenames.

    Only the ``audio`` field changes; every other field is copied verbatim.
    """
    audio_to_row: dict[str, str] = {}
    internal_by_id: dict[str, dict[str, Any]] = {}
    for record in internal_records:
        qa_id = record.get("qa_id")
        if qa_id is None:
            raise PortableAudioError("INTERNAL_MISSING_QA_ID", "")
        internal_by_id[str(qa_id)] = record
        rows = list(
            record.get("internal", {}).get("source_row_ids")
            or record.get("source_row_ids")
            or []
        )
        audio_ids = list(record.get("audio_ids") or [])
        if len(rows) != len(audio_ids):
            raise PortableAudioError(
                "AUDIO_ROW_ARITY_MISMATCH", f"{qa_id}:{len(audio_ids)}!={len(rows)}"
            )
        for audio_id, row_id in zip(audio_ids, rows):
            if verify_hash_link and opaque_audio_id(dataset, revision, row_id) != audio_id:
                raise PortableAudioError(
                    "AUDIO_ID_LINK_MISMATCH", f"{qa_id}:{audio_id}!={row_id}"
                )
            previous = audio_to_row.get(audio_id)
            if previous is not None and previous != row_id:
                raise PortableAudioError(
                    "AUDIO_ID_AMBIGUOUS", f"{audio_id}:{previous}/{row_id}"
                )
            audio_to_row[audio_id] = row_id

    portable: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    single = two = 0
    for record in model_records:
        qa_id = str(record.get("id", ""))
        internal = internal_by_id.get(qa_id)
        if internal is None:
            raise PortableAudioError("MODEL_RECORD_WITHOUT_INTERNAL", qa_id)
        internal_audio = list(internal.get("audio_ids") or [])
        model_audio = list(record.get("audio") or [])
        if preserve_audio_order and model_audio != internal_audio:
            raise PortableAudioError(
                "MODEL_INTERNAL_AUDIO_MISMATCH", f"{qa_id}:{model_audio}!={internal_audio}"
            )
        resolved: list[str] = []
        for audio_id in model_audio:
            row_id = audio_to_row.get(audio_id)
            if row_id is None:
                unresolved.append({"id": qa_id, "audio_id": audio_id, "reason": "NO_IDENTITY_LINK"})
                continue
            basename = index.get(row_id)
            if basename is None:
                unresolved.append({"id": qa_id, "audio_id": audio_id, "row_id": row_id, "reason": "UNRESOLVED_IDENTITY"})
                continue
            resolved.append(basename)
        if unresolved and fail_on_unresolved:
            raise PortableAudioError(
                "UNRESOLVED_AUDIO_REFERENCES",
                f"{qa_id}:{unresolved[-1]['reason']}",
            )
        new_record = dict(record)
        new_record["audio"] = resolved
        portable.append(new_record)
        if len(resolved) == 1:
            single += 1
        elif len(resolved) == 2:
            two += 1

    stats = {
        "records": len(portable),
        "single_audio_records": single,
        "two_audio_records": two,
        "unresolved": unresolved,
        "unresolved_count": len(unresolved),
        "unique_source_audio": len({name for record in portable for name in record["audio"]}),
    }
    return portable, stats
