"""ViMedCSS deployment audio mapper.

Bridges the LOGICAL QA release to a physically runnable server release:

    logical QA (qa_model_facing.jsonl)
        + audio_logical_mapping.jsonl   (logical_audio_id -> segment_id)
        + server audio manifest         (segment_id -> /server/path.wav)
        -> single-audio rows rewritten to physical paths
        -> pairwise rows composed as A+BEEP+B and rewritten to ONE physical path

No network. No LLM. Fails closed at mapping time (never publishes a partial
file). The input JSONL is never modified.

Usage:
    python map_vimedcss_audio_paths.py \
        --input qa_model_facing.jsonl \
        --output qa_model_facing_server.jsonl \
        --audio-manifest vimedcss_audio_mapping.jsonl \
        --logical-mapping audio_logical_mapping.jsonl \
        --merged-dir /server/audio_pairs_beep \
        --pair-manifest vimedcss_pair_merge_manifest.jsonl \
        [--recipe resources/semantics/p1_audio_recipe.json] \
        [--check-only] [--force] [--repo-root .]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PAIRWISE_TASK = "vimedcss_pairwise_topic_same"
LAYOUT = "A_BEEP_B"


class MappingError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _resolve(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from src.autonomous_qa.datasets.vimedcss import vimedcss_release as rel

    return rel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("qa_model_facing.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("qa_model_facing_server.jsonl"))
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--logical-mapping", type=Path, default=Path("audio_logical_mapping.jsonl"))
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("vimedcss_pair_merge_manifest.jsonl"))
    parser.add_argument("--recipe", type=Path, default=Path("resources/semantics/p1_audio_recipe.json"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    rel = _resolve(args.repo_root)
    from src.autonomous_qa.datasets.vimedcss import vimedcss_audio as audio

    if not args.input.exists():
        raise MappingError("INPUT_MISSING", str(args.input))
    if not args.logical_mapping.exists():
        raise MappingError("LOGICAL_MAPPING_MISSING", str(args.logical_mapping))
    if not args.recipe.exists():
        raise MappingError("RECIPE_MISSING", str(args.recipe))

    logical_rows = _load_jsonl(args.logical_mapping)
    logical_to_segment = {row["logical_audio_id"]: row["segment_id"] for row in logical_rows}
    try:
        server_manifest = rel.load_audio_manifest(args.audio_manifest)
    except rel.VimedcssReleaseError as exc:
        raise MappingError(exc.code, exc.detail) from exc
    recipe = audio.load_recipe(args.recipe)
    recipe_sha = hashlib.sha256(args.recipe.read_bytes()).hexdigest()
    layout = LAYOUT

    rows = _load_jsonl(args.input)

    def resolve_path(logical_id: str) -> Path:
        segment = logical_to_segment.get(logical_id)
        if not segment:
            raise MappingError("VIMEDCSS_LOGICAL_ID_UNRESOLVED", logical_id)
        try:
            return rel.resolve_source_audio(segment, server_manifest)
        except rel.VimedcssReleaseError as exc:
            raise MappingError(exc.code, exc.detail) from exc

    out_rows: list[dict] = []
    pair_manifest: list[dict] = []
    unresolved_segments: list[str] = []
    merged_planned: list[str] = []

    for row in rows:
        audios = list(row["audio"])
        if len(audios) == 1:
            physical = resolve_path(audios[0])
            out_rows.append({**row, "audio": [str(physical)]})
        elif len(audios) == 2 and row["type_id"] == PAIRWISE_TASK:
            path_a = resolve_path(audios[0])
            path_b = resolve_path(audios[1])
            pair_id = rel.pair_audio_id(
                source_a=audios[0], source_b=audios[1], recipe_sha256=recipe_sha, layout=layout
            )
            merged = args.merged_dir / f"{pair_id}.wav"
            merged_planned.append(str(merged))
            if not args.check_only:
                if not merged.exists() or args.force:
                    try:
                        entry = rel.materialize_pair(
                            source_a=str(path_a),
                            source_b=str(path_b),
                            logical_a=audios[0],
                            logical_b=audios[1],
                            output_dir=args.merged_dir,
                            recipe=recipe,
                            recipe_sha256=recipe_sha,
                            layout=layout,
                        )
                    except rel.VimedcssReleaseError as exc:
                        raise MappingError(exc.code, exc.detail) from exc
                else:
                    entry = {
                        "logical_audio_a": audios[0],
                        "logical_audio_b": audios[1],
                        "physical_audio_a": str(path_a),
                        "physical_audio_b": str(path_b),
                        "source_audio_ids": audios,
                        "merged_audio": str(merged),
                        "recipe_sha256": recipe_sha,
                        "layout": layout,
                        "output_sha256": hashlib.sha256(merged.read_bytes()).hexdigest(),
                    }
                pair_manifest.append(entry)
            out_rows.append({**row, "audio": [str(merged)]})
        else:
            raise MappingError("UNEXPECTED_AUDIO_ARITY", f"{row.get('id')}:{audios}")

    # fail-closed before publishing: every referenced physical output must exist
    if not args.check_only:
        for out_row in out_rows:
            if not Path(out_row["audio"][0]).exists():
                unresolved_segments.append(out_row["audio"][0])

    summary = {
        "rows": len(out_rows),
        "pair_rows": len(pair_manifest) if not args.check_only else len(merged_planned),
        "unresolved": unresolved_segments,
        "check_only": args.check_only,
    }
    if unresolved_segments:
        raise MappingError("VIMEDCSS_SOURCE_AUDIO_UNRESOLVED", ",".join(unresolved_segments[:5]))

    if args.check_only:
        summary["merged_planned"] = merged_planned
        summary["merged_missing"] = [p for p in merged_planned if not Path(p).exists()]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out_rows)
    _write_atomic(args.output, text)
    manifest_text = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
        for r in sorted(pair_manifest, key=lambda item: item["merged_audio"])
    )
    _write_atomic(args.pair_manifest, manifest_text)
    summary["output"] = str(args.output)
    summary["pair_manifest"] = str(args.pair_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MappingError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}), file=sys.stderr)
        raise SystemExit(3)
