"""ViMedCSS source adapter + deterministic metadata diagnostics.

This is a SOURCE-ACCESS-SPECIFIC module (dataset #3 onboarding). It knows how
to enumerate ViMedCSS splits, read metadata rows, parse the documented
semicolon-separated code-switch term list, and run deterministic source
audits. It does NOT decide semantic propositions, does NOT call an LLM, and
does NOT touch audio.

All audits are metadata-only and never mutate source rows.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.common.config import ROOT

DEFAULT_SOURCE_DIR = ROOT / "data_sources" / "vimedcss" / "source"

SPLITS = ("train", "validation", "test", "hard")
RESERVED_SPLITS = ("validation", "test", "hard")
TRAIN_SPLIT = "train"

SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
    "hard": "hard.jsonl",
}

LEGACY_SPLIT_FILES = {
    "train": "train_set.csv",
    "validation": "valid_set.csv",
    "test": "test_set.csv",
    "hard": "hard_set.csv",
}

CANONICAL_FIELDS = (
    "segment_id",
    "duration_seconds",
    "segment_text",
    "cs_terms_list",
    "cs_terms_count",
    "topic",
    "original_video_link",
    "original_video_title",
    "start_time",
    "end_time",
)

# Header casing varies between files (test_set.csv uses "Topic").
_HEADER_ALIASES = {"topic": "topic", "text": "segment_text"}


def _norm_header(name: str) -> str:
    key = name.strip()
    return _HEADER_ALIASES.get(key.lower(), key)


def load_split(source_dir: Path | str | None = None, split: str = "train") -> list[dict[str, Any]]:
    """Read one metadata split (JSONL or fallback CSV) into normalized dict rows (no audio)."""
    is_default = source_dir is None
    if source_dir is None:
        source_dir = DEFAULT_SOURCE_DIR
    source_dir = Path(source_dir)
    filename = SPLIT_FILES[split]
    path = source_dir / filename
    if not path.exists():
        if is_default or source_dir == DEFAULT_SOURCE_DIR:
            raise FileNotFoundError(f"CANONICAL_SOURCE_MISSING:{path}")
        legacy_filename = LEGACY_SPLIT_FILES.get(split, f"{split}_set.csv")
        legacy_path = source_dir / legacy_filename
        if legacy_path.exists():
            path = legacy_path
        else:
            raise FileNotFoundError(f"SOURCE_SPLIT_MISSING:{path}")
    return load_rows_from_path(path)


def load_all_splits(source_dir: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    if source_dir is None:
        source_dir = DEFAULT_SOURCE_DIR
    return {split: load_split(source_dir, split) for split in SPLITS}


def load_rows_from_path(path: Path | str) -> list[dict[str, Any]]:
    """Load normalized rows from an explicit metadata JSONL or CSV path."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                row: dict[str, Any] = {}
                for k, v in raw.items():
                    if k is None:
                        continue
                    key = _norm_header(k)
                    if key == "duration_seconds" and v not in (None, ""):
                        if isinstance(v, str):
                            try:
                                v = float(v) if "." in v else int(v)
                            except ValueError:
                                pass
                    elif key == "cs_terms_count" and v not in (None, ""):
                        try:
                            v = int(v)
                        except ValueError:
                            pass
                    row[key] = v
                rows.append(row)
        return rows
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if k is None:
                    continue
                key = _norm_header(k)
                if key == "duration_seconds" and v not in (None, ""):
                    if isinstance(v, str):
                        try:
                            v = float(v) if "." in v else int(v)
                        except ValueError:
                            pass
                elif key == "cs_terms_count" and v not in (None, ""):
                    try:
                        v = int(v)
                    except ValueError:
                        pass
                row[key] = v
            rows.append(row)
    return rows




# ---------------------------------------------------------------------------
# CS term list parsing (structural, diagnostic; preserves spelling)
# ---------------------------------------------------------------------------


def parse_cs_terms(value: Any) -> list[str]:
    """Split on the documented semicolon; trim only; preserve spelling."""
    if value is None:
        return []
    return [term.strip() for term in str(value).split(";") if term.strip()]


# ---------------------------------------------------------------------------
# Diagnostic normalization levels
# ---------------------------------------------------------------------------


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _strip_punctuation(text: str) -> str:
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)


def levels(text: str) -> dict[str, str]:
    return {
        "raw": text,
        "casefold": text.casefold(),
        "nopunct": _collapse(_strip_punctuation(text)),
        "nopunct_casefold": _collapse(_strip_punctuation(text)).casefold(),
    }


def _tokens(normalized: str) -> list[str]:
    return normalized.split()


def term_match_level(term: str, text: str) -> str:
    """Return the strongest diagnostic match level (A-D) or LEXICAL_MISMATCH."""
    if not term or not text:
        return "ANNOTATION_UNCLEAR"
    tl = levels(term)
    xl = levels(text)
    if tl["raw"] in xl["raw"]:
        return "A_RAW"
    if tl["casefold"] in xl["casefold"]:
        return "B_CASE_ONLY"
    if tl["nopunct_casefold"] in xl["nopunct_casefold"]:
        # distinguish punctuation-only vs whitespace-only by collapsing spaces
        return "C_PUNCT_OR_WS"
    # token-boundary containment on the punctuation/case-normalized forms
    term_tokens = _tokens(tl["nopunct_casefold"])
    text_tokens = _tokens(xl["nopunct_casefold"])
    if term_tokens and _contains_subsequence(text_tokens, term_tokens):
        return "D_TOKEN_BOUNDARY"
    # hyphenation variant: compare with hyphens removed
    if tl["nopunct_casefold"].replace("-", "").replace(" ", "") in xl[
        "nopunct_casefold"
    ].replace("-", "").replace(" ", ""):
        return "HYPHENATION_VARIANT"
    return "LEXICAL_MISMATCH"


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


# ---------------------------------------------------------------------------
# PROFILER (metadata only)
# ---------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def profile_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ids = [r.get("segment_id") for r in rows]
    durations = [
        float(r["duration_seconds"])
        for r in rows
        if isinstance(r.get("duration_seconds"), (int, float))
    ]
    invalid_duration = sum(
        1
        for r in rows
        if isinstance(r.get("duration_seconds"), (int, float))
        and r["duration_seconds"] <= 0
    )

    char_lengths = [len(str(r.get("segment_text") or "")) for r in rows]
    token_lengths = [len(str(r.get("segment_text") or "").split()) for r in rows]

    count_dist: Counter = Counter()
    term_counter: Counter = Counter()
    term_norm_counter: Counter = Counter()
    single = multi = 0
    for r in rows:
        c = r.get("cs_terms_count")
        count_dist[str(c)] += 1
        parsed = parse_cs_terms(r.get("cs_terms_list"))
        if len(parsed) == 1:
            single += 1
        elif len(parsed) > 1:
            multi += 1
        for t in parsed:
            term_counter[t] += 1
            term_norm_counter[_collapse(t).casefold()] += 1

    videos = Counter(
        r.get("original_video_link")
        for r in rows
        if not _is_missing(r.get("original_video_link"))
    )
    topics = Counter(r.get("topic") for r in rows if not _is_missing(r.get("topic")))

    return {
        "rows": total,
        "unique_segment_ids": len({i for i in ids if not _is_missing(i)}),
        "duplicate_segment_ids": len([i for i in ids if not _is_missing(i)])
        - len({i for i in ids if not _is_missing(i)}),
        "missing": {
            field: sum(1 for r in rows if _is_missing(r.get(field)))
            for field in CANONICAL_FIELDS
        },
        "invalid_or_nonpositive_duration": invalid_duration,
        "duration_seconds": {
            "min": min(durations) if durations else 0,
            "p1": _percentile(durations, 0.01),
            "p5": _percentile(durations, 0.05),
            "median": _percentile(durations, 0.5),
            "mean": round(sum(durations) / len(durations), 4) if durations else 0.0,
            "p95": _percentile(durations, 0.95),
            "p99": _percentile(durations, 0.99),
            "max": max(durations) if durations else 0,
            "sum_hours": round(sum(durations) / 3600.0, 4),
        },
        "segment_text_length": {
            "char_min": min(char_lengths) if char_lengths else 0,
            "char_median": _percentile([float(c) for c in char_lengths], 0.5),
            "char_max": max(char_lengths) if char_lengths else 0,
            "token_median": _percentile([float(t) for t in token_lengths], 0.5),
        },
        "cs_terms_count": {
            "distribution": dict(sorted(count_dist.items(), key=lambda kv: kv[0])),
            "mean": round(
                sum(
                    int(r["cs_terms_count"])
                    for r in rows
                    if isinstance(r.get("cs_terms_count"), int)
                )
                / total,
                4,
            )
            if total
            else 0.0,
            "max": max(
                [
                    int(r["cs_terms_count"])
                    for r in rows
                    if isinstance(r.get("cs_terms_count"), int)
                ]
                or [0]
            ),
            "rows_zero": sum(1 for r in rows if r.get("cs_terms_count") == 0),
            "rows_one": sum(1 for r in rows if r.get("cs_terms_count") == 1),
            "rows_two": sum(1 for r in rows if r.get("cs_terms_count") == 2),
            "rows_three_plus": sum(
                1
                for r in rows
                if isinstance(r.get("cs_terms_count"), int) and r["cs_terms_count"] >= 3
            ),
        },
        "unique_raw_cs_terms": len(term_counter),
        "unique_normalized_cs_terms": len(term_norm_counter),
        "single_term_rows": single,
        "multi_term_rows": multi,
        "topic_distribution": dict(topics.most_common()),
        "source_videos": {
            "unique": len(videos),
            "segments_per_video_max": max(videos.values()) if videos else 0,
            "segments_per_video_median": _percentile(
                [float(v) for v in videos.values()], 0.5
            ),
        },
    }


# ---------------------------------------------------------------------------
# AUDITS
# ---------------------------------------------------------------------------


def cs_count_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = mismatch = 0
    declared_zero_nonempty = declared_positive_empty = 0
    duplicate_term_rows = 0
    examples: list[dict[str, Any]] = []
    for r in rows:
        declared = r.get("cs_terms_count")
        parsed = parse_cs_terms(r.get("cs_terms_list"))
        if len(set(parsed)) != len(parsed):
            duplicate_term_rows += 1
        if declared == len(parsed):
            exact += 1
            continue
        mismatch += 1
        if declared == 0 and parsed:
            declared_zero_nonempty += 1
        if isinstance(declared, int) and declared > 0 and not parsed:
            declared_positive_empty += 1
        if len(examples) < 10:
            examples.append(
                {
                    "segment_id": r.get("segment_id"),
                    "declared": declared,
                    "parsed_count": len(parsed),
                    "cs_terms_list": r.get("cs_terms_list"),
                }
            )
    return {
        "rows": len(rows),
        "exact_matches": exact,
        "mismatches": mismatch,
        "declared_zero_nonempty_list": declared_zero_nonempty,
        "declared_positive_empty_list": declared_positive_empty,
        "duplicate_term_rows": duplicate_term_rows,
        "examples": examples,
    }


def term_transcript_match(rows: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts: Counter = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_terms = 0
    for r in rows:
        text = str(r.get("segment_text") or "")
        for term in parse_cs_terms(r.get("cs_terms_list")):
            total_terms += 1
            lvl = term_match_level(term, text)
            level_counts[lvl] += 1
            if lvl != "A_RAW" and len(examples[lvl]) < 5:
                examples[lvl].append(
                    {
                        "segment_id": r.get("segment_id"),
                        "term": term,
                        "segment_text": text[:160],
                    }
                )
    return {
        "total_annotated_terms": total_terms,
        "level_counts": dict(level_counts.most_common()),
        "examples": {k: v for k, v in examples.items()},
    }


def term_representation_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    multiword = acronym = digits = hyphen = slash = repeated = 0
    forms: Counter = Counter()
    for r in rows:
        parsed = parse_cs_terms(r.get("cs_terms_list"))
        if len(parsed) != len(set(parsed)):
            repeated += 1
        for term in parsed:
            if " " in term:
                multiword += 1
            if re.fullmatch(r"[A-Z]{2,}", term):
                acronym += 1
                forms["ACRONYM"] += 1
            elif re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", term):
                forms["CAPITALIZED"] += 1
            elif term.islower():
                forms["LOWERCASE"] += 1
            if any(ch.isdigit() for ch in term):
                digits += 1
            if "-" in term:
                hyphen += 1
            if "/" in term:
                slash += 1
    return {
        "multiword_terms": multiword,
        "acronym_terms": acronym,
        "terms_with_digits": digits,
        "hyphenated_terms": hyphen,
        "slash_terms": slash,
        "rows_with_duplicate_labels": repeated,
        "case_forms": dict(forms),
    }


def term_order_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    same = different = not_found = ambiguous = 0
    for r in rows:
        parsed = parse_cs_terms(r.get("cs_terms_list"))
        if len(parsed) < 2:
            continue
        text = levels(str(r.get("segment_text") or ""))["nopunct_casefold"]
        positions = []
        missing = False
        for term in parsed:
            t = levels(term)["nopunct_casefold"]
            idx = text.find(t)
            if idx < 0:
                missing = True
                break
            positions.append(idx)
        if missing:
            not_found += 1
        elif len(set(parsed)) != len(parsed):
            ambiguous += 1
        elif positions == sorted(positions):
            same += 1
        else:
            different += 1
    return {
        "multi_term_rows": same + different + not_found + ambiguous,
        "order_of_occurrence": same,
        "different_order": different,
        "term_not_found": not_found,
        "ambiguous_or_repeated": ambiguous,
        "recommended_semantics": (
            "ORDER_OF_OCCURRENCE"
            if same and different == 0 and not_found == 0
            else "UNORDERED_MULTISET"
        ),
    }


def topic_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = Counter(r.get("topic") for r in rows if not _is_missing(r.get("topic")))
    variants = Counter(
        str(r.get("topic")).strip().casefold()
        for r in rows
        if not _is_missing(r.get("topic"))
    )
    return {
        "raw_labels": dict(raw.most_common()),
        "missing": sum(1 for r in rows if _is_missing(r.get("topic"))),
        "normalized_label_count": len(variants),
    }


def topic_source_video_shortcut(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, set[str]] = defaultdict(set)
    video_rows: Counter = Counter()
    for r in rows:
        v = r.get("original_video_link")
        t = r.get("topic")
        if _is_missing(v) or _is_missing(t):
            continue
        by_video[str(v)].add(str(t))
        video_rows[str(v)] += 1
    single_topic = sum(1 for topics in by_video.values() if len(topics) == 1)
    predictable = sum(
        video_rows[v] for v, topics in by_video.items() if len(topics) == 1
    )
    total = sum(video_rows.values())
    return {
        "unique_videos": len(by_video),
        "videos_single_topic": single_topic,
        "videos_multi_topic": len(by_video) - single_topic,
        "rows_topic_predictable_from_video": predictable,
        "rows_total": total,
        "predictable_fraction": round(predictable / total, 4) if total else 0.0,
        "source_group_shortcut_risk": predictable / total >= 0.5 if total else False,
    }


def source_video_interval_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def parse_ts(value: Any) -> int | None:
        if value is None:
            return None
        parts = str(value).strip().split(":")
        if not parts or not all(p.isdigit() for p in parts):
            return None
        seconds = 0
        for p in parts:
            seconds = seconds * 60 + int(p)
        return seconds

    by_video: dict[str, list[tuple[int, int]]] = defaultdict(list)
    unparseable = 0
    for r in rows:
        v = r.get("original_video_link")
        start = parse_ts(r.get("start_time"))
        end = parse_ts(r.get("end_time"))
        if _is_missing(v) or start is None or end is None:
            unparseable += 1
            continue
        by_video[str(v)].append((start, end))
    duplicate_intervals = overlapping = 0
    for intervals in by_video.values():
        seen_final = set()
        for a, b in intervals:
            if (a, b) in seen_final:
                duplicate_intervals += 1
            seen_final.add((a, b))
        ordered = sorted(intervals)
        for i in range(1, len(ordered)):
            if ordered[i][0] < ordered[i - 1][1]:
                overlapping += 1
    return {
        "unparseable_timestamps": unparseable,
        "duplicate_intervals": duplicate_intervals,
        "overlapping_intervals": overlapping,
    }


def cross_split_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def ids(rows: Iterable[dict]) -> set[str]:
        return {
            str(r.get("segment_id"))
            for r in rows
            if not _is_missing(r.get("segment_id"))
        }

    def texts(rows: Iterable[dict]) -> set[str]:
        return {
            str(r.get("segment_text"))
            for r in rows
            if not _is_missing(r.get("segment_text"))
        }

    def videos(rows: Iterable[dict]) -> set[str]:
        return {
            str(r.get("original_video_link"))
            for r in rows
            if not _is_missing(r.get("original_video_link"))
        }

    def vocab(rows: Iterable[dict]) -> set[str]:
        return {
            _collapse(t).casefold()
            for r in rows
            for t in parse_cs_terms(r.get("cs_terms_list"))
        }

    result: dict[str, Any] = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            result[f"{a}__{b}"] = {
                "duplicate_ids": len(ids(splits[a]) & ids(splits[b])),
                "duplicate_transcripts": len(texts(splits[a]) & texts(splits[b])),
                "shared_source_videos": len(videos(splits[a]) & videos(splits[b])),
                "shared_cs_vocab": len(vocab(splits[a]) & vocab(splits[b])),
            }
    return result


def negative_capacity(
    rows: list[dict[str, Any]], *, same_topic: bool = False
) -> dict[str, Any]:
    """TRAIN-only negative pool capacity for a term-verification task.

    For each row, eligible negatives are TRAIN vocabulary terms that are not
    annotated for that row (and not a lexical substring of its transcript).
    No negatives are sampled or frozen here.
    """
    vocab: set[str] = set()
    for r in rows:
        for t in parse_cs_terms(r.get("cs_terms_list")):
            vocab.add(_collapse(t).casefold())

    sizes: list[int] = []
    for r in rows:
        annotated = {
            _collapse(t).casefold() for t in parse_cs_terms(r.get("cs_terms_list"))
        }
        text = levels(str(r.get("segment_text") or ""))["nopunct_casefold"]
        if same_topic:
            topic = r.get("topic")
            pool = {
                t
                for other in rows
                if other.get("topic") == topic
                for t in [
                    _collapse(x).casefold()
                    for x in parse_cs_terms(other.get("cs_terms_list"))
                ]
            }
        else:
            pool = vocab
        eligible = {t for t in pool if t not in annotated and t and t not in text}
        sizes.append(len(eligible))
    sizes.sort()
    return {
        "rows": len(sizes),
        "min": sizes[0] if sizes else 0,
        "p1": _percentile([float(s) for s in sizes], 0.01),
        "median": _percentile([float(s) for s in sizes], 0.5),
        "p99": _percentile([float(s) for s in sizes], 0.99),
        "max": sizes[-1] if sizes else 0,
        "same_topic": same_topic,
    }


def hard_vocab_audit(
    train_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Vocabulary boundary: TRAIN vocabulary only; reserved split isolated."""
    train_vocab: Counter = Counter()
    for r in train_rows:
        for t in parse_cs_terms(r.get("cs_terms_list")):
            train_vocab[_collapse(t).casefold()] += 1
    other_vocab: Counter = Counter()
    for r in other_rows:
        for t in parse_cs_terms(r.get("cs_terms_list")):
            other_vocab[_collapse(t).casefold()] += 1

    seen = [t for t in other_vocab if t in train_vocab]
    unseen = [t for t in other_vocab if t not in train_vocab]
    seen_occ = sum(other_vocab[t] for t in seen)
    unseen_occ = sum(other_vocab[t] for t in unseen)
    return {
        "train_vocab_size": len(train_vocab),
        "other_vocab_size": len(other_vocab),
        "seen_in_train_types": len(seen),
        "unseen_types": len(unseen),
        "type_unseen_rate": round(len(unseen) / len(other_vocab), 4)
        if other_vocab
        else 0.0,
        "seen_occurrences": seen_occ,
        "unseen_occurrences": unseen_occ,
        "occurrence_unseen_rate": round(unseen_occ / (seen_occ + unseen_occ), 4)
        if (seen_occ + unseen_occ)
        else 0.0,
        "rare_but_seen_types": sum(1 for t in seen if train_vocab[t] <= 2),
        "unseen_examples": sorted(unseen)[:10],
    }
