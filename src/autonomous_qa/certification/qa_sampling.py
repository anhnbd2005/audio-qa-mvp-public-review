"""Full-TRAIN O(N) indexes, capacity formulas, and bounded zero-LLM samplers.

Generic over operator + semantic_class + value_policy + TypeContract +
SemanticFieldSpec. No dataset field names are hardcoded here; ViMD bindings
arrive through the contracts/specs supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal

from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec, ValuePolicy
from src.autonomous_qa.language.template_contracts import TypeContract

ValueSampling = Literal["uniform_over_rows", "uniform_over_values"]


class BudgetShortfallError(RuntimeError):
    """Requested semantic budget exceeds feasible unique capacity."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        super().__init__(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        (str(seed) + "|" + "|".join(parts)).encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def meta_of(row: dict[str, Any]) -> dict[str, Any]:
    """Flat metadata rows and enveloped preview rows share one access path."""
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else row


def normalize_field_value(value: Any, policy: ValuePolicy) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if policy.normalization == "identity":
        return text
    if policy.normalization == "normalized_text_exact":
        return " ".join(text.split()).casefold()
    raise ValueError(f"UNKNOWN_VALUE_NORMALIZATION:{policy.normalization}")


def comb2(count: int) -> int:
    return count * (count - 1) // 2


class TrainIndex:
    """O(N) indexes over all TRAIN rows; never materializes row pairs."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        row_ids: list[str],
        specs: dict[str, SemanticFieldSpec],
        hidden_identifier_field: str | None = None,
    ) -> None:
        if len(rows) != len(row_ids):
            raise ValueError("ROW_ID_ALIGNMENT_MISMATCH")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("DUPLICATE_SOURCE_ROW_ID")
        self.rows = rows
        self.row_ids = list(row_ids)
        self.specs = dict(specs)
        self.row_by_id = dict(zip(row_ids, rows, strict=True))
        self.rows_by_value: dict[str, dict[str, list[str]]] = {}
        self.valid_row_ids: dict[str, list[str]] = {}
        self.display: dict[str, dict[str, Any]] = {}
        self._norm_cache: dict[tuple[str, str], str] = {}
        self.hidden_value_by_row: dict[str, str] = {}
        for field_name, spec in specs.items():
            groups: dict[str, list[str]] = {}
            valid: list[str] = []
            shown: dict[str, Any] = {}
            for row_id, row in zip(row_ids, rows, strict=True):
                raw = meta_of(row).get(field_name)
                norm = self._norm(raw, field_name, spec.value_policy)
                if not norm:
                    continue
                groups.setdefault(norm, []).append(row_id)
                valid.append(row_id)
                shown.setdefault(norm, raw)
            self.rows_by_value[field_name] = groups
            self.valid_row_ids[field_name] = valid
            self.display[field_name] = shown
        if hidden_identifier_field:
            for row_id, row in zip(row_ids, rows, strict=True):
                raw = meta_of(row).get(hidden_identifier_field)
                if raw not in (None, ""):
                    self.hidden_value_by_row[row_id] = str(raw)

    def _norm(self, raw: Any, field_name: str, policy: ValuePolicy) -> str:
        key = (field_name, str(raw if raw is not None else ""))
        if key not in self._norm_cache:
            self._norm_cache[key] = normalize_field_value(raw, policy)
        return self._norm_cache[key]

    def normalize(self, field_name: str, raw: Any) -> str:
        return self._norm(raw, field_name, self.specs[field_name].value_policy)

    def values(self, field_name: str) -> list[str]:
        return sorted(self.rows_by_value[field_name])

    def group(self, field_name: str, norm: str) -> list[str]:
        return self.rows_by_value[field_name].get(norm, [])

    def summary(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for field_name in sorted(self.rows_by_value):
            groups = self.rows_by_value[field_name]
            counts = [len(v) for v in groups.values()]
            fields[field_name] = {
                "valid_rows": len(self.valid_row_ids[field_name]),
                "empty_or_missing_rows": len(self.rows)
                - len(self.valid_row_ids[field_name]),
                "domain_size": len(groups),
                "max_group_size": max(counts, default=0),
                "groups_with_count_ge_2": sum(1 for c in counts if c >= 2),
            }
        return {
            "rows_indexed": len(self.rows),
            "fields": fields,
            "hidden_identifier_rows": len(self.hidden_value_by_row),
            "complexity": "O(N * F) grouping; no O(N^2) pair enumeration",
            "pair_enumeration": "NOT_PERFORMED",
        }


EqualityStrategy = Literal["default", "anchor_neighborhood"]
PairUniqueness = Literal["unordered", "ordered"]


@dataclass
class SamplingSettings:
    value_sampling: ValueSampling = "uniform_over_values"
    prefer_distinct_speaker: bool = True
    hidden_identifier_field: str | None = None
    positive_ratio: float | None = None
    randomized_position: bool = True
    length_bucket_match: bool = True
    max_source_row_reuse: int | None = None
    max_source_row_reuse_per_type: int | None = None
    max_sampling_attempts: int = 64
    # Opt-in EQUALITY strategy (default behaviour unchanged).
    equality_strategy: EqualityStrategy = "default"
    equality_candidate_pool_same: int = 8
    equality_candidate_pool_different: int = 8
    equality_positive_per_anchor: int = 2
    equality_negative_per_anchor: int = 2
    equality_pair_uniqueness: PairUniqueness = "unordered"


@dataclass
class ReuseTracker:
    """Bounded source-row reuse; caps apply only when the config requests them."""

    global_cap: int | None = None
    per_type_cap: int | None = None
    global_counts: Counter = dataclass_field(default_factory=Counter)
    type_counts: dict[str, Counter] = dataclass_field(default_factory=dict)

    def available(self, row_id: str, type_id: str) -> bool:
        if (
            self.global_cap is not None
            and self.global_counts[row_id] >= self.global_cap
        ):
            return False
        if self.per_type_cap is not None:
            count = self.type_counts.get(type_id, Counter()).get(row_id, 0)
            if count >= self.per_type_cap:
                return False
        return True

    def acquire(self, type_id: str, row_ids: list[str]) -> None:
        self.global_counts.update(row_ids)
        bucket = self.type_counts.setdefault(type_id, Counter())
        bucket.update(row_ids)

    def audit(self) -> dict[str, Any]:
        counts = sorted(self.global_counts.values())
        total = sum(counts)
        if not counts:
            return {
                "unique_source_rows": 0,
                "total_source_row_references": 0,
                "max_reuse": 0,
                "mean_reuse": 0.0,
                "median_reuse": 0.0,
                "p95_reuse": 0.0,
                "top_reused_source_row_ids": [],
                "global_cap": self.global_cap,
                "per_type_cap": self.per_type_cap,
            }

        def percentile(values: list[int], pct: float) -> float:
            index = min(len(values) - 1, round(pct * (len(values) - 1)))
            return float(values[index])

        top = sorted(self.global_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        return {
            "unique_source_rows": len(counts),
            "total_source_row_references": total,
            "max_reuse": max(counts),
            "mean_reuse": round(total / len(counts), 6),
            "median_reuse": percentile(counts, 0.5),
            "p95_reuse": percentile(counts, 0.95),
            "top_reused_source_row_ids": [
                {"source_row_id": row_id, "count": count} for row_id, count in top
            ],
            "global_cap": self.global_cap,
            "per_type_cap": self.per_type_cap,
            "cap_violations": 0,
        }


def capacity_for_contract(
    contract: TypeContract,
    index: TrainIndex,
    spec: SemanticFieldSpec,
    complete_train: bool,
) -> dict[str, Any]:
    """Per-type capacity from the loaded index; O(V), no pair enumeration."""
    field_name = contract.semantic_field
    counts = Counter(
        {norm: len(rows) for norm, rows in index.rows_by_value[field_name].items()}
    )
    valid_rows = sum(counts.values())
    distinct = len(counts)
    groups_ge2 = sum(1 for c in counts.values() if c >= 2)
    max_group = max(counts.values(), default=0)
    positive = valid_rows >= 1
    negative = distinct >= 2
    selection = positive and negative
    total_pairs = comb2(valid_rows)
    positive_pairs = sum(comb2(c) for c in counts.values())
    if contract.operator == "EQUALITY":
        positive = groups_ge2 >= 1
        status = (
            "SUFFICIENT"
            if positive and negative
            else (
                "INSUFFICIENT_POSITIVE_CAPACITY"
                if not positive
                else "INSUFFICIENT_NEGATIVE_CAPACITY"
            )
        )
        positive_theoretical = positive_pairs
        negative_theoretical = total_pairs - positive_pairs
        selection_capacity = None
    elif contract.operator == "DIRECT":
        status = "SUFFICIENT" if positive else "UNKNOWN"
        positive_theoretical = valid_rows
        negative_theoretical = None
        selection_capacity = None
    elif contract.operator == "TARGET_MATCH":
        status = "SUFFICIENT" if selection else "INSUFFICIENT_NEGATIVE_CAPACITY"
        positive_theoretical = valid_rows
        negative_theoretical = valid_rows * max(0, distinct - 1)
        selection_capacity = None
    else:  # PAIRWISE_SELECTION
        status = "SUFFICIENT" if selection else "INSUFFICIENT_NEGATIVE_CAPACITY"
        positive_theoretical = None
        negative_theoretical = None
        selection_capacity = sum(c * (valid_rows - c) for c in counts.values())
    if not complete_train:
        status = "NOT_AUDITED"
    return {
        "type_id": contract.type_id,
        "operator": contract.operator,
        "field": field_name,
        "semantic_class": spec.semantic_class,
        "metadata_rows_examined": len(index.rows),
        "source_scope": "COMPLETE_LOCAL_TRAIN"
        if complete_train
        else "PARTIAL_LOCAL_ONLY",
        "positive_capacity": positive,
        "negative_capacity": negative if contract.operator != "DIRECT" else None,
        "selection_capacity": selection_capacity
        if contract.operator == "PAIRWISE_SELECTION"
        else None,
        "full_train_capacity_status": status,
        "semantic_status": contract.source_status,
        "theoretical_semantic_capacity": {
            "positive": positive_theoretical,
            "negative": negative_theoretical,
            "selection": selection_capacity,
        },
        "evidence_summary": {
            "valid_rows": valid_rows,
            "unique_values": distinct,
            "groups_with_count_ge_2": groups_ge2,
            "maximum_group_size": max_group,
            "negative_value_availability": negative,
            "pair_enumeration": "NOT_PERFORMED",
        },
    }


def feasible_semantic_capacity(
    capacity: dict[str, Any], positive_ratio: float | None
) -> int:
    """Largest unique semantic budget a type can serve under its label ratio."""
    operator = capacity["operator"]
    if operator == "DIRECT":
        return int(capacity["theoretical_semantic_capacity"]["positive"] or 0)
    if operator == "PAIRWISE_SELECTION":
        return int(capacity["theoretical_semantic_capacity"]["selection"] or 0)
    positive_cap = int(capacity["theoretical_semantic_capacity"]["positive"] or 0)
    negative_cap = int(capacity["theoretical_semantic_capacity"]["negative"] or 0)
    ratio = positive_ratio if positive_ratio is not None else 0.5
    if ratio <= 0:
        return negative_cap
    if ratio >= 1:
        return positive_cap
    return int(min(positive_cap / ratio, negative_cap / (1.0 - ratio)))


def positive_count_for(total: int, positive_ratio: float | None) -> int:
    if positive_ratio is None:
        return (total + 1) // 2
    return round(total * positive_ratio)


class SemanticSampler:
    """Bounded sampler producing unique semantic drafts for one type."""

    def __init__(
        self,
        contract: TypeContract,
        spec: SemanticFieldSpec,
        index: TrainIndex,
        settings: SamplingSettings,
        reuse: ReuseTracker,
        seed: int,
    ) -> None:
        if spec.value_policy.match_policy != "exact":
            raise ValueError(
                f"UNSUPPORTED_MATCH_POLICY:{contract.type_id}:"
                f"{spec.value_policy.match_policy}"
            )
        self.contract = contract
        self.spec = spec
        self.index = index
        self.settings = settings
        self.reuse = reuse
        self.seed = seed
        self.field = contract.semantic_field
        self.rng = random.Random(stable_seed(seed, contract.type_id, "semantic"))
        self.seen: set[tuple[Any, ...]] = set()
        self.stats: Counter = Counter()

    def sample(self, total: int) -> list[dict[str, Any]]:
        if total <= 0:
            return []
        operator = self.contract.operator
        if operator == "DIRECT":
            drafts = self._direct(total)
        elif operator == "EQUALITY":
            drafts = self._equality(total)
        elif operator == "TARGET_MATCH":
            drafts = self._target_match(total)
        elif operator == "PAIRWISE_SELECTION":
            drafts = self._selection(total)
        else:
            raise ValueError(f"UNKNOWN_OPERATOR:{operator}")
        if len(drafts) != total:
            raise BudgetShortfallError(
                {
                    "code": "BUDGET_SHORTFALL",
                    "type_id": self.contract.type_id,
                    "requested": total,
                    "feasible": len(drafts),
                    "shortfall": total - len(drafts),
                }
            )
        return drafts

    # --- shared helpers -------------------------------------------------
    def _values(self) -> list[str]:
        return self.index.values(self.field)

    def _display(self, norm: str) -> Any:
        return self.index.display[self.field].get(norm)

    def _row_value(self, row_id: str) -> Any:
        return meta_of(self.index.row_by_id[row_id]).get(self.field)

    def _row_norm(self, row_id: str) -> str:
        return self.index.normalize(self.field, self._row_value(row_id))

    def _speaker(self, row_id: str) -> str | None:
        return self.index.hidden_value_by_row.get(row_id)

    def _available(self, row_id: str) -> bool:
        return self.reuse.available(row_id, self.contract.type_id)

    def _row_stream(self) -> Iterator[str]:
        """Deterministic shuffled stream honouring the value-sampling policy.

        With `uniform_over_values`, rows are interleaved round-robin across
        values (one row per value per cycle). Exhausting one value group
        before the next would collapse a bounded DIRECT sample onto a single
        value (e.g. all rows from one region), so value diversity is
        preserved without changing per-value row shuffling.
        """
        if self.settings.value_sampling == "uniform_over_values":
            values = self._values()
            self.rng.shuffle(values)
            groups: list[list[str]] = []
            for norm in values:
                rows = list(self.index.group(self.field, norm))
                self.rng.shuffle(rows)
                if rows:
                    groups.append(rows)
            while groups:
                remaining: list[list[str]] = []
                for rows in groups:
                    yield rows.pop()
                    if rows:
                        remaining.append(rows)
                groups = remaining
        else:
            rows = list(self.index.valid_row_ids[self.field])
            self.rng.shuffle(rows)
            yield from rows

    def _pick_in_group(
        self, norm: str, exclude: str | None = None, prefer_different: str | None = None
    ) -> str | None:
        pool = [
            row_id
            for row_id in self.index.group(self.field, norm)
            if row_id != exclude and self._available(row_id)
        ]
        if not pool:
            return None
        if prefer_different is not None and self.settings.prefer_distinct_speaker:
            distinct = [
                row_id for row_id in pool if self._speaker(row_id) != prefer_different
            ]
            if distinct:
                return distinct[self.rng.randrange(len(distinct))]
            self.stats["same_speaker_fallbacks"] += 1
        return pool[self.rng.randrange(len(pool))]

    def _different_value(self, norm: str) -> str | None:
        others = [value for value in self._values() if value != norm]
        if not others:
            return None
        return others[self.rng.randrange(len(others))]

    def _negative_target(self, row_norm: str) -> tuple[str, bool] | None:
        """Choose a real-domain negative target; text uses a length bucket."""
        others = [value for value in self._values() if value != row_norm]
        if not others:
            return None
        fallback = False
        if (
            self.spec.semantic_class == "text_content"
            and self.settings.length_bucket_match
        ):
            size = max(1, len(row_norm))
            close = [value for value in others if 0.6 <= len(value) / size <= 1.67]
            if close:
                others = close
            else:
                fallback = True
        chosen = others[self.rng.randrange(len(others))]
        if fallback:
            self.stats["text_negative_length_fallbacks"] += 1
        return chosen, fallback

    def _positive_total(self, total: int) -> int:
        return positive_count_for(total, self.settings.positive_ratio)

    # --- operator samplers ----------------------------------------------
    def _direct(self, total: int) -> list[dict[str, Any]]:
        drafts: list[dict[str, Any]] = []
        for row_id in self._row_stream():
            if len(drafts) >= total:
                break
            if not self._available(row_id):
                continue
            norm = self._row_norm(row_id)
            if not norm:
                continue
            key = ("direct", row_id)
            if key in self.seen:
                continue
            self.seen.add(key)
            draft = {
                "source_row_ids": [row_id],
                "target": None,
                "target_display": None,
                "gold_value": self._row_value(row_id),
                "flags": {},
            }
            self.reuse.acquire(self.contract.type_id, draft["source_row_ids"])
            drafts.append(draft)
        return drafts

    def _equality(self, total: int) -> list[dict[str, Any]]:
        if self.settings.equality_strategy == "anchor_neighborhood":
            return self._equality_anchor_neighborhood(total)
        positive_total = self._positive_total(total)
        negative_total = total - positive_total
        drafts: list[dict[str, Any]] = []
        drafts.extend(self._equality_positive(positive_total))
        drafts.extend(self._equality_negative(negative_total))
        return drafts

    # --- opt-in anchor-neighborhood EQUALITY strategy --------------------
    def _rank_key(self, *parts: str) -> str:
        return hashlib.sha256(
            ("|".join([str(self.seed), self.contract.type_id, *parts])).encode("utf-8")
        ).hexdigest()

    def _anchor_pair_draft(
        self, anchor: str, partner: str, is_same: bool
    ) -> dict[str, Any]:
        # Delivery order is ALWAYS anchor first (A then B).
        return {
            "source_row_ids": [anchor, partner],
            "target": None,
            "target_display": None,
            "gold_value": bool(is_same),
            "flags": {"pair_kind": "SAME_TOPIC" if is_same else "DIFFERENT_TOPIC"},
        }

    def _equality_anchor_neighborhood(self, total: int) -> list[dict[str, Any]]:
        """AF3-inspired bounded neighborhood strategy.

        Every eligible row is used exactly once as an anchor and receives
        ``positive_per_anchor`` SAME_TOPIC and ``negative_per_anchor``
        DIFFERENT_TOPIC comparisons, selected from a bounded, deterministic
        SHA256-ranked neighborhood. Unordered pairs are globally unique; the
        anchor always precedes its partner in delivery order.
        """
        field = self.field
        pos_needed = self.settings.equality_positive_per_anchor
        neg_needed = self.settings.equality_negative_per_anchor
        anchors = sorted(self.index.valid_row_ids[field])
        expected = len(anchors) * (pos_needed + neg_needed)
        if total != expected:
            raise BudgetShortfallError(
                {
                    "code": "BUDGET_SHORTFALL",
                    "type_id": self.contract.type_id,
                    "kind": "ANCHOR_NEIGHBORHOOD_TOTAL_MISMATCH",
                    "requested": total,
                    "feasible": expected,
                    "shortfall": total - expected,
                }
            )
        if not anchors:
            return []

        norm_by_row = {row_id: self._row_norm(row_id) for row_id in anchors}
        group_order: dict[str, list[str]] = {
            norm: sorted(members, key=lambda r: self._rank_key("order", r))
            for norm, members in self.index.rows_by_value[field].items()
        }
        position: dict[str, int] = {}
        for members in group_order.values():
            for index, row_id in enumerate(members):
                position[row_id] = index
        global_order = sorted(anchors, key=lambda r: self._rank_key("order", r))

        window_same = max(self.settings.equality_candidate_pool_same, pos_needed)
        window_diff = max(self.settings.equality_candidate_pool_different, neg_needed)

        partner_reuse: Counter = Counter()
        drafts: list[dict[str, Any]] = []

        for anchor in global_order:
            norm = norm_by_row[anchor]
            members = group_order.get(norm, [])
            group_size = len(members)
            start = (position[anchor] + 1) % group_size if group_size else 0

            # SAME_TOPIC neighborhood (bounded, expandable fallback).
            chosen_same: list[str] = []
            window = min(window_same, max(0, group_size - 1))
            while True:
                cands = [
                    members[(start + k) % group_size]
                    for k in range(window)
                ]
                chosen_same = self._select_partners(
                    anchor, cands, pos_needed, "same", partner_reuse
                )
                if len(chosen_same) >= pos_needed or window >= group_size - 1:
                    break
                window = min(group_size - 1, max(window * 2, window + 1))
            if len(chosen_same) < pos_needed:
                raise BudgetShortfallError(
                    {
                        "code": "BUDGET_SHORTFALL",
                        "type_id": self.contract.type_id,
                        "kind": "ANCHOR_NEIGHBORHOOD_SAME",
                        "anchor": anchor,
                        "requested": pos_needed,
                        "feasible": len(chosen_same),
                    }
                )

            # DIFFERENT_TOPIC neighborhood (bounded, expandable fallback).
            diff_order = [
                row_id for row_id in global_order if norm_by_row[row_id] != norm
            ]
            chosen_diff: list[str] = []
            window = min(window_diff, len(diff_order))
            while True:
                cands = diff_order[:window]
                chosen_diff = self._select_partners(
                    anchor, cands, neg_needed, "different", partner_reuse
                )
                if len(chosen_diff) >= neg_needed or window >= len(diff_order):
                    break
                window = min(len(diff_order), max(window * 2, window + 1))
            if len(chosen_diff) < neg_needed:
                raise BudgetShortfallError(
                    {
                        "code": "BUDGET_SHORTFALL",
                        "type_id": self.contract.type_id,
                        "kind": "ANCHOR_NEIGHBORHOOD_DIFFERENT",
                        "anchor": anchor,
                        "requested": neg_needed,
                        "feasible": len(chosen_diff),
                    }
                )

            for partner in chosen_same:
                self.seen.add(self._pair_key(anchor, partner))
                partner_reuse[partner] += 1
                drafts.append(self._anchor_pair_draft(anchor, partner, True))
            for partner in chosen_diff:
                self.seen.add(self._pair_key(anchor, partner))
                partner_reuse[partner] += 1
                drafts.append(self._anchor_pair_draft(anchor, partner, False))

        self.stats["anchor_neighborhood_same"] = pos_needed * len(global_order)
        self.stats["anchor_neighborhood_different"] = neg_needed * len(global_order)
        return drafts

    def _pair_key(self, anchor: str, partner: str) -> tuple[str, str]:
        if self.settings.equality_pair_uniqueness == "unordered":
            return tuple(sorted((anchor, partner)))
        return (anchor, partner)

    def _select_partners(
        self,
        anchor: str,
        candidates: list[str],
        needed: int,
        tag: str,
        partner_reuse: Counter,
    ) -> list[str]:
        """Pure selection: prefer lower partner reuse, tie-break by SHA256 rank."""
        rank = {row_id: self._rank_key(tag, anchor, row_id) for row_id in candidates}
        ordered = sorted(
            candidates, key=lambda row_id: (partner_reuse[row_id], rank[row_id], row_id)
        )
        chosen: list[str] = []
        for row_id in ordered:
            if len(chosen) >= needed:
                break
            if row_id == anchor:
                continue
            if self._pair_key(anchor, row_id) in self.seen:
                continue
            chosen.append(row_id)
        return chosen

    def _equality_positive(self, total: int) -> list[dict[str, Any]]:
        values = [
            v for v in self._values() if len(self.index.group(self.field, v)) >= 2
        ]
        if not values or total <= 0:
            if total > 0:
                raise BudgetShortfallError(
                    {
                        "code": "BUDGET_SHORTFALL",
                        "type_id": self.contract.type_id,
                        "kind": "EQUALITY_POSITIVE",
                        "requested": total,
                        "feasible": 0,
                        "shortfall": total,
                    }
                )
            return []
        flat_rows = [
            row_id
            for norm in values
            for row_id in self.index.group(self.field, norm)
            if self._available(row_id)
        ]
        drafts: list[dict[str, Any]] = []
        attempts = 0
        limit = self.settings.max_sampling_attempts * max(1, total)
        while len(drafts) < total and attempts < limit:
            attempts += 1
            if self.settings.value_sampling == "uniform_over_values":
                norm = values[self.rng.randrange(len(values))]
            else:
                if not flat_rows:
                    break
                norm = self._row_norm(flat_rows[self.rng.randrange(len(flat_rows))])
            first = self._pick_in_group(norm)
            if first is None:
                continue
            second = self._pick_in_group(
                norm,
                exclude=first,
                prefer_different=(
                    self._speaker(first)
                    if self.settings.prefer_distinct_speaker
                    else None
                ),
            )
            if second is None:
                continue
            pair = tuple(sorted((first, second)))
            if pair in self.seen:
                continue
            self.seen.add(pair)
            draft = {
                "source_row_ids": list(pair),
                "target": None,
                "target_display": None,
                "gold_value": True,
                "flags": {},
            }
            self.reuse.acquire(self.contract.type_id, draft["source_row_ids"])
            drafts.append(draft)
        return drafts

    def _equality_negative(self, total: int) -> list[dict[str, Any]]:
        values = self._values()
        if len(values) < 2:
            if total > 0:
                raise BudgetShortfallError(
                    {
                        "code": "BUDGET_SHORTFALL",
                        "type_id": self.contract.type_id,
                        "kind": "EQUALITY_NEGATIVE",
                        "requested": total,
                        "feasible": 0,
                        "shortfall": total,
                    }
                )
            return []
        drafts: list[dict[str, Any]] = []
        attempts = 0
        limit = self.settings.max_sampling_attempts * max(1, total)
        while len(drafts) < total and attempts < limit:
            attempts += 1
            left, right = self.rng.sample(values, 2)
            first = self._pick_in_group(left)
            second = self._pick_in_group(
                right,
                prefer_different=(
                    self._speaker(first)
                    if first is not None and self.settings.prefer_distinct_speaker
                    else None
                ),
            )
            if first is None or second is None:
                continue
            pair = tuple(sorted((first, second)))
            if pair in self.seen:
                continue
            self.seen.add(pair)
            draft = {
                "source_row_ids": list(pair),
                "target": None,
                "target_display": None,
                "gold_value": False,
                "flags": {},
            }
            self.reuse.acquire(self.contract.type_id, draft["source_row_ids"])
            drafts.append(draft)
        return drafts

    def _target_match(self, total: int) -> list[dict[str, Any]]:
        positive_total = self._positive_total(total)
        negative_total = total - positive_total
        drafts: list[dict[str, Any]] = []
        produced_positive = 0
        for row_id in self._row_stream():
            if produced_positive >= positive_total:
                break
            if not self._available(row_id):
                continue
            norm = self._row_norm(row_id)
            if not norm:
                continue
            key = ("tm_pos", row_id, norm)
            if key in self.seen:
                continue
            self.seen.add(key)
            produced_positive += 1
            draft = {
                "source_row_ids": [row_id],
                "target": norm,
                "target_display": self._row_value(row_id),
                "gold_value": True,
                "flags": {},
            }
            self.reuse.acquire(self.contract.type_id, draft["source_row_ids"])
            drafts.append(draft)
        rows = list(self.index.valid_row_ids[self.field])
        produced = 0
        attempts = 0
        limit = self.settings.max_sampling_attempts * max(1, negative_total)
        while produced < negative_total and attempts < limit:
            attempts += 1
            row_id = rows[self.rng.randrange(len(rows))]
            if not self._available(row_id):
                continue
            row_norm = self._row_norm(row_id)
            if not row_norm:
                continue
            negative = self._negative_target(row_norm)
            if negative is None:
                continue
            norm, fallback = negative
            key = ("tm_neg", row_id, norm)
            if key in self.seen:
                continue
            self.seen.add(key)
            produced += 1
            draft = {
                "source_row_ids": [row_id],
                "target": norm,
                "target_display": self._display(norm),
                "gold_value": False,
                "flags": {"text_negative_length_fallback": fallback},
            }
            self.reuse.acquire(self.contract.type_id, draft["source_row_ids"])
            drafts.append(draft)
        return drafts

    def _selection(self, total: int) -> list[dict[str, Any]]:
        values = self._values()
        if len(values) < 2:
            raise BudgetShortfallError(
                {
                    "code": "BUDGET_SHORTFALL",
                    "type_id": self.contract.type_id,
                    "kind": "PAIRWISE_SELECTION",
                    "requested": total,
                    "feasible": 0,
                    "shortfall": total,
                }
            )
        rows = list(self.index.valid_row_ids[self.field])
        drafts: list[dict[str, Any]] = []
        attempts = 0
        limit = self.settings.max_sampling_attempts * max(1, total)
        while len(drafts) < total and attempts < limit:
            attempts += 1
            if self.settings.value_sampling == "uniform_over_values":
                target_norm = values[self.rng.randrange(len(values))]
            else:
                pre_row = rows[self.rng.randrange(len(rows))]
                if not self._available(pre_row):
                    continue
                target_norm = self._row_norm(pre_row)
            match = self._pick_in_group(target_norm)
            if match is None:
                continue
            nonmatch = self._pick_nonmatch(target_norm, match)
            if nonmatch is None:
                continue
            low, high = sorted((match, nonmatch))
            key = ("ps", target_norm, low, high, match)
            if key in self.seen:
                continue
            self.seen.add(key)
            if self.settings.randomized_position:
                digest = hashlib.sha256(
                    f"{self.seed}|{self.contract.type_id}|{target_norm}|"
                    f"{low}|{high}|{match}".encode()
                ).digest()
                swap = bool(digest[0] & 1)
            else:
                swap = False
            ordered = [nonmatch, match] if swap else [match, nonmatch]
            draft = {
                "source_row_ids": ordered,
                "target": target_norm,
                "target_display": self._display(target_norm),
                "gold_value": "B" if swap else "A",
                "flags": {"match_row_id": match},
            }
            self.reuse.acquire(self.contract.type_id, draft["source_row_ids"])
            drafts.append(draft)
        return drafts

    def _pick_nonmatch(self, target_norm: str, match_row: str) -> str | None:
        others = [value for value in self._values() if value != target_norm]
        if not others:
            return None
        speaker = (
            self._speaker(match_row) if self.settings.prefer_distinct_speaker else None
        )
        last: str | None = None
        for _ in range(min(self.settings.max_sampling_attempts, 16)):
            norm = others[self.rng.randrange(len(others))]
            candidate = self._pick_in_group(
                norm,
                prefer_different=speaker if speaker is not None else None,
            )
            if candidate is None:
                continue
            last = candidate
            if speaker is None or self._speaker(candidate) != speaker:
                return candidate
        return last
