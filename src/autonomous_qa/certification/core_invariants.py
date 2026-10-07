"""CoreInvariantsSuite: Gatekeeper for Autonomous Compiler.

Executes mandatory architectural, mathematical, and schema invariant assertions
BEFORE SemanticCatalog freeze, PromotionManifest freeze, or ProductionPlan resolution.
Raises CompileError on any fatal invariant violation.
"""

from __future__ import annotations

from typing import Any

from src.autonomous_qa.production.production_plan import ProductionPlan
from src.autonomous_qa.compiler.semantic_contract import FORBIDDEN_SAMPLING_FIELDS, SemanticContract


class CompileError(ValueError):
    """Raised when a core compiler invariant is violated."""


class CoreInvariantsSuite:
    """Mandatory gatekeeper suite for the Autonomous Audio-QA Compiler."""

    @staticmethod
    def assert_mutually_exclusive_partition(
        promoted: set[str],
        promoted_with_fallback: set[str],
        rejected: set[str],
        all_candidates: set[str],
    ) -> None:
        """1. Candidate final-state partitions must be mutually exclusive and complete."""
        # Check pairwise disjointness
        if promoted & promoted_with_fallback:
            raise CompileError(
                f"PARTITION_OVERLAP: promoted & fallback = {promoted & promoted_with_fallback}"
            )
        if promoted & rejected:
            raise CompileError(
                f"PARTITION_OVERLAP: promoted & rejected = {promoted & rejected}"
            )
        if promoted_with_fallback & rejected:
            raise CompileError(
                f"PARTITION_OVERLAP: fallback & rejected = {promoted_with_fallback & rejected}"
            )

        partition_union = promoted | promoted_with_fallback | rejected
        if partition_union != all_candidates:
            missing = all_candidates - partition_union
            extra = partition_union - all_candidates
            raise CompileError(
                f"PARTITION_INCOMPLETE: missing={missing}, extra={extra}"
            )

    @staticmethod
    def assert_sanitization_monotonicity(
        raw_capacity: int, sanitized_capacity: int, task_id: str
    ) -> None:
        """2. Task-scoped sanitation must not accidentally increase capacity."""
        if sanitized_capacity > raw_capacity:
            raise CompileError(
                f"SANITIZATION_NON_MONOTONIC:{task_id}: sanitized={sanitized_capacity} > raw={raw_capacity}"
            )

    @staticmethod
    def assert_concept_space_declared(contract: SemanticContract) -> None:
        """3. Every term/concept contract must explicitly declare concept_space."""
        if (
            contract.task_id
            in (
                "vimedcss_cs_term_extraction",
                "vimedcss_cs_term_presence",
                "vimedcss_t4k_medical_knowledge",
            )
            and (
                not contract.concept_space
                or contract.concept_space
                not in ("LEXICAL_SURFACE_STRING", "NORMALIZED_KG_CONCEPT")
            )
        ):
                raise CompileError(
                    f"UNDECLARED_CONCEPT_SPACE:{contract.task_id}: concept_space='{contract.concept_space}'"
                )

    @staticmethod
    def assert_topology_maximum(
        topology: str, n_anchors: int, capacity: int, allow_self: bool = False
    ) -> None:
        """4. Unordered pair capacity must not exceed n*(n-1)/2."""
        if topology == "UNORDERED_PAIR" and not allow_self:
            max_allowed = n_anchors * (n_anchors - 1) // 2 if n_anchors >= 2 else 0
            if capacity > max_allowed:
                raise CompileError(
                    f"TOPOLOGY_CAPACITY_EXCEEDED: reported={capacity} > max_allowed={max_allowed}"
                )

    @staticmethod
    def assert_no_sampling_fields_in_contract(contract_data: dict[str, Any]) -> None:
        """5. SemanticContract must not contain production sampling fields."""
        for key in contract_data:
            if key.lower() in FORBIDDEN_SAMPLING_FIELDS:
                raise CompileError(
                    f"SAMPLING_FIELD_IN_CONTRACT:{contract_data.get('task_id')}:{key}"
                )

    @staticmethod
    def assert_planner_capacity_bounds(
        plan: ProductionPlan, contract: SemanticContract
    ) -> None:
        """6. ProductionPlan cannot select more instances than the universe permits."""
        if plan.pos_neg_ratio > 0:
            target_pos = round(plan.target_qa_count * (plan.pos_neg_ratio / (1.0 + plan.pos_neg_ratio)))
            target_neg = plan.target_qa_count - target_pos
        else:
            target_pos = plan.target_qa_count
            target_neg = 0

        if target_pos > contract.positive_universe_capacity:
            raise CompileError(
                f"PLANNER_BOUNDS_EXCEEDED:{plan.task_id}: target_pos={target_pos} > pos_universe={contract.positive_universe_capacity}"
            )
        if target_neg > contract.negative_universe_capacity:
            raise CompileError(
                f"PLANNER_BOUNDS_EXCEEDED:{plan.task_id}: target_neg={target_neg} > neg_universe={contract.negative_universe_capacity}"
            )

    @staticmethod
    def assert_determinism(manifest_a_json: str, manifest_b_json: str) -> None:
        """7. Same contract + source + plan + seed must produce byte-identical manifest."""
        if manifest_a_json != manifest_b_json:
            raise CompileError("DETERMINISM_VIOLATION: manifests differ for same inputs")

    @staticmethod
    def assert_cartesian_conservation(
        raw_cartesian: int, pos: int, neg: int, collision: int, invalid: int = 0
    ) -> None:
        """8. Cartesian presence buckets must conserve full Cartesian product A x C."""
        if pos < 0 or neg < 0 or collision < 0 or invalid < 0:
            raise CompileError(
                f"NEGATIVE_CAPACITY_BUCKET: pos={pos}, neg={neg}, collision={collision}, invalid={invalid}"
            )
        total = pos + neg + collision + invalid
        if total != raw_cartesian:
            raise CompileError(
                f"CARTESIAN_CONSERVATION_VIOLATION: sum={total} != raw_cartesian={raw_cartesian}"
            )

    @staticmethod
    def assert_formula_enumeration_equivalence(
        formula_res: dict[str, int], enum_res: dict[str, int]
    ) -> None:
        """9. Formula-based capacity must equal exact enumerated capacity."""
        for key in ("pos_capacity", "neg_capacity", "collision_capacity", "total_capacity"):
            f_val = formula_res.get(key)
            e_val = enum_res.get(key)
            if f_val != e_val:
                raise CompileError(
                    f"FORMULA_ENUMERATION_DISCREPANCY:{key}: formula={f_val} != enumerated={e_val}"
                )

    @staticmethod
    def assert_manifest_instance_uniqueness(instance_ids: list[str]) -> None:
        """10. FrozenInstanceManifest instance IDs must be strictly unique."""
        if len(instance_ids) != len(set(instance_ids)):
            duplicates = len(instance_ids) - len(set(instance_ids))
            raise CompileError(
                f"MANIFEST_INSTANCE_ID_DUPLICATION: {duplicates} duplicate instance IDs found in manifest"
            )

    @staticmethod
    def assert_manifest_universe_membership(
        selected_instance_ids: list[str], valid_universe_ids: set[str]
    ) -> None:
        """11. Every selected manifest instance must belong to the valid semantic universe."""
        invalid_instances = [
            iid for iid in selected_instance_ids if iid not in valid_universe_ids
        ]
        if invalid_instances:
            raise CompileError(
                f"MANIFEST_UNIVERSE_MEMBERSHIP_VIOLATION: {len(invalid_instances)} selected instances not in universe (sample: {invalid_instances[:3]})"
            )

    @staticmethod
    def assert_registry_hash_separation(lexical_hash: str, kg_hash: str) -> None:
        """12. Lexical comparator and KG resolver registries must be separately hashed."""
        if not lexical_hash or not kg_hash:
            raise CompileError("EMPTY_REGISTRY_HASH: registry hashes cannot be empty")
        if lexical_hash == kg_hash:
            raise CompileError(
                f"REGISTRY_HASH_COLLISION: lexical_hash equals kg_hash='{lexical_hash}'"
            )

    @staticmethod
    def assert_fallback_decision_provenance(
        fallback_decision_hash: str, effective_contract_hash: str
    ) -> None:
        """13. Fallback decision hash must be distinct from effective execution contract hash."""
        if not fallback_decision_hash or not effective_contract_hash:
            raise CompileError("EMPTY_FALLBACK_HASH: fallback decision hashes cannot be empty")
        if fallback_decision_hash == effective_contract_hash:
            raise CompileError(
                f"FALLBACK_HASH_ALIASING: fallback_decision_hash equals effective_contract_hash='{fallback_decision_hash}'"
            )

    @staticmethod
    def assert_semantic_universe_source_provenance(source_artifact_kind: str) -> None:
        """14. Semantic universe cannot be constructed from historical production artifacts."""
        forbidden = {
            "HISTORICAL_QA_RELEASE",
            "HISTORICAL_GENERATION_PLAN",
            "PRODUCTION_PLAN",
            "FROZEN_INSTANCE_MANIFEST",
        }
        if source_artifact_kind in forbidden:
            raise CompileError(
                f"FORBIDDEN_SOURCE_PROVENANCE_KIND:{source_artifact_kind}: "
                f"Semantic universe must be constructed from SOURCE_DATASET or SOURCE_PROFILE only."
            )

    @staticmethod
    def assert_topology_role_compatibility(contract: SemanticContract) -> None:
        """15. SemanticContract role mapping must be structurally compatible with topology."""
        if contract.topology == "UNORDERED_PAIR":
            if contract.anchor_type != "audio_segment":
                raise CompileError(
                    f"TOPOLOGY_ROLE_MISMATCH:{contract.task_id}: UNORDERED_PAIR requires peer audio anchor domain"
                )
        elif contract.topology in ("CARTESIAN_PRESENCE", "CARTESIAN_RELATION"):
            if not contract.candidate_type and not any(
                r in contract.visible_roles for r in ("candidate_term", "target_text", "candidate_label", "target_value")
            ):
                raise CompileError(
                    f"TOPOLOGY_ROLE_MISMATCH:{contract.task_id}: {contract.topology} requires anchor + candidate domain"
                )
        elif contract.topology == "TARGET_CONDITIONED_UNORDERED_PAIR":
            if not any(
                r in contract.visible_roles for r in ("candidate_term", "target_text", "candidate_label", "target_value")
            ):
                raise CompileError(
                    f"TOPOLOGY_ROLE_MISMATCH:{contract.task_id}: TARGET_CONDITIONED_UNORDERED_PAIR requires visible target role"
                )

    @staticmethod
    def assert_topology_usage_aggregation(
        topology_counts: dict[str, int], total_executable_contracts: int
    ) -> None:
        """16. Sum of topology usage counts across datasets must equal total executable contracts."""
        sum_counts = sum(topology_counts.values())
        if sum_counts != total_executable_contracts:
            raise CompileError(
                f"TOPOLOGY_USAGE_AGGREGATION_DISCREPANCY: sum={sum_counts} != total={total_executable_contracts}"
            )

    @staticmethod
    def assert_task_binding_referential_integrity(
        task_id: str,
        contract_task_id: str,
        universe_task_id: str,
        capacity_task_id: str,
        anchor_reg_hash: str | None = None,
        universe_anchor_hash: str | None = None,
        candidate_reg_hash: str | None = None,
        universe_candidate_hash: str | None = None,
    ) -> None:
        """17. Task binding referential integrity invariant across contract, universe, and capacity result."""
        if not (task_id == contract_task_id == universe_task_id == capacity_task_id):
            raise CompileError(
                f"TASK_BINDING_REFERENTIAL_INTEGRITY_FAIL:{task_id}: IDs mismatch "
                f"contract={contract_task_id}, universe={universe_task_id}, capacity={capacity_task_id}"
            )
        if anchor_reg_hash and universe_anchor_hash and anchor_reg_hash != universe_anchor_hash:
            raise CompileError(
                f"TASK_BINDING_REFERENTIAL_INTEGRITY_FAIL:{task_id}: Anchor hash mismatch "
                f"binding={anchor_reg_hash} != universe={universe_anchor_hash}"
            )
        if candidate_reg_hash and universe_candidate_hash and candidate_reg_hash != universe_candidate_hash:
            raise CompileError(
                f"TASK_BINDING_REFERENTIAL_INTEGRITY_FAIL:{task_id}: Candidate hash mismatch "
                f"binding={candidate_reg_hash} != universe={universe_candidate_hash}"
            )

    @staticmethod
    def assert_historical_replay_count_closure(
        dataset_id: str, per_task_counts: dict[str, int], expected_total: int
    ) -> None:
        """18. Sum of per-task historical selected counts must strictly equal canonical release count."""
        actual_total = sum(per_task_counts.values())
        if actual_total != expected_total:
            raise CompileError(
                f"HISTORICAL_REPLAY_COUNT_CLOSURE_FAIL:{dataset_id}: "
                f"actual sum={actual_total} != expected canonical release total={expected_total}"
            )

    @staticmethod
    def assert_target_conditioned_bucket_exactness(
        candidate_buckets: list[Any],
        capacity_result: Any,
    ) -> None:
        """19. CandidatePairBucket exactness & conservation invariant for TARGET_CONDITIONED_UNORDERED_PAIR."""
        if capacity_result.topology != "TARGET_CONDITIONED_UNORDERED_PAIR":
            return  # NOT_APPLICABLE for other topologies

        if not candidate_buckets:
            return  # Omitted for fallback uniform approximations

        # Verify per-bucket conservation and exact sum
        sum_raw = sum(b.raw_pair_capacity for b in candidate_buckets)
        sum_valid = sum(b.valid_pair_capacity for b in candidate_buckets)
        sum_both = sum(b.both_match_excluded_capacity for b in candidate_buckets)
        sum_neither = sum(b.neither_match_excluded_capacity for b in candidate_buckets)

        for b in candidate_buckets:
            if b.raw_pair_capacity != b.valid_pair_capacity + b.both_match_excluded_capacity + b.neither_match_excluded_capacity:
                raise CompileError(
                    f"TARGET_CONDITIONED_PER_CANDIDATE_CONSERVATION_FAIL:{b.candidate_id}"
                )

        if sum_raw != capacity_result.raw_combinatorial_space_capacity:
            raise CompileError(
                f"TARGET_CONDITIONED_BUCKET_EXACTNESS_FAIL: raw sum={sum_raw} != result={capacity_result.raw_combinatorial_space_capacity}"
            )

        if sum_valid != capacity_result.valid_semantic_universe_capacity:
            raise CompileError(
                f"TARGET_CONDITIONED_BUCKET_EXACTNESS_FAIL: valid sum={sum_valid} != result={capacity_result.valid_semantic_universe_capacity}"
            )

        if sum_both != capacity_result.collision_capacity:
            raise CompileError(
                f"TARGET_CONDITIONED_BUCKET_EXACTNESS_FAIL: both_match sum={sum_both} != collision={capacity_result.collision_capacity}"
            )

        if sum_neither != capacity_result.invalid_capacity:
            raise CompileError(
                f"TARGET_CONDITIONED_BUCKET_EXACTNESS_FAIL: neither_match sum={sum_neither} != invalid={capacity_result.invalid_capacity}"
            )

    @staticmethod
    def run_all_gatekeeper_checks(
        contracts: list[SemanticContract],
        promoted: set[str],
        promoted_with_fallback: set[str],
        rejected: set[str],
        all_candidates: set[str],
    ) -> dict[str, str]:
        """Execute all gatekeeper assertions across all candidate contracts."""
        CoreInvariantsSuite.assert_mutually_exclusive_partition(
            promoted, promoted_with_fallback, rejected, all_candidates
        )

        for c in contracts:
            CoreInvariantsSuite.assert_concept_space_declared(c)
            CoreInvariantsSuite.assert_no_sampling_fields_in_contract(c.model_dump())
            CoreInvariantsSuite.assert_topology_role_compatibility(c)
            if c.topology == "UNORDERED_PAIR":
                n_anchors = c.split_policy.get("n_anchors", 11832)
                CoreInvariantsSuite.assert_topology_maximum(
                    c.topology, n_anchors, c.total_universe_capacity
                )

        return {"status": "PASS", "gatekeeper_checks": "ALL_CORE_INVARIANTS_PASSED"}

