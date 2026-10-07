"""Pydantic schemas for the v3 discovery loop (structured output).

Each round runs Style -> Template -> Paraphrase -> Quality. Only the
Quality verdict is ids + reasons; Python holds every text.

Note on `answer` (structured) vs `answer_rule` (human-readable): the
renderer needs a machine-readable gold derivation to instantiate QA pairs
with zero LLM calls, so every proposed type carries BOTH. `answer_rule`
is what humans audit; `answer` is what Python executes. They must agree —
the validity gate checks the structured part, Quality judges the whole.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerSpec(BaseModel):
    kind: str = Field(description="field_value | equality | derived_field")
    key: str | None = None
    keys: list[str] | None = None
    source_key: str | None = None
    target_key: str | None = None


class RoundQuestionType(BaseModel):
    id: str = Field(description="Round-scoped id, e.g. QS_R2_03")
    name: str
    goal: str
    uses: list[str]
    input_count: int = 1
    answer_rule: str = Field(
        description="Human-readable deterministic gold derivation, "
        "e.g. 'same speakerID => yes, otherwise no'"
    )
    answer: AnswerSpec = Field(
        description="Machine-readable gold derivation Python executes"
    )
    input_context_fields: list[str] = Field(
        default_factory=list,
        description="Optional visible context fields required by this question type (e.g. target_sentence)",
    )


class StyleRoundOutput(BaseModel):
    """Raw structured response of one Style round call."""

    round: int
    question_types: list[RoundQuestionType]








class RoundTemplate(BaseModel):
    template_id: str = Field(description="E.g. T_R2_03_01 or BF_EQSEM_001")
    question_type_id: str = Field(description="Owning candidate type or family id")
    text: str = Field(
        description="Generic template. Only [KEY]/[VALUE] placeholders "
        "or no placeholder at all."
    )


class TemplateRoundOutput(BaseModel):
    """Raw structured response of one Template round call.

    key_realizations maps QUESTION TYPE IDs (copied exactly from the
    input round types) to natural Vietnamese [KEY] phrases. Keys are
    NEVER raw dataset field names. Python derives the raw field via
    get_answer_concept_field().
    """

    round: int
    key_realizations: dict[str, str] = Field(
        default_factory=dict,
        description="QUESTION TYPE ID -> natural Vietnamese [KEY] phrase",
    )
    templates: list[RoundTemplate] = Field(default_factory=list)
    family_templates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional shared family base templates (base_template_id, text)",
    )


class ParaphraseVariant(BaseModel):
    template_id: str = Field(description="E.g. T_R2_03_01_P01")
    source_template_id: str
    text: str


class ParaphraseRoundOutput(BaseModel):
    """Raw structured response of one Paraphrase round call."""

    round: int
    paraphrases: list[ParaphraseVariant]


class AcceptedNewType(BaseModel):
    type_id: str
    keep_template_ids: list[str] = Field(
        description="Surviving template/paraphrase ids for this type. "
        "Bad wording is dropped without rejecting the type."
    )


class TypeDuplicate(BaseModel):
    type_id: str
    duplicate_of: str = Field(description="Approved type id from any round")


class TypeRejection(BaseModel):
    type_id: str
    reason: str = ""


class ProfileField(BaseModel):
    """One raw source field as described by the dataset card.

    `type` is the card-declared dtype, verbatim (string / int64 / audio).
    `entity_scope` is null when the card supports no canonical scope.
    """

    type: str = Field(description="Source dtype exactly as declared in the card")
    role: str = Field(
        description="audio | semantic | hidden_identifier | provenance | context_only"
    )
    entity_scope: str | None = Field(
        default=None,
        description="speaker | utterance | recording | conversation | null",
    )
    description: str
    is_categorical: bool = False
    evaluation: str = Field(description="eligible | discovery_only")
    context_visibility: str = Field(description="visible | hidden | forbidden")
    # Optional vNext rendering hints. Historical profiles remain valid because
    # every new field defaults to None. These are semantic lexical labels, not
    # question templates, and are emitted by the existing single profiler call.
    semantic_class: (
        Literal[
            "categorical_attribute",
            "numeric_attribute",
            "text_content",
            "ordinal_attribute",
            "boolean_attribute",
            "multi_label_attribute",
            "identifier_relation",
        ]
        | None
    ) = None
    verbalization: ProfileFieldVerbalization | None = None
    value_policy: ProfileFieldValuePolicy | None = None


class ProfileFieldVerbalization(BaseModel):
    entity_phrase: str | None = None
    attribute_phrase: str | None = None
    content_phrase: str | None = None
    value_phrase: str | None = None
    unit: str | None = None


class ProfileFieldValuePolicy(BaseModel):
    normalization: str = "identity"
    match_policy: str = "exact"
    comparison_policy: str | None = None
    bins: list[float] | None = None
    tolerance: float | None = None


class FieldRelation(BaseModel):
    source_field: str
    target_field: str
    description: str


class DatasetProfile(BaseModel):
    """Canonical, single-source dataset profile (one LLM call).

    This is the ONLY artifact the LLM produces. schema.json and
    metadata_context.json are exported deterministically from it by
    Python, so the two compatibility files can never drift apart.
    """

    dataset: str
    pretty_name: str
    description: str
    locale: str
    task_categories: list[str] = Field(default_factory=list)

    fields: dict[str, ProfileField]

    field_roles: dict[str, str]
    entity_scopes: dict[str, str | None]
    field_evaluations: dict[str, str]
    context_visibilities: dict[str, str]

    hidden_fields: list[str] = Field(default_factory=list)
    semantic_fields: list[str] = Field(default_factory=list)

    relations: list[FieldRelation] = Field(default_factory=list)
    qa_constraints: list[str] = Field(default_factory=list)


class QualityRoundOutput(BaseModel):
    """Raw structured response of one Quality round call.

    Judgement operates at QUESTION-TYPE capability level, never at
    paraphrase-wording level: two phrasings with the same semantics are
    the goal of paraphrasing, not a duplicate type.
    """

    round: int
    accepted_new: list[AcceptedNewType] = []
    duplicates: list[TypeDuplicate] = []
    rejected: list[TypeRejection] = []
