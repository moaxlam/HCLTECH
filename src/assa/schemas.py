"""Pydantic v2 models for the fixed API contract (Section 6) and the Source
Register metadata (Annex B).

The field names of ``AskResponse`` are exactly the names in the participant
guide, in the same order: trace_id, answer, answer_type, citations,
tools_invoked, applied_rules, conflicts_detected, explanation, as_of_date.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AnswerType(str, Enum):
    retrieved_fact = "retrieved_fact"
    calculated = "calculated"
    not_found = "not_found"
    clarification_needed = "clarification_needed"
    refused = "refused"
    conflict_flagged = "conflict_flagged"


NOT_FOUND_MESSAGE = "I could not find this information in the authorised university sources."


class AskRequest(BaseModel):
    """Body of POST /ask. Identity never comes from here (R7)."""

    question: str = Field(min_length=3, max_length=2000)
    as_of_date: date | None = Field(
        default=None,
        description="Date used to decide which document version is in force. Defaults to today.",
    )
    student_id: str | None = Field(
        default=None,
        description="Optional body hint. The X-Student-Id header always takes precedence.",
    )

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class Citation(BaseModel):
    doc_id: str
    title: str
    section: str
    page: int | None = None
    version: str
    effective_from: str


class ToolInvocation(BaseModel):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "skipped"] = "ok"
    ms: int = 0
    error: str | None = None


class AppliedRule(BaseModel):
    rule_id: str
    parameter: str | None = None
    value: str
    source_doc_id: str
    source_section: str | None = None
    effective_from: str | None = None


class Conflict(BaseModel):
    reason: str
    doc_ids: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    resolution: str | None = None
    resolved_by: str | None = Field(
        default=None, description="step-2 supersession, step-3 authority, step-4 recency or unresolved"
    )


class AskResponse(BaseModel):
    trace_id: str
    answer: str
    answer_type: AnswerType
    citations: list[Citation] = Field(default_factory=list)
    tools_invoked: list[ToolInvocation] = Field(default_factory=list)
    applied_rules: list[AppliedRule] = Field(default_factory=list)
    conflicts_detected: list[Conflict] = Field(default_factory=list)
    explanation: str = ""
    as_of_date: str


class IngestMetadata(BaseModel):
    """Source Register fields (Annex B) sent with POST /ingest."""

    doc_id: str = Field(min_length=2, max_length=64)
    title: str = Field(min_length=2)
    issuer: str = Field(min_length=2)
    authority_level: int = Field(ge=1, le=5)
    doc_type: Literal["regulation", "circular", "notice", "faq", "handbook", "unofficial"]
    version: str = "1.0"
    effective_from: str
    effective_to: str = ""
    supersedes: str = ""
    scope_programmes: str = "ALL"
    scope_batches: str = "ALL"
    provenance: str = "uploaded via POST /ingest"
    retrieved_on: str = ""
    synthetic: Literal["Y", "N"] = "N"

    @field_validator("effective_from")
    @classmethod
    def _check_date(cls, value: str) -> str:
        _parse_iso(value)
        return value

    @field_validator("effective_to")
    @classmethod
    def _check_optional_date(cls, value: str) -> str:
        if value:
            _parse_iso(value)
        return value


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    status: Literal["indexed", "replaced", "needs_ocr", "failed"]
    authority_level: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ComponentHealth(BaseModel):
    status: Literal["ok", "degraded", "down"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    api: ComponentHealth
    vector_store: ComponentHealth
    sqlite: ComponentHealth
    llm: ComponentHealth
    documents_indexed: int = 0
    mock_llm: bool = False


class SourceRegisterRow(BaseModel):
    doc_id: str
    title: str
    issuer: str
    authority_level: int
    doc_type: str
    version: str
    effective_from: str
    effective_to: str = ""
    supersedes: str = ""
    scope_programmes: str = "ALL"
    scope_batches: str = "ALL"
    provenance: str = ""
    retrieved_on: str = ""
    synthetic: str = "N"
    chunks_indexed: int = 0
    ingested_at: str = ""


class AuditRecord(BaseModel):
    trace_id: str
    timestamp: str
    student_id: str | None = None
    question: str = ""
    question_category: str = ""
    sources_retrieved: list[dict[str, Any]] = Field(default_factory=list)
    precedence_decision: str = ""
    tools_invoked: list[dict[str, Any]] = Field(default_factory=list)
    applied_rules: list[dict[str, Any]] = Field(default_factory=list)
    conflicts_detected: list[dict[str, Any]] = Field(default_factory=list)
    answer_type: str = ""
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    llm_calls: int = 0
    tokens: int = 0
    latency_ms: int = 0
    guardrail_flags: list[str] = Field(default_factory=list)
    notes: str = ""


def _parse_iso(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - message is what matters
        raise ValueError(f"expected a YYYY-MM-DD date, got {value!r}") from exc
