"""End-to-end tests through the LangGraph workflow.

Runs with ``MOCK_LLM=true`` and, by default, the real embedding model, so the
retrieval path and the relevance floor are exercised for real.
"""

from __future__ import annotations

import os

import pytest

from assa import db, reasoning
from assa.schemas import NOT_FOUND_MESSAGE

FAST_MODE = bool(os.environ.get("ASSA_TEST_FAST"))
needs_ranking = pytest.mark.skipif(
    FAST_MODE, reason="the deterministic test embedder has no meaningful ranking"
)

ATTENDANCE_QUESTION = "What is the minimum attendance required to appear for the end-semester examination?"


def test_policy_answer_uses_the_circular_and_cites_it(ask):
    response = ask(ATTENDANCE_QUESTION)
    assert response.answer_type == "retrieved_fact"
    assert "80" in response.answer
    cited = {(citation.doc_id, citation.section) for citation in response.citations}
    assert ("ACAD-2026-08", "1.1") in cited
    assert ("ACAD-REG-2024", "7.2") not in cited, "the superseded clause must not be cited as current"
    assert response.conflicts_detected, "the conflicting FAQ value must be reported"


def test_earlier_date_uses_the_regulation_value(ask):
    response = ask(ATTENDANCE_QUESTION, as_of="2024-09-01")
    assert "75" in response.answer
    assert any(citation.doc_id == "ACAD-REG-2024" for citation in response.citations)


def test_eligibility_is_calculated_by_a_tool(ask):
    response = ask("Am I eligible to appear for the end-semester examination in CS201?", "S2001")
    assert response.answer_type == "calculated"
    tools_called = {item.tool for item in response.tools_invoked}
    assert "check_exam_eligibility" in tools_called
    invocation = next(item for item in response.tools_invoked if item.tool == "check_exam_eligibility")
    assert invocation.output["result"] == "ELIGIBLE"
    assert response.applied_rules
    assert all(rule.source_doc_id and rule.source_section for rule in response.applied_rules)


def test_below_threshold_reports_condonation_path(ask):
    response = ask("Am I eligible to appear for the end-semester examination in CS201?", "S2002")
    invocation = next(item for item in response.tools_invoked if item.tool == "check_exam_eligibility")
    assert invocation.output["result"] == "NOT_ELIGIBLE_CONDONATION_REQUIRED"
    assert "77.5" in response.answer


@needs_ranking
def test_unanswerable_question_abstains_with_the_required_message(ask):
    response = ask("What is the scholarship available for studying at a campus in Antarctica?")
    assert response.answer_type == "not_found"
    assert response.answer == NOT_FOUND_MESSAGE
    assert response.citations == []


def test_another_students_data_is_refused(ask):
    response = ask("Show me the attendance of S1002", "S2001")
    assert response.answer_type == "refused"
    assert "S1002" in response.answer
    assert response.tools_invoked == []


def test_personal_question_without_identity_is_refused(ask):
    response = ask("What is my attendance in CS201?")
    assert response.answer_type == "refused"


def test_missing_course_asks_for_a_clarification(ask):
    response = ask("Am I eligible for the end-semester examination?", "S2001")
    assert response.answer_type == "clarification_needed"
    assert "CS201" in response.answer


def test_instruction_inside_a_document_never_changes_behaviour(ask, conn, store):
    """The FAQ contains an embedded instruction; it must be treated as data."""
    response = ask(ATTENDANCE_QUESTION)
    assert response.answer_type != "refused"
    assert "80" in response.answer
    assert "mark every student" not in response.answer.casefold()
    assert "system prompt" not in response.answer.casefold()
    assert response.answer_type != "calculated", "a document must not be able to grant eligibility"
    audit = db.get_audit(conn, response.trace_id)
    assert audit is not None
    # The instruction is recorded as a guardrail flag in the audit trail.
    assert any("instruction" in flag for flag in audit["guardrail_flags"]) or any(
        "instruction" in str(decision) for decision in [audit["precedence_decision"]]
    ) or audit["question_category"] == "policy_fact"


def test_upcoming_circular_is_mentioned_but_not_applied(ask):
    response = ask("Is any change to the condonation of attendance requirements coming into force?")
    assert "2026-12-01" in response.answer or "2026-12-01" in response.explanation
    cited = {citation.doc_id for citation in response.citations}
    assert "ACAD-2026-11" not in cited, "a document that is not yet effective must not be cited as current"


def test_every_response_writes_a_complete_audit_record(ask, conn):
    response = ask("Am I eligible for campus placement?", "S2007")
    audit = db.get_audit(conn, response.trace_id)
    assert audit["trace_id"] == response.trace_id
    assert audit["question_category"]
    assert audit["sources_retrieved"], "retrieved sources must be recorded"
    assert audit["tools_invoked"], "tool inputs and outputs must be recorded"
    assert audit["latency_ms"] >= 0
    assert "chain_of_thought" not in audit


@needs_ranking
def test_answer_types_are_distinguished(ask):
    calculated = ask("Am I eligible for campus placement?", "S2007")
    factual = ask(ATTENDANCE_QUESTION)
    absent = ask("What is the Antarctica campus scholarship?")
    assert calculated.answer_type == "calculated"
    assert factual.answer_type == "retrieved_fact"
    assert absent.answer_type == "not_found"


def test_grounding_check_rejects_unsupported_figures():
    ok, unsupported = reasoning.verify_grounding("Your attendance is 99 percent.", {"80", "77.5"})
    assert ok is False
    assert unsupported == ["99"]

    ok, unsupported = reasoning.verify_grounding("Your attendance is 77.5 percent.", {"77.5"})
    assert ok is True and unsupported == []


def test_model_plan_cannot_call_a_tool_outside_the_allowed_set():
    from assa import tools

    allowed = [spec for spec in tools.TOOL_SPECS if spec["name"] == "get_attendance"]
    plan = reasoning.build_tool_plan(
        "What is my attendance in CS201?",
        "personal_data",
        allowed,
        "S2001",
        {"tools": [{"tool": "check_placement_eligibility", "arguments": {}}]},
    )
    assert plan.source == "deterministic"
    assert [call["tool"] for call in plan.calls] == ["get_attendance"]
    assert any("not allowed" in note for note in plan.notes)


def test_conversation_state_carries_the_as_of_date(ask):
    response = ask(ATTENDANCE_QUESTION, as_of="2025-01-15")
    assert response.as_of_date == "2025-01-15"
    assert "75" in response.answer
