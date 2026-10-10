"""Annex A precedence policy tests, including the worked example from the guide."""

from __future__ import annotations

from datetime import date

from assa.precedence import (
    Candidate,
    document_covers,
    document_is_effective,
    parse_supersedes,
    resolve_candidates,
    resolve_parameter,
    scope_matches,
    topic_of,
)


def candidate(doc_id, section, text, authority, effective_from, supersedes="", **kwargs):
    return Candidate(
        doc_id=doc_id,
        section=section,
        text=text,
        authority_level=authority,
        effective_from=effective_from,
        supersedes=supersedes,
        **kwargs,
    )


ANNEX_A_WORKED_EXAMPLE = [
    candidate(
        "ACAD-REG-2024",
        "7.2",
        "A student shall have a minimum of 75 percent of the classes held in each registered course.",
        1,
        "2024-07-01",
        claim_key="param:min_attendance_pct",
    ),
    candidate(
        "ACAD-2026-08",
        "1.1",
        "The minimum attendance required is 80 percent of the classes held in that course.",
        2,
        "2026-08-01",
        supersedes="ACAD-REG-2024#7.2",
        claim_key="param:min_attendance_pct",
    ),
    candidate(
        "DEPT-FAQ-2026",
        "Q1",
        "As per the help desk's understanding, 65 percent attendance is enough.",
        4,
        "2026-09-15",
        claim_key="param:min_attendance_pct",
    ),
]


def test_worked_example_resolves_to_the_circular():
    outcome = resolve_candidates(
        ANNEX_A_WORKED_EXAMPLE,
        as_of_date=date(2026, 10, 6),
        resolved_parameter_values={"min_attendance_pct": 80.0},
    )
    assert [item.doc_id for item in outcome.kept] == ["ACAD-2026-08"]
    assert any("supersedes ACAD-REG-2024#7.2 (step 2)" in text for text in outcome.decisions)
    assert outcome.conflicts and outcome.conflicts[0]["resolved_by"] == "step-3-authority"


def test_before_the_circular_only_the_regulation_applies():
    outcome = resolve_candidates(
        ANNEX_A_WORKED_EXAMPLE,
        as_of_date=date(2024, 9, 1),
        resolved_parameter_values={"min_attendance_pct": 75.0},
    )
    assert [item.doc_id for item in outcome.kept] == ["ACAD-REG-2024"]
    assert outcome.upcoming and outcome.upcoming[0].doc_id == "ACAD-2026-08"


def test_level_five_cannot_override_a_regulation():
    candidates = [
        candidate("ACAD-2026-08", "1.1", "The minimum attendance required is 80 percent.", 2, "2026-08-01",
                  claim_key="param:min_attendance_pct"),
        candidate("STUCOUNCIL-NOTICE-2026-09", "1.1", "The requirement shall be 60 percent.", 5, "2026-09-20",
                  claim_key="param:min_attendance_pct"),
    ]
    outcome = resolve_candidates(
        candidates, as_of_date=date(2026, 10, 6), resolved_parameter_values={"min_attendance_pct": 80.0}
    )
    assert [item.doc_id for item in outcome.kept] == ["ACAD-2026-08"]
    assert "STUCOUNCIL-NOTICE-2026-09" in outcome.decisions[-1]


def test_unresolvable_conflict_is_reported_not_guessed():
    candidates = [
        candidate("DEPT-A", "1", "A maximum of 10 percent is permitted.", 3, "2026-01-01",
                  claim_key="param:max_condonation_pct"),
        candidate("DEPT-B", "1", "A maximum of 15 percent is permitted.", 3, "2026-01-01",
                  claim_key="param:max_condonation_pct"),
    ]
    outcome = resolve_candidates(candidates, as_of_date=date(2026, 10, 6))
    assert outcome.unresolved is True
    assert {item.doc_id for item in outcome.kept} == {"DEPT-A", "DEPT-B"}
    assert outcome.conflicts[0]["resolved_by"] == "unresolved"


def test_scope_filtering_uses_programme_and_batch():
    scoped = {
        "effective_from": "2026-09-01",
        "effective_to": "",
        "scope_programmes": "B.Tech CSE",
        "scope_batches": "2023+",
    }
    assert document_covers(scoped, "B.Tech CSE", 2024) is True
    assert document_covers(scoped, "B.Tech ECE", 2024) is False
    assert document_covers(scoped, "B.Tech CSE", 2022) is False
    assert scope_matches("2023-2025", "2024") is True
    assert scope_matches("2023-2025", "2026") is False


def test_effective_window_and_supersedes_parsing():
    assert document_is_effective({"effective_from": "2026-08-01", "effective_to": ""}, date(2026, 10, 6))
    assert not document_is_effective({"effective_from": "2026-12-01", "effective_to": ""}, date(2026, 10, 6))
    assert not document_is_effective(
        {"effective_from": "2020-01-01", "effective_to": "2026-01-01"}, date(2026, 10, 6)
    )
    assert parse_supersedes("ACAD-REG-2024#7.2; OLD-DOC") == [("ACAD-REG-2024", "7.2"), ("OLD-DOC", None)]


def test_rule_registry_supersession_prefers_the_newer_rule():
    rules = [
        {
            "rule_id": "ATT-MIN-01",
            "parameter": "min_attendance_pct",
            "operator": ">=",
            "value": "75",
            "effective_from": "2024-07-01",
            "effective_to": "",
            "scope_programmes": "ALL",
            "scope_batches": "2023+",
            "authority_level": 1,
            "source_doc_id": "ACAD-REG-2024",
            "source_section": "7.2",
            "supersedes_rule_id": "",
        },
        {
            "rule_id": "ATT-MIN-02",
            "parameter": "min_attendance_pct",
            "operator": ">=",
            "value": "80",
            "effective_from": "2026-08-01",
            "effective_to": "",
            "scope_programmes": "ALL",
            "scope_batches": "2023+",
            "authority_level": 2,
            "source_doc_id": "ACAD-2026-08",
            "source_section": "1.1",
            "supersedes_rule_id": "ATT-MIN-01",
        },
    ]
    resolution = resolve_parameter(rules, "min_attendance_pct", date(2026, 10, 6), "B.Tech CSE", 2023)
    assert resolution.status == "resolved"
    assert resolution.value == 80.0
    assert resolution.winner["rule_id"] == "ATT-MIN-02"
    assert "supersedes ATT-MIN-01" in resolution.decision

    earlier = resolve_parameter(rules, "min_attendance_pct", date(2025, 1, 1), "B.Tech CSE", 2023)
    assert earlier.value == 75.0


def test_not_yet_effective_rule_is_reported_as_upcoming():
    rules = [
        {
            "rule_id": "FUTURE-01",
            "parameter": "min_attendance_pct",
            "operator": ">=",
            "value": "85",
            "effective_from": "2026-12-01",
            "effective_to": "",
            "scope_programmes": "ALL",
            "scope_batches": "ALL",
            "authority_level": 2,
            "source_doc_id": "ACAD-2026-11",
            "source_section": "1.1",
            "supersedes_rule_id": "",
        }
    ]
    resolution = resolve_parameter(rules, "min_attendance_pct", date(2026, 10, 6))
    assert resolution.status == "not_found"
    assert [rule["rule_id"] for rule in resolution.upcoming] == ["FUTURE-01"]


def test_topic_taxonomy():
    assert topic_of("What is the minimum attendance required?") == "attendance"
    assert topic_of("How do I apply for the supplementary examination?") == "supplementary_exam"
    assert topic_of("Am I eligible for campus placement?") == "placement"
