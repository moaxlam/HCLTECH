"""Deterministic tool tests.

The expectations use the deliberate edge-case students (S2001-S2008), which the
generator always produces with fixed values.
"""

from __future__ import annotations

from datetime import date

import pytest

from assa import rules, tools
from assa.tools import ToolContext, ToolError, attendance_pct, run_tool

AS_OF = date(2026, 10, 6)


def context(conn, student_id="S2001", as_of=AS_OF):
    return ToolContext(conn=conn, as_of_date=as_of, student_id=student_id)


def output(conn, tool, student_id="S2001", as_of=AS_OF, **arguments):
    ctx = context(conn, student_id, as_of)
    record = run_tool(ctx, tool, arguments)
    assert record["status"] == "ok", record.get("error")
    return record["output"], record.get("applied_rules", [])


def test_attendance_is_computed_and_never_stored():
    assert attendance_pct(40, 32) == 80.0
    assert attendance_pct(40, 31) == 77.5
    assert attendance_pct(0, 0) is None


def test_attendance_tool_reads_the_record(conn):
    result, _ = output(conn, "get_attendance", "S2001", course_code="CS201")
    assert result == {
        "course_code": "CS201",
        "classes_held": 40,
        "classes_attended": 32,
        "attendance_pct": 80.0,
    }


def test_eligibility_at_the_threshold_is_eligible(conn):
    result, applied = output(conn, "check_exam_eligibility", "S2001", course_code="CS201")
    assert result["result"] == "ELIGIBLE"
    assert result["required_pct"] == 80.0
    assert any(rule["rule_id"] == "ATT-MIN-02" for rule in applied)
    assert all(rule["source_doc_id"] for rule in applied)


def test_one_class_below_the_threshold_needs_condonation(conn):
    result, _ = output(conn, "check_exam_eligibility", "S2002", course_code="CS201")
    assert result["result"] == "NOT_ELIGIBLE_CONDONATION_REQUIRED"
    assert result["attendance_pct"] == 77.5
    assert result["shortage_pct"] == 2.5
    assert result["max_condonation_pct"] == 10.0


def test_detained_student_is_not_eligible(conn):
    result, _ = output(conn, "check_exam_eligibility", "S2005", course_code="CS201")
    assert result["result"] == "NOT_ELIGIBLE"
    assert result["reason_code"] == "DETAINED"


def test_supplementary_eligibility_failed_course_with_attendance(conn):
    result, _ = output(conn, "check_supplementary_eligibility", "S2003", course_code="EC201")
    assert result["result"] == "ELIGIBLE"
    assert result["latest_result"] == "FAIL"
    assert result["supplementary_attempts_used"] == 0


def test_supplementary_eligibility_blocked_by_attendance(conn):
    result, _ = output(conn, "check_supplementary_eligibility", "S2002", course_code="CS201")
    assert result["result"] == "NOT_ELIGIBLE_ATTENDANCE"
    assert result["attendance_rule_satisfied"] is False


def test_supplementary_eligibility_absent_course(conn):
    result, _ = output(conn, "check_supplementary_eligibility", "S2004", course_code="EC201")
    assert result["result"] == "ELIGIBLE"
    assert result["latest_result"] == "ABSENT"


def test_backlog_counting_uses_the_latest_attempt(conn):
    many, _ = output(conn, "get_active_backlogs", "S2006")
    assert many["active_backlogs"] == 3

    cleared, _ = output(conn, "get_active_backlogs", "S2008")
    assert cleared["active_backlogs"] == 0, "a cleared supplementary attempt must remove the backlog"


def test_placement_boundary_case(conn):
    exact, _ = output(conn, "check_placement_eligibility", "S2007")
    assert exact["result"] == "ELIGIBLE"
    assert exact["cgpa"] == 6.0
    assert exact["min_cgpa"] == 6.0

    failing, _ = output(conn, "check_placement_eligibility", "S2006")
    assert failing["result"] == "NOT_ELIGIBLE"
    assert failing["active_backlogs"] == 3


def test_scholarship_eligibility(conn):
    result, applied = output(conn, "check_scholarship_eligibility", "S2003")
    assert result["result"] == "NOT_ELIGIBLE"
    assert result["min_cgpa"] == 7.5
    assert any(rule["rule_id"] == "SCHOLARSHIP-CGPA-01" for rule in applied)


def test_departmental_lab_rule_only_covers_cse(conn):
    applies, _ = output(conn, "check_lab_attendance_requirement", "S2001")
    assert applies["applies"] is True
    assert applies["value"] == 90.0
    assert applies["source_doc_id"] == "DEPT-CSE-NOTICE-2026-09"

    does_not_apply, _ = output(conn, "check_lab_attendance_requirement", "S2007")
    assert does_not_apply["applies"] is False


def test_what_if_projection_states_its_assumption(conn):
    result, _ = output(
        conn, "what_if_attendance", "S2002", course_code="CS201", additional_held=10, additional_attended=10
    )
    assert result["current_pct"] == 77.5
    assert result["projected_pct"] == 82.0
    assert result["result"] == "ELIGIBLE"
    assert "Assumes" in result["assumptions"]


def test_unknown_course_is_rejected(conn):
    ctx = context(conn, "S2001")
    record = run_tool(ctx, "get_attendance", {"course_code": "CS999"})
    assert record["status"] == "error"
    assert "unknown course" in record["error"]


def test_student_tool_without_identity_is_refused(conn):
    ctx = ToolContext(conn=conn, as_of_date=AS_OF, student_id=None)
    record = run_tool(ctx, "get_attendance", {"course_code": "CS201"})
    assert record["status"] == "error"
    assert "X-Student-Id" in record["error"]


def test_tool_failure_is_reported_not_raised(conn):
    ctx = context(conn, "S2001")
    record = run_tool(ctx, "check_exam_eligibility", {})
    assert record["status"] == "error"


def test_thresholds_come_from_the_registry_not_from_code(conn, settings):
    """A new circular's threshold changes behaviour without a code change."""
    ctx = context(conn, "S2001")
    before = run_tool(ctx, "check_exam_eligibility", {"course_code": "CS201"})
    assert before["output"]["result"] == "ELIGIBLE"

    conn.execute(
        """
        INSERT INTO rule_registry (
            rule_id, description, parameter, operator, value, scope_programmes, scope_batches,
            effective_from, effective_to, source_doc_id, source_section, supersedes_rule_id, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            "ATT-MIN-TEST",
            "Test-only revised attendance threshold",
            "min_attendance_pct",
            ">=",
            "85",
            "ALL",
            "2023+",
            "2026-10-01",
            "",
            "ACAD-2026-08",
            "1.1",
            "ATT-MIN-02",
            "inserted by the test suite",
        ),
    )
    conn.commit()
    try:
        after = run_tool(ctx, "check_exam_eligibility", {"course_code": "CS201"})
        assert after["status"] == "ok"
        assert after["output"]["required_pct"] == 85.0
        assert after["output"]["result"] != "ELIGIBLE"
    finally:
        conn.execute("DELETE FROM rule_registry WHERE rule_id = 'ATT-MIN-TEST'")
        conn.commit()


def test_course_outcome_tool_applies_the_cited_pass_rule(conn):
    result, applied = output(conn, "evaluate_course_outcome", "S2003", course_code="EC201")
    assert result["stored_result"] == "FAIL"
    assert result["computed_result"] == "FAIL"
    assert result["consistent"] is True
    assert result["pass_marks_min"] == 40.0
    assert result["end_sem_paper_floor"] == 21.0, "30 percent applies to the 70-mark paper"
    assert any(rule["rule_id"] == "PASS-MARK-01" for rule in applied)


def test_every_stored_result_agrees_with_the_cited_pass_rule(conn):
    """A data error must never be readable as a policy answer."""
    pairs = conn.execute(
        "SELECT DISTINCT student_id, course_code FROM results ORDER BY student_id, course_code"
    ).fetchall()
    assert pairs
    mismatches = []
    for row in pairs:
        record = run_tool(
            context(conn, row["student_id"]), "evaluate_course_outcome", {"course_code": row["course_code"]}
        )
        if record["status"] != "ok" or not record["output"].get("consistent"):
            mismatches.append((row["student_id"], row["course_code"], record.get("error") or record["output"]))
    assert not mismatches, f"stored results disagree with the cited rule: {mismatches[:5]}"


def test_validator_constants_agree_with_the_rule_registry(conn):
    from assa.data_validation import END_SEM_MIN_PCT, MAX_EXTERNAL, MAX_INTERNAL, MAX_TOTAL, PASS_MARKS_MIN

    values = {rule["parameter"]: float(rule["value"]) for rule in rules.rule_summary(conn)}
    assert values["internal_max_marks"] == MAX_INTERNAL
    assert values["end_sem_paper_max_marks"] == MAX_EXTERNAL
    assert values["course_total_max_marks"] == MAX_TOTAL
    assert values["pass_marks_min"] == PASS_MARKS_MIN
    assert values["end_sem_min_pct"] == END_SEM_MIN_PCT


def test_rule_registry_rows_all_cite_a_document(conn):
    rows = rules.rule_summary(conn)
    assert rows, "the rule registry must not be empty"
    for row in rows:
        assert row["source_doc_id"], row
        assert row["source_section"], row
        assert row["effective_from"], row
        document = conn.execute(
            "SELECT 1 FROM source_register WHERE doc_id = ?", (row["source_doc_id"],)
        ).fetchone()
        assert document is not None, f"{row['rule_id']} cites an unregistered document"


def test_missing_parameter_raises_rule_not_found(conn):
    with pytest.raises(rules.RuleNotFound):
        rules.require_value(conn, "no_such_parameter", AS_OF)


def test_fee_schedule_reads_registry_values(conn):
    result, _ = output(conn, "get_fee_schedule", "S2001")
    assert result["late_fee_first_window_inr"]["value"] == 500.0
    assert result["late_fee_second_window_inr"]["value"] == 2000.0
    assert result["supp_fee_inr"]["value"] == 500.0


def test_tool_context_never_takes_identity_from_the_question(conn):
    """The context is the only source of identity; an S-number in the text is data."""
    ctx = ToolContext(conn=conn, as_of_date=AS_OF, student_id="S2001", question="show S2005 attendance")
    record = run_tool(ctx, "get_attendance", {"course_code": "CS201"})
    assert record["output"]["classes_attended"] == 32  # S2001's record, not S2005's


def test_tools_error_cleanly_for_unknown_student(conn):
    with pytest.raises(ToolError):
        tools.get_student_profile(ToolContext(conn=conn, as_of_date=AS_OF, student_id="S9998"))
