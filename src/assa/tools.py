"""Deterministic tools (R5).

Every authoritative number in an answer -- attendance percentages, eligibility
decisions, CGPA comparisons, fee amounts -- is produced here, in Python, over the
SQLite database and the rule registry. The LLM only selects a tool and explains
the result it gets back. No arithmetic and no eligibility judgement happens in
the model.

Each tool receives a :class:`ToolContext` whose ``student_id`` comes from the
request context (the ``X-Student-Id`` header) and never from the question text
(R7).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable, Sequence

from . import rules

BACKLOG_RESULTS = ("FAIL", "ABSENT", "DETAINED")
SESSION_MONTH_ORDER = {"MAY": 1, "DEC": 2}


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    as_of_date: date
    student_id: str | None = None
    question: str = ""
    #: Set by the caller when the student record has been loaded.
    programme: str | None = None
    batch_year: int | None = None


@dataclass
class ToolResult:
    output: dict[str, Any]
    applied_rules: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class ToolError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def session_key(session: str) -> tuple[int, int]:
    """``"2026-MAY"`` -> ``(2026, 1)`` so sessions sort chronologically."""
    parts = (session or "").split("-")
    year = int(parts[0]) if parts and parts[0].isdigit() else 0
    month = SESSION_MONTH_ORDER.get(parts[1].upper(), 0) if len(parts) > 1 else 0
    return year, month


def attendance_pct(classes_held: int | None, classes_attended: int | None) -> float | None:
    """Attendance is always computed by a tool and never stored (Annex C)."""
    if not classes_held:
        return None
    return round((classes_attended or 0) / classes_held * 100, 1)


def _student_row(conn: sqlite3.Connection, student_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
    if row is None:
        raise ToolError(f"no student record for {student_id}")
    return dict(row)


def _require_student(ctx: ToolContext) -> dict[str, Any]:
    if not ctx.student_id:
        raise ToolError("no student identity in the request context")
    student = _student_row(ctx.conn, ctx.student_id)
    ctx.programme = student["programme"]
    ctx.batch_year = int(student["batch_year"])
    return student


def _course_row(conn: sqlite3.Connection, course_code: str, student: dict[str, Any]) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM courses WHERE course_code = ?", (course_code,)).fetchone()
    if row is None:
        raise ToolError(f"unknown course {course_code}")
    course = dict(row)
    programme_scope = (course.get("programme") or "ALL").strip()
    if programme_scope.upper() != "ALL" and programme_scope.casefold() != str(student["programme"]).casefold():
        raise ToolError(
            f"course {course_code} is not offered to programme {student['programme']}"
        )
    return course


def _attendance_rows(conn: sqlite3.Connection, student_id: str, course_code: str | None = None):
    sql = "SELECT * FROM attendance WHERE student_id = ?"
    params: list[Any] = [student_id]
    if course_code:
        sql += " AND course_code = ?"
        params.append(course_code)
    return [dict(row) for row in conn.execute(sql + " ORDER BY course_code", params).fetchall()]


def _result_rows(conn: sqlite3.Connection, student_id: str, course_code: str | None = None):
    sql = "SELECT * FROM results WHERE student_id = ?"
    params: list[Any] = [student_id]
    if course_code:
        sql += " AND course_code = ?"
        params.append(course_code)
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def latest_results(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest result per course. A supplementary attempt after a regular attempt wins."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        course = row["course_code"]
        current = latest.get(course)
        if current is None:
            latest[course] = row
            continue
        current_key = (session_key(current["exam_session"]), 1 if current["exam_type"] == "SUPPLEMENTARY" else 0)
        row_key = (session_key(row["exam_session"]), 1 if row["exam_type"] == "SUPPLEMENTARY" else 0)
        if row_key >= current_key:
            latest[course] = row
    return latest


def _rule_value(
    ctx: ToolContext, parameter: str
) -> tuple[float, dict[str, Any], list[dict[str, Any]], list[str]]:
    """Read a threshold from the rule registry (never a constant in code)."""
    try:
        value, winner, resolution = rules.require_value(
            ctx.conn, parameter, ctx.as_of_date, ctx.programme, ctx.batch_year
        )
    except rules.RuleNotFound as exc:
        raise ToolError(str(exc)) from exc
    notes: list[str] = []
    applied = [rules.applied_rule(rule) for rule in (resolution.applied or [winner])]
    if resolution.decision:
        notes.append(resolution.decision)
    for upcoming in resolution.upcoming:
        notes.append(
            f"{upcoming['rule_id']} will change {parameter} to {upcoming['value']} "
            f"from {upcoming['effective_from']} (not yet effective)"
        )
    return value, winner, applied, notes


def _merge_rules(*rule_lists: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in rule_lists:
        for rule in group or []:
            merged[rule["rule_id"]] = rule
    return list(merged.values())


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


def get_student_profile(ctx: ToolContext) -> ToolResult:
    student = _require_student(ctx)
    return ToolResult(
        output={
            "student_id": student["student_id"],
            "programme": student["programme"],
            "batch_year": student["batch_year"],
            "current_semester": student["current_semester"],
            "cgpa": student["cgpa"],
            "active_backlogs": student["active_backlogs"],
        }
    )


def get_attendance(ctx: ToolContext, course_code: str) -> ToolResult:
    student = _require_student(ctx)
    _course_row(ctx.conn, course_code, student)
    rows = _attendance_rows(ctx.conn, student["student_id"], course_code)
    if not rows:
        raise ToolError(f"no attendance record for {course_code}")
    row = rows[0]
    return ToolResult(
        output={
            "course_code": course_code,
            "classes_held": row["classes_held"],
            "classes_attended": row["classes_attended"],
            "attendance_pct": attendance_pct(row["classes_held"], row["classes_attended"]),
        }
    )


def get_all_attendance(ctx: ToolContext) -> ToolResult:
    student = _require_student(ctx)
    rows = _attendance_rows(ctx.conn, student["student_id"])
    if not rows:
        raise ToolError("no attendance records for this student")
    courses = [
        {
            "course_code": row["course_code"],
            "classes_held": row["classes_held"],
            "classes_attended": row["classes_attended"],
            "attendance_pct": attendance_pct(row["classes_held"], row["classes_attended"]),
        }
        for row in rows
    ]
    percentages = [course["attendance_pct"] for course in courses if course["attendance_pct"] is not None]
    return ToolResult(
        output={
            "courses": courses,
            "lowest_attendance_pct": min(percentages) if percentages else None,
            "highest_attendance_pct": max(percentages) if percentages else None,
            "courses_below_any_threshold": [
                course["course_code"]
                for course in courses
                if course["attendance_pct"] is not None and course["attendance_pct"] < 80
            ],
        }
    )


def get_results(ctx: ToolContext, course_code: str | None = None) -> ToolResult:
    student = _require_student(ctx)
    rows = _result_rows(ctx.conn, student["student_id"], course_code)
    if not rows:
        raise ToolError("no result records for this student")
    latest = latest_results(rows)
    return ToolResult(
        output={
            "results": [
                {
                    "course_code": row["course_code"],
                    "exam_session": row["exam_session"],
                    "exam_type": row["exam_type"],
                    "internal_marks": row["internal_marks"],
                    "external_marks": row["external_marks"],
                    "total_marks": row["total_marks"],
                    "max_marks": row["max_marks"],
                    "result": row["result"],
                    "is_latest": latest.get(row["course_code"]) is row,
                }
                for row in sorted(rows, key=lambda item: (item["course_code"], session_key(item["exam_session"])))
            ]
        }
    )


def evaluate_course_outcome(ctx: ToolContext, course_code: str) -> ToolResult:
    """Recompute PASS/FAIL from the marks using the cited pass rules.

    The stored result column stays authoritative for reporting, but this tool
    applies the registry thresholds (clause 3.2 plus the 30 percent paper floor of
    clause 3.3, taken against the 70-mark paper) and reports any disagreement, so
    a data error can never be read as a policy answer.
    """
    student = _require_student(ctx)
    _course_row(ctx.conn, course_code, student)
    rows = _result_rows(ctx.conn, student["student_id"], course_code)
    if not rows:
        raise ToolError(f"no result record for {course_code}")
    latest = latest_results(rows)[course_code]

    pass_marks, pass_rule, pass_applied, notes = _rule_value(ctx, "pass_marks_min")
    paper_pct, paper_rule, paper_applied, _ = _rule_value(ctx, "end_sem_min_pct")
    paper_max, paper_max_rule, paper_max_applied, _ = _rule_value(ctx, "end_sem_paper_max_marks")
    applied = _merge_rules(pass_applied, paper_applied, paper_max_applied)

    total = int(latest["total_marks"])
    external = int(latest["external_marks"])
    paper_floor = round(paper_max * paper_pct / 100, 2)
    stored = str(latest["result"]).upper()
    output: dict[str, Any] = {
        "course_code": course_code,
        "exam_session": latest["exam_session"],
        "stored_result": stored,
        "total_marks": total,
        "external_marks": external,
        "pass_marks_min": pass_marks,
        "end_sem_paper_floor": paper_floor,
        "rule_id": pass_rule["rule_id"],
        "paper_rule_id": paper_rule["rule_id"],
        "paper_max_rule_id": paper_max_rule["rule_id"],
    }
    if stored in {"ABSENT", "DETAINED"}:
        output.update(
            computed_result=stored,
            consistent=True,
            note=f"The recorded result is {stored}, so the pass rule is not applied.",
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    computed = "PASS" if total >= pass_marks and external >= paper_floor else "FAIL"
    output.update(computed_result=computed, consistent=computed == stored)
    if computed != stored:
        output["note"] = (
            f"The stored result ({stored}) disagrees with the cited rule for these marks "
            f"({computed}); the record needs review."
        )
    return ToolResult(output=output, applied_rules=applied, notes=notes)


def get_active_backlogs(ctx: ToolContext) -> ToolResult:
    student = _require_student(ctx)
    latest = latest_results(_result_rows(ctx.conn, student["student_id"]))
    backlogs = [
        {
            "course_code": course,
            "exam_session": row["exam_session"],
            "result": row["result"],
            "total_marks": row["total_marks"],
        }
        for course, row in sorted(latest.items())
        if row["result"] in BACKLOG_RESULTS
    ]
    return ToolResult(
        output={"active_backlogs": len(backlogs), "courses": backlogs, "cgpa": student["cgpa"]}
    )


def get_policy_rule(ctx: ToolContext, parameter: str) -> ToolResult:
    """Read a single threshold from the rule registry, with its cited clause."""
    value, winner, applied, notes = _rule_value(ctx, parameter)
    return ToolResult(
        output={
            "parameter": parameter,
            "value": value,
            "operator": winner.get("operator"),
            "rule_id": winner["rule_id"],
            "source_doc_id": winner.get("source_doc_id"),
            "source_section": winner.get("source_section"),
            "effective_from": winner.get("effective_from"),
        },
        applied_rules=applied,
        notes=notes,
    )


def check_exam_eligibility(ctx: ToolContext, course_code: str) -> ToolResult:
    student = _require_student(ctx)
    _course_row(ctx.conn, course_code, student)
    attendance = _attendance_rows(ctx.conn, student["student_id"], course_code)
    if not attendance:
        raise ToolError(f"no attendance record for {course_code}")
    row = attendance[0]
    pct = attendance_pct(row["classes_held"], row["classes_attended"])

    required, required_rule, required_applied, notes = _rule_value(ctx, "min_attendance_pct")
    applied = list(required_applied)

    latest = latest_results(_result_rows(ctx.conn, student["student_id"], course_code)).get(course_code)

    output: dict[str, Any] = {
        "course_code": course_code,
        "classes_held": row["classes_held"],
        "classes_attended": row["classes_attended"],
        "attendance_pct": pct,
        "required_pct": required,
        "rule_id": required_rule["rule_id"],
        "source_doc_id": required_rule.get("source_doc_id"),
        "source_section": required_rule.get("source_section"),
    }
    if latest and latest["result"] == "DETAINED":
        output.update(
            result="NOT_ELIGIBLE",
            reason_code="DETAINED",
            note="The student is detained in this course and may not appear for the end-semester examination.",
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    if pct is not None and pct >= required:
        output.update(result="ELIGIBLE", reason_code="ATTENDANCE_OK")
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    shortage = round(required - (pct or 0), 1)
    detention_limit, _, detention_applied, _ = _rule_value(ctx, "detention_threshold_pct")
    max_condonation, _, condonation_applied, _ = _rule_value(ctx, "max_condonation_pct")
    floor, _, floor_applied, _ = _rule_value(ctx, "min_attendance_after_condonation_pct")
    applied = _merge_rules(applied, detention_applied, condonation_applied, floor_applied)

    output.update(shortage_pct=shortage, max_condonation_pct=max_condonation, detention_threshold_pct=detention_limit)

    if pct is not None and pct <= detention_limit:
        output.update(
            result="NOT_ELIGIBLE",
            reason_code="BELOW_DETENTION_LIMIT",
            note=(
                f"Attendance is at or below the detention limit of {detention_limit}%; a detention under the "
                "Academic Regulations applies unless condonation is granted."
            ),
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    if shortage <= max_condonation and (pct or 0) + max_condonation >= floor:
        output.update(
            result="NOT_ELIGIBLE_CONDONATION_REQUIRED",
            reason_code="CONDONATION_MAY_BE_CONSIDERED",
            note=(
                f"The shortage of {shortage} percentage points is within the {max_condonation}% that the Dean "
                f"(Academics) may condone, provided attendance after condonation is at least {floor}%. "
                "Condonation is not automatic and must be applied for with documentary evidence."
            ),
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    output.update(
        result="NOT_ELIGIBLE",
        reason_code="ATTENDANCE_SHORTAGE",
        note=f"Attendance is {pct}%, which is {shortage} percentage points below the required {required}%.",
    )
    return ToolResult(output=output, applied_rules=applied, notes=notes)


def check_supplementary_eligibility(ctx: ToolContext, course_code: str) -> ToolResult:
    student = _require_student(ctx)
    _course_row(ctx.conn, course_code, student)
    rows = _result_rows(ctx.conn, student["student_id"], course_code)
    applied: list[dict[str, Any]] = []
    notes: list[str] = []

    if not rows:
        raise ToolError(f"no result record for {course_code}")

    latest = latest_results(rows)[course_code]
    attempts = len([row for row in rows if row["exam_type"] == "SUPPLEMENTARY"])
    max_attempts, max_rule, max_applied, max_notes = _rule_value(ctx, "max_supplementary_attempts")
    applied = _merge_rules(applied, max_applied)
    notes += max_notes

    output: dict[str, Any] = {
        "course_code": course_code,
        "latest_result": latest["result"],
        "exam_session": latest["exam_session"],
        "supplementary_attempts_used": attempts,
        "max_supplementary_attempts": int(max_attempts),
        "rule_id": max_rule["rule_id"],
    }

    if latest["result"] == "PASS":
        output.update(
            result="NOT_APPLICABLE",
            reason_code="COURSE_CLEARED",
            note="The course is already cleared, so no supplementary examination is required.",
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    if latest["result"] == "DETAINED":
        output.update(
            result="NOT_ELIGIBLE",
            reason_code="DETAINED",
            note=(
                "A student detained in a course is not eligible for the supplementary examination of that "
                "course and must repeat the course."
            ),
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    if attempts >= max_attempts:
        output.update(
            result="NOT_ELIGIBLE",
            reason_code="ATTEMPTS_EXHAUSTED",
            note=f"{attempts} supplementary attempt(s) have already been used; the limit is {int(max_attempts)}.",
        )
        return ToolResult(output=output, applied_rules=applied, notes=notes)

    attendance = check_exam_eligibility(ctx, course_code)
    applied = _merge_rules(applied, attendance.applied_rules)
    notes += attendance.notes
    output.update(
        attendance_rule_satisfied=attendance.output["result"] == "ELIGIBLE",
        attendance_pct=attendance.output.get("attendance_pct"),
        attendance_requirement_pct=attendance.output.get("required_pct"),
        attendance_rule_id=attendance.output.get("rule_id"),
    )

    if attendance.output["result"] == "ELIGIBLE":
        output.update(
            result="ELIGIBLE",
            reason_code="FAILED_WITH_ATTENDANCE_OK",
            note=(
                "The course is not cleared and the attendance requirement for the course is met, so a "
                "supplementary examination may be applied for within the notified window."
            ),
        )
    else:
        output.update(
            result="NOT_ELIGIBLE_ATTENDANCE",
            reason_code=attendance.output.get("reason_code"),
            note=(
                "A supplementary examination requires the attendance requirement of the course to be met. "
                + str(attendance.output.get("note", ""))
            ).strip(),
        )
    return ToolResult(output=output, applied_rules=applied, notes=notes)


def check_placement_eligibility(ctx: ToolContext) -> ToolResult:
    student = _require_student(ctx)
    min_cgpa, cgpa_rule, cgpa_applied, cgpa_notes = _rule_value(ctx, "min_cgpa_placement")
    max_backlogs, backlog_rule, backlog_applied, backlog_notes = _rule_value(
        ctx, "max_active_backlogs_placement"
    )
    backlogs = get_active_backlogs(ctx)
    detained = [
        course
        for course, row in latest_results(_result_rows(ctx.conn, student["student_id"])).items()
        if row["result"] == "DETAINED"
    ]

    reasons: list[str] = []
    if float(student["cgpa"]) < min_cgpa:
        reasons.append(f"CGPA {student['cgpa']} is below the required {min_cgpa}")
    if backlogs.output["active_backlogs"] > max_backlogs:
        reasons.append(
            f"{backlogs.output['active_backlogs']} active backlog(s) exceed the allowed {int(max_backlogs)}"
        )
    if detained:
        reasons.append(f"detention recorded in {', '.join(sorted(detained))}")

    return ToolResult(
        output={
            "result": "ELIGIBLE" if not reasons else "NOT_ELIGIBLE",
            "cgpa": student["cgpa"],
            "min_cgpa": min_cgpa,
            "active_backlogs": backlogs.output["active_backlogs"],
            "max_active_backlogs": int(max_backlogs),
            "backlog_courses": [entry["course_code"] for entry in backlogs.output["courses"]],
            "rule_id": cgpa_rule["rule_id"],
            "backlog_rule_id": backlog_rule["rule_id"],
            "reasons": reasons,
            "pending_unfair_means_checked": False,
        },
        applied_rules=_merge_rules(cgpa_applied, backlog_applied),
        notes=cgpa_notes + backlog_notes,
    )


def check_scholarship_eligibility(ctx: ToolContext) -> ToolResult:
    student = _require_student(ctx)
    min_cgpa, cgpa_rule, cgpa_applied, cgpa_notes = _rule_value(ctx, "min_cgpa_scholarship")
    min_attendance, att_rule, att_applied, att_notes = _rule_value(ctx, "min_attendance_scholarship_pct")
    max_backlogs, backlog_rule, backlog_applied, _ = _rule_value(ctx, "max_active_backlogs_scholarship")
    attendance = get_all_attendance(ctx)
    backlogs = get_active_backlogs(ctx)

    lowest = attendance.output["lowest_attendance_pct"]
    reasons: list[str] = []
    if float(student["cgpa"]) < min_cgpa:
        reasons.append(f"CGPA {student['cgpa']} is below the required {min_cgpa}")
    if lowest is not None and lowest < min_attendance:
        reasons.append(f"lowest course attendance {lowest}% is below the required {min_attendance}%")
    if backlogs.output["active_backlogs"] > max_backlogs:
        reasons.append(f"{backlogs.output['active_backlogs']} active backlog(s) exceed the allowed {int(max_backlogs)}")

    return ToolResult(
        output={
            "result": "ELIGIBLE" if not reasons else "NOT_ELIGIBLE",
            "cgpa": student["cgpa"],
            "min_cgpa": min_cgpa,
            "lowest_attendance_pct": lowest,
            "min_attendance_pct": min_attendance,
            "active_backlogs": backlogs.output["active_backlogs"],
            "rule_id": cgpa_rule["rule_id"],
            "attendance_rule_id": att_rule["rule_id"],
            "backlog_rule_id": backlog_rule["rule_id"],
            "reasons": reasons,
        },
        applied_rules=_merge_rules(cgpa_applied, att_applied, backlog_applied),
        notes=cgpa_notes + att_notes,
    )


def check_lab_attendance_requirement(ctx: ToolContext) -> ToolResult:
    student = _require_student(ctx)
    try:
        value, winner, applied, notes = _rule_value(ctx, "min_lab_attendance_pct")
    except ToolError:
        return ToolResult(
            output={
                "applies": False,
                "reason": f"no departmental laboratory attendance rule covers programme {student['programme']}",
            }
        )
    return ToolResult(
        output={
            "applies": True,
            "parameter": "min_lab_attendance_pct",
            "value": value,
            "rule_id": winner["rule_id"],
            "source_doc_id": winner.get("source_doc_id"),
            "source_section": winner.get("source_section"),
        },
        applied_rules=applied,
        notes=notes,
    )


def get_fee_schedule(ctx: ToolContext) -> ToolResult:
    results = {}
    applied: list[dict[str, Any]] = []
    notes: list[str] = []
    for parameter in (
        "late_fee_first_window_inr",
        "late_fee_second_window_inr",
        "supp_fee_inr",
        "supp_late_fee_inr",
        "hostel_late_fee_inr",
    ):
        try:
            value, winner, rule_applied, rule_notes = _rule_value(ctx, parameter)
        except ToolError:
            continue
        results[parameter] = {
            "value": value,
            "rule_id": winner["rule_id"],
            "source_doc_id": winner.get("source_doc_id"),
            "source_section": winner.get("source_section"),
        }
        applied = _merge_rules(applied, rule_applied)
        notes += rule_notes
    return ToolResult(output=results, applied_rules=applied, notes=notes)


def what_if_attendance(
    ctx: ToolContext, course_code: str, additional_held: int = 0, additional_attended: int = 0
) -> ToolResult:
    """What-if: recompute eligibility for a hypothetical future attendance."""
    student = _require_student(ctx)
    _course_row(ctx.conn, course_code, student)
    rows = _attendance_rows(ctx.conn, student["student_id"], course_code)
    if not rows:
        raise ToolError(f"no attendance record for {course_code}")
    row = rows[0]

    projection = check_exam_eligibility(ctx, course_code)
    required = float(projection.output["required_pct"])
    held = row["classes_held"] + max(0, int(additional_held))
    attended = row["classes_attended"] + max(0, int(additional_attended))
    if held and attended > held:
        raise ToolError("attended classes cannot exceed classes held")

    projected_pct = attendance_pct(held, attended)
    classes_needed = None
    if projected_pct is not None and projected_pct < required:
        needed_attended = -(-held * required // 100)  # ceil
        classes_needed = max(0, int(needed_attended) - attended)

    return ToolResult(
        output={
            "course_code": course_code,
            "current_pct": projection.output["attendance_pct"],
            "projected_pct": projected_pct,
            "required_pct": required,
            "additional_classes_assumed": int(additional_held),
            "additional_classes_attended_assumed": int(additional_attended),
            "result": "ELIGIBLE" if (projected_pct or 0) >= required else "NOT_ELIGIBLE",
            "additional_classes_needed": classes_needed,
            "rule_id": projection.output["rule_id"],
            "assumptions": (
                f"Assumes {int(additional_attended)} of the next {int(additional_held)} classes are attended "
                "and that no other rule changes before the examination."
            ),
        },
        applied_rules=projection.applied_rules,
        notes=projection.notes,
    )


# --------------------------------------------------------------------------- #
# registry of tools exposed to the orchestrator
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    "get_student_profile": get_student_profile,
    "get_attendance": get_attendance,
    "get_all_attendance": get_all_attendance,
    "get_results": get_results,
    "evaluate_course_outcome": evaluate_course_outcome,
    "get_active_backlogs": get_active_backlogs,
    "get_policy_rule": get_policy_rule,
    "check_exam_eligibility": check_exam_eligibility,
    "check_supplementary_eligibility": check_supplementary_eligibility,
    "check_placement_eligibility": check_placement_eligibility,
    "check_scholarship_eligibility": check_scholarship_eligibility,
    "check_lab_attendance_requirement": check_lab_attendance_requirement,
    "get_fee_schedule": get_fee_schedule,
    "what_if_attendance": what_if_attendance,
}

#: Tools that need a student identity. They are never offered when no
#: X-Student-Id header is present (R7).
STUDENT_TOOLS = {
    "get_student_profile",
    "get_attendance",
    "get_all_attendance",
    "get_results",
    "evaluate_course_outcome",
    "get_active_backlogs",
    "check_exam_eligibility",
    "check_supplementary_eligibility",
    "check_placement_eligibility",
    "check_scholarship_eligibility",
    "check_lab_attendance_requirement",
    "what_if_attendance",
}

#: JSON-schema description of every tool, used by the LLM to select a tool.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_student_profile",
        "description": "Programme, batch, semester, CGPA and stored backlog count of the logged-in student.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_attendance",
        "description": "Attendance of the logged-in student in one course, with the computed percentage.",
        "parameters": {
            "type": "object",
            "properties": {"course_code": {"type": "string", "description": "For example CS201"}},
            "required": ["course_code"],
        },
    },
    {
        "name": "get_all_attendance",
        "description": "Attendance of the logged-in student in every registered course.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_results",
        "description": "Examination results of the logged-in student, optionally for one course.",
        "parameters": {
            "type": "object",
            "properties": {"course_code": {"type": "string"}},
            "required": [],
        },
    },
    {
        "name": "evaluate_course_outcome",
        "description": (
            "Recompute PASS or FAIL for a course from the marks using the cited pass rules, and report "
            "whether the stored result agrees."
        ),
        "parameters": {
            "type": "object",
            "properties": {"course_code": {"type": "string"}},
            "required": ["course_code"],
        },
    },
    {
        "name": "get_active_backlogs",
        "description": "Courses not yet cleared by the logged-in student, with CGPA.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_policy_rule",
        "description": "Value in force of a named policy threshold from the rule registry, with its cited clause.",
        "parameters": {
            "type": "object",
            "properties": {
                "parameter": {
                    "type": "string",
                    "description": (
                        "One of min_attendance_pct, max_condonation_pct, min_attendance_after_condonation_pct, "
                        "detention_threshold_pct, pass_marks_min, end_sem_min_pct, max_supplementary_attempts, "
                        "min_cgpa_placement, max_active_backlogs_placement, min_cgpa_scholarship, "
                        "min_attendance_scholarship_pct"
                    ),
                }
            },
            "required": ["parameter"],
        },
    },
    {
        "name": "check_exam_eligibility",
        "description": "Deterministic eligibility check for the end-semester examination of one course.",
        "parameters": {
            "type": "object",
            "properties": {"course_code": {"type": "string"}},
            "required": ["course_code"],
        },
    },
    {
        "name": "check_supplementary_eligibility",
        "description": "Deterministic eligibility check for the supplementary examination of one course.",
        "parameters": {
            "type": "object",
            "properties": {"course_code": {"type": "string"}},
            "required": ["course_code"],
        },
    },
    {
        "name": "check_placement_eligibility",
        "description": "Deterministic eligibility check for campus placement drives.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_scholarship_eligibility",
        "description": "Deterministic eligibility check for the merit scholarship.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_lab_attendance_requirement",
        "description": "Departmental laboratory attendance requirement, if one applies to the student.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_fee_schedule",
        "description": "Late fee, supplementary examination fee and hostel late fee amounts in force.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "what_if_attendance",
        "description": "Project attendance and eligibility for a hypothetical number of additional classes.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_code": {"type": "string"},
                "additional_held": {"type": "integer", "description": "Classes to be held in future"},
                "additional_attended": {"type": "integer", "description": "How many of those will be attended"},
            },
            "required": ["course_code"],
        },
    },
]


def run_tool(ctx: ToolContext, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one tool and return an audit-friendly invocation record."""
    arguments = dict(arguments or {})
    started = time.perf_counter()
    record: dict[str, Any] = {"tool": name, "input": arguments, "output": {}, "status": "ok", "ms": 0}
    function = TOOL_REGISTRY.get(name)
    if function is None:
        record.update(status="error", error=f"unknown tool {name}")
        return record
    if name in STUDENT_TOOLS and not ctx.student_id:
        record.update(
            status="error",
            error="this tool needs a student identity (X-Student-Id header)",
        )
        return record
    try:
        result = function(ctx, **arguments)
    except (ToolError, TypeError, ValueError) as exc:
        record.update(status="error", error=str(exc))
        record["ms"] = int((time.perf_counter() - started) * 1000)
        return record
    record["output"] = result.output
    record["applied_rules"] = result.applied_rules
    record["notes"] = result.notes
    record["ms"] = int((time.perf_counter() - started) * 1000)
    return record
