"""Validator tests: the checks that keep the synthetic data honest."""

from __future__ import annotations

from assa.data_validation import (
    active_backlog_count,
    expected_result,
    validate_csv_dir,
    validate_dataset,
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def students():
    return [
        {"student_id": "S3001", "full_name": "Test One", "programme": "B.Tech CSE", "batch_year": 2023,
         "current_semester": 5, "cgpa": 7.0, "active_backlogs": 1},
        {"student_id": "S3002", "full_name": "Test Two", "programme": "B.Tech ECE", "batch_year": 2024,
         "current_semester": 3, "cgpa": 6.0, "active_backlogs": 0},
    ]


def courses():
    return [
        {"course_code": "CS201", "course_name": "Data Structures", "programme": "B.Tech CSE", "semester": 3, "credits": 4},
        {"course_code": "EC201", "course_name": "Digital Circuits", "programme": "B.Tech ECE", "semester": 3, "credits": 4},
    ]


def attendance():
    return [
        {"student_id": "S3001", "course_code": "CS201", "classes_held": 40, "classes_attended": 31},
        {"student_id": "S3002", "course_code": "EC201", "classes_held": 40, "classes_attended": 36},
    ]


def results():
    return [
        {"student_id": "S3001", "course_code": "CS201", "exam_session": "2026-MAY", "exam_type": "REGULAR",
         "internal_marks": 20, "external_marks": 19, "total_marks": 39, "max_marks": 100, "result": "FAIL"},
        {"student_id": "S3002", "course_code": "EC201", "exam_session": "2026-DEC", "exam_type": "REGULAR",
         "internal_marks": 25, "external_marks": 30, "total_marks": 55, "max_marks": 100, "result": "PASS"},
    ]


def test_pass_rule_matches_the_examination_rules():
    assert expected_result(40, 21, 100) == "PASS"
    assert expected_result(39, 20, 100) == "FAIL"
    assert expected_result(70, 20, 100) == "FAIL", "21 of the 70 paper marks is also required"
    assert expected_result(45, 25, 100) == "PASS", "clause 3.3 applies to the 70-mark paper, not the total"
    assert expected_result(100, 70, 100) == "PASS"


def test_attended_greater_than_held_is_a_violation():
    broken = attendance()
    broken[0]["classes_attended"] = 45
    report = validate_dataset(students(), courses(), broken, results())
    assert any("attended > held" in violation.rule for violation in report.violations)


def test_total_must_equal_internal_plus_external():
    broken = results()
    broken[0]["total_marks"] = 60
    report = validate_dataset(students(), courses(), attendance(), broken)
    assert any("total != internal + external" in violation.rule for violation in report.violations)


def test_result_must_agree_with_marks():
    broken = results()
    broken[0]["result"] = "PASS"  # 39 marks cannot be a pass
    report = validate_dataset(students(), courses(), attendance(), broken)
    assert any("result inconsistent with marks" in violation.rule for violation in report.violations)


def test_active_backlogs_must_match_the_results():
    broken = students()
    broken[0]["active_backlogs"] = 0
    report = validate_dataset(broken, courses(), attendance(), results())
    assert any("active_backlogs inconsistent" in violation.rule for violation in report.violations)


def test_reserved_identifiers_are_rejected():
    reserved = students()
    reserved[0]["student_id"] = "S9001"
    report = validate_dataset(reserved, courses(), attendance(), results())
    assert any("reserved id" in violation.rule for violation in report.violations)


def test_reserved_course_codes_are_rejected():
    reserved = courses()
    reserved[0]["course_code"] = "JDG001"
    report = validate_dataset(students(), reserved, attendance(), results())
    assert any("reserved code" in violation.rule for violation in report.violations)


def test_course_from_another_programme_is_a_violation():
    mixed = attendance()
    mixed.append({"student_id": "S3002", "course_code": "CS201", "classes_held": 40, "classes_attended": 30})
    report = validate_dataset(students(), courses(), mixed, results())
    assert any("programme mismatch" in violation.rule for violation in report.violations)


def test_latest_attempt_wins_when_counting_backlogs():
    rows = [
        {"course_code": "EC201", "exam_session": "2026-MAY", "exam_type": "REGULAR", "result": "FAIL"},
        {"course_code": "EC201", "exam_session": "2026-DEC", "exam_type": "SUPPLEMENTARY", "result": "PASS"},
    ]
    assert active_backlog_count(rows) == 0


def test_the_committed_dataset_validates_with_all_edge_cases():
    report = validate_csv_dir(PROJECT_ROOT / "data")
    assert report.ok, report.summary()
    edge_cases = report.coverage["edge_cases"]
    for tag, ids in edge_cases.items():
        assert ids, f"edge case {tag} is missing from the committed dataset"
    assert report.counts["students"] >= 30
    assert report.counts["courses"] >= 6
