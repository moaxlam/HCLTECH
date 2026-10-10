"""Schema and logical validation for the synthetic student dataset (Annex C).

Deliberately written with the standard library only so that the validation script,
the data generator and the tests can all run without the heavy runtime
dependencies (ChromaDB, sentence-transformers) installed.

Checks performed:

* column names and order of every required table
* identifier formats, and the student IDs and course codes reserved for judges
* ``attended <= held`` and ``0 <= held``
* marks ranges, ``total = internal + external``, ``total <= max_marks``
* result consistent with marks (PASS/FAIL/ABSENT/DETAINED)
* referential integrity across the four files
* ``students.active_backlogs`` equal to the value recomputed from ``results``
* the minimum coverage required by the guide (30 students, 2 programmes,
  2 batches, 6 courses)
* the deliberately requested edge cases
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

STUDENT_ID_RE = re.compile(r"^S\d{4}$")
COURSE_CODE_RE = re.compile(r"^[A-Z]{2,4}\d{3}$")

RESERVED_STUDENT_IDS = {f"S{number}" for number in range(9000, 10000)}

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "students": [
        "student_id",
        "full_name",
        "programme",
        "batch_year",
        "current_semester",
        "cgpa",
        "active_backlogs",
    ],
    "courses": ["course_code", "course_name", "programme", "semester", "credits"],
    "attendance": ["student_id", "course_code", "classes_held", "classes_attended"],
    "results": [
        "student_id",
        "course_code",
        "exam_session",
        "exam_type",
        "internal_marks",
        "external_marks",
        "total_marks",
        "max_marks",
        "result",
    ],
}

PROGRAMMES = ("B.Tech CSE", "B.Tech ECE")
BATCH_YEARS = (2023, 2024)
EXAM_TYPES = ("REGULAR", "SUPPLEMENTARY")
RESULT_VALUES = ("PASS", "FAIL", "ABSENT", "DETAINED")

MIN_STUDENTS = 30
MIN_PROGRAMMES = 2
MIN_BATCHES = 2
MIN_COURSES = 6

# These mirror the cited rows of the rule registry, which is the single source
# of truth the tools read at run time (EXAM-RULES-2025 clauses 3.1-3.3). The test
# suite asserts that these constants and the registry agree.
PASS_MARKS_MIN = 40  # rule PASS-MARK-01, clause 3.2, out of course_total_max_marks
END_SEM_MIN_PCT = 30  # rule PASS-PAPER-01, clause 3.3, of the end-semester paper
MAX_INTERNAL = 30  # rule INTERNAL-MAX-01, clause 3.1
MAX_EXTERNAL = 70  # rule PAPER-MAX-01, clause 3.1
MAX_TOTAL = 100  # rule COURSE-MAX-01, clause 3.1

EDGE_CASE_TAGS = (
    "attendance_at_threshold",
    "attendance_below_threshold",
    "marks_just_below_pass",
    "absent_result",
    "detained_student",
    "multiple_backlogs",
    "cgpa_at_placement_cutoff",
)


@dataclass
class Violation:
    table: str
    row: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.table}] {self.row}: {self.rule} -- {self.detail}"


@dataclass
class ValidationReport:
    violations: list[Violation]
    counts: dict[str, int]
    coverage: dict[str, Any]

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        lines = [f"rows: {self.counts}", f"coverage: {self.coverage}"]
        if self.ok:
            lines.append("no violations found")
        else:
            lines.append(f"{len(self.violations)} violation(s):")
            lines.extend(f"  - {violation}" for violation in self.violations)
        return "\n".join(lines)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def expected_result(total: int, external: int, max_marks: int = MAX_TOTAL) -> str:
    """EXAM-RULES-2025 clauses 3.2 and 3.3, expressed as code.

    The 30 percent in clause 3.3 applies to the **end-semester paper** (70 marks),
    not to the 100-mark course total, so the paper floor is 21 marks. Applying it
    to ``max_marks`` was an early mistake in this repository; it is fixed here and
    covered by a test.
    """
    paper_min = MAX_EXTERNAL * END_SEM_MIN_PCT / 100
    if total >= PASS_MARKS_MIN and external >= paper_min:
        return "PASS"
    return "FAIL"


SESSION_MONTH_ORDER = {"MAY": 1, "DEC": 2}


def session_key(session: str) -> tuple[int, int]:
    """``"2026-MAY"`` -> ``(2026, 1)`` and ``"2026-DEC"`` -> ``(2026, 2)``.

    A plain string comparison would put 2026-DEC before 2026-MAY, which would
    make a supplementary attempt look older than the regular attempt it replaces.
    """
    parts = str(session or "").split("-")
    year = int(parts[0]) if parts and parts[0].isdigit() else 0
    month = SESSION_MONTH_ORDER.get(parts[1].upper(), 0) if len(parts) > 1 else 0
    return year, month


def attempt_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        *session_key(str(row.get("exam_session", ""))),
        1 if str(row.get("exam_type", "")).upper() == "SUPPLEMENTARY" else 0,
    )


def latest_result_per_course(results: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in results:
        course = str(row.get("course_code", "")).strip()
        if course not in latest or attempt_key(row) >= attempt_key(latest[course]):
            latest[course] = row
    return latest


def active_backlog_count(results: Iterable[dict[str, Any]]) -> int:
    latest = latest_result_per_course(results)
    return len(
        [
            row
            for row in latest.values()
            if str(row.get("result", "")).upper() in {"FAIL", "ABSENT", "DETAINED"}
        ]
    )


def validate_dataset(
    students: list[dict[str, Any]],
    courses: list[dict[str, Any]],
    attendance: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> ValidationReport:
    violations: list[Violation] = []
    counts = {
        "students": len(students),
        "courses": len(courses),
        "attendance": len(attendance),
        "results": len(results),
    }

    course_codes = {str(row.get("course_code", "")).strip() for row in courses}
    student_ids = {str(row.get("student_id", "")).strip() for row in students}
    course_programmes = {str(row.get("course_code", "")).strip(): str(row.get("programme", "")).strip() for row in courses}
    student_programmes = {
        str(row.get("student_id", "")).strip(): str(row.get("programme", "")).strip() for row in students
    }

    # ---- students ----------------------------------------------------------
    for index, row in enumerate(students, start=1):
        label = f"row {index} ({row.get('student_id', '?')})"
        student_id = str(row.get("student_id", "")).strip()
        if not STUDENT_ID_RE.match(student_id):
            violations.append(Violation("students", label, "id format", "student_id must be S followed by 4 digits"))
        if student_id in RESERVED_STUDENT_IDS:
            violations.append(
                Violation("students", label, "reserved id", "S9000-S9999 are reserved for the judges")
            )
        if str(row.get("programme", "")).strip() not in PROGRAMMES:
            violations.append(
                Violation("students", label, "programme", f"programme must be one of {PROGRAMMES}")
            )
        if _to_int(row.get("batch_year")) not in BATCH_YEARS:
            violations.append(
                Violation("students", label, "batch_year", f"batch_year must be one of {BATCH_YEARS}")
            )
        semester = _to_int(row.get("current_semester"))
        if not 1 <= semester <= 10:
            violations.append(Violation("students", label, "current_semester", "must be between 1 and 10"))
        cgpa = _to_float(row.get("cgpa"), default=-1.0)
        if not 0.0 <= cgpa <= 10.0:
            violations.append(Violation("students", label, "cgpa", "must be between 0.00 and 10.00"))
        if not str(row.get("full_name", "")).strip():
            violations.append(Violation("students", label, "full_name", "must not be empty"))

    # ---- courses -----------------------------------------------------------
    for index, row in enumerate(courses, start=1):
        label = f"row {index} ({row.get('course_code', '?')})"
        course_code = str(row.get("course_code", "")).strip()
        if not COURSE_CODE_RE.match(course_code):
            violations.append(Violation("courses", label, "code format", "course codes look like CS201"))
        if course_code.startswith("JDG"):
            violations.append(
                Violation("courses", label, "reserved code", "course codes starting with JDG are reserved for judges")
            )
        if str(row.get("programme", "")).strip() not in PROGRAMMES + ("ALL",):
            violations.append(
                Violation("courses", label, "programme", "must match a students.programme value or be ALL")
            )
        if not 1 <= _to_int(row.get("semester")) <= 8:
            violations.append(Violation("courses", label, "semester", "must be between 1 and 8"))
        if _to_int(row.get("credits")) <= 0:
            violations.append(Violation("courses", label, "credits", "must be positive"))

    # ---- attendance --------------------------------------------------------
    attendance_by_student: dict[str, dict[str, dict[str, Any]]] = {}
    for index, row in enumerate(attendance, start=1):
        label = f"row {index} ({row.get('student_id', '?')}/{row.get('course_code', '?')})"
        student_id = str(row.get("student_id", "")).strip()
        course_code = str(row.get("course_code", "")).strip()
        if student_id not in student_ids:
            violations.append(Violation("attendance", label, "foreign key", f"unknown student {student_id}"))
        if course_code not in course_codes:
            violations.append(Violation("attendance", label, "foreign key", f"unknown course {course_code}"))
        programme = student_programmes.get(student_id)
        course_programme = course_programmes.get(course_code, "ALL")
        if programme and course_programme not in (programme, "ALL"):
            violations.append(
                Violation(
                    "attendance",
                    label,
                    "programme mismatch",
                    f"course {course_code} is offered to {course_programme}, student is {programme}",
                )
            )
        held = _to_int(row.get("classes_held"), default=-1)
        attended = _to_int(row.get("classes_attended"), default=-1)
        if held <= 0:
            violations.append(Violation("attendance", label, "classes held", "must be greater than zero"))
        if attended < 0:
            violations.append(Violation("attendance", label, "classes attended", "must not be negative"))
        if attended > held:
            violations.append(
                Violation("attendance", label, "attended > held", f"{attended} attended of {held} held")
            )
        attendance_by_student.setdefault(student_id, {})[course_code] = row

    # ---- results -----------------------------------------------------------
    results_by_student: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(results, start=1):
        label = f"row {index} ({row.get('student_id', '?')}/{row.get('course_code', '?')})"
        student_id = str(row.get("student_id", "")).strip()
        course_code = str(row.get("course_code", "")).strip()
        if student_id not in student_ids:
            violations.append(Violation("results", label, "foreign key", f"unknown student {student_id}"))
        if course_code not in course_codes:
            violations.append(Violation("results", label, "foreign key", f"unknown course {course_code}"))
        exam_type = str(row.get("exam_type", "")).strip().upper()
        if exam_type not in EXAM_TYPES:
            violations.append(Violation("results", label, "exam_type", f"must be one of {EXAM_TYPES}"))
        if not re.match(r"^\d{4}-(MAY|DEC)$", str(row.get("exam_session", "")).strip()):
            violations.append(
                Violation("results", label, "exam_session", "must look like 2026-MAY or 2026-DEC")
            )
        internal = _to_int(row.get("internal_marks"), default=-1)
        external = _to_int(row.get("external_marks"), default=-1)
        total = _to_int(row.get("total_marks"), default=-1)
        max_marks = _to_int(row.get("max_marks"), default=-1)
        result = str(row.get("result", "")).strip().upper()
        if not 0 <= internal <= MAX_INTERNAL:
            violations.append(Violation("results", label, "internal_marks", f"must be 0..{MAX_INTERNAL}"))
        if not 0 <= external <= MAX_EXTERNAL:
            violations.append(Violation("results", label, "external_marks", f"must be 0..{MAX_EXTERNAL}"))
        if max_marks != MAX_TOTAL:
            violations.append(Violation("results", label, "max_marks", f"must be {MAX_TOTAL}"))
        if total != internal + external:
            violations.append(
                Violation("results", label, "total != internal + external", f"{total} != {internal} + {external}")
            )
        if total > MAX_TOTAL:
            violations.append(Violation("results", label, "total_marks", "must not exceed max_marks"))
        if result not in RESULT_VALUES:
            violations.append(Violation("results", label, "result", f"must be one of {RESULT_VALUES}"))
        elif result in {"PASS", "FAIL"} and _to_float(row.get("internal_marks"), default=-1) >= 0:
            implied = expected_result(total, external)
            if result != implied:
                violations.append(
                    Violation(
                        "results",
                        label,
                        "result inconsistent with marks",
                        f"total={total} external={external} (paper floor "
                        f"{MAX_EXTERNAL * END_SEM_MIN_PCT / 100:g}) implies {implied}",
                    )
                )
        elif result == "ABSENT" and external != 0:
            violations.append(
                Violation("results", label, "absent marks", "an ABSENT result must carry 0 external marks")
            )
        results_by_student.setdefault(student_id, []).append(row)

    # ---- cross-file consistency -------------------------------------------
    for student in students:
        student_id = str(student.get("student_id", "")).strip()
        declared = _to_int(student.get("active_backlogs"), default=-1)
        computed = active_backlog_count(results_by_student.get(student_id, []))
        if declared != computed:
            violations.append(
                Violation(
                    "students",
                    student_id,
                    "active_backlogs inconsistent",
                    f"declared {declared}, computed {computed} from results",
                )
            )

    # ---- coverage and edge cases ------------------------------------------
    programmes = {str(row.get("programme", "")).strip() for row in students}
    batches = {_to_int(row.get("batch_year")) for row in students}
    coverage: dict[str, Any] = {
        "programmes": sorted(programmes),
        "batches": sorted(batches),
        "courses": len(course_codes),
        "students": len(students),
    }
    if len(students) < MIN_STUDENTS:
        violations.append(
            Violation("students", "dataset", "minimum rows", f"need at least {MIN_STUDENTS} students")
        )
    if len(programmes) < MIN_PROGRAMMES:
        violations.append(
            Violation("students", "dataset", "minimum programmes", f"need at least {MIN_PROGRAMMES}")
        )
    if len(batches) < MIN_BATCHES:
        violations.append(Violation("students", "dataset", "minimum batches", f"need at least {MIN_BATCHES}"))
    if len(course_codes) < MIN_COURSES:
        violations.append(Violation("courses", "dataset", "minimum courses", f"need at least {MIN_COURSES}"))

    edge_cases: dict[str, list[str]] = {tag: [] for tag in EDGE_CASE_TAGS}
    threshold_pct = 80.0
    for student in students:
        student_id = str(student.get("student_id", "")).strip()
        rows = attendance_by_student.get(student_id, {})
        percentages = [
            round(_to_int(row.get("classes_attended")) / _to_int(row.get("classes_held"), 1) * 100, 1)
            for row in rows.values()
            if _to_int(row.get("classes_held")) > 0
        ]
        if any(abs(pct - threshold_pct) < 0.05 for pct in percentages):
            edge_cases["attendance_at_threshold"].append(student_id)
        if any(threshold_pct - 5.0 < pct < threshold_pct for pct in percentages):
            edge_cases["attendance_below_threshold"].append(student_id)
        if abs(_to_float(student.get("cgpa")) - 6.0) < 0.005:
            edge_cases["cgpa_at_placement_cutoff"].append(student_id)
        student_results = results_by_student.get(student_id, [])
        if any(_to_int(row.get("total_marks")) in (38, 39) for row in student_results):
            edge_cases["marks_just_below_pass"].append(student_id)
        if any(str(row.get("result", "")).upper() == "ABSENT" for row in student_results):
            edge_cases["absent_result"].append(student_id)
        if any(str(row.get("result", "")).upper() == "DETAINED" for row in student_results):
            edge_cases["detained_student"].append(student_id)
        if active_backlog_count(student_results) >= 2:
            edge_cases["multiple_backlogs"].append(student_id)

    coverage["edge_cases"] = {tag: sorted(set(ids)) for tag, ids in edge_cases.items()}
    for tag, ids in edge_cases.items():
        if not ids:
            violations.append(
                Violation("dataset", "edge cases", "missing edge case", f"no student carries the {tag} edge case")
            )

    return ValidationReport(violations=violations, counts=counts, coverage=coverage)


def validate_csv_dir(directory: Path | str) -> ValidationReport:
    """Validate a directory of the four CSV files in the Annex C schema."""
    import csv

    directory = Path(directory)
    tables: dict[str, list[dict[str, Any]]] = {}
    violations: list[Violation] = []
    for table, required in REQUIRED_COLUMNS.items():
        path = directory / f"{table}.csv"
        if not path.exists():
            violations.append(Violation(table, str(path), "missing file", "file not found"))
            tables[table] = []
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing = [column for column in required if column not in header]
            if missing:
                violations.append(
                    Violation(table, path.name, "missing columns", f"required columns absent: {missing}")
                )
            tables[table] = [dict(row) for row in reader]

    report = validate_dataset(
        tables.get("students", []),
        tables.get("courses", []),
        tables.get("attendance", []),
        tables.get("results", []),
    )
    report.violations = violations + report.violations
    return report
