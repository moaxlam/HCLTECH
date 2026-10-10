#!/usr/bin/env python
"""Load students from CSV files in the fixed Annex C schema.

This is the loader the judges use (R "Test-student loader" in Section 6):

    python scripts/load_students.py --dir test_students/
    python scripts/load_students.py --dir test_students/ --replace

Files expected in the directory: ``students.csv``, ``courses.csv``,
``attendance.csv``, ``results.csv`` with the column names fixed by Annex C.
Every file is validated before anything is written; a single violation aborts the
load and nothing is inserted. Student IDs in the reserved range S9000-S9999 are
accepted, because that range is reserved for the judges.

The same path is available over HTTP as ``POST /admin/students`` (multipart CSV).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assa import db  # noqa: E402
from assa.config import get_settings  # noqa: E402
from assa.data_validation import active_backlog_count, validate_csv_dir  # noqa: E402

TABLES = ("students", "courses", "attendance", "results")
COLUMNS = {
    "students": ["student_id", "full_name", "programme", "batch_year", "current_semester", "cgpa", "active_backlogs"],
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

#: Added columns that Annex C allows; the loader tolerates them in judge files.
OPTIONAL_COLUMNS = {
    "students": ["gender", "email", "hostel_resident", "phone"],
}


def read_table(directory: Path, table: str) -> list[dict[str, Any]]:
    path = directory / f"{table}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_directory(
    directory: Path, conn, replace: bool = True, quiet: bool = False
) -> dict[str, int]:
    report = validate_csv_dir(directory)
    if not report.ok:
        print(f"Refusing to load {directory}: the dataset has {len(report.violations)} violation(s)")
        for violation in report.violations:
            print(f"  - {violation}")
        raise SystemExit(2)

    counts: dict[str, int] = {}
    with conn:
        for table in TABLES:
            rows = read_table(directory, table)
            counts[table] = len(rows)
            if not rows:
                continue
            columns = COLUMNS[table]
            statement = (
                f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) "
                f"VALUES ({','.join('?' for _ in columns)})"
            )
            conn.executemany(
                statement,
                [tuple(row.get(column, "") for column in columns) for row in rows],
            )

        # Keep students.active_backlogs consistent with the results just loaded.
        corrected = 0
        for student in read_table(directory, "students"):
            student_id = str(student.get("student_id", "")).strip()
            if not student_id:
                continue
            results = conn.execute(
                "SELECT * FROM results WHERE student_id = ?", (student_id,)
            ).fetchall()
            computed = active_backlog_count([dict(row) for row in results])
            if int(student.get("active_backlogs") or -1) != computed:
                conn.execute(
                    "UPDATE students SET active_backlogs = ? WHERE student_id = ?",
                    (computed, student_id),
                )
                corrected += 1
        if corrected:
            print(f"note: recomputed active_backlogs for {corrected} student(s) from the results table")
    counts["active_backlogs_corrected"] = corrected
    if not quiet:
        print(f"loaded from {directory}: {counts}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Load students in the Annex C schema into SQLite")
    parser.add_argument("--dir", required=True, help="directory with students.csv, courses.csv, attendance.csv, results.csv")
    parser.add_argument("--db", default=None, help="override the SQLite path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    conn = db.connect(args.db or settings.sqlite_path)
    db.init_db(conn)
    load_directory(Path(args.dir), conn, quiet=args.quiet)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
