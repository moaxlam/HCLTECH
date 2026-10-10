#!/usr/bin/env python
"""Create and populate the SQLite database.

    python scripts/init_db.py                # create schema, rules, register, students
    python scripts/init_db.py --reset        # delete the database file first
    python scripts/init_db.py --skip-students

Steps, in order:

1. Annex C schema (required tables) plus ``source_register`` and ``audit_log``.
2. Rule registry from ``data/rule_registry_seed.csv`` -- every threshold a tool
   uses, each row carrying the clause it came from.
3. Source Register from ``data/source_register.csv`` (Annex B).
4. Student data from the four CSVs in ``data/`` (validated first).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from assa import db  # noqa: E402
from assa.config import get_settings  # noqa: E402
from load_students import load_directory  # noqa: E402


def load_source_register(conn, path: Path) -> int:
    if not path.exists():
        print(f"note: no source register at {path}")
        return 0
    written = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            existing = db.get_register_row(conn, row["doc_id"]) or {}
            db.upsert_register_row(
                conn,
                {
                    **row,
                    # Preserve how many chunks a document has after a re-ingest.
                    "chunks_indexed": existing.get("chunks_indexed", 0),
                },
            )
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the assistant database")
    parser.add_argument("--reset", action="store_true", help="delete the database file first")
    parser.add_argument("--skip-students", action="store_true")
    parser.add_argument("--students-dir", default=None)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    settings = get_settings()
    db_path = Path(args.db) if args.db else settings.sqlite_path
    if args.reset and db_path.exists():
        db_path.unlink()

    conn = db.connect(db_path)
    db.init_db(conn)

    rules_path = settings.data_dir / "rule_registry_seed.csv"
    rules = db.seed_rules_from_csv(conn, rules_path)
    print(f"rule registry rows: {rules}")

    register = load_source_register(conn, settings.source_register_path)
    print(f"source register rows: {register}")

    if not args.skip_students:
        students_dir = Path(args.students_dir) if args.students_dir else settings.data_dir
        try:
            load_directory(students_dir, conn)
        except SystemExit as exit_code:
            print(f"student load aborted ({exit_code})")
            return 1

    print("table counts:", db.table_counts(conn))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
