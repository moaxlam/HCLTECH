#!/usr/bin/env python
"""Validate a synthetic student dataset against the Annex C schema and logic.

    python scripts/validate_students.py                     # validates data/
    python scripts/validate_students.py --dir test_students/
    python scripts/validate_students.py --json              # machine-readable report

Exit code 0 means no violations. Exit code 1 means at least one violation, and
every violation is printed with the table, row, rule and detail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assa.data_validation import validate_csv_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the synthetic student dataset")
    parser.add_argument("--dir", default=str(PROJECT_ROOT / "data"), help="directory holding the four CSV files")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    directory = Path(args.dir)
    report = validate_csv_dir(directory)

    if args.json:
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "ok": report.ok,
                    "counts": report.counts,
                    "coverage": report.coverage,
                    "violations": [str(violation) for violation in report.violations],
                },
                indent=2,
            )
        )
    else:
        print(f"Validating {directory}")
        print(report.summary())
        edge_cases = report.coverage.get("edge_cases", {})
        if edge_cases:
            print("edge cases:")
            for tag, ids in edge_cases.items():
                print(f"  {tag}: {', '.join(ids) if ids else 'MISSING'}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
