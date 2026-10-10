#!/usr/bin/env python
"""Add or change one threshold in the rule registry, with its cited clause.

This is the documented procedure for "what happens when a new circular changes a
threshold". Nothing in the code changes: a row is added and the row it replaces
is named, so the precedence engine picks the new value up automatically and the
old value stays auditable.

    python scripts/add_rule.py \
        --rule-id ATT-MIN-03 --parameter min_attendance_pct --operator ">=" --value 85 \
        --description "Minimum attendance after circular ACAD-2026-12" \
        --source-doc-id ACAD-2026-12 --source-section 1.1 \
        --effective-from 2026-12-01 --supersedes ATT-MIN-02 \
        --scope-programmes ALL --scope-batches 2023+

The rule is rejected if the source document is not in the Source Register,
because every threshold must trace back to a cited clause (Annex C).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assa import db  # noqa: E402
from assa.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Add a rule to the registry")
    parser.add_argument("--rule-id", required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--operator", default=">=", choices=[">=", "<=", "==", "between"])
    parser.add_argument("--value", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--source-doc-id", required=True)
    parser.add_argument("--source-section", required=True)
    parser.add_argument("--effective-from", required=True)
    parser.add_argument("--effective-to", default="")
    parser.add_argument("--supersedes", default="", help="rule_id this row replaces")
    parser.add_argument("--scope-programmes", default="ALL")
    parser.add_argument("--scope-batches", default="ALL")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    settings = get_settings()
    conn = db.connect(settings.sqlite_path)
    db.init_db(conn)

    document = db.get_register_row(conn, args.source_doc_id)
    if document is None:
        print(
            f"refused: source document {args.source_doc_id!r} is not in the Source Register. "
            "Ingest the document and register it first, so the rule can cite a clause."
        )
        return 1

    conn.execute(
        """
        INSERT INTO rule_registry (
            rule_id, description, parameter, operator, value, scope_programmes, scope_batches,
            effective_from, effective_to, source_doc_id, source_section, supersedes_rule_id, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(rule_id) DO UPDATE SET
            description=excluded.description, parameter=excluded.parameter, operator=excluded.operator,
            value=excluded.value, scope_programmes=excluded.scope_programmes,
            scope_batches=excluded.scope_batches, effective_from=excluded.effective_from,
            effective_to=excluded.effective_to, source_doc_id=excluded.source_doc_id,
            source_section=excluded.source_section, supersedes_rule_id=excluded.supersedes_rule_id,
            notes=excluded.notes
        """,
        (
            args.rule_id,
            args.description,
            args.parameter,
            args.operator,
            args.value,
            args.scope_programmes,
            args.scope_batches,
            args.effective_from,
            args.effective_to,
            args.source_doc_id,
            args.source_section,
            args.supersedes,
            args.notes,
        ),
    )
    conn.commit()
    print(
        f"rule {args.rule_id} registered: {args.parameter} {args.operator} {args.value}, "
        f"citing {args.source_doc_id}#{args.source_section}, effective {args.effective_from}"
    )
    if args.supersedes:
        print(f"it supersedes {args.supersedes}; the precedence policy will prefer it from {args.effective_from}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
