#!/usr/bin/env python
"""Ingest every document in the Source Register into the vector store.

    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --reset        # clear the collection first
    python scripts/ingest_documents.py --dir documents

The Source Register CSV (Annex B) is the list of documents to ingest. A document
file is matched to a register row by the ``doc_id`` prefix of its file name, for
example ``ACAD-2026-08_minimum_attendance_circular.md`` for ``ACAD-2026-08``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assa import db  # noqa: E402
from assa.config import get_settings  # noqa: E402
from assa.vectorstore import VectorStore, ingest_document  # noqa: E402


def load_register_into_db(conn, settings) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not settings.source_register_path.exists():
        print(f"no Source Register at {settings.source_register_path}")
        return rows
    with settings.source_register_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row.setdefault("retrieved_on", "")
            existing = db.get_register_row(conn, row["doc_id"]) or {}
            db.upsert_register_row(conn, {**row, "chunks_indexed": existing.get("chunks_indexed", 0)})
            rows[row["doc_id"]] = row
    return rows


def ingest_registered_documents(conn, settings, store=None, directory: Path | None = None) -> list[dict]:
    store = store or VectorStore()
    directory = Path(directory or settings.documents_dir)
    rows = load_register_into_db(conn, settings)
    outcomes: list[dict] = []
    for doc_id, metadata in sorted(rows.items()):
        candidates = sorted(directory.glob(f"{doc_id}*")) + sorted(
            (settings.data_dir / "uploads").glob(f"{doc_id}*")
        )
        if not candidates:
            print(f"  ! {doc_id}: no file found in {directory}")
            continue
        try:
            outcome = ingest_document(conn, store, candidates[0], metadata)
        except ValueError as exc:
            print(f"  ! {doc_id}: {exc}")
            outcomes.append({"doc_id": doc_id, "chunks_indexed": 0, "status": "failed", "warnings": [str(exc)]})
            continue
        outcomes.append(outcome.as_dict())
        print(f"  · {outcome.doc_id}: {outcome.chunks_indexed} chunk(s), {outcome.status}")
        for warning in outcome.warnings:
            print(f"      warning: {warning}")
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the registered university documents")
    parser.add_argument("--dir", default=None, help="directory holding the document files")
    parser.add_argument("--reset", action="store_true", help="delete the whole collection first")
    args = parser.parse_args()

    settings = get_settings()
    conn = db.connect(settings.sqlite_path)
    db.init_db(conn)
    store = VectorStore()
    if args.reset:
        store._client.delete_collection(store.collection_name)  # noqa: SLF001 - explicit reset
        store = VectorStore()
        print("collection cleared")

    outcomes = ingest_registered_documents(conn, settings, store=store, directory=Path(args.dir) if args.dir else None)
    total = sum(outcome["chunks_indexed"] for outcome in outcomes)
    print(f"\nindexed {total} chunk(s) from {len(outcomes)} document(s); collection now holds {store.count()}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
