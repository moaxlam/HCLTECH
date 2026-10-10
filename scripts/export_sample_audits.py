#!/usr/bin/env python
"""Export three sample audit records for different answer types.

The deliverables checklist asks for three sample audit records covering different
answer types. This script asks one question of each kind, keeps the audit record
of the responses and writes ``evaluation/sample_audits.json``:

* a cited policy answer (``retrieved_fact``)
* a tool-based eligibility answer (``calculated``)
* an unanswerable question (``not_found``)

Run with ``MOCK_LLM=true`` for a deterministic set, or with a model running to
record the model name and token counts in the audit.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from assa import db  # noqa: E402
from assa.config import get_settings  # noqa: E402
from assa.graph import AssistantWorkflow, run_ask  # noqa: E402
from assa.llm import LLMClient  # noqa: E402
from assa.schemas import AskRequest  # noqa: E402
from assa.vectorstore import VectorStore  # noqa: E402

QUESTIONS = [
    ("retrieved_fact", "What is the minimum attendance required to appear for the end-semester examination?", None),
    ("calculated", "Am I eligible to appear for the end-semester examination in CS201?", "S2002"),
    ("not_found", "What is the scholarship available for studying at a campus in Antarctica?", None),
]


def main() -> int:
    settings = get_settings()
    conn = db.connect(settings.sqlite_path)
    db.init_db(conn)
    store = VectorStore()
    if store.count() == 0:
        print("the vector store is empty; run python scripts/ingest_documents.py first")
        return 1

    workflow = AssistantWorkflow(conn, store, LLMClient())
    samples = []
    for expected, question, student_id in QUESTIONS:
        response = run_ask(
            conn,
            store,
            AskRequest(question=question, as_of_date=date.today()),
            student_id,
            workflow=workflow,
        )
        record = db.get_audit(conn, response.trace_id)
        status = "as expected" if response.answer_type == expected else f"expected {expected}"
        print(f"{response.answer_type:16s} {response.trace_id}  {status}")
        samples.append(
            {
                "expected_answer_type": expected,
                "question": question,
                "student_id": student_id,
                "response": response.model_dump(mode="json"),
                "audit_record": record,
            }
        )

    out = PROJECT_ROOT / "evaluation" / "sample_audits.json"
    out.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
