"""Test fixtures.

The suite runs with ``MOCK_LLM=true`` and the deterministic hashing embedder so
that no model and no model download are needed. That exercises the same
graph, retrieval filtering, precedence engine, tools and grounding checks that
run in production; only the wording of the explanation and the semantic quality
of the vectors are out of scope here (they are measured by the evaluation
harness instead).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Must be set before assa.config is imported anywhere.
os.environ["MOCK_LLM"] = "true"

# The suite uses the real embedding model (all-MiniLM-L6-v2, cached after the first
# run) so that retrieval filtering, the relevance floor and abstention are exercised
# for real. ASSA_TEST_FAST=1 switches to the deterministic hashing embedder, which
# needs no model but has no meaningful ranking: the abstention tests skip themselves.
FAST_MODE = bool(os.environ.get("ASSA_TEST_FAST"))
if FAST_MODE:
    os.environ["ASSA_EMBEDDING_PROVIDER"] = "hash"
    os.environ["ASSA_MIN_SCORE"] = "0.0"

DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("assa")
    os.environ["ASSA_DATA_DIR"] = str(workdir / "data")
    os.environ["ASSA_SQLITE_PATH"] = str(workdir / "data" / "university.db")
    os.environ["ASSA_CHROMA_DIR"] = str(workdir / "data" / "chroma")
    os.environ["ASSA_DOCUMENTS_DIR"] = str(PROJECT_ROOT / "documents")
    Path(os.environ["ASSA_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

    from assa.config import reset_settings_cache

    reset_settings_cache()
    from assa.config import get_settings

    resolved = get_settings()
    # The generated dataset and register live with the repository; copy them in.
    for name in ("students.csv", "courses.csv", "attendance.csv", "results.csv", "source_register.csv"):
        (resolved.data_dir / name).write_text((DATA_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    (resolved.data_dir / "rule_registry_seed.csv").write_text(
        (DATA_DIR / "rule_registry_seed.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return resolved


@pytest.fixture(scope="session")
def conn(settings):
    from assa import db
    from load_students import load_directory

    connection = db.connect(settings.sqlite_path)
    db.init_db(connection)
    db.seed_rules_from_csv(connection, settings.data_dir / "rule_registry_seed.csv")
    with (settings.data_dir / "source_register.csv").open(newline="", encoding="utf-8") as handle:
        import csv

        for row in csv.DictReader(handle):
            db.upsert_register_row(connection, row)
    load_directory(settings.data_dir, connection, quiet=True)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def store(conn, settings):
    from assa.vectorstore import VectorStore
    from ingest_documents import ingest_registered_documents

    vector_store = VectorStore()
    ingest_registered_documents(conn, settings, store=vector_store)
    return vector_store


@pytest.fixture(scope="session")
def workflow(conn, store):
    from assa.graph import AssistantWorkflow
    from assa.llm import LLMClient

    return AssistantWorkflow(conn, store, LLMClient())


@pytest.fixture(scope="session")
def ask(conn, store, workflow):
    from assa.schemas import AskRequest
    from assa.graph import run_ask
    from datetime import date

    def run(question: str, student_id: str | None = None, as_of: str = "2026-10-06"):
        return run_ask(
            conn,
            store,
            AskRequest(question=question, as_of_date=date.fromisoformat(as_of)),
            student_id,
            workflow=workflow,
        )

    return run
