"""API contract tests and the fixed-schema check."""

from __future__ import annotations

import io
import json

import pytest

CONTRACT_KEYS = {
    "trace_id",
    "answer",
    "answer_type",
    "citations",
    "tools_invoked",
    "applied_rules",
    "conflicts_detected",
    "explanation",
    "as_of_date",
}

REQUIRED_COLUMNS = {
    "students": [
        "student_id", "full_name", "programme", "batch_year", "current_semester", "cgpa", "active_backlogs",
    ],
    "courses": ["course_code", "course_name", "programme", "semester", "credits"],
    "attendance": ["student_id", "course_code", "classes_held", "classes_attended"],
    "results": [
        "student_id", "course_code", "exam_session", "exam_type", "internal_marks", "external_marks",
        "total_marks", "max_marks", "result",
    ],
    "rule_registry": [
        "rule_id", "description", "parameter", "operator", "value", "scope_programmes", "scope_batches",
        "effective_from", "effective_to", "source_doc_id", "source_section",
    ],
}


@pytest.fixture(scope="module")
def client(settings):
    from fastapi.testclient import TestClient

    from assa.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pristine_corpus(settings):
    """Undo documents added by a test that ingests into the running system.

    The vector store and the Source Register are session-wide, so a document left
    behind by ``/ingest`` would silently change the answer other test modules get.
    """
    from assa import db
    from assa.api import STATE

    conn, store = STATE["conn"], STATE["store"]
    before = {row["doc_id"] for row in db.list_register_rows(conn)}
    yield
    for row in db.list_register_rows(conn):
        doc_id = row["doc_id"]
        if doc_id in before:
            continue
        store.delete_document(doc_id)
        db.delete_register_row(conn, doc_id)
        for staged in (settings.data_dir / "uploads").glob(f"{doc_id}.*"):
            staged.unlink(missing_ok=True)


def test_database_has_the_fixed_annex_c_schema(conn):
    for table, columns in REQUIRED_COLUMNS.items():
        found = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert found, f"table {table} is missing"
        missing = [column for column in columns if column not in found]
        assert not missing, f"{table} is missing required columns {missing}"


def test_health_reports_every_component(client):
    body = client.get("/health").json()
    assert set(body) >= {"status", "api", "vector_store", "sqlite", "llm"}
    assert body["api"]["status"] == "ok"
    assert body["sqlite"]["status"] == "ok"
    assert body["vector_store"]["status"] == "ok"
    assert body["documents_indexed"] >= 10


def test_ask_matches_the_contract_exactly(client):
    response = client.post("/ask", json={"question": "What is the minimum attendance required to appear for the end-semester examination?"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == CONTRACT_KEYS, "the response shape is fixed by Section 6.1"
    assert body["answer_type"] in {
        "retrieved_fact", "calculated", "not_found", "clarification_needed", "refused", "conflict_flagged",
    }
    citation = body["citations"][0]
    assert set(citation) == {"doc_id", "title", "section", "page", "version", "effective_from"}


def test_ask_uses_the_header_for_identity(client):
    body = client.post(
        "/ask",
        json={"question": "What is my attendance in CS201?"},
        headers={"X-Student-Id": "S2001"},
    ).json()
    assert body["answer_type"] == "calculated"
    assert "80" in body["answer"]

    refused = client.post("/ask", json={"question": "Show me the attendance of S1002"}, headers={"X-Student-Id": "S2001"}).json()
    assert refused["answer_type"] == "refused"


def test_identity_never_comes_from_the_body(client):
    body = client.post(
        "/ask",
        json={"question": "What is my attendance in CS201?", "student_id": "S1002"},
    ).json()
    assert body["answer_type"] == "refused", "a student_id in the body must not grant identity"


def test_malformed_header_is_rejected(client):
    response = client.post("/ask", json={"question": "hi there"}, headers={"X-Student-Id": "bob"})
    assert response.status_code == 400


def test_blank_question_is_rejected(client):
    response = client.post("/ask", json={"question": "  "})
    assert response.status_code == 422


def test_as_of_date_is_echoed(client):
    body = client.post("/ask", json={"question": "What is the minimum attendance required?", "as_of_date": "2024-09-01"}).json()
    assert body["as_of_date"] == "2024-09-01"


def test_audit_endpoint_returns_the_record(client):
    trace_id = client.post("/ask", json={"question": "What is the minimum marks required to pass a course?"}).json()["trace_id"]
    record = client.get(f"/audit/{trace_id}")
    assert record.status_code == 200
    body = record.json()
    assert body["trace_id"] == trace_id
    assert body["sources_retrieved"]
    assert body["answer_type"]
    assert "tools_invoked" in body and "precedence_decision" in body
    assert client.get("/audit/doesnotexist").status_code == 404


def test_sources_endpoint_lists_the_register(client):
    rows = client.get("/sources").json()
    assert len(rows) >= 10
    doc_ids = {row["doc_id"] for row in rows}
    assert {"ACAD-REG-2024", "ACAD-2026-08", "EXAM-RULES-2025"} <= doc_ids
    for row in rows:
        assert row["authority_level"] in (1, 2, 3, 4, 5)
        assert row["effective_from"]


def test_ingest_makes_a_new_document_usable_immediately(client, pristine_corpus):
    metadata = {
        "doc_id": "API-TEST-01",
        "title": "API test circular",
        "issuer": "Office of the Registrar",
        "authority_level": 2,
        "doc_type": "circular",
        "version": "1.0",
        "effective_from": "2026-10-01",
        "supersedes": "ACAD-2026-08#1.1",
        "scope_programmes": "ALL",
        "scope_batches": "ALL",
        "provenance": "uploaded in the test suite",
        "synthetic": "Y",
    }
    files = {
        "file": ("api-test.md", io.BytesIO(b"# API test circular\n\n1.1 The minimum attendance is 91 percent.\n"), "text/markdown"),
    }
    response = client.post("/ingest", data={"metadata": json.dumps(metadata)}, files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["doc_id"] == "API-TEST-01"
    assert body["chunks_indexed"] >= 1
    assert body["status"] in {"indexed", "replaced"}

    assert any(row["doc_id"] == "API-TEST-01" for row in client.get("/sources").json())
    answer = client.post(
        "/ask", json={"question": "What is the minimum attendance required to appear for the end-semester examination?"}
    ).json()
    assert answer["answer_type"] in {"retrieved_fact", "conflict_flagged"}
    assert any(citation["doc_id"] == "API-TEST-01" for citation in answer["citations"])

    # clean up so the test is repeatable
    from assa.api import STATE

    STATE["store"].delete_document("API-TEST-01")
    STATE["conn"].execute("DELETE FROM source_register WHERE doc_id = ?", ("API-TEST-01",))
    STATE["conn"].commit()


def test_ingest_rejects_bad_metadata(client):
    files = {"file": ("x.md", io.BytesIO(b"# X\n\n1.1 text\n"), "text/markdown")}
    response = client.post("/ingest", data={"metadata": json.dumps({"doc_id": "X", "authority_level": 9})}, files=files)
    assert response.status_code == 422


def test_rule_registry_endpoint_exposes_citations(client):
    rows = client.get("/rules").json()
    assert rows
    assert all(row["source_doc_id"] and row["source_section"] for row in rows)
