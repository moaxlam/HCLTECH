"""SQLite layer.

Tables in ``REQUIRED_SCHEMA`` use exactly the names and columns fixed by Annex C
of the participant guide. Extra columns (``supersedes_rule_id``, ``notes`` on the
rule registry) and extra tables (``source_register``, ``audit_log``) are allowed
by Annex C ("You may add more; do not rename or remove these").
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import get_settings

REQUIRED_SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id       TEXT PRIMARY KEY,
    full_name        TEXT NOT NULL,
    programme        TEXT NOT NULL,
    batch_year       INTEGER NOT NULL,
    current_semester INTEGER NOT NULL,
    cgpa             REAL NOT NULL,
    active_backlogs  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    course_code TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    programme   TEXT NOT NULL,
    semester    INTEGER NOT NULL,
    credits     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    student_id       TEXT NOT NULL REFERENCES students(student_id),
    course_code      TEXT NOT NULL REFERENCES courses(course_code),
    classes_held     INTEGER NOT NULL,
    classes_attended INTEGER NOT NULL,
    PRIMARY KEY (student_id, course_code)
);

CREATE TABLE IF NOT EXISTS results (
    student_id     TEXT NOT NULL REFERENCES students(student_id),
    course_code    TEXT NOT NULL REFERENCES courses(course_code),
    exam_session   TEXT NOT NULL,
    exam_type      TEXT NOT NULL,
    internal_marks INTEGER NOT NULL,
    external_marks INTEGER NOT NULL,
    total_marks    INTEGER NOT NULL,
    max_marks      INTEGER NOT NULL,
    result         TEXT NOT NULL,
    PRIMARY KEY (student_id, course_code, exam_session, exam_type)
);

CREATE TABLE IF NOT EXISTS rule_registry (
    rule_id            TEXT PRIMARY KEY,
    description        TEXT NOT NULL,
    parameter          TEXT NOT NULL,
    operator           TEXT NOT NULL,
    value              TEXT NOT NULL,
    scope_programmes   TEXT NOT NULL DEFAULT 'ALL',
    scope_batches      TEXT NOT NULL DEFAULT 'ALL',
    effective_from     TEXT NOT NULL,
    effective_to       TEXT NOT NULL DEFAULT '',
    source_doc_id      TEXT NOT NULL,
    source_section     TEXT NOT NULL,
    supersedes_rule_id TEXT NOT NULL DEFAULT '',
    notes              TEXT NOT NULL DEFAULT ''
);
"""

EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_register (
    doc_id           TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    issuer           TEXT NOT NULL,
    authority_level  INTEGER NOT NULL,
    doc_type         TEXT NOT NULL,
    version          TEXT NOT NULL DEFAULT '1.0',
    effective_from   TEXT NOT NULL,
    effective_to     TEXT NOT NULL DEFAULT '',
    supersedes       TEXT NOT NULL DEFAULT '',
    scope_programmes TEXT NOT NULL DEFAULT 'ALL',
    scope_batches    TEXT NOT NULL DEFAULT 'ALL',
    provenance       TEXT NOT NULL DEFAULT '',
    retrieved_on     TEXT NOT NULL DEFAULT '',
    synthetic        TEXT NOT NULL DEFAULT 'N',
    chunks_indexed   INTEGER NOT NULL DEFAULT 0,
    ingested_at      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS audit_log (
    trace_id            TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    student_id          TEXT,
    question            TEXT NOT NULL DEFAULT '',
    question_category   TEXT NOT NULL DEFAULT '',
    sources_retrieved   TEXT NOT NULL DEFAULT '[]',
    precedence_decision TEXT NOT NULL DEFAULT '',
    tools_invoked       TEXT NOT NULL DEFAULT '[]',
    applied_rules       TEXT NOT NULL DEFAULT '[]',
    conflicts_detected  TEXT NOT NULL DEFAULT '[]',
    answer_type         TEXT NOT NULL DEFAULT '',
    answer              TEXT NOT NULL DEFAULT '',
    citations           TEXT NOT NULL DEFAULT '[]',
    model               TEXT NOT NULL DEFAULT '',
    llm_calls           INTEGER NOT NULL DEFAULT 0,
    tokens              INTEGER NOT NULL DEFAULT 0,
    latency_ms          INTEGER NOT NULL DEFAULT 0,
    guardrail_flags     TEXT NOT NULL DEFAULT '[]',
    notes               TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_results_student ON results(student_id);
CREATE INDEX IF NOT EXISTS idx_rules_parameter ON rule_registry(parameter);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else get_settings().sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(REQUIRED_SCHEMA)
    conn.executescript(EXTRA_SCHEMA)
    conn.commit()


def seed_rules_from_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    """Idempotently load the rule registry. Returns the number of rows written."""
    import csv

    if not Path(csv_path).exists():
        return 0
    written = 0
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            conn.execute(
                """
                INSERT INTO rule_registry (
                    rule_id, description, parameter, operator, value, scope_programmes,
                    scope_batches, effective_from, effective_to, source_doc_id,
                    source_section, supersedes_rule_id, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    description=excluded.description,
                    parameter=excluded.parameter,
                    operator=excluded.operator,
                    value=excluded.value,
                    scope_programmes=excluded.scope_programmes,
                    scope_batches=excluded.scope_batches,
                    effective_from=excluded.effective_from,
                    effective_to=excluded.effective_to,
                    source_doc_id=excluded.source_doc_id,
                    source_section=excluded.source_section,
                    supersedes_rule_id=excluded.supersedes_rule_id,
                    notes=excluded.notes
                """,
                (
                    row["rule_id"],
                    row.get("description", ""),
                    row["parameter"],
                    row["operator"],
                    row["value"],
                    row.get("scope_programmes", "ALL") or "ALL",
                    row.get("scope_batches", "ALL") or "ALL",
                    row["effective_from"],
                    row.get("effective_to", "") or "",
                    row["source_doc_id"],
                    row.get("source_section", "") or "",
                    row.get("supersedes_rule_id", "") or "",
                    row.get("notes", "") or "",
                ),
            )
            written += 1
    conn.commit()
    return written


def upsert_register_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    columns = [
        "doc_id",
        "title",
        "issuer",
        "authority_level",
        "doc_type",
        "version",
        "effective_from",
        "effective_to",
        "supersedes",
        "scope_programmes",
        "scope_batches",
        "provenance",
        "retrieved_on",
        "synthetic",
    ]
    values = {column: row.get(column, "") for column in columns}
    values["authority_level"] = int(values.get("authority_level") or 3)
    values["doc_id"] = str(values["doc_id"])
    values["chunks_indexed"] = int(row.get("chunks_indexed", 0) or 0)
    values["ingested_at"] = row.get("ingested_at") or utc_now_iso()
    placeholders = ",".join("?" for _ in range(len(columns) + 2))
    assignments = ",".join(f"{column}=excluded.{column}" for column in columns)
    conn.execute(
        f"""
        INSERT INTO source_register ({",".join(columns)}, chunks_indexed, ingested_at)
        VALUES ({placeholders})
        ON CONFLICT(doc_id) DO UPDATE SET {assignments},
            chunks_indexed=excluded.chunks_indexed,
            ingested_at=excluded.ingested_at
        """,
        tuple(values[column] for column in columns) + (values["chunks_indexed"], values["ingested_at"]),
    )
    conn.commit()


def list_register_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM source_register ORDER BY authority_level ASC, effective_from DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_register_row(conn: sqlite3.Connection, doc_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM source_register WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def delete_register_row(conn: sqlite3.Connection, doc_id: str) -> None:
    """Remove a document from the Source Register (used when an ingest is undone)."""
    conn.execute("DELETE FROM source_register WHERE doc_id = ?", (doc_id,))
    conn.commit()


def all_documents(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every registered document, including ones not yet effective."""
    return list_register_rows(conn)


def insert_audit(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    json_fields = (
        "sources_retrieved",
        "tools_invoked",
        "applied_rules",
        "conflicts_detected",
        "citations",
        "guardrail_flags",
    )
    payload = {**record}
    for field in json_fields:
        payload[field] = json.dumps(payload.get(field, []), default=str)
    conn.execute(
        """
        INSERT INTO audit_log (
            trace_id, timestamp, student_id, question, question_category, sources_retrieved,
            precedence_decision, tools_invoked, applied_rules, conflicts_detected, answer_type,
            answer, citations, model, llm_calls, tokens, latency_ms, guardrail_flags, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(trace_id) DO UPDATE SET
            timestamp=excluded.timestamp, answer=excluded.answer,
            answer_type=excluded.answer_type, notes=excluded.notes
        """,
        (
            payload["trace_id"],
            payload.get("timestamp") or utc_now_iso(),
            payload.get("student_id"),
            payload.get("question", ""),
            payload.get("question_category", ""),
            payload["sources_retrieved"],
            payload.get("precedence_decision", ""),
            payload["tools_invoked"],
            payload["applied_rules"],
            payload["conflicts_detected"],
            payload.get("answer_type", ""),
            payload.get("answer", ""),
            payload["citations"],
            payload.get("model", ""),
            int(payload.get("llm_calls", 0) or 0),
            int(payload.get("tokens", 0) or 0),
            int(payload.get("latency_ms", 0) or 0),
            payload["guardrail_flags"],
            payload.get("notes", ""),
        ),
    )
    conn.commit()


def get_audit(conn: sqlite3.Connection, trace_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM audit_log WHERE trace_id = ?", (trace_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    for field in (
        "sources_retrieved",
        "tools_invoked",
        "applied_rules",
        "conflicts_detected",
        "citations",
        "guardrail_flags",
    ):
        record[field] = json.loads(record[field] or "[]")
    return record


def fetch_all(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in ("students", "courses", "attendance", "results", "rule_registry", "source_register"):
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
        except sqlite3.Error:
            counts[table] = 0
    return counts


def executemany(conn: sqlite3.Connection, sql: str, rows: Iterable[Sequence[Any]]) -> None:
    conn.executemany(sql, rows)
    conn.commit()
