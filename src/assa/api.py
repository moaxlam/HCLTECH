"""FastAPI application implementing the mandatory API contract (Section 6).

    POST /ask                  ask a question (X-Student-Id identifies the student)
    POST /ingest               add a document while running (multipart + metadata JSON)
    GET  /health               readiness of API, vector store, SQLite and LLM
    GET  /audit/{trace_id}     full audit record for one response
    GET  /sources              the Source Register
    POST /admin/students       load judge test students (Annex C CSV files)

Two extra read-only endpoints support the UI and the presentation:
``GET /rules`` (rule registry with citations) and ``GET /audit`` (recent traces).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import sqlite3
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, guardrails, rules
from .config import get_settings
from .embeddings import embedder_health
from .graph import AssistantWorkflow, run_ask
from .llm import LLMClient
from .schemas import (
    AskRequest,
    AskResponse,
    AuditRecord,
    ComponentHealth,
    HealthResponse,
    IngestMetadata,
    IngestResponse,
    SourceRegisterRow,
)
from .vectorstore import VectorStore, ingest_document

STATE: dict[str, Any] = {}


def _auto_ingest(conn: sqlite3.Connection, store: VectorStore) -> list[dict[str, Any]]:
    """Ingest registered documents that are not in the vector store yet."""
    settings = get_settings()
    outcomes = []
    for row in db.list_register_rows(conn):
        if int(row.get("chunks_indexed") or 0) > 0 and row["doc_id"] in store.documents_indexed():
            continue
        candidates = sorted(settings.documents_dir.glob(f"{row['doc_id']}*"))
        if not candidates:
            continue
        outcome = ingest_document(conn, store, candidates[0], row)
        outcomes.append(outcome.as_dict())
    return outcomes


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = db.connect(settings.sqlite_path)
    db.init_db(conn)
    db.seed_rules_from_csv(conn, settings.data_dir / "rule_registry_seed.csv")
    store = VectorStore()
    uploaded = settings.data_dir / "uploads"
    uploaded.mkdir(parents=True, exist_ok=True)
    STATE.update(
        {
            "conn": conn,
            "store": store,
            "llm": LLMClient(),
            "uploads": uploaded,
            "workflow": None,
        }
    )
    STATE["workflow"] = AssistantWorkflow(conn, store, STATE["llm"])
    if store.count() == 0:
        outcomes = _auto_ingest(conn, store)
        if outcomes:
            print(f"[startup] ingested {len(outcomes)} registered document(s)")
    yield
    conn.close()


app = FastAPI(
    title="AI-Powered University Student Services Assistant",
    version="1.0.0",
    description=(
        "Grounded, cited answers over authorised university documents, with deterministic tools "
        "for eligibility and calculations. HCLTech Future Ready AI Engineer Hackathon, Use Case 2."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rewrite_api_prefix(request: Request, call_next):
    if request.scope.get("path", "").startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
    return await call_next(request)


def _services() -> tuple[sqlite3.Connection, VectorStore, AssistantWorkflow]:
    conn = STATE.get("conn")
    store = STATE.get("store")
    workflow = STATE.get("workflow")
    if conn is None or store is None or workflow is None:
        raise HTTPException(status_code=503, detail="service is still starting")
    return conn, store, workflow


# --------------------------------------------------------------------------- #
# POST /ask
# --------------------------------------------------------------------------- #
@app.post("/ask", response_model=AskResponse, tags=["contract"])
def ask(
    request: AskRequest,
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
) -> AskResponse:
    """Answer one question. Identity comes only from the header (R7)."""
    conn, store, workflow = _services()
    if x_student_id:
        x_student_id = x_student_id.strip().upper()
        if not guardrails.find_student_ids(x_student_id):
            raise HTTPException(status_code=400, detail="X-Student-Id must look like S1001")
    return run_ask(conn, store, request, x_student_id, workflow=workflow)


# --------------------------------------------------------------------------- #
# POST /ingest
# --------------------------------------------------------------------------- #
@app.post("/ingest", response_model=IngestResponse, tags=["contract"])
async def ingest(
    file: UploadFile = File(..., description="The document: .md, .txt, .pdf or .docx"),
    metadata: str = Form(..., description="Source Register metadata (Annex B) as a JSON object"),
) -> IngestResponse:
    """Ingest a document while the system is running, usable immediately (R11)."""
    conn, store, _ = _services()
    try:
        payload = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"metadata is not valid JSON: {exc}") from exc
    try:
        parsed = IngestMetadata(**payload)
    except Exception as exc:  # pydantic ValidationError
        raise HTTPException(status_code=422, detail=f"metadata failed validation: {exc}") from exc

    suffix = Path(file.filename or "document.md").suffix or ".md"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=STATE["uploads"]) as handle:
        shutil.copyfileobj(file.file, handle)
        staged = Path(handle.name)

    try:
        outcome = ingest_document(conn, store, staged, parsed.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Keep a copy so a restart can re-index the same document.
    final_path = Path(STATE["uploads"]) / f"{parsed.doc_id}{suffix}"
    shutil.move(str(staged), final_path)
    return IngestResponse(**outcome.as_dict())


# --------------------------------------------------------------------------- #
# GET /health, /audit, /sources
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["contract"])
def health() -> HealthResponse:
    settings = get_settings()
    conn, store, _ = _services()
    llm_ok, llm_detail = LLMClient().health()
    store_ok, store_detail = store.health()
    embedder_ok, embedder_detail = embedder_health()
    try:
        counts = db.table_counts(conn)
        sqlite_ok, sqlite_detail = True, f"{counts.get('students', 0)} students, {counts.get('rule_registry', 0)} rules"
    except Exception as exc:  # pragma: no cover
        sqlite_ok, sqlite_detail = False, str(exc)

    vector_detail = f"{store_detail}; embeddings: {embedder_detail}"
    status = "ok"
    if not (sqlite_ok and store_ok):
        status = "down"
    elif not (llm_ok and embedder_ok):
        status = "degraded"

    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        api=ComponentHealth(status="ok", detail=f"v{app.version}"),
        vector_store=ComponentHealth(status="ok" if store_ok else "down", detail=vector_detail),
        sqlite=ComponentHealth(status="ok" if sqlite_ok else "down", detail=sqlite_detail),
        llm=ComponentHealth(status="ok" if llm_ok else "degraded", detail=llm_detail),
        documents_indexed=len(store.documents_indexed()),
        mock_llm=settings.mock_llm,
    )


@app.get("/audit/{trace_id}", response_model=AuditRecord, tags=["contract"])
def audit(trace_id: str) -> AuditRecord:
    conn, _, _ = _services()
    record = db.get_audit(conn, trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no audit record for trace_id {trace_id}")
    return AuditRecord(**record)


@app.get("/audit", tags=["ui"])
def recent_audit(limit: int = 20) -> list[dict[str, Any]]:
    conn, _, _ = _services()
    rows = db.fetch_all(
        conn,
        "SELECT trace_id, timestamp, student_id, question, question_category, answer_type, latency_ms, model "
        "FROM audit_log ORDER BY timestamp DESC LIMIT ?",
        (max(1, min(limit, 100)),),
    )
    return rows


@app.get("/sources", response_model=list[SourceRegisterRow], tags=["contract"])
def sources() -> list[SourceRegisterRow]:
    conn, _, _ = _services()
    return [SourceRegisterRow(**row) for row in db.list_register_rows(conn)]


@app.get("/rules", tags=["ui"])
def rule_registry() -> list[dict[str, Any]]:
    conn, _, _ = _services()
    return rules.rule_summary(conn)


@app.get("/evaluation", tags=["ui"])
def evaluation() -> dict[str, Any]:
    """Serve the committed evaluation report if the harness has been run."""
    settings = get_settings()
    report_path = settings.base_dir / "evaluation" / "report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="no evaluation report yet; run python scripts/run_evaluation.py",
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# POST /admin/students  (the loader, over HTTP)
# --------------------------------------------------------------------------- #
@app.post("/admin/students", tags=["contract"])
async def load_students(
    students: UploadFile = File(...),
    courses: UploadFile = File(...),
    attendance: UploadFile = File(...),
    results: UploadFile = File(...),
) -> dict[str, Any]:
    """Load judge test students from the four Annex C CSV files."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from load_students import load_directory  # type: ignore

    conn, _, workflow = _services()
    with tempfile.TemporaryDirectory(dir=STATE["uploads"]) as directory:
        target = Path(directory)
        for name, upload in (
            ("students", students),
            ("courses", courses),
            ("attendance", attendance),
            ("results", results),
        ):
            (target / f"{name}.csv").write_bytes(await upload.read())
        try:
            counts = load_directory(target, conn, quiet=True)
        except SystemExit as exit_code:
            raise HTTPException(
                status_code=422, detail=f"the uploaded dataset failed validation (exit {exit_code})"
            ) from exit_code
    workflow.retriever.invalidate()
    return {"status": "loaded", **counts}


_UI_DIST = get_settings().base_dir / "ui" / "dist"
if _UI_DIST.exists() and (_UI_DIST / "index.html").exists():
    if (_UI_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(_UI_DIST / "assets")), name="assets")

    @app.get("/app", tags=["ui"])
    def app_ui() -> FileResponse:
        return FileResponse(_UI_DIST / "index.html")


@app.get("/", tags=["ui"])
def root(request: Request) -> Any:
    accept = request.headers.get("accept", "")
    if "text/html" in accept and _UI_DIST.exists() and (_UI_DIST / "index.html").exists():
        return FileResponse(_UI_DIST / "index.html")
    return {
        "service": "AI-Powered University Student Services Assistant",
        "contract": ["POST /ask", "POST /ingest", "GET /health", "GET /audit/{trace_id}", "GET /sources"],
        "docs": "/docs",
        "ui": "/app" if _UI_DIST.exists() else None,
        "today": date.today().isoformat(),
    }
