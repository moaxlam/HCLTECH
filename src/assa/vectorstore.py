"""ChromaDB vector store, persisted to disk, plus the ingestion pipeline (R11).

``chromadb.PersistentClient`` keeps the index between restarts, so the assistant
does not re-ingest on boot. ``POST /ingest`` writes a new document and its
metadata while the process is running; the chunks are searchable on the next
request with no code change and no restart.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import db
from .chunking import Chunk, chunk_document, read_document
from .config import get_settings
from .embeddings import get_embedder
from .guardrails import scan_text_for_injections


@dataclass
class IngestOutcome:
    doc_id: str
    chunks_indexed: int
    status: str
    authority_level: int | None = None
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunks_indexed": self.chunks_indexed,
            "status": self.status,
            "authority_level": self.authority_level,
            "warnings": self.warnings,
        }


class VectorStore:
    def __init__(self, persist_dir: Path | None = None, collection_name: str | None = None) -> None:
        import chromadb

        settings = get_settings()
        self.persist_dir = Path(persist_dir or settings.chroma_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name or settings.collection_name
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = None

    # -- embedding ----------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embedder.encode(texts)

    # -- writes -------------------------------------------------------------
    def index_chunks(self, chunks: Sequence[Chunk], replace: bool = True) -> int:
        if not chunks:
            return 0
        if replace:
            doc_ids = {chunk.doc_id for chunk in chunks}
            for doc_id in doc_ids:
                self.delete_document(doc_id)
        ids, documents, metadatas, retrieval_texts = [], [], [], []
        for chunk in chunks:
            chunk_id, text, metadata = chunk.as_record()
            ids.append(chunk_id)
            documents.append(text)  # clean text is what an answer cites
            metadatas.append(metadata)
            # The title and heading carry the subject of the clause ("Attendance",
            # "Placement Policy"), which lifts the right clause above a same-topic
            # clause from an unrelated document. Only the vector is affected.
            retrieval_texts.append(
                f"{metadata.get('title', '')} | {chunk.heading} | {chunk.section} | {text}"
            )
        vectors = self._embed(retrieval_texts)
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors)
        return len(ids)

    def delete_document(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # -- reads --------------------------------------------------------------
    def query(
        self, text: str, top_k: int = 6, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        try:
            response = self._collection.query(
                query_embeddings=self._embed([text]),
                n_results=top_k,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:  # empty collection or a filter that matches nothing
            return []
        results: list[dict[str, Any]] = []
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        for index, chunk_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else 1.0
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index],
                    "metadata": dict(metadatas[index] or {}),
                    # cosine distance -> similarity in [0, 1]
                    "score": round(max(0.0, 1.0 - float(distance)), 4),
                }
            )
        return results

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:  # pragma: no cover
            return 0

    def documents_indexed(self) -> list[str]:
        try:
            records = self._collection.get(include=["metadatas"])
        except Exception:  # pragma: no cover
            return []
        return sorted({str((meta or {}).get("doc_id", "")) for meta in records.get("metadatas") or []} - {""})

    def health(self) -> tuple[bool, str]:
        try:
            count = self.count()
            return True, f"{count} chunks in {self.persist_dir}"
        except Exception as exc:  # pragma: no cover
            return False, str(exc)


# --------------------------------------------------------------------------- #
# ingestion
# --------------------------------------------------------------------------- #


def ingest_document(
    conn: sqlite3.Connection,
    store: VectorStore,
    path: Path,
    metadata: dict[str, Any],
    max_chars: int = 1400,
    overlap_chars: int = 150,
) -> IngestOutcome:
    """Read one file, chunk it, index it and record it in the Source Register."""
    settings = get_settings()
    doc_id = str(metadata["doc_id"])
    warnings: list[str] = []

    read = read_document(path)
    if read.note:
        warnings.append(read.note)
    if read.needs_ocr:
        db.upsert_register_row(conn, {**metadata, "provenance": f"{metadata.get('provenance', '')}", "chunks_indexed": 0})
        return IngestOutcome(doc_id, 0, "needs_ocr", metadata.get("authority_level"), warnings)

    injections = scan_text_for_injections(read.text)
    if injections:
        warnings.append(
            "document content contains imperative text that looks like an instruction "
            f"({', '.join(injections)}); it is stored as data and can never change behaviour"
        )

    chunks = chunk_document(
        read.text,
        metadata,
        max_chars=max_chars or settings.chunk_max_chars,
        overlap_chars=overlap_chars or settings.chunk_overlap_chars,
    )
    existing = store.documents_indexed()
    indexed = store.index_chunks(chunks, replace=True)
    status = "replaced" if doc_id in existing else "indexed"

    db.upsert_register_row(
        conn,
        {
            **metadata,
            "chunks_indexed": indexed,
            "ingested_at": db.utc_now_iso(),
        },
    )
    return IngestOutcome(doc_id, indexed, status, metadata.get("authority_level"), warnings)


def ingest_directory(
    conn: sqlite3.Connection,
    store: VectorStore,
    directory: Path,
    register_rows: dict[str, dict[str, Any]],
) -> list[IngestOutcome]:
    """Ingest every registered document found in ``directory``."""
    outcomes: list[IngestOutcome] = []
    for doc_id, row in sorted(register_rows.items()):
        candidates = sorted(directory.glob(f"{doc_id}*"))
        if not candidates:
            outcomes.append(IngestOutcome(doc_id, 0, "failed", row.get("authority_level"), ["file not found"]))
            continue
        outcomes.append(ingest_document(conn, store, candidates[0], row))
    return outcomes
