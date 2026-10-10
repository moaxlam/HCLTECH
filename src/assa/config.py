"""Runtime configuration.

Every value can be overridden with an environment variable so the same image can
run against a host Ollama, a different vector store path or a cloud LLM fallback
without a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(os.environ.get("ASSA_BASE_DIR", Path(__file__).resolve().parents[2]))


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    documents_dir: Path
    data_dir: Path
    chroma_dir: Path
    sqlite_path: Path
    source_register_path: Path
    test_students_dir: Path

    embedding_model: str
    collection_name: str
    top_k: int
    chunk_max_chars: int
    chunk_overlap_chars: int

    ollama_host: str
    ollama_model: str
    llm_timeout_s: int
    mock_llm: bool

    cloud_fallback_enabled: bool
    cloud_base_url: str
    cloud_api_key: str
    cloud_model: str

    cors_origins: tuple[str, ...]
    log_level: str
    min_confidence: float
    min_score: float

    llm_provider: str = "ollama"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("ASSA_DATA_DIR", BASE_DIR / "data"))
    groq_api_key = (
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("ASSA_GROQ_API_KEY", "").strip()
        or (
            os.environ.get("ASSA_CLOUD_API_KEY", "").strip()
            if "groq" in os.environ.get("ASSA_CLOUD_BASE_URL", "").lower()
            or os.environ.get("ASSA_CLOUD_API_KEY", "").strip().startswith("gsk_")
            else ""
        )
    )
    groq_model = (
        os.environ.get("GROQ_MODEL", "").strip()
        or os.environ.get("ASSA_GROQ_MODEL", "").strip()
        or "llama-3.3-70b-versatile"
    )
    groq_base_url = (
        os.environ.get("GROQ_BASE_URL", "").strip()
        or os.environ.get("ASSA_GROQ_BASE_URL", "").strip()
        or "https://api.groq.com/openai/v1"
    ).rstrip("/")
    llm_provider = (
        os.environ.get("ASSA_LLM_PROVIDER", "").strip().lower()
        or os.environ.get("LLM_PROVIDER", "").strip().lower()
        or ("groq" if groq_api_key else "ollama")
    )
    return Settings(
        base_dir=BASE_DIR,
        documents_dir=Path(os.environ.get("ASSA_DOCUMENTS_DIR", BASE_DIR / "documents")),
        data_dir=data_dir,
        chroma_dir=Path(os.environ.get("ASSA_CHROMA_DIR", data_dir / "chroma")),
        sqlite_path=Path(os.environ.get("ASSA_SQLITE_PATH", data_dir / "university.db")),
        source_register_path=Path(
            os.environ.get("ASSA_SOURCE_REGISTER", data_dir / "source_register.csv")
        ),
        test_students_dir=Path(os.environ.get("ASSA_TEST_STUDENTS_DIR", BASE_DIR / "test_students")),
        embedding_model=os.environ.get("ASSA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        collection_name=os.environ.get("ASSA_CHROMA_COLLECTION", "university_documents"),
        top_k=_int("ASSA_TOP_K", 6),
        chunk_max_chars=_int("ASSA_CHUNK_MAX_CHARS", 1400),
        chunk_overlap_chars=_int("ASSA_CHUNK_OVERLAP_CHARS", 150),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=os.environ.get("ASSA_OLLAMA_MODEL", "llama3.1:8b"),
        llm_timeout_s=_int("ASSA_LLM_TIMEOUT_S", 120),
        mock_llm=_flag("MOCK_LLM", False),
        cloud_fallback_enabled=_flag("ASSA_CLOUD_FALLBACK", False),
        cloud_base_url=os.environ.get("ASSA_CLOUD_BASE_URL", ""),
        cloud_api_key=os.environ.get("ASSA_CLOUD_API_KEY", ""),
        cloud_model=os.environ.get("ASSA_CLOUD_MODEL", ""),
        cors_origins=tuple(
            origin.strip()
            for origin in os.environ.get(
                "ASSA_CORS_ORIGINS", "http://localhost:5173,http://localhost:4173,http://localhost:8080"
            ).split(",")
            if origin.strip()
        ),
        log_level=os.environ.get("ASSA_LOG_LEVEL", "INFO").upper(),
        min_confidence=float(os.environ.get("ASSA_MIN_CONFIDENCE", "0.60")),
        # Relevance floor: chunks below this cosine similarity are not evidence.
        # Measured on this corpus with all-MiniLM-L6-v2: the best clause for a
        # question the corpus answers scores 0.587-0.836, and the best clause for a
        # question it cannot answer scores 0.354-0.495, so 0.55 separates them.
        # The harness compares 0.30 and 0.55 in evaluation/comparison.md.
        min_score=float(os.environ.get("ASSA_MIN_SCORE", "0.55")),
        llm_provider=llm_provider,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        groq_base_url=groq_base_url,
    )


def reset_settings_cache() -> None:
    """Used by tests that patch the environment."""
    get_settings.cache_clear()
