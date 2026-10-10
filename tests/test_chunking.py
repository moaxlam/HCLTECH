"""Chunking tests: clause sections, page markers, FAQ blocks and long clauses."""

from __future__ import annotations

from assa.chunking import chunk_document, read_document

METADATA = {
    "doc_id": "TEST-DOC",
    "title": "Test document",
    "version": "1.0",
    "effective_from": "2026-01-01",
    "authority_level": 2,
    "doc_type": "circular",
}

SAMPLE = """# Test document

<!-- page: 1 -->
## 1. Preamble

1.1 This document is a test fixture used by the unit tests for chunking.

1.2 A second clause that follows the first one.

<!-- page: 3 -->
## 2. Attendance

2.1 A student shall have a minimum of 80 percent of the classes held.
"""

FAQ = """# Help desk FAQ

<!-- page: 1 -->
## General

**Q1. What is the minimum attendance?**

A. As per the help desk, 65 percent is enough.

**Q2. Where do I apply?**

A. On the student portal.
"""


def test_clause_numbers_become_sections():
    chunks = chunk_document(SAMPLE, METADATA)
    sections = [chunk.section for chunk in chunks]
    assert "1.1" in sections
    assert "1.2" in sections
    assert "2.1" in sections


def test_page_numbers_are_tracked_from_markers():
    chunks = chunk_document(SAMPLE, METADATA)
    by_section = {chunk.section: chunk for chunk in chunks}
    assert by_section["1.1"].page == 1
    assert by_section["2.1"].page == 3


def test_faq_questions_become_sections():
    chunks = chunk_document(FAQ, METADATA)
    sections = [chunk.section for chunk in chunks]
    assert "Q1" in sections and "Q2" in sections
    q1 = next(chunk for chunk in chunks if chunk.section == "Q1")
    assert "65 percent" in q1.text


def test_metadata_is_carried_onto_every_chunk():
    chunks = chunk_document(SAMPLE, METADATA)
    for chunk in chunks:
        chunk_id, text, metadata = chunk.as_record()
        assert chunk_id.startswith("TEST-DOC::")
        assert metadata["authority_level"] == 2
        assert metadata["effective_from"] == "2026-01-01"
        assert metadata["doc_id"] == "TEST-DOC"
        assert text.strip()


def test_long_clauses_are_split_with_overlap():
    body = "3.1 " + " ".join(f"This is sentence number {index} of a long clause." for index in range(120))
    chunks = chunk_document(f"# Doc\n\n{body}", METADATA, max_chars=400, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 700 for chunk in chunks)
    assert all(chunk.section == "3.1" for chunk in chunks)


def test_reading_a_markdown_file(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nSome content about attendance.", encoding="utf-8")
    result = read_document(path)
    assert result.needs_ocr is False
    assert "attendance" in result.text


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    try:
        read_document(path)
    except ValueError as error:
        assert "unsupported document type" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
