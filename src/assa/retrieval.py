"""Retrieval: vector search + Source Register metadata + precedence resolution.

The retriever never returns raw similarity hits. It returns *clauses that are in
force on the requested date*, with the precedence decision that produced them,
which is what R2, R4 and R10 need downstream.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import rules
from .config import get_settings
from .guardrails import chunk_is_unsafe
from .precedence import (
    TOPIC_PARAMETER,
    Candidate,
    PrecedenceOutcome,
    asserts_threshold,
    numbers_in,
    parse_iso,
    parse_supersedes,
    resolve_candidates,
    topic_of,
)
from .vectorstore import VectorStore


@dataclass
class RetrievalResult:
    question_topic: str
    kept: list[Candidate] = field(default_factory=list)
    dropped: list[Candidate] = field(default_factory=list)
    upcoming: list[Candidate] = field(default_factory=list)
    sources_retrieved: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    unresolved_conflict: bool = False
    parameter_values: dict[str, float] = field(default_factory=dict)
    applied_rules: list[dict[str, Any]] = field(default_factory=list)
    unsafe_chunks: list[str] = field(default_factory=list)
    has_evidence: bool = False
    upcoming_notes: list[str] = field(default_factory=list)

    @property
    def decision_text(self) -> str:
        return "; ".join(self.decisions)

    @property
    def context(self) -> str:
        blocks = []
        for candidate in self.kept:
            blocks.append(
                f"[{candidate.doc_id} | {candidate.section} | page {candidate.page or 'n/a'} | "
                f"v{candidate.version} | effective {candidate.effective_from} | "
                f"authority {candidate.authority_level}]\n{candidate.text}"
            )
        return "\n\n".join(blocks)


class Retriever:
    def __init__(self, conn: sqlite3.Connection, store: VectorStore) -> None:
        self.conn = conn
        self.store = store
        self._rule_lookup: dict[tuple[str, str], str] | None = None

    # -- rule registry mapping ---------------------------------------------
    @property
    def rule_lookup(self) -> dict[tuple[str, str], str]:
        """``(source_doc_id, source_section) -> parameter`` for every registry row."""
        if self._rule_lookup is None:
            rows = self.conn.execute(
                "SELECT source_doc_id, source_section, parameter FROM rule_registry"
            ).fetchall()
            self._rule_lookup = {
                (str(row["source_doc_id"]), str(row["source_section"])): str(row["parameter"])
                for row in rows
            }
        return self._rule_lookup

    def invalidate(self) -> None:
        self._rule_lookup = None

    # -- retrieval ----------------------------------------------------------
    def retrieve(
        self,
        question: str,
        as_of_date: date,
        programme: str | None = None,
        batch_year: int | str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        settings = get_settings()
        top_k = top_k or settings.top_k
        topic = topic_of(question)
        result = RetrievalResult(question_topic=topic)

        # The authoritative value for the topic, when the registry governs it.
        topic_parameter = TOPIC_PARAMETER.get(topic)
        if topic_parameter:
            resolution = rules.resolve(self.conn, topic_parameter, as_of_date, programme, batch_year)
            if resolution.status == "resolved" and resolution.value is not None:
                result.parameter_values[topic_parameter] = float(resolution.value)
                if resolution.winner:
                    result.applied_rules.append(rules.applied_rule(resolution.winner))
                if resolution.conflict:
                    result.conflicts.append(resolution.conflict)

        hits = self.store.query(question, top_k=max(top_k * 3, 12))

        # R4: a document that is not yet effective must never be applied, but when it
        # is relevant the answer has to mention it as an upcoming change. Similarity
        # alone will not surface it -- the rules in force legitimately rank higher --
        # so the nearest not-yet-effective clauses are fetched with a metadata filter
        # (ISO dates sort correctly as strings) and reported separately.
        seen_chunks = {hit["chunk_id"] for hit in hits}
        upcoming_floor = max(0.30, settings.min_score - 0.20)
        for hit in self._upcoming_hits(question, as_of_date):
            if hit["chunk_id"] in seen_chunks or hit["score"] < upcoming_floor:
                continue
            seen_chunks.add(hit["chunk_id"])
            hits.append(hit)

        candidates: list[Candidate] = []
        for hit in hits:
            metadata = hit["metadata"]
            doc_id = str(metadata.get("doc_id", ""))
            section = str(metadata.get("section", ""))
            page = metadata.get("page")
            candidate = Candidate(
                doc_id=doc_id,
                section=section,
                text=hit["text"],
                authority_level=int(metadata.get("authority_level", 3) or 3),
                effective_from=str(metadata.get("effective_from", "")),
                effective_to="" if str(metadata.get("effective_to", "-")) in {"", "-"} else str(metadata["effective_to"]),
                supersedes="" if str(metadata.get("supersedes", "-")) in {"", "-"} else str(metadata["supersedes"]),
                scope_programmes=str(metadata.get("scope_programmes", "ALL")),
                scope_batches=str(metadata.get("scope_batches", "ALL")),
                title=str(metadata.get("title", "")),
                version=str(metadata.get("version", "")),
                page=None if page in (-1, None, "-") else int(page),
                score=float(hit.get("score", 0.0)),
                parameter=self.rule_lookup.get((doc_id, section)),
            )
            if not candidate.parameter:
                # Annex A step 2: a clause that supersedes a registered clause
                # replaces that clause's rule, so it belongs to the same claim
                # group even though the new clause is not in the registry yet.
                candidate.parameter = self._parameter_of_superseded(candidate.supersedes)
            if candidate.parameter:
                candidate.claim_key = f"param:{candidate.parameter}"
            elif topic_parameter and asserts_threshold(candidate.text):
                # An unmapped document that states a requirement for the same
                # parameter competes with the registry-backed clause, so the
                # precedence policy can show that it cannot override it.
                candidate.claim_key = f"param:{topic_parameter}"
            if chunk_is_unsafe(candidate.text):
                result.unsafe_chunks.append(f"{doc_id}#{section}")
            candidates.append(candidate)

        result.sources_retrieved = _source_entries(candidates[: max(top_k * 2, 12)], as_of_date)
        # A clause that is not yet effective is recorded in the audit with its
        # status, even when it falls outside the nearest-neighbour window.
        upcoming_entries = _source_entries(
            [candidate for candidate in candidates if _is_upcoming(candidate, as_of_date)], as_of_date
        )
        known = {(entry["doc_id"], entry["section"]) for entry in result.sources_retrieved}
        for entry in upcoming_entries:
            if (entry["doc_id"], entry["section"]) not in known:
                result.sources_retrieved.append(entry)

        outcome: PrecedenceOutcome = resolve_candidates(
            candidates,
            as_of_date=as_of_date,
            programme=programme,
            batch_year=batch_year,
            question=question,
            topic=topic,
            resolved_parameter_values=result.parameter_values,
        )
        result.dropped = outcome.dropped
        result.upcoming = outcome.upcoming
        result.decisions = outcome.decisions
        result.conflicts.extend(outcome.conflicts)
        result.unresolved_conflict = outcome.unresolved

        kept = [candidate for candidate in outcome.kept if candidate.doc_id not in {""}]

        # Relevance floor: a chunk that is not actually about the question must
        # not become evidence. This is what makes honest abstention possible.
        floor = settings.min_score
        below_floor = [candidate for candidate in kept if candidate.score < floor]
        if below_floor and len(below_floor) < len(kept):
            kept = [candidate for candidate in kept if candidate.score >= floor]
            result.decisions.append(
                f"{len(below_floor)} clause(s) scored below the relevance floor of {floor}; "
                "they were not used as evidence"
            )
        elif below_floor and len(below_floor) == len(kept):
            result.decisions.append(
                f"every retrieved clause scored below the relevance floor of {floor}; no evidence was used"
            )
            kept = []

        # A chunk carrying an embedded instruction is never the sole basis of an
        # answer; it is dropped when anything else survives.
        unsafe = [candidate for candidate in kept if chunk_is_unsafe(candidate.text)]
        if unsafe and len(kept) > len(unsafe):
            kept = [candidate for candidate in kept if candidate not in unsafe]
            result.decisions.append(
                "Text in "
                + ", ".join(sorted({f"{candidate.doc_id}#{candidate.section}" for candidate in unsafe}))
                + " attempted to give instructions; it was treated as data and not used as evidence"
            )

        ranked = _lead_with_authoritative(kept, result.parameter_values)
        ranked = _limit_per_document(ranked, max_per_document=3)
        result.kept = ranked[: max(top_k, 4)]
        result.has_evidence = bool(result.kept)
        result.upcoming_notes = [
            f"{candidate.doc_id} ({candidate.title}) takes effect on {candidate.effective_from} and is not "
            "in force on the requested date"
            for candidate in outcome.upcoming
        ]
        result.citations = _citations(result.kept)
        _attach_rules_from_chunks(self.conn, result)
        return result


    def _parameter_of_superseded(self, supersedes: str) -> str | None:
        """The registry parameter of the clause this clause replaces, if any."""
        for target_doc, target_section in parse_supersedes(supersedes):
            parameter = self.rule_lookup.get((target_doc, target_section))
            if parameter:
                return parameter
        return None

    def _upcoming_hits(self, question: str, as_of_date: date) -> list[dict[str, Any]]:
        """The nearest clauses that take effect *after* ``as_of_date`` (R4).

        Similarity alone will not surface them -- the clauses in force legitimately
        rank higher -- so they are fetched with a range filter over the numeric
        effective-date key. An index built before that key existed would silently
        return nothing, so the wide unfiltered window is used as a fallback and the
        dates are compared in Python.
        """
        ordinal = int(as_of_date.strftime("%Y%m%d"))
        filtered = self.store.query(
            question, top_k=4, where={"effective_from_ord": {"$gt": ordinal}}
        )
        if filtered:
            return filtered
        return [
            hit
            for hit in self.store.query(question, top_k=40)
            if _effective_after(hit["metadata"], as_of_date)
        ][:4]


def _effective_after(metadata: dict[str, Any], as_of_date: date) -> bool:
    value = str(metadata.get("effective_from", "-")).strip()
    return len(value) == 10 and value > as_of_date.isoformat()


def _is_upcoming(candidate: Candidate, as_of_date: date) -> bool:
    return candidate.effective_from > as_of_date.isoformat()


def _source_entries(candidates: list[Candidate], as_of_date: date) -> list[dict[str, Any]]:
    entries = []
    for candidate in candidates:
        entry: dict[str, Any] = {
            "doc_id": candidate.doc_id,
            "section": candidate.section,
            "score": round(candidate.score, 4),
            "authority_level": candidate.authority_level,
            "effective_from": candidate.effective_from,
        }
        if _is_upcoming(candidate, as_of_date):
            entry["status"] = "not_yet_effective"
        entries.append(entry)
    return entries


def _lead_with_authoritative(
    candidates: list[Candidate], parameter_values: dict[str, float]
) -> list[Candidate]:
    """Put the clause that carries the rule in force in front of its neighbours.

    A document often has several clauses about the same subject (the circular that
    supersedes clause 7.2 also has a "clarifications" clause mentioning attendance).
    Similarity alone would let the tangential clause lead, so the clause that is
    mapped to the resolved parameter, or that states the authoritative value, is
    ordered first, and the rest keep their similarity order.
    """
    if not parameter_values:
        return candidates
    values = {round(value, 2) for value in parameter_values.values()}

    def rank(candidate: Candidate) -> int:
        if candidate.parameter in parameter_values:
            return 0
        stated = {round(value, 2) for value in numbers_in(candidate.text)}
        return 1 if stated & values else 2

    return sorted(candidates, key=lambda candidate: (rank(candidate), -candidate.score))


def _limit_per_document(candidates: list[Candidate], max_per_document: int = 3) -> list[Candidate]:
    """Keep the evidence set diverse.

    One long circular has many clauses that mention the same subject; without a cap
    they fill the evidence set and push out the clause from another document that
    actually supersedes or contradicts them. At most ``max_per_document`` clauses
    from any single document are kept.
    """
    counts: dict[str, int] = {}
    limited: list[Candidate] = []
    for candidate in candidates:
        seen = counts.get(candidate.doc_id, 0)
        if seen >= max_per_document:
            continue
        counts[candidate.doc_id] = seen + 1
        limited.append(candidate)
    return limited


def _citations(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """Citations in the order the clauses were ranked, deduplicated by clause."""
    best: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        best.setdefault((candidate.doc_id, candidate.section), candidate)
    ordered = list(best.values())
    return [
        {
            "doc_id": candidate.doc_id,
            "title": candidate.title,
            "section": candidate.section,
            "page": candidate.page,
            "version": candidate.version,
            "effective_from": candidate.effective_from,
        }
        for candidate in ordered
    ]


def _attach_rules_from_chunks(conn: sqlite3.Connection, result: RetrievalResult) -> None:
    """Add registry rows whose cited clause is one of the kept chunks."""
    seen = {rule["rule_id"] for rule in result.applied_rules}
    candidate_effective = parse_iso(result.kept[0].effective_from) if result.kept else None
    for candidate in result.kept:
        rows = conn.execute(
            "SELECT * FROM rule_registry WHERE source_doc_id = ? AND source_section = ?",
            (candidate.doc_id, candidate.section),
        ).fetchall()
        for row in rows:
            rule = dict(row)
            if rule["rule_id"] in seen:
                continue
            # A registry row that only takes effect later is not in force here.
            rule_effective = parse_iso(rule.get("effective_from"))
            if rule_effective and candidate_effective and rule_effective > candidate_effective:
                continue
            seen.add(rule["rule_id"])
            result.applied_rules.append(rules.applied_rule(rule))
