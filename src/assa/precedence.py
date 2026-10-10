"""Annex A -- Source Precedence Policy, implemented as deterministic code.

Nothing in this module calls an LLM. The same inputs always produce the same
decision, which is what the participant guide asks for (R4).

Resolution order implemented exactly as Annex A.2:

1. Applicability   -- only documents effective on ``as_of_date`` whose programme
                      and batch scope cover the student (or are general).
                      Documents that are not yet effective are excluded from the
                      answer and reported separately as upcoming changes.
2. Supersession    -- a document that explicitly supersedes another document or
                      clause replaces it, provided the superseding document is
                      issued at authority level 1 or 2.
3. Authority       -- a higher-authority document prevails over a lower-authority
                      one, regardless of date.
4. Recency         -- between documents of the same authority, the later
                      ``effective_from`` prevails.
5. Unresolved      -- otherwise the conflict is reported, both documents are
                      cited and the student is advised to contact the issuing
                      office (``answer_type: conflict_flagged``).

Level 5 content (unofficial) can never override another document and may only be
cited as informational.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

AUTHORITY_LABELS = {
    1: "statutes/ordinances/academic regulations",
    2: "official circular or notification of an authorised office",
    3: "department notice",
    4: "handbook or help-desk FAQ",
    5: "unofficial content",
}

#: Questions are mapped to a small, documented topic taxonomy so that competing
#: claims can be compared deterministically. Add topics here, nowhere else.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "attendance": ("attendance", "classes", "detained", "detention", "condonation", "shortage"),
    "supplementary_exam": ("supplementary", "backlog", "reappear", "re-examination", "improvement"),
    "end_sem_exam": ("end-semester", "end semester", "hall ticket", "pass mark", "exam eligibility"),
    "placement": ("placement", "campus drive", "recruiter", "internship", "offer"),
    "scholarship": ("scholarship", "fee concession", "merit award"),
    "fees": ("fee", "late fee", "payment", "due date", "refund"),
    "results": ("result", "cgpa", "grade", "revaluation", "rechecking", "marks"),
    "hostel": ("hostel", "warden", "mess", "room", "out-pass"),
    "discipline": ("discipline", "unfair means", "misconduct"),
}

#: The authoritative registry parameter each topic is governed by, if any. Used
#: only for the plain-language contradiction check in step 3.
TOPIC_PARAMETER: dict[str, str] = {
    "attendance": "min_attendance_pct",
    "supplementary_exam": "pass_marks_min",
    "placement": "min_cgpa_placement",
    "scholarship": "min_cgpa_scholarship",
    "fees": "late_fee_inr",
}

PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:percent|%)", re.IGNORECASE)
AMOUNT_RE = re.compile(r"(?:INR|Rs\.?)\s*([\d,]{2,})", re.IGNORECASE)


def parse_iso(value: str | None) -> date | None:
    if not value or not ISO_RE.match(value.strip()):
        return None
    return date.fromisoformat(value.strip())


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]


def scope_matches(scope_value: str | None, actual: str | None) -> bool:
    """Does ``scope_value`` (from a document) cover ``actual`` (the student)?

    ``ALL`` matches anything. Batch scopes support ``2023+``, ``2023`` and
    ``2023-2025``. Programme scopes are compared case-insensitively.
    """
    scope_value = (scope_value or "ALL").strip()
    if not scope_value or scope_value.upper() == "ALL":
        return True
    if actual is None:
        # A scoped document cannot be shown to cover a student we know nothing about;
        # general questions are answered from ALL-scoped documents first.
        return False
    for entry in _split_list(scope_value):
        if entry.upper() == "ALL":
            return True
        if entry.endswith("+"):
            year = parse_iso_year(entry[:-1])
            if year is not None and parse_iso_year(actual) is not None:
                if parse_iso_year(actual) >= year:
                    return True
                continue
        if "-" in entry and re.match(r"^\d{4}-\d{4}$", entry):
            low, high = entry.split("-")
            if low <= actual <= high:
                return True
            continue
        if entry.casefold() == actual.casefold():
            return True
    return False


def parse_iso_year(value: str) -> int | None:
    match = re.match(r"^(\d{4})", value.strip())
    return int(match.group(1)) if match else None


def _field(document: Any, name: str, default: Any = "") -> Any:
    """Read a metadata field from a mapping (a SQLite row or CSV row) or an object."""
    if isinstance(document, dict):
        return document.get(name, default)
    return getattr(document, name, default)


def document_is_effective(document: Any, as_of_date: date) -> bool:
    start = parse_iso(_field(document, "effective_from"))
    if start is None:
        return False
    if start > as_of_date:
        return False
    end = parse_iso(_field(document, "effective_to"))
    if end is not None and end < as_of_date:
        return False
    return True


def document_covers(
    document: Any, programme: str | None = None, batch_year: int | str | None = None
) -> bool:
    batch = str(batch_year) if batch_year is not None else None
    return scope_matches(_field(document, "scope_programmes", "ALL"), programme) and scope_matches(
        _field(document, "scope_batches", "ALL"), batch
    )


def parse_supersedes(value: str | None) -> list[tuple[str, str | None]]:
    """``"ACAD-REG-2024#7.2; OLD-DOC"`` -> ``[("ACAD-REG-2024", "7.2"), ("OLD-DOC", None)]``."""
    targets: list[tuple[str, str | None]] = []
    for entry in _split_list(value):
        if "#" in entry:
            doc_id, section = entry.split("#", 1)
            targets.append((doc_id.strip(), section.strip() or None))
        else:
            targets.append((entry.strip(), None))
    return targets


def _section_matches(target: str | None, candidate: str | None) -> bool:
    if target is None:
        return True
    if candidate is None:
        return False
    return candidate.strip().startswith(target.strip()) or target.strip().startswith(candidate.strip())


@dataclass
class Candidate:
    """One retrieved clause competing to answer the question."""

    doc_id: str
    section: str
    text: str
    authority_level: int
    effective_from: str
    effective_to: str = ""
    supersedes: str = ""
    scope_programmes: str = "ALL"
    scope_batches: str = "ALL"
    title: str = ""
    version: str = "1.0"
    page: int | None = None
    score: float = 0.0
    claim_key: str = ""
    parameter: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "section": self.section,
            "text": self.text,
            "title": self.title,
            "version": self.version,
            "page": self.page,
            "score": round(self.score, 4),
            "authority_level": self.authority_level,
            "effective_from": self.effective_from,
        }


@dataclass
class PrecedenceOutcome:
    kept: list[Candidate] = field(default_factory=list)
    dropped: list[Candidate] = field(default_factory=list)
    upcoming: list[Candidate] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    unresolved: bool = False

    @property
    def decision_text(self) -> str:
        return "; ".join(self.decisions)


THRESHOLD_WORDS = (
    "minimum",
    "at least",
    "not less than",
    "enough",
    "required",
    "below",
    "above",
    "or more",
    "or less",
    "cut-off",
    "cutoff",
)


def numbers_in(text: str) -> list[float]:
    values = [float(match) for match in PERCENT_RE.findall(text)]
    values += [float(match.replace(",", "")) for match in AMOUNT_RE.findall(text)]
    return values


def asserts_threshold(text: str) -> bool:
    """Does this text state a value as a requirement rather than mention one in passing?"""
    lowered = text.casefold()
    if not numbers_in(text):
        return False
    return any(word in lowered for word in THRESHOLD_WORDS)


def topic_of(question: str) -> str:
    lowered = question.casefold()
    best, best_hits = "general", 0
    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits > best_hits:
            best, best_hits = topic, hits
    return best


def resolve_candidates(
    candidates: Sequence[Candidate],
    as_of_date: date,
    programme: str | None = None,
    batch_year: int | str | None = None,
    question: str = "",
    topic: str | None = None,
    resolved_parameter_values: dict[str, float] | None = None,
) -> PrecedenceOutcome:
    """Apply Annex A.2 to a set of retrieved clauses."""
    outcome = PrecedenceOutcome()
    topic = topic or topic_of(question)
    resolved_parameter_values = resolved_parameter_values or {}

    # ---- Step 1: applicability -------------------------------------------------
    applicable: list[Candidate] = []
    for candidate in candidates:
        if not document_is_effective(candidate, as_of_date):
            start = parse_iso(candidate.effective_from)
            if start is not None and start > as_of_date:
                outcome.upcoming.append(candidate)
                outcome.decisions.append(
                    f"{candidate.doc_id} is not yet effective on {as_of_date.isoformat()} "
                    f"(effective_from {candidate.effective_from}) and was excluded (step 1)"
                )
            else:
                outcome.dropped.append(candidate)
                outcome.decisions.append(
                    f"{candidate.doc_id} is not in force on {as_of_date.isoformat()} (step 1)"
                )
            continue
        scoped = (candidate.scope_programmes or "ALL").upper() != "ALL" or (
            candidate.scope_batches or "ALL").upper() != "ALL"
        if programme is None and batch_year is None:
            # A general question has no student scope to compare against, so
            # scoped documents are kept and reported instead of silently dropped.
            if scoped:
                outcome.decisions.append(
                    f"{candidate.doc_id} applies only to {candidate.scope_programmes}/{candidate.scope_batches}; "
                    "no student scope was given, so it was kept and marked as scoped (step 1)"
                )
            applicable.append(candidate)
            continue
        if not document_covers(candidate, programme, batch_year):
            outcome.dropped.append(candidate)
            outcome.decisions.append(
                f"{candidate.doc_id} does not cover programme={programme!r} batch={batch_year!r} (step 1)"
            )
            continue
        applicable.append(candidate)

    # ---- Step 2: explicit supersession ----------------------------------------
    superseders = [
        candidate
        for candidate in applicable
        if candidate.authority_level <= 2 and parse_supersedes(candidate.supersedes)
    ]
    survivors: list[Candidate] = []
    for candidate in applicable:
        superseded_by: Candidate | None = None
        for superseder in superseders:
            if superseder.doc_id == candidate.doc_id:
                continue
            for target_doc, target_section in parse_supersedes(superseder.supersedes):
                if target_doc == candidate.doc_id and _section_matches(target_section, candidate.section):
                    superseded_by = superseder
                    break
            if superseded_by:
                break
        if superseded_by is not None:
            outcome.dropped.append(candidate)
            target = f"{candidate.doc_id}#{candidate.section}" if candidate.section else candidate.doc_id
            outcome.decisions.append(
                f"{superseded_by.doc_id} supersedes {target} (step 2)"
            )
        else:
            survivors.append(candidate)

    # ---- Steps 3-5: conflicts --------------------------------------------------
    kept, conflicts, unresolved = _resolve_conflicts(
        survivors, topic, outcome, resolved_parameter_values
    )
    outcome.kept = kept
    outcome.conflicts = conflicts
    outcome.unresolved = unresolved
    kept.sort(key=lambda candidate: (-candidate.score, candidate.authority_level))
    return outcome


def _resolve_conflicts(
    survivors: list[Candidate],
    topic: str,
    outcome: PrecedenceOutcome,
    resolved_parameter_values: dict[str, float],
) -> tuple[list[Candidate], list[dict[str, Any]], bool]:
    """Group surviving clauses by the claim they make and resolve each group."""
    groups: dict[str, list[Candidate]] = {}
    for candidate in survivors:
        key = candidate.claim_key or (f"param:{candidate.parameter}" if candidate.parameter else f"topic:{topic}")
        groups.setdefault(key, []).append(candidate)

    kept: list[Candidate] = []
    conflicts: list[dict[str, Any]] = []
    unresolved = False

    primary_parameter = TOPIC_PARAMETER.get(topic)

    for key, members in groups.items():
        parameter = primary_parameter if key.startswith("topic:") else key.split(":", 1)[1]
        if len(members) == 1:
            kept.append(members[0])
            continue

        # Detect a genuine numeric disagreement. Two clauses from the same
        # authority that say the same thing are not a conflict.
        claim_values: dict[float, list[Candidate]] = {}
        for member in members:
            for value in numbers_in(member.text):
                claim_values.setdefault(value, []).append(member)
            if member.parameter and member.parameter in resolved_parameter_values:
                claim_values.setdefault(
                    float(resolved_parameter_values[member.parameter]), []
                ).append(member)

        # Add the authoritative registry value when the clause group has no
        # parameter mapping (for example a help-desk FAQ answering the same topic).
        if parameter and parameter in resolved_parameter_values:
            authoritative = float(resolved_parameter_values[parameter])
            for member in members:
                if not member.parameter and authoritative in numbers_in(member.text):
                    claim_values.setdefault(authoritative, []).append(member)

        distinct = {value for value, who in claim_values.items() if who}
        if len(distinct) <= 1:
            kept.extend(members)
            continue

        doc_ids = sorted({member.doc_id for member in members})
        sections = [f"{member.doc_id}#{member.section}" for member in members]
        conflict: dict[str, Any] = {
            "reason": (
                f"Clauses in {', '.join(doc_ids)} state different values for the same question "
                f"({', '.join(_format_number(value) for value in sorted(distinct))})."
            ),
            "doc_ids": doc_ids,
            "sections": sections,
        }

        # Step 3: authority. Only clauses asserting the authoritative value survive.
        by_authority = min(member.authority_level for member in members)
        candidates_at_top = [member for member in members if member.authority_level == by_authority]

        authoritative_value = (
            float(resolved_parameter_values[parameter]) if parameter and parameter in resolved_parameter_values else None
        )
        if authoritative_value is not None:
            winners = [
                member
                for member in members
                if authoritative_value in numbers_in(member.text)
                or (member.parameter == parameter)
            ]
            winners = [member for member in winners if member.authority_level <= by_authority + 2]
            losers = [member for member in members if member not in winners]
            if winners and losers:
                kept.extend(winners)
                conflict["resolution"] = (
                    f"{', '.join(sorted({m.doc_id for m in winners}))} prevails; "
                    f"{', '.join(sorted({m.doc_id for m in losers}))} cannot override it (step 3 on authority)"
                )
                conflict["resolved_by"] = "step-3-authority"
                decision = (
                    f"{conflict['resolution']} for {parameter}="
                    f"{_format_number(authoritative_value)} (step 3)"
                )
                outcome.decisions.append(decision)
                conflicts.append(conflict)
                continue

        # Plain authority comparison.
        if len(candidates_at_top) < len(members):
            higher = candidates_at_top
            lower = [member for member in members if member.authority_level != by_authority]
            rejected = _restrict_losers(lower, higher)
            if not rejected:
                # No lower-authority clause restates a winner's value, so the
                # numbers belong to different rules. Dropping them here would
                # delete evidence that never contradicted anything.
                kept.extend(members)
                continue
            kept.extend(higher)
            conflict["resolution"] = (
                f"{', '.join(sorted({m.doc_id for m in higher}))} prevails on authority "
                f"(level {by_authority}) over {', '.join(sorted({m.doc_id for m in rejected}))}"
            )
            conflict["resolved_by"] = "step-3-authority"
            outcome.decisions.append(conflict["resolution"] + " (step 3)")
            conflicts.append(conflict)
            continue

        # Step 4: recency among the same authority.
        latest = max(parse_iso(member.effective_from) or date.min for member in members)
        newest = [member for member in members if (parse_iso(member.effective_from) or date.min) == latest]
        if len(newest) < len(members):
            older = [member for member in members if member not in newest]
            rejected = _restrict_losers(older, newest)
            kept.extend(newest)
            conflict["resolution"] = (
                f"{', '.join(sorted({m.doc_id for m in newest}))} prevails as the later document "
                f"effective {newest[0].effective_from} over {', '.join(sorted({m.doc_id for m in rejected}))}"
            )
            conflict["resolved_by"] = "step-4-recency"
            outcome.decisions.append(conflict["resolution"] + " (step 4)")
            conflicts.append(conflict)
            continue

        # Step 5: unresolved.
        unresolved = True
        kept.extend(members)
        conflict["resolution"] = None
        conflict["resolved_by"] = "unresolved"
        conflict["reason"] += " The precedence policy cannot resolve this; both documents are cited."
        conflicts.append(conflict)
        outcome.decisions.append(
            "Conflict could not be resolved by the precedence policy (step 5); both sources are cited"
        )

    return kept, conflicts, unresolved


def _restrict_losers(losers: Iterable[Candidate], winners: Sequence[Candidate]) -> list[Candidate]:
    """A clause is only 'rejected' if it makes the same claim as a winner."""
    winner_values = {value for winner in winners for value in numbers_in(winner.text)}
    restricted: list[Candidate] = []
    for loser in losers:
        loser_values = set(numbers_in(loser.text))
        if loser.parameter or not winner_values or loser_values & winner_values:
            restricted.append(loser)
    return restricted


def _format_number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text


# --------------------------------------------------------------------------- #
# Rule registry resolution (thresholds are read from rule_registry, never from
# constants in code -- Annex C).
# --------------------------------------------------------------------------- #


@dataclass
class RuleResolution:
    parameter: str
    status: str  # resolved | not_found | conflict
    value: float | None = None
    operator: str | None = None
    winner: dict[str, Any] | None = None
    applied: list[dict[str, Any]] = field(default_factory=list)
    upcoming: list[dict[str, Any]] = field(default_factory=list)
    decision: str = ""
    conflict: dict[str, Any] | None = None


def _rule_value(rule: dict[str, Any]) -> float | None:
    raw = str(rule.get("value", "")).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        return float(match.group(0)) if match else None


def _rule_applicable(
    rule: dict[str, Any], as_of_date: date, programme: str | None, batch_year: int | str | None
) -> bool:
    fake_document = {
        "effective_from": rule.get("effective_from", ""),
        "effective_to": rule.get("effective_to", ""),
        "scope_programmes": rule.get("scope_programmes", "ALL"),
        "scope_batches": rule.get("scope_batches", "ALL"),
    }
    if not document_is_effective(fake_document, as_of_date):
        return False
    return document_covers(fake_document, programme, batch_year)


def resolve_parameter(
    rules: Sequence[dict[str, Any]],
    parameter: str,
    as_of_date: date,
    programme: str | None = None,
    batch_year: int | str | None = None,
) -> RuleResolution:
    """Resolve one registry parameter to a single value for a scope and date.

    ``rules`` must be the rows of ``rule_registry`` for ``parameter`` joined with
    ``authority_level`` from the Source Register.
    """
    resolution = RuleResolution(parameter=parameter, status="not_found")

    # When the caller knows nothing about the student (for example a fee schedule
    # that is not student-specific), scope is not used to exclude a rule; the same
    # choice is made in chunk retrieval for general questions.
    scope_unknown = programme is None and batch_year is None
    applicable: list[dict[str, Any]] = []
    for rule in rules:
        effective = document_is_effective(
            {
                "effective_from": rule.get("effective_from", ""),
                "effective_to": rule.get("effective_to", ""),
            },
            as_of_date,
        )
        if not effective:
            start = parse_iso(str(rule.get("effective_from", "")))
            if start is not None and start > as_of_date:
                resolution.upcoming.append(rule)
            continue
        if scope_unknown or document_covers(rule, programme, batch_year):
            applicable.append(rule)
            if scope_unknown and (
                str(rule.get("scope_programmes", "ALL")).upper() != "ALL"
                or str(rule.get("scope_batches", "ALL")).upper() != "ALL"
            ):
                resolution.decision = (
                    f"{rule['rule_id']} applies to {rule.get('scope_programmes')}/"
                    f"{rule.get('scope_batches')}; no student scope was given, so it was kept"
                )

    if not applicable:
        return resolution

    # Supersession between registry rows.
    superseded_ids = {
        str(rule.get("supersedes_rule_id") or "")
        for rule in applicable
        if str(rule.get("supersedes_rule_id") or "")
    }
    survivors = [rule for rule in applicable if rule["rule_id"] not in superseded_ids]
    decisions: list[str] = []
    for rule in applicable:
        if rule not in survivors:
            winner = next(
                (
                    other
                    for other in applicable
                    if str(other.get("supersedes_rule_id") or "") == rule["rule_id"]
                ),
                None,
            )
            if winner is not None:
                decisions.append(
                    f"{winner['rule_id']} supersedes {rule['rule_id']} (step 2)"
                )
    if not survivors:
        survivors = applicable

    # Group by asserted value.
    by_value: dict[float, list[dict[str, Any]]] = {}
    for rule in survivors:
        value = _rule_value(rule)
        if value is None:
            continue
        by_value.setdefault(value, []).append(rule)

    resolution.applied = survivors

    if len(by_value) == 1:
        value, group = next(iter(by_value.items()))
        winner = max(group, key=lambda rule: (str(rule.get("effective_from") or ""), rule["rule_id"]))
        resolution.status = "resolved"
        resolution.value = value
        resolution.operator = winner.get("operator")
        resolution.winner = winner
        resolution.decision = "; ".join(decisions)
        return resolution

    if not by_value:
        return resolution

    # Step 3: authority of the source document.
    top_authority = min(int(rule.get("authority_level") or 3) for rule in survivors)
    at_top = [rule for rule in survivors if int(rule.get("authority_level") or 3) == top_authority]
    top_values = {_rule_value(rule) for rule in at_top}
    if len(top_values) == 1:
        value = next(iter(top_values))
        assert value is not None
        winner = max(at_top, key=lambda rule: (str(rule.get("effective_from") or ""), rule["rule_id"]))
        resolution.status = "resolved"
        resolution.value = value
        resolution.operator = winner.get("operator")
        resolution.winner = winner
        resolution.decision = "; ".join(
            decisions
            + [
                f"{winner['rule_id']} prevails on authority (level {top_authority}) "
                f"over {', '.join(sorted({r['rule_id'] for r in survivors if r not in at_top}))} (step 3)"
            ]
        )
        return resolution

    # Step 4: recency among the top authority.
    latest = max(str(rule.get("effective_from") or "") for rule in at_top)
    newest = [rule for rule in at_top if str(rule.get("effective_from") or "") == latest]
    newest_values = {_rule_value(rule) for rule in newest}
    if len(newest_values) == 1:
        value = next(iter(newest_values))
        assert value is not None
        winner = max(newest, key=lambda rule: rule["rule_id"])
        resolution.status = "resolved"
        resolution.value = value
        resolution.operator = winner.get("operator")
        resolution.winner = winner
        resolution.decision = "; ".join(
            decisions + [f"{winner['rule_id']} prevails as the later rule effective {latest} (step 4)"]
        )
        return resolution

    # Step 5: unresolved.
    resolution.status = "conflict"
    resolution.conflict = {
        "reason": (
            f"Rules {', '.join(sorted(rule['rule_id'] for rule in newest))} assert different values "
            f"for {parameter} effective {latest} at the same authority level."
        ),
        "doc_ids": sorted({str(rule.get("source_doc_id")) for rule in newest}),
        "sections": [f"{rule.get('source_doc_id')}#{rule.get('source_section')}" for rule in newest],
        "resolution": None,
        "resolved_by": "unresolved",
    }
    resolution.decision = "; ".join(decisions + ["conflict could not be resolved (step 5)"])
    return resolution
