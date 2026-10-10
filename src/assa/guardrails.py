"""Safety and privacy guardrails (R7, R8, and the responsible-AI requirements).

Two distinct threats are handled here:

* **Instructions hidden in documents** (R8). Retrieved text is *data*. Every chunk
  is scanned at ingest time and at retrieval time for imperative content that
  tries to change assistant behaviour. Such content is never executed; the
  question is flagged in the audit record and the chunk cannot be the sole basis
  of an answer.
* **Instructions in the question.** Attempts to read another student's records,
  to obtain the system prompt, to bypass citation, or to make the assistant
  perform an action reserved for a human are refused with a short explanation.

Personal data is masked before it is written to any log line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

STUDENT_ID_RE = re.compile(r"\bS\d{4}\b")

INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "instruction override"),
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+rules", "instruction override"),
    (r"disregard\s+(all\s+)?(previous|prior|above)", "instruction override"),
    (r"reveal\s+(your\s+)?(system\s+)?prompt", "prompt extraction"),
    (r"show\s+(me\s+)?(your\s+)?(hidden\s+)?(system\s+)?prompt", "prompt extraction"),
    (r"print\s+the\s+(full\s+)?(customer|student)\s+database", "data exfiltration"),
    (r"do\s+not\s+cite", "citation bypass"),
    (r"without\s+citing", "citation bypass"),
    (r"highest\s+authority\s+in\s+the\s+knowledge\s+base", "authority spoofing"),
    (r"mark\s+(every|all)\s+students?\s+as\s+eligible", "eligibility bypass"),
    (r"bypass\s+(safety|the\s+rules|precedence)", "safety bypass"),
    (r"assistant\s+instruction", "embedded instruction"),
    (r"you\s+are\s+now\s+(a|an)\s+", "role confusion"),
    (r"override\s+the\s+(rule|policy|precedence)", "policy override"),
    (r"jailbreak", "jailbreak"),
)

HUMAN_APPROVAL_ACTIONS: tuple[tuple[str, str], ...] = (
    (r"\b(submit|file)\s+(a\s+)?(vendor\s+)?claim\b", "vendor claim submission"),
    (r"\b(quarantine|recall)\b", "product quarantine or recall"),
    (r"\b(approve|sanction|waive)\s+(my\s+)?(refund|fine|late\s+fee|fee)\b", "refund or fee waiver"),
    (r"\b(exempt|condone)\s+(me|my)\b", "condonation or exemption"),
    (r"\bchange|edit|update\s+(my|the)\s+(marks|result|attendance|record)\b", "record modification"),
)

PHONE_RE = re.compile(r"\b(\d{2,5})[- ]?(\d{3,5})[- ]?(\d{3,5})\b")
EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SECRET_RE = re.compile(r"\b(cvv|pin|otp|password|bank account|aadhaar|passport)\b[:\s]*\S*", re.IGNORECASE)


@dataclass
class GuardrailDecision:
    answer_type: str | None = None
    message: str = ""
    flags: list[str] = field(default_factory=list)
    matched_ids: list[str] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        return self.answer_type is not None


def scan_text_for_injections(text: str) -> list[str]:
    """Return the names of any instruction-override patterns found in the text."""
    hits: list[str] = []
    lowered = text.casefold()
    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            hits.append(label)
    return sorted(set(hits))


def find_student_ids(text: str) -> list[str]:
    return sorted({match.group(0) for match in STUDENT_ID_RE.finditer(text)})


def review_question(
    question: str,
    student_id: str | None,
    personal_terms: Iterable[str] = (),
) -> GuardrailDecision:
    """Decide whether a question may be answered at all (R7, R8).

    ``personal_terms`` are words that make a question personal ("my", "am I",
    "eligible"), supplied by the caller so this module stays free of routing
    logic.
    """
    decision = GuardrailDecision()
    lowered = question.casefold()

    requested_ids = find_student_ids(question)
    if requested_ids:
        if student_id and requested_ids == [student_id]:
            decision.flags.append("question names the logged-in student")
        else:
            decision.answer_type = "refused"
            decision.matched_ids = requested_ids
            decision.flags.append("request for another student's record")
            decision.message = (
                "I can only answer questions about the student who is signed in, and identity is taken "
                f"from the session, never from the message. I cannot look up {', '.join(requested_ids)}. "
                "If you are asking about your own record, sign in with your own account and ask again."
            )
            return decision

    for pattern, label in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            decision.answer_type = "refused"
            decision.flags.append(f"question contains a {label} attempt")
            decision.message = (
                "I do not follow instructions that ask me to ignore the rules, hide my sources or reveal "
                "internal configuration. I can answer questions about university rules and about your own "
                "academic records, with citations."
            )
            return decision

    for pattern, label in HUMAN_APPROVAL_ACTIONS:
        if re.search(pattern, lowered):
            decision.flags.append(f"requests a human-approved action ({label})")

    is_personal = bool(student_id) is False and any(term in lowered for term in personal_terms)
    if is_personal:
        decision.answer_type = "refused"
        decision.flags.append("personal question without an identity")
        decision.message = (
            "I need to know which student is asking before I can look at academic records. "
            "Sign in (the X-Student-Id header carries the identity) and ask again."
        )
    return decision


def mask_pii(text: str) -> str:
    """Mask personal and payment data before it reaches a log line."""
    masked = CARD_RE.sub("[REDACTED_CARD]", text)
    masked = EMAIL_RE.sub(lambda match: f"{match.group(1)}***{match.group(2)}", masked)
    masked = SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", masked)
    masked = PHONE_RE.sub(lambda match: f"******{match.group(3)[-4:]}", masked)
    return masked


def chunk_is_unsafe(text: str) -> bool:
    return bool(scan_text_for_injections(text))


def safe_audit_text(value: Any) -> str:
    return mask_pii(str(value))
