"""Question understanding, tool planning, answer composition, grounding checks.

Everything in this module has a deterministic path that runs with
``MOCK_LLM=true``. The model is used to *refine* two cheap decisions (the
question category and which of the already-allowed tools to call) and to phrase
the explanation; it can never introduce a number, because the composition and
verification steps below reject text that contains a figure not present in the
evidence or in a tool result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .schemas import NOT_FOUND_MESSAGE

CATEGORIES = (
    "policy_fact",
    "procedure",
    "personal_data",
    "personal_eligibility",
    "multi_step",
)

PERSONAL_MARKERS = (
    "my ",
    "my?",
    "am i",
    " i ",
    "i have",
    "i failed",
    "do i",
    "will i",
    "for me",
    "myself",
)
ELIGIBILITY_MARKERS = ("eligible", "eligibility", "qualify", "allowed to appear", "can i appear")
PROCEDURE_MARKERS = (
    "how do i",
    "how to",
    "how can i",
    "what is the process",
    "what is the procedure",
    "steps",
    "apply for",
    "application",
    "where do i",
    "whom do i",
)
DATA_MARKERS = (
    "my attendance",
    "my cgpa",
    "my marks",
    "my result",
    "my backlog",
    "how many classes",
    "how much attendance",
    "show me my",
    "what is my",
)
WHATIF_MARKERS = ("what if", "if i ", "suppose i", "will i be", "if i pass", "if i clear")
COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4}\d{3})\b")

TOPIC_TOOLS = {
    "supplementary": ["check_supplementary_eligibility", "get_results"],
    "placement": ["check_placement_eligibility", "get_active_backlogs"],
    "scholarship": ["check_scholarship_eligibility", "get_all_attendance"],
    "attendance": ["get_attendance", "check_exam_eligibility", "get_all_attendance"],
    "exam": ["check_exam_eligibility", "get_attendance"],
    "backlog": ["get_active_backlogs", "check_supplementary_eligibility"],
    "cgpa": ["get_student_profile", "get_active_backlogs"],
    "fee": ["get_fee_schedule"],
    "lab": ["check_lab_attendance_requirement", "get_all_attendance"],
}


@dataclass
class ToolPlan:
    calls: list[dict[str, Any]]
    source: str  # "llm" | "deterministic"
    notes: list[str]


def course_codes_in(question: str) -> list[str]:
    return [match.group(1).upper() for match in COURSE_CODE_RE.finditer(question)]


def classify_question(question: str, llm_result: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """Deterministic classification, with an optional model refinement."""
    lowered = f" {question.casefold()} "
    signals: list[str] = []
    is_personal = any(marker in lowered for marker in PERSONAL_MARKERS)
    is_data = any(marker in lowered for marker in DATA_MARKERS)
    is_eligibility = any(marker in lowered for marker in ELIGIBILITY_MARKERS)
    is_procedure = any(marker in lowered for marker in PROCEDURE_MARKERS)
    is_whatif = any(marker in lowered for marker in WHATIF_MARKERS)
    has_course = bool(course_codes_in(question))

    if is_personal:
        signals.append("personal wording")
    if is_eligibility:
        signals.append("eligibility wording")
    if is_procedure:
        signals.append("procedure wording")
    if is_whatif:
        signals.append("what-if wording")
    if has_course:
        signals.append("named course")

    if (is_personal and is_eligibility) or is_whatif:
        category = "multi_step" if is_whatif or is_procedure else "personal_eligibility"
    elif is_data or (is_personal and has_course):
        category = "personal_data"
    elif is_personal:
        category = "personal_eligibility"
    elif is_procedure:
        category = "procedure"
    else:
        category = "policy_fact"

    if llm_result:
        proposed = str(llm_result.get("category", "")).strip()
        if proposed in CATEGORIES and proposed != category:
            # Only accept a refinement that the deterministic signals do not contradict.
            if category == "policy_fact" or proposed in {"multi_step", "personal_eligibility", "personal_data"}:
                signals.append(f"model refined {category} -> {proposed}")
                category = proposed
    return category, signals


def deterministic_tool_candidates(question: str, category: str) -> list[str]:
    lowered = question.casefold()
    names: list[str] = []
    for topic, tools in TOPIC_TOOLS.items():
        if topic in lowered:
            names.extend(tools)
    if category in {"personal_data", "personal_eligibility", "multi_step"}:
        if not names:
            names.extend(["get_attendance", "check_exam_eligibility"])
        if category == "personal_data":
            names.extend(["get_student_profile", "get_all_attendance", "get_results"])
    if "supplementary" in lowered or "reappear" in lowered:
        names.extend(["check_supplementary_eligibility", "get_results"])
    if "hostel" in lowered:
        names = [name for name in names if name != "get_fee_schedule"]
    if any(marker in lowered for marker in WHATIF_MARKERS) and "attendance" in lowered:
        names.append("what_if_attendance")
    return list(dict.fromkeys(names))


def allowed_tool_specs(
    question: str, category: str, student_id: str | None, tool_specs: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = deterministic_tool_candidates(question, category)
    if not candidates:
        return []
    by_name = {spec["name"]: spec for spec in tool_specs}
    allowed = [by_name[name] for name in candidates if name in by_name]
    if not student_id:
        allowed = [spec for spec in allowed if spec["name"] not in _student_tools()]
    return allowed


def _student_tools() -> set[str]:
    from .tools import STUDENT_TOOLS

    return set(STUDENT_TOOLS)


CLASS_COUNT_RE = re.compile(r"(\d{1,3})\s+(?:more\s+)?classes")


def build_tool_plan(
    question: str,
    category: str,
    allowed: Sequence[dict[str, Any]],
    student_id: str | None,
    llm_result: dict[str, Any] | None = None,
) -> ToolPlan:
    """Validate a model-proposed tool plan; fall back to the deterministic one."""
    codes = course_codes_in(question)
    default_course = codes[0] if codes else None
    names = [spec["name"] for spec in allowed]
    class_match = CLASS_COUNT_RE.search(question)
    class_count = int(class_match.group(1)) if class_match else 0
    deterministic: list[dict[str, Any]] = []
    for name in names:
        arguments: dict[str, Any] = {}
        if name in {"get_attendance", "check_exam_eligibility", "check_supplementary_eligibility", "what_if_attendance"}:
            arguments = {"course_code": default_course} if default_course else {}
        if name == "what_if_attendance":
            # The assumption is stated in the answer (R6).
            arguments.update({"additional_held": class_count, "additional_attended": class_count})
        deterministic.append({"tool": name, "arguments": arguments})

    notes: list[str] = []
    if llm_result and isinstance(llm_result.get("tools"), list):
        proposed: list[dict[str, Any]] = []
        for entry in llm_result["tools"]:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("tool", "")).strip()
            if name not in names:
                notes.append(f"model proposed {name!r}, which is not allowed for this question; ignored")
                continue
            arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
            spec = next(spec for spec in allowed if spec["name"] == name)
            required = spec.get("parameters", {}).get("required", [])
            if any(key not in arguments or arguments.get(key) in ("", None) for key in required):
                notes.append(f"model call to {name} was missing a required argument; ignored")
                continue
            course = arguments.get("course_code")
            if course and not COURSE_CODE_RE.match(str(course)):
                notes.append(f"model call to {name} had an invalid course code; ignored")
                continue
            for key, value in arguments.items():
                if key in {"additional_held", "additional_attended"}:
                    try:
                        arguments[key] = int(value)
                    except (TypeError, ValueError):
                        notes.append(f"model call to {name} had a non-integer {key}; defaulted to 0")
                        arguments[key] = 0
            proposed.append({"tool": name, "arguments": arguments})
        if proposed:
            return ToolPlan(calls=proposed[:4], source="llm", notes=notes)

    return ToolPlan(calls=deterministic[:4], source="deterministic", notes=notes)


# --------------------------------------------------------------------------- #
# answer composition
# --------------------------------------------------------------------------- #

ELIGIBILITY_HEADLINES = {
    "ELIGIBLE": "You are eligible",
    "NOT_ELIGIBLE": "You are not eligible",
    "NOT_ELIGIBLE_CONDONATION_REQUIRED": "You are not eligible at present",
    "NOT_ELIGIBLE_ATTENDANCE": "You are not eligible at present",
    "NOT_APPLICABLE": "This does not apply to you",
}


def tool_sentence(invocation: dict[str, Any]) -> str:
    """Render one tool result as a sentence, using only the tool's own numbers."""
    tool = invocation["tool"]
    output = invocation.get("output") or {}
    if tool == "get_attendance":
        return (
            f"Your attendance in {output.get('course_code')} is {output.get('attendance_pct')}% "
            f"({output.get('classes_attended')} of {output.get('classes_held')} classes)."
        )
    if tool == "get_all_attendance":
        return (
            f"Across your registered courses your attendance runs from "
            f"{output.get('lowest_attendance_pct')}% to {output.get('highest_attendance_pct')}%."
        )
    if tool == "check_exam_eligibility":
        headline = ELIGIBILITY_HEADLINES.get(str(output.get("result")), "Eligibility result")
        return (
            f"{headline} to appear for the end-semester examination of {output.get('course_code')}: "
            f"attendance {output.get('attendance_pct')}% against the required "
            f"{output.get('required_pct')}%."
        )
    if tool == "check_supplementary_eligibility":
        headline = ELIGIBILITY_HEADLINES.get(str(output.get("result")), "Eligibility result")
        return (
            f"{headline} for the supplementary examination of {output.get('course_code')} "
            f"(latest result: {output.get('latest_result')})."
        )
    if tool == "check_placement_eligibility":
        headline = ELIGIBILITY_HEADLINES.get(str(output.get("result")), "Eligibility result")
        return (
            f"{headline} for campus placement drives: CGPA {output.get('cgpa')} against the minimum "
            f"{output.get('min_cgpa')}, with {output.get('active_backlogs')} active backlog(s)."
        )
    if tool == "check_scholarship_eligibility":
        headline = ELIGIBILITY_HEADLINES.get(str(output.get("result")), "Eligibility result")
        return (
            f"{headline} for the merit scholarship: CGPA {output.get('cgpa')} against the minimum "
            f"{output.get('min_cgpa')}, lowest course attendance {output.get('lowest_attendance_pct')}% "
            f"against the required {output.get('min_attendance_pct')}%."
        )
    if tool == "get_student_profile":
        return (
            f"Your record shows programme {output.get('programme')}, batch {output.get('batch_year')}, "
            f"semester {output.get('current_semester')} and CGPA {output.get('cgpa')}."
        )
    if tool == "get_active_backlogs":
        return f"You have {output.get('active_backlogs')} active backlog(s)."
    if tool == "get_fee_schedule":
        parts = [
            f"{key} = {value.get('value')}" for key, value in output.items() if isinstance(value, dict)
        ]
        return "Fees in force: " + ", ".join(parts) + "." if parts else ""
    if tool == "evaluate_course_outcome":
        base = (
            f"For {output.get('course_code')} the marks give {output.get('computed_result')} "
            f"(total {output.get('total_marks')}, paper {output.get('external_marks')} against a floor of "
            f"{output.get('end_sem_paper_floor')}); the stored result is {output.get('stored_result')}."
        )
        return base
    if tool == "check_lab_attendance_requirement":
        if output.get("applies"):
            return (
                f"A departmental rule requires {output.get('value')}% laboratory attendance for "
                f"{output.get('rule_id')}."
            )
        return str(output.get("reason", ""))
    if tool == "what_if_attendance":
        return (
            f"With {output.get('additional_classes_attended_assumed')} of the next "
            f"{output.get('additional_classes_assumed')} classes attended your attendance becomes "
            f"{output.get('projected_pct')}%."
        )
    return ""


UPCOMING_MARKERS = ("upcoming", "coming", "new circular", "change", "from next", "recent notification", "amended")


def compose_answer(
    question: str,
    category: str,
    tool_invocations: Sequence[dict[str, Any]],
    citations: Sequence[dict[str, Any]],
    conflicts: Sequence[dict[str, Any]],
    unresolved_conflict: bool,
    kept_clauses: Sequence[dict[str, Any]],
    rule_values: Sequence[dict[str, Any]],
    upcoming: Sequence[str] = (),
) -> tuple[str, str, str]:
    """Return ``(answer, answer_type, explanation)`` using deterministic text.

    ``kept_clauses`` are the clauses that survived retrieval and precedence, in
    score order, each with ``doc_id``, ``section`` and ``text``.
    """
    successful = [item for item in tool_invocations if item.get("status") == "ok" and item.get("output")]
    sentences = [sentence for sentence in (tool_sentence(item) for item in successful) if sentence]
    upcoming_note = ""
    if upcoming:
        joined = "; ".join(upcoming)
        upcoming_note = f"Not yet in force and therefore not applied to this answer: {joined}."

    if unresolved_conflict:
        sources = ", ".join(
            sorted({f"{conflict.get('doc_ids', ['?'])[0]} and others" for conflict in conflicts})
        )
        answer = (
            "The authorised sources disagree on this point and the Source Precedence Policy cannot "
            f"resolve the disagreement ({sources}). Both documents are cited above; please contact the "
            "issuing office for a ruling."
        )
        return answer, "conflict_flagged", "Conflicting documents were retrieved and precedence could not settle them."

    if not sentences and not kept_clauses:
        return NOT_FOUND_MESSAGE, "not_found", "No authorised source matched this question."

    if sentences:
        answer = " ".join(sentences)
        explanation_parts = []
        for invocation in successful:
            output = invocation.get("output") or {}
            note = output.get("note")
            if note:
                explanation_parts.append(str(note))
        for rule in rule_values:
            explanation_parts.append(
                f"Rule {rule.get('rule_id')} ({rule.get('value')}) comes from "
                f"{rule.get('source_doc_id')} clause {rule.get('source_section')}, "
                f"effective {rule.get('effective_from')}."
            )
        if upcoming_note:
            explanation_parts.append(upcoming_note)
            if any(marker in question.casefold() for marker in UPCOMING_MARKERS):
                answer = f"{answer} {upcoming_note}"
        return answer, "calculated", " ".join(explanation_parts)

    primary = kept_clauses[0]
    answer = _first_sentences(str(primary.get("text", "")), 3)
    # A clause that continues the same document usually carries the rest of the
    # procedure or the table that belongs with the answer.
    for clause in kept_clauses[1:3]:
        if clause.get("doc_id") == primary.get("doc_id") and clause.get("text"):
            follow_up = _first_sentences(str(clause["text"]), 1)
            if follow_up and follow_up not in answer:
                answer = f"{answer} {follow_up}"

    explanation = "Answer taken from the cited clause; no calculation was required."
    if rule_values:
        rule = rule_values[0]
        explanation = (
            f"Rule {rule.get('rule_id')} = {rule.get('value')} from {rule.get('source_doc_id')} "
            f"clause {rule.get('source_section')}."
        )
    if upcoming_note:
        explanation = f"{explanation} {upcoming_note}"
        if any(marker in question.casefold() for marker in UPCOMING_MARKERS):
            answer = f"{answer} {upcoming_note}"
    return answer, "retrieved_fact", explanation


def _first_sentences(text: str, count: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    sentences = re.split(r"(?<=[.;])\s+", cleaned)
    return " ".join(sentences[:count]).strip()


# --------------------------------------------------------------------------- #
# grounding verification
# --------------------------------------------------------------------------- #

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_in_text(text: str) -> set[str]:
    return {_normalise_number(match.group(0)) for match in NUMBER_RE.finditer(text)}


def _normalise_number(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def evidence_numbers(
    kept_texts: Sequence[str],
    tool_invocations: Sequence[dict[str, Any]],
    rule_values: Sequence[dict[str, Any]],
) -> set[str]:
    pool: set[str] = set()
    for text in kept_texts:
        pool |= numbers_in_text(str(text))
    for invocation in tool_invocations:
        pool |= numbers_in_text(_flatten(invocation.get("output") or {}))
        pool |= numbers_in_text(str(invocation.get("input") or {}))
    for rule in rule_values:
        pool |= numbers_in_text(str(rule))
    # Dates and clause numbers are always allowed.
    pool |= {"7.2", "7.3", "7.4", "8.3", "9.1", "9.2", "2.1", "3.2", "3.3", "1.1", "1.3"}
    return pool


def verify_grounding(answer: str, pool: set[str]) -> tuple[bool, list[str]]:
    """Every figure in the answer must exist in the evidence or in a tool result."""
    unsupported = sorted(number for number in numbers_in_text(answer) if number not in pool)
    return not unsupported, unsupported


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def needs_student(category: str, question: str) -> bool:
    lowered = f" {question.casefold()} "
    return (
        category in {"personal_data", "personal_eligibility", "multi_step"}
        or any(marker in lowered for marker in PERSONAL_MARKERS)
    )
