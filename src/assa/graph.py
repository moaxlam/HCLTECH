"""The LangGraph workflow.

One fixed workflow, not a swarm. Each node does one thing, every node has a
deterministic path, and the only decisions the model is allowed to influence are
the question category, which of the already-allowed tools to call, and the
wording of the explanation. Everything authoritative -- retrieval filtering,
precedence, thresholds, eligibility, arithmetic, abstention -- is code.

    classify -> security_check -> student_context -> retrieve -> plan_tools
             -> run_tools -> compose -> verify_grounding -> audit -> END

The graph is linear except for the refusal branch after ``security_check`` and the
clarification branch in ``run_tools``.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from datetime import date, datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import db, guardrails, reasoning, rules, tools
from .config import get_settings
from .llm import LLMClient
from .precedence import parse_iso
from .retrieval import RetrievalResult, Retriever
from .schemas import NOT_FOUND_MESSAGE, AskRequest, AskResponse
from .vectorstore import VectorStore

CLASSIFY_PROMPT = """You label a student's question for a university services assistant.

Answer with one JSON object: {{"category": "<one of policy_fact, procedure, personal_data,
personal_eligibility, multi_step>"}}

policy_fact            a general question about a rule, with no personal situation
procedure              how to do something: steps, forms, where to apply
personal_data          asks for the student's own records (attendance, marks, CGPA)
personal_eligibility   asks whether the student qualifies under a rule
multi_step             needs several rules and the student's records, including what-if

Question: {question}
"""

TOOL_PROMPT = """You choose which of the offered tools to run for a student-services question.

Use only the tools listed below, and only with an argument that appears in the question.
Answer with one JSON object: {{"tools": [{{"tool": "name", "arguments": {{...}}}}]}}

Tools:
{tools}

Question: {question}
Category: {category}
"""

EXPLAIN_PROMPT = """You explain an eligibility or policy result to a university student in one or two short
sentences of plain English.

Rules for your explanation:
- Use only the facts given below. Never introduce a number, date, clause or document that is not given.
- Do not repeat the answer sentence, add context on what it means for the student.
- Do not invent contacts, exceptions, approvals or promises.
- Polite, neutral tone. Never blame the student, a teacher or an office.

Answer already given: {answer}
Facts:
{facts}

Explanation:
"""


class AskState(TypedDict, total=False):
    trace_id: str
    question: str
    as_of_date: str
    student_id: str | None
    programme: str | None
    batch_year: int | None
    category: str
    signals: list[str]
    started: float
    guardrail_flags: list[str]
    blocked: bool
    blocked_message: str
    blocked_type: str
    clarification: str
    retrieval: dict[str, Any]
    sources_retrieved: list[dict[str, Any]]
    tool_plan: dict[str, Any]
    tool_invocations: list[dict[str, Any]]
    applied_rules: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    unresolved_conflict: bool
    citations: list[dict[str, Any]]
    answer: str
    answer_type: str
    explanation: str
    model: str
    llm_calls: int
    tokens: int
    notes: list[str]


class AssistantWorkflow:
    def __init__(
        self,
        conn: sqlite3.Connection,
        store: VectorStore,
        llm: LLMClient | None = None,
    ) -> None:
        self.conn = conn
        self.store = store
        self.llm = llm or LLMClient()
        self.retriever = Retriever(conn, store)
        self.graph = self._build()

    # -- nodes --------------------------------------------------------------
    def classify(self, state: AskState) -> dict[str, Any]:
        llm_result = None
        if self.llm.enabled:
            llm_result = self.llm.generate_json(
                CLASSIFY_PROMPT.format(question=state["question"])
            )
        category, signals = reasoning.classify_question(state["question"], llm_result)
        return {"category": category, "signals": signals}

    def security_check(self, state: AskState) -> dict[str, Any]:
        decision = guardrails.review_question(
            state["question"],
            state.get("student_id"),
            personal_terms=reasoning.PERSONAL_MARKERS,
        )
        flags = list(decision.flags)
        if decision.blocks:
            return {
                "blocked": True,
                "blocked_message": decision.message,
                "blocked_type": decision.answer_type or "refused",
                "guardrail_flags": flags,
            }
        if reasoning.needs_student(state.get("category", ""), state["question"]) and not state.get(
            "student_id"
        ):
            return {
                "blocked": True,
                "blocked_type": "refused",
                "blocked_message": (
                    "This question needs your academic records, so I need to know which student is asking. "
                    "Identity comes only from the request context (the X-Student-Id header), never from the "
                    "question text. Sign in and ask again."
                ),
                "guardrail_flags": flags + ["personal question without identity"],
            }
        return {"blocked": False, "guardrail_flags": flags}

    def student_context(self, state: AskState) -> dict[str, Any]:
        student_id = state.get("student_id")
        if not student_id:
            return {}
        ctx = tools.ToolContext(
            conn=self.conn,
            as_of_date=parse_iso(state["as_of_date"]) or date.today(),
            student_id=student_id,
        )
        invocation = tools.run_tool(ctx, "get_student_profile")
        if invocation["status"] != "ok":
            return {
                "blocked": True,
                "blocked_type": "refused",
                "blocked_message": (
                    f"I cannot find a student record for the signed-in identity {student_id}. "
                    "Ask the academic section to check the record, or load the student file again."
                ),
                "guardrail_flags": list(state.get("guardrail_flags", []))
                + ["unrecognised student identity"],
            }
        output = invocation["output"]
        return {
            "programme": output.get("programme"),
            "batch_year": output.get("batch_year"),
        }

    def retrieve(self, state: AskState) -> dict[str, Any]:
        # Scope-aware retrieval: a student is answered from documents that cover
        # their programme and batch, a general question from general documents.
        result = self.retriever.retrieve(
            state["question"],
            as_of_date=parse_iso(state["as_of_date"]) or date.today(),
            programme=state.get("programme"),
            batch_year=state.get("batch_year"),
        )
        return {
            "retrieval": _retrieval_payload(result),
            "citations": result.citations,
            "applied_rules": list(result.applied_rules),
            "conflicts": list(result.conflicts),
            "unresolved_conflict": result.unresolved_conflict,
            "sources_retrieved": result.sources_retrieved,
            "guardrail_flags": list(state.get("guardrail_flags", []))
            + ([f"instruction found in {', '.join(result.unsafe_chunks)}"] if result.unsafe_chunks else []),
        }

    def plan_tools(self, state: AskState) -> dict[str, Any]:
        allowed = reasoning.allowed_tool_specs(
            state["question"], state.get("category", "policy_fact"), state.get("student_id"), tools.TOOL_SPECS
        )
        llm_result = None
        if allowed and self.llm.enabled:
            llm_result = self.llm.generate_json(
                TOOL_PROMPT.format(
                    tools="\n".join(f"- {spec['name']}: {spec['description']}" for spec in allowed),
                    question=state["question"],
                    category=state.get("category"),
                )
            )
        plan = reasoning.build_tool_plan(
            state["question"],
            state.get("category", "policy_fact"),
            allowed,
            state.get("student_id"),
            llm_result,
        )
        return {"tool_plan": {"calls": plan.calls, "source": plan.source}, "notes": plan.notes}

    def run_tools(self, state: AskState) -> dict[str, Any]:
        calls = (state.get("tool_plan") or {}).get("calls", [])
        if not calls:
            return {"tool_invocations": [], "applied_rules": list(state.get("applied_rules", []))}

        ctx = tools.ToolContext(
            conn=self.conn,
            as_of_date=parse_iso(state["as_of_date"]) or date.today(),
            student_id=state.get("student_id"),
            question=state["question"],
            programme=state.get("programme"),
            batch_year=state.get("batch_year"),
        )
        invocations: list[dict[str, Any]] = []
        applied: list[dict[str, Any]] = list(state.get("applied_rules", []))
        notes: list[str] = list(state.get("notes", []))
        missing_course = False

        for call in calls:
            name = call["tool"]
            arguments = dict(call.get("arguments") or {})
            needs_course = name in {
                "get_attendance",
                "check_exam_eligibility",
                "check_supplementary_eligibility",
                "what_if_attendance",
            }
            if needs_course and not arguments.get("course_code"):
                missing_course = True
                invocations.append(
                    {
                        "tool": name,
                        "input": arguments,
                        "output": {},
                        "status": "skipped",
                        "ms": 0,
                        "error": "no course was named in the question",
                    }
                )
                continue
            invocation = tools.run_tool(ctx, name, arguments)
            invocations.append(invocation)
            for rule in invocation.get("applied_rules", []) or []:
                applied.append(rule)
            for note in invocation.get("notes", []) or []:
                if note not in notes:
                    notes.append(note)

        update: dict[str, Any] = {
            "tool_invocations": invocations,
            "applied_rules": applied,
            "notes": notes,
        }
        if missing_course and not any(item["status"] == "ok" for item in invocations):
            courses = self._student_courses(state.get("student_id"))
            update["clarification"] = (
                "Which course do you mean? " + ("Your registered courses are " + ", ".join(courses) + "." if courses else "")
            ).strip()
        return update

    def compose(self, state: AskState) -> dict[str, Any]:
        if state.get("clarification"):
            return {
                "answer": state["clarification"],
                "answer_type": "clarification_needed",
                "explanation": "The question did not name a course, so the tool could not be run.",
            }

        retrieval = state.get("retrieval") or {}
        kept_texts: list[str] = retrieval.get("kept_texts", [])
        rule_values = [rule for rule in state.get("applied_rules", [])]
        answer, answer_type, explanation = reasoning.compose_answer(
            state["question"],
            state.get("category", "policy_fact"),
            state.get("tool_invocations", []),
            state.get("citations", []),
            state.get("conflicts", []),
            bool(state.get("unresolved_conflict")),
            retrieval.get("kept", []),
            rule_values,
            retrieval.get("upcoming_notes", []),
        )

        update: dict[str, Any] = {
            "answer": answer,
            "answer_type": answer_type,
            "explanation": explanation,
        }
        if not self.llm.enabled or answer_type not in {"calculated", "retrieved_fact"}:
            return update

        facts = "\n".join(
            ["Retrieved clauses:"]
            + [f"- {text[:400]}" for text in kept_texts[:3]]
            + ["Rule values:"]
            + [f"- {rule}" for rule in rule_values]
            + ["Tool results:"]
            + [
                f"- {item['tool']}: {item['output']}"
                for item in state.get("tool_invocations", [])
                if item.get("status") == "ok"
            ]
        )
        reply = self.llm.generate(
            EXPLAIN_PROMPT.format(answer=answer, facts=facts[:6000]), temperature=0.2
        )
        if reply is None or not reply.text.strip():
            return update
        candidate = reply.text.strip()
        pool = reasoning.evidence_numbers(kept_texts, state.get("tool_invocations", []), rule_values)
        ok, unsupported = reasoning.verify_grounding(candidate, pool)
        if not ok:
            notes = list(state.get("notes", []))
            notes.append(
                "model explanation rejected: it contained figures not present in the evidence "
                f"({', '.join(unsupported)})"
            )
            update["notes"] = notes
            return update
        update["explanation"] = candidate
        return update

    def verify_grounding(self, state: AskState) -> dict[str, Any]:
        answer_type = state.get("answer_type", "not_found")
        citations = state.get("citations", [])
        notes = list(state.get("notes", []))
        retrieval = state.get("retrieval") or {}
        kept_texts: list[str] = retrieval.get("kept_texts", [])

        if answer_type == "retrieved_fact" and not citations:
            notes.append("no citation could be attached to a factual answer; abstaining")
            return {"answer": NOT_FOUND_MESSAGE, "answer_type": "not_found", "notes": notes}

        if state.get("unresolved_conflict") and answer_type not in {"refused", "clarification_needed"}:
            return {"answer_type": "conflict_flagged"}

        if answer_type in {"calculated", "retrieved_fact"}:
            pool = reasoning.evidence_numbers(
                kept_texts, state.get("tool_invocations", []), state.get("applied_rules", [])
            )
            ok, unsupported = reasoning.verify_grounding(state.get("answer", ""), pool)
            if not ok:
                notes.append(
                    "answer rejected by the grounding check for unsupported figures "
                    f"({', '.join(unsupported)}); abstaining"
                )
                return {"answer": NOT_FOUND_MESSAGE, "answer_type": "not_found", "notes": notes}

        return {"notes": notes}

    def audit(self, state: AskState) -> dict[str, Any]:
        latency_ms = int((time.perf_counter() - state.get("started", time.perf_counter())) * 1000)
        record = {
            "trace_id": state["trace_id"],
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "student_id": state.get("student_id"),
            "question": guardrails.safe_audit_text(state["question"]),
            "question_category": state.get("category", ""),
            "sources_retrieved": state.get("sources_retrieved", []),
            "precedence_decision": (state.get("retrieval") or {}).get("decision_text", ""),
            "tools_invoked": [
                {
                    "tool": item["tool"],
                    "status": item.get("status", "ok"),
                    "ms": item.get("ms", 0),
                    "input": item.get("input", {}),
                    "output": item.get("output", {}),
                }
                for item in state.get("tool_invocations", [])
            ],
            "applied_rules": state.get("applied_rules", []),
            "conflicts_detected": state.get("conflicts", []),
            "answer_type": state.get("answer_type", ""),
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "model": self.llm.stats.model if self.llm.enabled else "MOCK_LLM",
            "llm_calls": self.llm.stats.calls,
            "tokens": self.llm.stats.tokens,
            "latency_ms": latency_ms,
            "guardrail_flags": state.get("guardrail_flags", []),
            "notes": "; ".join(state.get("notes", []) or []),
        }
        db.insert_audit(self.conn, record)
        return {"model": record["model"], "llm_calls": record["llm_calls"], "tokens": record["tokens"]}

    # -- graph --------------------------------------------------------------
    def _build(self):
        graph = StateGraph(AskState)
        graph.add_node("classify", self.classify)
        graph.add_node("security_check", self.security_check)
        graph.add_node("student_context", self.student_context)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("plan_tools", self.plan_tools)
        graph.add_node("run_tools", self.run_tools)
        graph.add_node("compose", self.compose)
        graph.add_node("verify_grounding", self.verify_grounding)
        graph.add_node("audit", self.audit)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "security_check")
        graph.add_conditional_edges(
            "security_check",
            lambda state: "blocked" if state.get("blocked") else "continue",
            {"blocked": "verify_grounding", "continue": "student_context"},
        )
        graph.add_conditional_edges(
            "student_context",
            lambda state: "blocked" if state.get("blocked") else "continue",
            {"blocked": "verify_grounding", "continue": "retrieve"},
        )
        graph.add_edge("retrieve", "plan_tools")
        graph.add_edge("plan_tools", "run_tools")
        graph.add_edge("run_tools", "compose")
        graph.add_edge("compose", "verify_grounding")
        graph.add_edge("verify_grounding", "audit")
        graph.add_edge("audit", END)
        return graph.compile()

    # -- helpers ------------------------------------------------------------
    def _student_courses(self, student_id: str | None) -> list[str]:
        if not student_id:
            return []
        rows = self.conn.execute(
            """
            SELECT a.course_code FROM attendance AS a
            JOIN students AS s ON s.student_id = a.student_id
            JOIN courses AS c ON c.course_code = a.course_code
            WHERE a.student_id = ? AND (c.programme = s.programme OR c.programme = 'ALL')
            ORDER BY a.course_code
            """,
            (student_id,),
        ).fetchall()
        return [str(row["course_code"]) for row in rows]


def _retrieval_payload(result: RetrievalResult) -> dict[str, Any]:
    return {
        "topic": result.question_topic,
        "kept_texts": [candidate.text for candidate in result.kept],
        "kept": [candidate.as_dict() for candidate in result.kept],
        "dropped": [candidate.as_dict() for candidate in result.dropped],
        "upcoming": [candidate.as_dict() for candidate in result.upcoming],
        "decision_text": result.decision_text,
        "has_evidence": result.has_evidence,
        "parameter_values": result.parameter_values,
        "upcoming_notes": result.upcoming_notes,
        "unsafe_chunks": result.unsafe_chunks,
    }


def run_ask(
    conn: sqlite3.Connection,
    store: VectorStore,
    request: AskRequest,
    student_id: str | None,
    workflow: AssistantWorkflow | None = None,
) -> AskResponse:
    """Run one question end to end and return the contracted response."""
    settings = get_settings()
    as_of = request.as_of_date or date.today()
    # Identity comes only from the request context, never from the body (R7).
    effective_student = student_id or None
    trace_id = uuid.uuid4().hex[:8]
    workflow = workflow or AssistantWorkflow(conn, store)

    initial: AskState = {
        "trace_id": trace_id,
        "question": request.question,
        "as_of_date": as_of.isoformat(),
        "student_id": effective_student,
        "started": time.perf_counter(),
        "guardrail_flags": [],
        "notes": [],
        "citations": [],
        "applied_rules": [],
        "conflicts": [],
        "tool_invocations": [],
    }
    final: AskState = workflow.graph.invoke(initial)

    answer_type = final.get("answer_type", "not_found")
    answer = final.get("answer") or NOT_FOUND_MESSAGE
    if final.get("blocked"):
        answer_type = final.get("blocked_type", "refused")
        answer = final.get("blocked_message", answer)

    # A response that cannot clear the confidence floor is escalated, not guessed.
    explanation = final.get("explanation", "")
    if answer_type == "retrieved_fact" and not final.get("citations"):
        answer_type = "not_found"
        answer = NOT_FOUND_MESSAGE

    return AskResponse(
        trace_id=trace_id,
        answer=answer,
        answer_type=answer_type,  # type: ignore[arg-type]
        citations=final.get("citations", [])[:4],
        tools_invoked=[_invocation_payload(item) for item in final.get("tool_invocations", [])],
        applied_rules=_dedupe_rules(final.get("applied_rules", [])),
        conflicts_detected=final.get("conflicts", []),
        explanation=explanation,
        as_of_date=as_of.isoformat(),
    )


def _invocation_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "tool": item["tool"],
        "input": item.get("input", {}),
        "output": item.get("output", {}),
        "status": item.get("status", "ok"),
        "ms": item.get("ms", 0),
    }
    if item.get("error"):
        payload["error"] = item["error"]
    return payload


def _dedupe_rules(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for entry in entries:
        seen[str(entry.get("rule_id"))] = entry
    return list(seen.values())


def build_services(conn: sqlite3.Connection | None = None, store: VectorStore | None = None):
    """Convenience factory used by the API, the CLI and the tests."""
    settings = get_settings()
    connection = conn or db.connect(settings.sqlite_path)
    db.init_db(connection)
    vector_store = store or VectorStore()
    return connection, vector_store, AssistantWorkflow(connection, vector_store)


def rule_registry_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return rules.rule_summary(conn)
