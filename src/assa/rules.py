"""Rule registry access (Annex C).

Every threshold a tool uses is read from the ``rule_registry`` table through this
module. No module in this package hard-codes a policy number, and every registry
row carries the ``source_doc_id`` and ``source_section`` of the clause it came
from. When a new circular changes a threshold, a new row is added (with
``supersedes_rule_id`` pointing at the old row) and no code changes.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

from .precedence import RuleResolution, resolve_parameter


class RuleNotFound(LookupError):
    """Raised when no rule in force can be found for a parameter."""

    def __init__(self, parameter: str, as_of_date: date) -> None:
        super().__init__(
            f"No rule for parameter {parameter!r} is in force on {as_of_date.isoformat()}. "
            "Add a cited clause to the rule registry."
        )
        self.parameter = parameter


def parameter_rules(conn: sqlite3.Connection, parameter: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT r.rule_id, r.description, r.parameter, r.operator, r.value,
               r.scope_programmes, r.scope_batches, r.effective_from, r.effective_to,
               r.source_doc_id, r.source_section, r.supersedes_rule_id, r.notes,
               COALESCE(s.authority_level, 3) AS authority_level
        FROM rule_registry AS r
        LEFT JOIN source_register AS s ON s.doc_id = r.source_doc_id
        WHERE r.parameter = ?
        ORDER BY r.effective_from ASC
        """,
        (parameter,),
    ).fetchall()
    return [dict(row) for row in rows]


def resolve(
    conn: sqlite3.Connection,
    parameter: str,
    as_of_date: date,
    programme: str | None = None,
    batch_year: int | str | None = None,
) -> RuleResolution:
    return resolve_parameter(
        parameter_rules(conn, parameter),
        parameter=parameter,
        as_of_date=as_of_date,
        programme=programme,
        batch_year=batch_year,
    )


def require_value(
    conn: sqlite3.Connection,
    parameter: str,
    as_of_date: date,
    programme: str | None = None,
    batch_year: int | str | None = None,
) -> tuple[float, dict[str, Any], RuleResolution]:
    """Return ``(value, winning_rule, resolution)`` or raise ``RuleNotFound``."""
    resolution = resolve(conn, parameter, as_of_date, programme, batch_year)
    if resolution.status != "resolved" or resolution.value is None or resolution.winner is None:
        raise RuleNotFound(parameter, as_of_date)
    return resolution.value, resolution.winner, resolution


def applied_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """Shape a registry row as the ``applied_rules`` entry of POST /ask."""
    return {
        "rule_id": rule["rule_id"],
        "parameter": rule.get("parameter"),
        "value": _with_operator(rule),
        "source_doc_id": rule.get("source_doc_id"),
        "source_section": rule.get("source_section"),
        "effective_from": rule.get("effective_from"),
    }


def _with_operator(rule: dict[str, Any]) -> str:
    operator = (rule.get("operator") or "").strip()
    value = str(rule.get("value", "")).strip()
    if operator in {"", "=="}:
        return value
    return f"{operator} {value}"


def describe_rule(rule: dict[str, Any]) -> str:
    return (
        f"{rule['rule_id']} ({_with_operator(rule)}) from "
        f"{rule.get('source_doc_id')}#{rule.get('source_section')}"
    )


def upcoming_rules(conn: sqlite3.Connection, parameter: str, as_of_date: date) -> list[dict[str, Any]]:
    """Rules registered for this parameter that are not yet in force."""
    return [
        rule
        for rule in parameter_rules(conn, parameter)
        if (rule.get("effective_from") or "") > as_of_date.isoformat()
    ]


def rule_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM rule_registry ORDER BY parameter, effective_from").fetchall()
    return [dict(row) for row in rows]
