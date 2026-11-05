"""Orchestration helper shared by Task 1/2 (prompt v1 only) and Task 3 (v1/v2/v3)."""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from eduassist.config import PROMPT_TEMPLATES, QUESTIONS
from eduassist.costs import compute_cost, is_valid_schema_json
from eduassist.llm_client import call_llm


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).isoformat(timespec="milliseconds") if ts else ""


def run_prompt_version(prompt_version: str, questions: Optional[list[dict]] = None) -> list[dict]:
    """Run every question in `questions` (default: the 5 assignment questions)
    through the given prompt version ("v1" | "v2" | "v3").

    Returns one record per question with timing, token, cost, and JSON-validity
    fields - the single schema shared by both the Task 1/2 and Task 3 CSV exports.
    """
    questions = questions if questions is not None else QUESTIONS
    template = PROMPT_TEMPLATES[prompt_version]["template"]
    records = []
    for q in questions:
        prompt_text = template.replace("{{question}}", q["question"])
        result = call_llm(prompt_text)
        cost = compute_cost(result.input_tokens, result.output_tokens)
        records.append(
            {
                "question_id": q["question_id"],
                "topic": q["topic"],
                "question": q["question"],
                "prompt_version": prompt_version,
                "model_name": result.model_name,
                "call_path": result.call_path,
                "start_time": _iso(result.start_time),
                "end_time": _iso(result.end_time),
                "latency_s": round(result.latency_s, 4),
                "status": result.status,
                "error_message": result.error_message or "",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
                "input_cost_usd": cost["input_cost_usd"],
                "output_cost_usd": cost["output_cost_usd"],
                "total_cost_usd": cost["total_cost_usd"],
                "valid_json": (
                    is_valid_schema_json(result.response_text) if result.status == "success" else False
                ),
                "response_text": result.response_text,
            }
        )
    return records
