"""Token-cost calculations (Task 2) and JSON-validity scoring (Task 3, v3)."""

from __future__ import annotations

import json
from typing import Any

from eduassist.config import PRICING


def compute_cost(input_tokens: int, output_tokens: int) -> dict[str, float]:
    """Return input/output/total cost in USD for one request.

    Input Cost = input_tokens / 1,000,000 * input_price_per_million
    Output Cost = output_tokens / 1,000,000 * output_price_per_million
    """
    input_cost = (input_tokens / 1_000_000) * PRICING["input_price_per_million"]
    output_cost = (output_tokens / 1_000_000) * PRICING["output_price_per_million"]
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def projected_monthly_cost(avg_cost_per_request: float, requests: int) -> float:
    """Projected Monthly Cost = Average Cost per Request x requests."""
    return avg_cost_per_request * requests


REQUIRED_JSON_KEYS = {"answer", "confidence", "key_points"}
VALID_CONFIDENCE_VALUES = {"high", "medium", "low"}


def is_valid_schema_json(raw_text: str) -> bool:
    """Strictly check whether `raw_text` is valid JSON matching the v3 schema.

    A response only counts as valid if it:
      1. Parses as JSON on its own (no markdown/code-fence stripping - the v3
         prompt explicitly asks the model not to use them, so leniency here
         would hide real compliance failures).
      2. Is a JSON object with exactly the three required keys.
      3. Has `confidence` in {"high", "medium", "low"} and `key_points` as a list.
    """
    if not raw_text or not raw_text.strip():
        return False
    try:
        parsed: Any = json.loads(raw_text.strip())
    except (json.JSONDecodeError, ValueError):
        return False

    if not isinstance(parsed, dict):
        return False
    if not REQUIRED_JSON_KEYS.issubset(parsed.keys()):
        return False
    if not isinstance(parsed.get("answer"), str):
        return False
    if parsed.get("confidence") not in VALID_CONFIDENCE_VALUES:
        return False
    if not isinstance(parsed.get("key_points"), list):
        return False
    return True
