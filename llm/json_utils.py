"""Robust extraction of JSON from raw LLM text.

Handles the *safe* recoveries — markdown code fences, and JSON preceded/followed
by prose — WITHOUT ever fabricating structure. Every path returns only JSON that
``json.loads`` accepts verbatim on a real substring of the input; nothing closes
unbalanced brackets or trims to "the last good value". Truncated output (cut off
at max_tokens) is detected upstream via the provider ``finish_reason`` and
rejected by the router (CLAUDE.md Zero Silent Failure) — so heuristic repair of
incomplete JSON, which could serve a partial clinical annotation as if complete,
is deliberately NOT done here.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json(raw: str) -> Any:
    """Parse the first complete JSON value out of ``raw``.

    Raises ``json.JSONDecodeError`` if no complete, verbatim-parseable JSON value
    can be found. Never repairs/closes truncated JSON.
    """
    text = (raw or "").strip()

    # Strip markdown code fences (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        inner = [line for line in lines[1:] if line.strip() != "```"]
        text = "\n".join(inner).strip()

    # Strict parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Trailing prose after the JSON: raw_decode stops after the first COMPLETE
    # value (a truncated value still fails here — no repair).
    try:
        result, _ = json.JSONDecoder().raw_decode(text)
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Prose BEFORE the JSON: decode from the earliest opener so we grab the
    # outermost structure (not a nested array inside an object).
    opener_positions = [pos for pos in (text.find("["), text.find("{")) if pos != -1]
    if opener_positions:
        start = min(opener_positions)
        try:
            result, _ = json.JSONDecoder().raw_decode(text[start:])
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    raise json.JSONDecodeError("Could not extract JSON from LLM response", raw[:200], 0)
