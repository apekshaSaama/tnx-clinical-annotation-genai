"""extract_json recovers only SAFE cases (fences, surrounding prose) and never
fabricates structure from truncated/incomplete JSON (CLAUDE.md Zero Silent Failure)."""

from __future__ import annotations

import json

import pytest

from llm.json_utils import extract_json


def test_plain_object_and_array():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_strips_code_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_leading_and_trailing_prose():
    assert extract_json('Here is the JSON:\n{"a": 1}\nHope that helps!') == {"a": 1}
    assert extract_json('{"a": 1} — done') == {"a": 1}


def test_outer_object_with_nested_array_not_confused():
    # Must return the whole object, not the nested [1, 2].
    assert extract_json('prefix {"a": [1, 2]} suffix') == {"a": [1, 2]}


def test_truncated_json_raises_not_salvaged():
    # Cut off mid-structure — previously salvaged; now must fail loudly.
    with pytest.raises(json.JSONDecodeError):
        extract_json('{"entities": [{"text": "smok')
    with pytest.raises(json.JSONDecodeError):
        extract_json('[{"a": 1}, {"b": 2')


def test_empty_and_garbage_raise():
    for bad in ("", "   ", "not json", "```\n```"):
        with pytest.raises(json.JSONDecodeError):
            extract_json(bad)
