"""Rule #1 guardrail: the LLMRouter exists and exposes its mandated surface,
and application code imports it (not any provider SDK directly)."""

from __future__ import annotations


def test_router_importable_with_full_surface():
    from llm import LLMRouter

    for method in ("complete", "complete_json", "run_skill", "flush"):
        assert callable(getattr(LLMRouter, method)), f"LLMRouter missing {method}"


def test_extractor_routes_through_router():
    import clinical_ner_extractor as extractor

    # The CLI must depend on the router, never on a provider adapter directly.
    source = (extractor.__file__ or "").rstrip("co")
    with open(extractor.__file__, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert "from llm import" in text
    assert "connectors" not in text
