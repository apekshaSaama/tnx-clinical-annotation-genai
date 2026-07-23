"""Cost tracking: ModelPrice computes USD from token usage, and the router
derives cost_details for models present in pricing config (None otherwise)."""

from __future__ import annotations

from llm.config import ModelPrice


def test_model_price_cost():
    price = ModelPrice(input_per_1m=3.0, output_per_1m=15.0)
    cost = price.cost(1_000_000, 2_000_000)
    assert cost["input"] == 3.0
    assert cost["output"] == 30.0
    assert cost["total"] == 33.0


def test_model_price_handles_zero_and_none():
    price = ModelPrice(input_per_1m=1.0, output_per_1m=2.0)
    assert price.cost(0, 0) == {"input": 0.0, "output": 0.0, "total": 0.0}


def test_router_cost_details_from_config(tmp_path):
    from test_router_json import _make_router

    router, _ = _make_router(tmp_path, ["{}"])
    router.settings.pricing["stub-model"] = ModelPrice(
        input_per_1m=10.0, output_per_1m=20.0
    )
    cost = router._cost_details("stub-model", 500_000, 250_000)
    assert cost == {"input": 5.0, "output": 5.0, "total": 10.0}
    # Unknown model -> no router-computed cost (defers to Langfuse registry)
    assert router._cost_details("unknown-model", 1000, 1000) is None
