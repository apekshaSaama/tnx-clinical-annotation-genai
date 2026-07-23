"""Structured-output guardrail (CLAUDE.md Pattern 1): complete_json parses,
validates against a schema, retries on bad output, and fails deterministically
when the model never produces valid output. Uses an offline stub provider so no
network or credentials are needed."""

from __future__ import annotations

import pytest

from llm.config import (
    GuardrailsConfig,
    LangfuseConfig,
    ProviderConfig,
    Settings,
    TaskConfig,
)
from llm.errors import AllProvidersFailedError, ProviderCallError, StructuredOutputError
from llm.providers.base import ProviderResult
from llm.router import LLMRouter


class StubProvider:
    name = "stub"

    def __init__(self, responses: list[str], finish_reason: str | None = None) -> None:
        self._responses = list(responses)
        self.finish_reason = finish_reason
        self.calls = 0

    @property
    def model(self) -> str:
        return "stub-model"

    def complete(self, **_kwargs) -> ProviderResult:
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ProviderResult(
            text=text,
            model=self.model,
            provider=self.name,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason=self.finish_reason,
        )


def _make_router(tmp_path, responses):
    (tmp_path / "echo.txt").write_text("note: {{x}}", encoding="utf-8")
    settings = Settings(
        providers={
            "stub": ProviderConfig(
                name="stub",
                type="stub",
                max_tokens=100,
                token_param="max_tokens",
                temperature=0.0,
                timeout=5,
                retries=1,
                backoff_base_seconds=0.0,
                supports_json_mode=True,
                prompt_suffix="",
                model_version="stub_v1",
                model_class="reasoning",
            )
        },
        tasks={"t": TaskConfig(name="t", chain=["stub"], temperature=0.0)},
        model_aliases={},
        langfuse=LangfuseConfig(
            public_key=None, secret_key=None, base_url=None, environment="sandbox"
        ),
        guardrails=GuardrailsConfig(
            session_token_cap=10_000_000,
            default_environment="sandbox",
            warn_if_sandbox_in_prod=False,
        ),
        prompts_dir=tmp_path,
        skills_dir=tmp_path,
    )
    router = LLMRouter(settings=settings)
    stub = StubProvider(responses)
    router._providers["stub"] = stub
    return router, stub


def test_valid_json_first_try(tmp_path):
    router, stub = _make_router(tmp_path, ['{"ok": true}'])
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.data == {"ok": True}
    assert result.provider == "stub"
    assert stub.calls == 1


def test_retries_then_succeeds(tmp_path):
    router, stub = _make_router(tmp_path, ["not json at all", '{"ok": 1}'])
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.data == {"ok": 1}
    assert stub.calls == 2  # retried once with augmented prompt


def test_unparseable_after_retries_raises(tmp_path):
    router, _ = _make_router(tmp_path, ["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})


def test_schema_validation_retries_then_passes(tmp_path):
    schema = {"type": "array"}
    router, stub = _make_router(tmp_path, ["{}", "[]"])
    result = router.complete_json(
        task="t", prompt_name="echo", variables={"x": "hi"}, schema=schema
    )
    assert result.data == []
    assert stub.calls == 2  # object failed schema, array passed


def test_schema_violation_always_raises(tmp_path):
    schema = {"type": "array"}
    router, _ = _make_router(tmp_path, ["{}", "{}"])
    with pytest.raises(StructuredOutputError):
        router.complete_json(
            task="t", prompt_name="echo", variables={"x": "hi"}, schema=schema
        )


class FailingProvider:
    name = "primary"

    @property
    def model(self) -> str:
        return "primary-model"

    def complete(self, **_kwargs) -> ProviderResult:
        raise ProviderCallError(
            "simulated 503", provider="primary", status_code=503, retryable=True
        )


def _two_provider_router(tmp_path, secondary_responses):
    router, secondary = _make_router(tmp_path, secondary_responses)
    # Reconfigure to a two-provider chain: primary (fails) -> stub (succeeds).
    primary_cfg = ProviderConfig(
        name="primary",
        type="primary",
        max_tokens=100,
        token_param="max_tokens",
        temperature=None,
        timeout=5,
        retries=2,
        backoff_base_seconds=0.0,
        supports_json_mode=True,
        prompt_suffix="",
        model_version="primary_v1",
        model_class="reasoning",
    )
    router.settings.providers["primary"] = primary_cfg
    router.settings.tasks["t"] = TaskConfig(
        name="t", chain=["primary", "stub"], temperature=None
    )
    router._providers["primary"] = FailingProvider()
    return router, secondary


def test_fails_over_to_secondary(tmp_path):
    router, secondary = _two_provider_router(tmp_path, ['{"ok": true}'])
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.provider == "stub"
    assert result.data == {"ok": True}
    assert secondary.calls == 1  # secondary served the request after failover


def test_all_providers_failing_raises(tmp_path):
    router, _ = _two_provider_router(tmp_path, ["ignored"])
    # Make the secondary fail too by pointing the chain at only the failing one.
    router.settings.tasks["t"] = TaskConfig(name="t", chain=["primary"], temperature=None)
    with pytest.raises(AllProvidersFailedError):
        router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})


def test_truncated_json_is_refused_not_salvaged(tmp_path):
    # A single provider that returns valid-but-truncated JSON must NOT be
    # silently accepted for structured output — it fails over / errors out.
    (tmp_path / "echo.txt").write_text("note: {{x}}", encoding="utf-8")
    router, stub = _make_router(tmp_path, ['{"ok": true}'])
    router._providers["stub"]._responses = ['{"ok": true}']
    router._providers["stub"].finish_reason = "length"  # cut off at max_tokens
    with pytest.raises(AllProvidersFailedError):
        router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})


def test_truncation_fails_over_to_larger_budget_provider(tmp_path):
    # primary truncates -> fail over to stub which completes cleanly.
    router, secondary = _two_provider_router(tmp_path, ['{"ok": 1}'])

    class TruncatingProvider:
        name = "primary"

        @property
        def model(self):
            return "primary-model"

        def complete(self, **_kwargs):
            return ProviderResult(
                text='{"ok": tr',  # cut off
                model="primary-model",
                provider="primary",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                finish_reason="length",
            )

    router._providers["primary"] = TruncatingProvider()
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.provider == "stub"
    assert result.data == {"ok": 1}


def test_non_json_truncation_is_returned_with_partial_text(tmp_path):
    # For plain completion, a truncated response is returned (partial text is
    # acceptable there) rather than triggering failover.
    router, stub = _make_router(tmp_path, ["partial answer"])
    router._providers["stub"].finish_reason = "length"
    result = router.complete(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.text == "partial answer"
    assert result.provider == "stub"


def test_empty_output_single_provider_raises(tmp_path):
    router, _ = _make_router(tmp_path, [""])
    with pytest.raises(AllProvidersFailedError):
        router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})


def test_empty_output_fails_over(tmp_path):
    router, secondary = _two_provider_router(tmp_path, ['{"ok": 1}'])

    class EmptyProvider:
        name = "primary"

        @property
        def model(self):
            return "primary-model"

        def complete(self, **_kwargs):
            return ProviderResult(
                text="   ", model="primary-model", provider="primary",
                usage={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
                finish_reason="stop",
            )

    router._providers["primary"] = EmptyProvider()
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.provider == "stub"


def test_usage_accumulates_across_json_retries(tmp_path):
    # attempt 1 = bad JSON, attempt 2 = good. Reported usage must sum BOTH calls.
    router, stub = _make_router(tmp_path, ["not json", '{"ok": 1}'])
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.data == {"ok": 1}
    assert stub.calls == 2
    assert result.usage["total_tokens"] == 4  # 2 tokens/call * 2 calls


def test_unknown_provider_in_chain_is_skipped(tmp_path):
    router, stub = _make_router(tmp_path, ['{"ok": 1}'])
    # A typo'd chain entry must not abort — it skips to the healthy provider.
    router.settings.tasks["t"] = TaskConfig(
        name="t", chain=["ghost", "stub"], temperature=None
    )
    result = router.complete_json(task="t", prompt_name="echo", variables={"x": "hi"})
    assert result.provider == "stub"
