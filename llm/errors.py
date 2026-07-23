"""Typed errors raised by the LLMRouter layer.

Errors carry enough context (provider, model, task) to be actionable in logs and
CLI output, without leaking secrets. See CLAUDE.md "Error Messages".
"""

from __future__ import annotations


class LLMRouterError(RuntimeError):
    """Base class for all router-layer failures."""


class ProviderCallError(LLMRouterError):
    """A single provider call failed. Carries whether it is worth failing over.

    ``retryable`` marks transient conditions (rate limit / 5xx / timeout) that
    justify retry-with-backoff and cross-provider failover; non-retryable errors
    (e.g. auth/config) short-circuit the chain for that provider.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class AllProvidersFailedError(LLMRouterError):
    """Every provider in the configured chain failed for a task."""

    def __init__(self, task: str, attempts: list[str]) -> None:
        detail = "; ".join(attempts) if attempts else "no providers configured"
        super().__init__(
            f"All providers failed for task={task!r}: {detail}"
        )
        self.task = task
        self.attempts = attempts


class StructuredOutputError(LLMRouterError):
    """The model could not produce parseable / schema-valid JSON after retries."""

    def __init__(self, task: str, reason: str) -> None:
        super().__init__(
            f"Structured output validation failed for task={task!r}: {reason}"
        )
        self.task = task
        self.reason = reason


class TokenBudgetExceededError(LLMRouterError):
    """The per-session token cap from config/guardrails.json was reached."""

    def __init__(self, spent: int, cap: int) -> None:
        super().__init__(
            f"Session token budget exceeded: spent={spent} cap={cap}"
        )
        self.spent = spent
        self.cap = cap


class PromptResolutionError(LLMRouterError):
    """A named prompt could not be resolved from Langfuse or the bundled files."""

    def __init__(self, name: str, searched: str) -> None:
        super().__init__(
            f"Could not resolve prompt {name!r}. Searched: {searched}"
        )
        self.name = name
