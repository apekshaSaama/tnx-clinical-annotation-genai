"""Provider protocol and the uniform result all providers return.

Every provider adapter lifts the raw urllib REST logic from the old connectors
but conforms to one interface so the router can treat them interchangeably and
fail over between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


_TRUNCATION_REASONS = {"length", "max_tokens", "model_length", "content_filter"}


@dataclass
class ProviderResult:
    """Normalized output of a single provider call."""

    text: str
    model: str
    provider: str
    # Uniform usage keys across providers (Anthropic/Azure name them differently).
    usage: dict[str, int | None] = field(default_factory=dict)
    # Provider-native stop/finish reason ("stop"/"end_turn" = clean;
    # "length"/"max_tokens" = cut off). Used to detect silent truncation.
    finish_reason: str | None = None

    @property
    def input_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or 0)

    @property
    def output_tokens(self) -> int:
        return int(self.usage.get("completion_tokens") or 0)

    @property
    def cached_tokens(self) -> int:
        """Prompt tokens served from a provider-side cache (Azure prompt cache
        reuse / Anthropic cache_read_input_tokens). 0 when not cached or the
        provider doesn't report it."""
        return int(self.usage.get("cached_tokens") or 0)

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() in _TRUNCATION_REASONS


class Provider(Protocol):
    """The contract the router depends on. Implemented by each adapter."""

    name: str

    @property
    def model(self) -> str: ...

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ProviderResult: ...
