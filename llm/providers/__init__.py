"""Provider adapters. These are the ONLY modules that talk to an LLM directly;
nothing outside the ``llm`` package may import them (CLAUDE.md Rule #1)."""

from __future__ import annotations

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .azure_openai import AzureOpenAIProvider
from .base import Provider, ProviderResult
from .gemini import GeminiProvider

_BY_TYPE = {
    "azure_openai": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def build_provider(config: ProviderConfig) -> Provider:
    """Instantiate a provider adapter from its config entry."""
    provider_cls = _BY_TYPE.get(config.type)
    if provider_cls is None:
        raise ValueError(
            f"No adapter for provider type {config.type!r}; known: {list(_BY_TYPE)}"
        )
    if config.type == "anthropic":
        return provider_cls(
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            cache_system_prompt=config.cache_system_prompt,
        )
    if config.type == "gemini":
        return provider_cls(
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            prompt_cache_enabled=config.prompt_cache_enabled,
            prompt_cache_ttl=config.prompt_cache_ttl,
        )
    return provider_cls(
        timeout=config.timeout,
        token_param=config.token_param,
        prompt_cache_key=config.prompt_cache_key,
        prompt_cache_retention=config.prompt_cache_retention,
    )


__all__ = ["Provider", "ProviderResult", "build_provider"]
