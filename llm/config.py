"""Typed configuration for the LLMRouter.

Loads credentials from ``.env`` (via python-dotenv) and behaviour from
``config/*.json``. Everything entity/provider-specific lives in config, not code
(CLAUDE.md Zero Hardcoding + Pattern 5). Application code never reads os.environ
for LLM concerns — it goes through the settings object the router owns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    type: str
    max_tokens: int
    token_param: str
    temperature: float | None
    timeout: int
    retries: int
    backoff_base_seconds: float
    supports_json_mode: bool
    prompt_suffix: str
    model_version: str
    model_class: str
    # Azure OpenAI manual prompt cache controls (2025-04-01-preview+): a stable
    # prompt_cache_key scopes the cache so unrelated workloads don't evict each
    # other; prompt_cache_retention keeps it warm past the default ~5-10 min
    # ephemeral window. None on providers that don't support them.
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    # Anthropic: wrap the system prompt in a cache_control breakpoint. Off by
    # default for providers where it doesn't apply.
    cache_system_prompt: bool = False
    # Gemini: explicit context caching via the cachedContents API. Off by
    # default for providers where it doesn't apply.
    prompt_cache_enabled: bool = False
    prompt_cache_ttl: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    name: str
    chain: list[str]
    temperature: float | None


@dataclass(frozen=True)
class LangfuseConfig:
    public_key: str | None
    secret_key: str | None
    base_url: str | None
    environment: str

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


@dataclass(frozen=True)
class GuardrailsConfig:
    session_token_cap: int
    default_environment: str
    warn_if_sandbox_in_prod: bool


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float

    def cost(self, input_tokens: int, output_tokens: int) -> dict[str, float]:
        in_cost = (input_tokens or 0) / 1_000_000 * self.input_per_1m
        out_cost = (output_tokens or 0) / 1_000_000 * self.output_per_1m
        return {
            "input": round(in_cost, 6),
            "output": round(out_cost, 6),
            "total": round(in_cost + out_cost, 6),
        }


@dataclass(frozen=True)
class Settings:
    providers: dict[str, ProviderConfig]
    tasks: dict[str, TaskConfig]
    model_aliases: dict[str, str]
    langfuse: LangfuseConfig
    guardrails: GuardrailsConfig
    pricing: dict[str, ModelPrice] = field(default_factory=dict)
    prompts_dir: Path = field(default=PROJECT_ROOT / "prompts")
    skills_dir: Path = field(default=PROJECT_ROOT / "skills")

    def price_for(self, model: str) -> ModelPrice | None:
        return self.pricing.get(model)

    def provider(self, name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(f"Unknown provider {name!r}; configured: {list(self.providers)}")
        return self.providers[name]

    def task(self, name: str) -> TaskConfig:
        if name not in self.tasks:
            raise KeyError(f"Unknown task {name!r}; configured: {list(self.tasks)}")
        return self.tasks[name]

    def resolve_provider_alias(self, model_name: str | None) -> str | None:
        """Map a user-supplied --model value to a provider key, or None."""
        if not model_name:
            return None
        normalized = model_name.strip().lower()
        if normalized in self.model_aliases:
            return self.model_aliases[normalized]
        for prefix, provider in self.model_aliases.items():
            if normalized.startswith(prefix):
                return provider
        return None


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load and cache settings. Reads .env once at process start."""
    load_dotenv(PROJECT_ROOT / ".env")

    models = _load_json(CONFIG_DIR / "models.json")
    guardrails_raw = _load_json(CONFIG_DIR / "guardrails.json")
    pricing_raw = _load_json(CONFIG_DIR / "pricing.json")

    providers: dict[str, ProviderConfig] = {}
    for name, raw in models.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name=name,
            type=raw["type"],
            max_tokens=int(raw.get("max_tokens", 4096)),
            token_param=raw.get("token_param", "max_tokens"),
            temperature=raw.get("temperature"),
            timeout=int(raw.get("timeout", 60)),
            retries=int(raw.get("retries", 3)),
            backoff_base_seconds=float(raw.get("backoff_base_seconds", 2.0)),
            supports_json_mode=bool(raw.get("supports_json_mode", False)),
            prompt_suffix=raw.get("prompt_suffix", ""),
            model_version=raw.get("model_version", name),
            model_class=raw.get("model_class", "reasoning"),
            prompt_cache_key=raw.get("prompt_cache_key"),
            prompt_cache_retention=raw.get("prompt_cache_retention"),
            cache_system_prompt=bool(raw.get("cache_system_prompt", False)),
            prompt_cache_enabled=bool(raw.get("prompt_cache_enabled", False)),
            prompt_cache_ttl=raw.get("prompt_cache_ttl"),
        )

    tasks: dict[str, TaskConfig] = {}
    for name, raw in models.get("tasks", {}).items():
        tasks[name] = TaskConfig(
            name=name,
            chain=list(raw["chain"]),
            temperature=raw.get("temperature"),
        )

    guardrails = GuardrailsConfig(
        session_token_cap=int(guardrails_raw.get("session_token_cap", 2_000_000)),
        default_environment=guardrails_raw.get("default_environment", "sandbox"),
        warn_if_sandbox_in_prod=bool(guardrails_raw.get("warn_if_sandbox_in_prod", True)),
    )

    langfuse = LangfuseConfig(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
        environment=os.environ.get("LANGFUSE_ENVIRONMENT", guardrails.default_environment),
    )

    pricing = {
        model: ModelPrice(
            input_per_1m=float(rates["input_per_1m"]),
            output_per_1m=float(rates["output_per_1m"]),
        )
        for model, rates in pricing_raw.get("models", {}).items()
    }

    return Settings(
        providers=providers,
        tasks=tasks,
        model_aliases={k.lower(): v for k, v in models.get("model_aliases", {}).items()},
        langfuse=langfuse,
        guardrails=guardrails,
        pricing=pricing,
    )
