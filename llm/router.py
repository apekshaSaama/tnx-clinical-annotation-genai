"""LLMRouter — the single chokepoint for every LLM call in this project.

All application code routes through ``complete`` / ``complete_json`` / ``run_skill``.
Direct provider calls anywhere outside ``llm/providers`` are forbidden
(CLAUDE.md Rule #1). This class owns, in one place:

  * provider strategy + deterministic fallback chain (per task, config-driven)
  * per-call timeout + retry-with-backoff on transient failures
  * Langfuse tracing: one generation span per provider call, token usage pushed
    for cost dashboards, tags + request metadata attached, flush on shutdown
  * Langfuse-first prompt (and skill) resolution with bundled-file fallback
  * structured output: JSON-mode + robust parse + schema validation + retry
  * cost guardrails: per-session token cap, sandbox tagging by default
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from .config import Settings, TaskConfig, load_settings
from .errors import (
    AllProvidersFailedError,
    PromptResolutionError,
    ProviderCallError,
    StructuredOutputError,
    TokenBudgetExceededError,
)
from .json_utils import extract_json
from .observability import Observability, ResolvedPrompt
from .providers import Provider, ProviderResult, build_provider

_JSON_MAX_RETRIES = 2


@dataclass
class RouterResult:
    """What every router method returns."""

    text: str
    provider: str
    model: str
    usage: dict[str, int | None] = field(default_factory=dict)
    trace_id: str | None = None
    data: Any = None  # parsed JSON for complete_json / run_skill


def _trace_id(trace) -> str | None:
    # Langfuse 4.x spans expose the observation id as `.id` and the trace id
    # (what create_score()/get_trace_url() need) as `.trace_id`.
    return getattr(trace, "trace_id", None)


class LLMRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.obs = Observability(self.settings.langfuse)
        self._providers: dict[str, Provider] = {}
        self._session_tokens = 0
        self._json_max_retries = _JSON_MAX_RETRIES
        self._log_startup()

    # ---- startup / guardrails -------------------------------------------------

    def _log_startup(self) -> None:
        env = self.settings.langfuse.environment
        tracing = "on" if self.obs.enabled else "off (no Langfuse keys)"
        print(f"[LLMRouter] environment={env} langfuse={tracing}", file=sys.stderr)
        guardrails = self.settings.guardrails
        deploy_env = (os.environ.get("APP_ENV") or os.environ.get("ENVIRONMENT") or "").lower()
        if (
            guardrails.warn_if_sandbox_in_prod
            and env == "sandbox"
            and deploy_env == "production"
        ):
            print(
                "[LLMRouter] WARNING: LANGFUSE_ENVIRONMENT=sandbox in a production "
                "deployment — traces will be excluded from prod cost aggregations. "
                "Set LANGFUSE_ENVIRONMENT=production.",
                file=sys.stderr,
            )

    def _check_budget(self) -> None:
        cap = self.settings.guardrails.session_token_cap
        if self._session_tokens >= cap:
            raise TokenBudgetExceededError(self._session_tokens, cap)

    def _record_usage(self, result: ProviderResult) -> None:
        total = result.usage.get("total_tokens")
        if total is None:
            total = result.input_tokens + result.output_tokens
        self._session_tokens += int(total or 0)

    @property
    def session_tokens(self) -> int:
        return self._session_tokens

    # ---- prompt / skill resolution -------------------------------------------

    def _bundled(self, name: str, subdir: str, suffix: str) -> str | None:
        base = self.settings.prompts_dir if subdir == "prompts" else self.settings.skills_dir
        path = base / f"{name}{suffix}"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def _resolve_prompt(
        self, name: str | None, variables: dict[str, Any]
    ) -> ResolvedPrompt | None:
        if not name:
            return None
        fallback = self._bundled(name, "prompts", ".txt")
        if fallback is None and not self.obs.enabled:
            raise PromptResolutionError(name, str(self.settings.prompts_dir / f"{name}.txt"))
        return self.obs.resolve_prompt(name, fallback=fallback or "", variables=variables)

    def _resolve_skill(self, name: str, variables: dict[str, Any]) -> ResolvedPrompt:
        fallback = self._bundled(name, "skills", ".skill.md")
        if fallback is None and not self.obs.enabled:
            raise PromptResolutionError(
                name, str(self.settings.skills_dir / f"{name}.skill.md")
            )
        resolved = self.obs.resolve_prompt(name, fallback=fallback or "", variables=variables)
        if not resolved.text.strip():
            raise PromptResolutionError(name, f"{self.settings.skills_dir} + Langfuse")
        return resolved

    def _cost_details(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> dict[str, float] | None:
        price = self.settings.price_for(model)
        if price is None:
            return None
        return price.cost(input_tokens, output_tokens)

    # ---- provider chain -------------------------------------------------------

    def _get_provider(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = build_provider(self.settings.provider(name))
        return self._providers[name]

    def _chain_for(self, task_cfg: TaskConfig, model_preference: str | None) -> list[str]:
        chain = list(task_cfg.chain)
        preferred = self.settings.resolve_provider_alias(model_preference)
        if preferred:
            chain = [preferred] + [p for p in chain if p != preferred]
        elif model_preference:
            # Explicit request that maps to no configured provider — don't
            # silently serve a different one; say so, then use the task default.
            print(
                f"[LLMRouter] --model {model_preference!r} matches no configured "
                f"provider; using default chain {chain}",
                file=sys.stderr,
            )
        return chain

    def _run_chain(
        self,
        *,
        task: str,
        task_cfg: TaskConfig,
        chain: list[str],
        user_prompt: str,
        system_prompt: str | None,
        json_mode: bool,
        trace,
        metadata: dict[str, Any] | None,
        prompt_obj: Any = None,
    ) -> ProviderResult:
        """Transport reliability: try each provider in order, with per-provider
        retry-with-backoff on transient errors, failing over on exhaustion."""
        attempts: list[str] = []

        for provider_name in chain:
            # A bad chain entry (unknown provider name/type) or missing
            # credentials must NOT abort the request — skip to the next provider
            # so a healthy fallback can still serve it.
            try:
                pcfg = self.settings.provider(provider_name)
                provider = self._get_provider(provider_name)
            except (ProviderCallError, KeyError, ValueError) as exc:
                attempts.append(f"{provider_name}: {exc}")
                continue

            temperature = (
                task_cfg.temperature
                if task_cfg.temperature is not None
                else pcfg.temperature
            )
            prompt = user_prompt + (pcfg.prompt_suffix or "")
            # Only enable API JSON-mode on providers that support it; others rely
            # on the prompt's JSON contract + extract_json (see config flag).
            provider_json_mode = json_mode and pcfg.supports_json_mode
            retries = max(1, pcfg.retries)  # a misconfigured 0 must still call once

            for attempt in range(1, retries + 1):
                self._check_budget()
                gen = self.obs.start_generation(
                    trace,
                    name=f"{task}:{provider_name}",
                    model=provider.model,
                    model_parameters={
                        "temperature": temperature,
                        "max_tokens": pcfg.max_tokens,
                        "json_mode": provider_json_mode,
                    },
                    input={"system": system_prompt, "user": prompt},
                    metadata={**(metadata or {}), "provider": provider_name, "attempt": attempt},
                    prompt=prompt_obj,
                )
                try:
                    result = provider.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=pcfg.max_tokens,
                        json_mode=provider_json_mode,
                    )
                except ProviderCallError as exc:
                    self.obs.end_generation(
                        gen, output=str(exc), level="ERROR", status_message=str(exc)
                    )
                    if not exc.retryable:
                        attempts.append(f"{provider_name}: {exc} (non-retryable)")
                        break
                    if attempt < retries:
                        delay = pcfg.backoff_base_seconds * (2 ** (attempt - 1))
                        print(
                            f"[LLMRouter] {provider_name} transient error "
                            f"(attempt {attempt}/{retries}), retrying in {delay:.1f}s: {exc}",
                            file=sys.stderr,
                        )
                        time.sleep(delay)
                    else:
                        attempts.append(f"{provider_name}: {exc} (retries exhausted)")
                    continue

                self._record_usage(result)  # tokens were spent even if unusable

                # Empty/whitespace output is never a valid result — treat as a
                # transient failure: retry this provider, then fail over.
                if not (result.text or "").strip():
                    self.obs.end_generation(
                        gen, output="", input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_details=self._cost_details(
                            result.model, result.input_tokens, result.output_tokens
                        ),
                        level="WARNING", status_message="empty output",
                    )
                    if attempt < retries:
                        delay = pcfg.backoff_base_seconds * (2 ** (attempt - 1))
                        print(
                            f"[LLMRouter] {provider_name} returned empty output "
                            f"(attempt {attempt}/{retries}), retrying in {delay:.1f}s",
                            file=sys.stderr,
                        )
                        time.sleep(delay)
                        continue
                    attempts.append(f"{provider_name}: empty output (retries exhausted)")
                    break

                truncated = result.truncated
                self.obs.end_generation(
                    gen,
                    output=result.text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cached_tokens=result.cached_tokens,
                    cost_details=self._cost_details(
                        result.model, result.input_tokens, result.output_tokens
                    ),
                    level="WARNING" if truncated else None,
                    status_message=(
                        f"output truncated at max_tokens={pcfg.max_tokens}"
                        if truncated else None
                    ),
                )

                # Silent-truncation guard (CLAUDE.md Zero Silent Failure): a
                # response cut off at max_tokens yields incomplete JSON that the
                # salvage parser would pass off as complete. For structured
                # output, refuse it and fail over — another provider in the
                # chain may have a larger token budget. Retrying the SAME
                # provider is pointless (same cap), so break to the next one.
                if json_mode and truncated:
                    msg = (
                        f"{provider_name}: output truncated at "
                        f"max_tokens={pcfg.max_tokens} (raise config max_tokens)"
                    )
                    print(f"[LLMRouter] {msg}; trying next provider", file=sys.stderr)
                    attempts.append(msg)
                    break

                if provider_name != chain[0]:
                    print(f"[LLMRouter] failed over to {provider_name}", file=sys.stderr)
                return result

        raise AllProvidersFailedError(task, attempts)

    # ---- core -----------------------------------------------------------------

    def _complete_core(
        self,
        *,
        task: str,
        user_prompt: ResolvedPrompt,
        system_prompt: ResolvedPrompt | None,
        model_preference: str | None,
        want_json: bool,
        schema: dict | None,
        trace_name: str,
        tags: list[str],
        metadata: dict[str, Any] | None,
    ) -> RouterResult:
        task_cfg = self.settings.task(task)
        chain = self._chain_for(task_cfg, model_preference)
        user_text = user_prompt.text
        prompt_obj = user_prompt.prompt
        system_text = system_prompt.text if system_prompt else None

        trace = self.obs.start_trace(
            trace_name,
            input={**(metadata or {}), "task": task},
            metadata={**(metadata or {}), "task": task},
            tags=[task, *tags],
            release=os.environ.get("APP_RELEASE"),
        )
        trace_id = _trace_id(trace)

        if not want_json:
            result = self._run_chain(
                task=task,
                task_cfg=task_cfg,
                chain=chain,
                user_prompt=user_text,
                system_prompt=system_text,
                json_mode=False,
                trace=trace,
                metadata=metadata,
                prompt_obj=prompt_obj,
            )
            cost = self._cost_details(result.model, result.input_tokens, result.output_tokens)
            self._finalize_trace(
                trace, trace_id, result, usage=result.usage,
                cost_total=(cost or {}).get("total"), json_attempts=None, valid=None,
            )
            return RouterResult(
                text=result.text,
                provider=result.provider,
                model=result.model,
                usage=result.usage,
                trace_id=trace_id,
            )

        # Structured output: deterministic validation layer around the LLM
        # (CLAUDE.md Pattern 1). Retry with an augmented prompt on bad output.
        # Usage/cost accumulate across every (billable) attempt so reported spend
        # is the true total, not just the last call.
        last_reason = "no output produced"
        prompt = user_text
        cum_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0,
        }
        cum_cost = 0.0
        for attempt in range(1, self._json_max_retries + 1):
            result = self._run_chain(
                task=task,
                task_cfg=task_cfg,
                chain=chain,
                user_prompt=prompt,
                system_prompt=system_text,
                json_mode=True,
                trace=trace,
                metadata={**(metadata or {}), "json_attempt": attempt},
                prompt_obj=prompt_obj,
            )
            self._accumulate_usage(cum_usage, result.usage)
            cost = self._cost_details(result.model, result.input_tokens, result.output_tokens)
            if cost:
                cum_cost += cost["total"]

            try:
                parsed = extract_json(result.text)
            except json.JSONDecodeError as exc:
                last_reason = f"unparseable JSON ({exc})"
            else:
                if schema is not None:
                    try:
                        jsonschema.validate(parsed, schema)
                    except jsonschema.ValidationError as exc:
                        last_reason = f"schema violation: {exc.message}"
                    else:
                        self._finalize_trace(
                            trace, trace_id, result, usage=cum_usage,
                            cost_total=round(cum_cost, 6), json_attempts=attempt, valid=True,
                        )
                        return self._json_result(result, parsed, trace_id, usage=cum_usage)
                else:
                    self._finalize_trace(
                        trace, trace_id, result, usage=cum_usage,
                        cost_total=round(cum_cost, 6), json_attempts=attempt, valid=True,
                    )
                    return self._json_result(result, parsed, trace_id, usage=cum_usage)

            prompt = (
                user_text
                + f"\n\nYour previous response was invalid: {last_reason}. "
                "Return ONLY valid JSON that satisfies the required format."
            )

        # Exhausted retries — record the failure as an evaluation score, then raise.
        self.obs.score(
            trace_id=trace_id, name="structured_output_valid", value=0,
            data_type="BOOLEAN", comment=last_reason,
        )
        self.obs.update_trace(trace, output={"error": last_reason})
        raise StructuredOutputError(task, last_reason)

    @staticmethod
    def _accumulate_usage(acc: dict[str, int], usage: dict[str, int | None]) -> None:
        prompt_t = usage.get("prompt_tokens") or 0
        completion_t = usage.get("completion_tokens") or 0
        total_t = usage.get("total_tokens")
        acc["prompt_tokens"] += prompt_t
        acc["completion_tokens"] += completion_t
        acc["total_tokens"] += total_t if total_t is not None else prompt_t + completion_t
        acc["cached_tokens"] += usage.get("cached_tokens") or 0

    def _finalize_trace(
        self, trace, trace_id: str | None, result: ProviderResult,
        *, usage: dict[str, int | None], cost_total: float | None,
        json_attempts: int | None, valid: bool | None,
    ) -> None:
        """Attach the outcome (output summary + evaluation scores) to the trace."""
        self.obs.update_trace(
            trace,
            output={
                "provider": result.provider,
                "model": result.model,
                "total_tokens": usage.get("total_tokens"),
                "cost_usd": cost_total,
            },
        )
        if valid is not None:
            self.obs.score(
                trace_id=trace_id, name="structured_output_valid",
                value=1 if valid else 0, data_type="BOOLEAN",
            )
        if json_attempts is not None:
            self.obs.score(
                trace_id=trace_id, name="json_attempts",
                value=json_attempts, data_type="NUMERIC",
            )

    @staticmethod
    def _json_result(
        result: ProviderResult, parsed: Any, trace_id: str | None,
        *, usage: dict[str, int | None],
    ) -> RouterResult:
        return RouterResult(
            text=result.text,
            data=parsed,
            provider=result.provider,
            model=result.model,
            usage=usage,
            trace_id=trace_id,
        )

    # ---- public API -----------------------------------------------------------

    def complete(
        self,
        *,
        task: str,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
        system_prompt_name: str | None = None,
        model_preference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RouterResult:
        variables = variables or {}
        return self._complete_core(
            task=task,
            user_prompt=self._resolve_prompt(prompt_name, variables),
            system_prompt=self._resolve_prompt(system_prompt_name, variables),
            model_preference=model_preference,
            want_json=False,
            schema=None,
            trace_name=f"router.complete:{task}",
            tags=[],
            metadata=metadata,
        )

    def complete_json(
        self,
        *,
        task: str,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
        system_prompt_name: str | None = None,
        model_preference: str | None = None,
        schema: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RouterResult:
        variables = variables or {}
        return self._complete_core(
            task=task,
            user_prompt=self._resolve_prompt(prompt_name, variables),
            system_prompt=self._resolve_prompt(system_prompt_name, variables),
            model_preference=model_preference,
            want_json=True,
            schema=schema,
            trace_name=f"router.complete_json:{task}",
            tags=[],
            metadata=metadata,
        )

    def run_skill(
        self,
        *,
        skill_name: str,
        task: str,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
        model_preference: str | None = None,
        schema: dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RouterResult:
        """Run a named skill: the skill markdown (Langfuse-first, file fallback)
        drives behaviour as the system prompt; ``prompt_name`` supplies the user
        input. The skill name is stamped on the trace for cost/latency slicing."""
        variables = variables or {}
        skill_system = self._resolve_skill(skill_name, variables)
        return self._complete_core(
            task=task,
            user_prompt=self._resolve_prompt(prompt_name, variables),
            system_prompt=skill_system,
            model_preference=model_preference,
            want_json=schema is not None,
            schema=schema,
            trace_name=f"router.run_skill:{skill_name}",
            tags=[f"skill:{skill_name}"],
            metadata={**(metadata or {}), "skill": skill_name},
        )

    def score(
        self,
        *,
        trace_id: str | None,
        name: str,
        value: float | str,
        data_type: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Attach an evaluation score to a trace (e.g. domain-specific quality
        signals the caller computes). No-op when tracing is disabled."""
        self.obs.score(
            trace_id=trace_id, name=name, value=value,
            data_type=data_type, comment=comment,
        )

    def sync_prompts(self, *, labels: list[str] | None = None) -> list[str]:
        """Publish every bundled ``prompts/*.txt`` to Langfuse so the router's
        Langfuse-first resolution finds managed, versioned prompts. Returns the
        list of prompt names synced. No-op (empty list) when disabled."""
        if not self.obs.enabled:
            return []
        synced: list[str] = []
        for path in sorted(self.settings.prompts_dir.glob("*.txt")):
            name = path.stem
            self.obs.create_prompt(
                name=name,
                text=path.read_text(encoding="utf-8"),
                labels=labels or ["production", "latest"],
                tags=[self.obs.environment],
                commit_message="Synced from bundled prompts/ by LLMRouter.sync_prompts",
            )
            synced.append(name)
        return synced

    def flush(self) -> None:
        """Flush in-flight Langfuse traces. Call in a shutdown hook."""
        self.obs.flush()
