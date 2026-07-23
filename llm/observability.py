"""Langfuse observability wrapper — the single place that talks to Langfuse.

Owns: tracing, generation spans (token usage + cost so cost/latency dashboards
populate), tag/metadata attachment, Langfuse-first prompt resolution *with
prompt-version linking*, prompt publishing, evaluation scores, and flush. Every
method is defensive: observability must NEVER break the application, and when
Langfuse keys are absent the whole surface degrades to a silent no-op
(CLAUDE.md Rule #2 "graceful no-op when keys are absent").
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any

from .config import LangfuseConfig

_VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _interpolate(template: str, variables: dict[str, Any]) -> str:
    """Substitute {{var}} placeholders (used when Langfuse is unavailable)."""
    return _VAR_PATTERN.sub(
        lambda m: str(variables.get(m.group(1), m.group(0))), template
    )


@dataclass
class ResolvedPrompt:
    """A resolved prompt: the compiled text plus the managed Langfuse prompt
    object when it came from Langfuse (``None`` for a bundled-file fallback).
    The object is used to LINK the generation to its prompt version."""

    text: str
    prompt: Any = None


class Observability:
    def __init__(self, cfg: LangfuseConfig) -> None:
        self.environment = cfg.environment
        self._client = None
        self._attrs = None
        self.enabled = False

        if cfg.enabled:
            try:
                from langfuse import Langfuse, LangfuseOtelSpanAttributes

                self._client = Langfuse(
                    public_key=cfg.public_key,
                    secret_key=cfg.secret_key,
                    host=cfg.base_url,
                    environment=cfg.environment,
                )
                self._attrs = LangfuseOtelSpanAttributes
                self.enabled = True
            except Exception as exc:  # never fatal
                print(f"[observability] Langfuse init failed, tracing disabled: {exc}",
                      file=sys.stderr)

    @property
    def client(self):
        return self._client

    def _tags(self, extra: list[str] | None) -> list[str]:
        tags = [self.environment]
        if extra:
            tags.extend(t for t in extra if t and t != self.environment)
        return tags

    # ---- traces -----------------------------------------------------------
    #
    # Langfuse 4.x replaced the v2 `client.trace()` / `client.generation()`
    # API with an OTEL-based `start_observation()` surface: there is no
    # standalone "trace" object anymore, only spans — a trace is just the
    # root span's trace_id. We use a root span (as_type="span") as the
    # "trace" handle returned to callers; `_trace_id()` in router.py must
    # read `.trace_id` off it, not `.id` (that's the observation/span id).

    def start_trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        release: str | None = None,
    ):
        if not self.enabled:
            return None
        try:
            span = self._client.start_observation(
                name=name,
                as_type="span",
                input=input,
                metadata=metadata,
            )
            attrs: dict[str, Any] = {}
            all_tags = self._tags(tags)
            if all_tags:
                attrs[self._attrs.TRACE_TAGS] = all_tags
            if session_id:
                attrs[self._attrs.TRACE_SESSION_ID] = session_id
            if release:
                attrs[self._attrs.RELEASE] = release
            if attrs:
                span._otel_span.set_attributes(attrs)
            return span
        except Exception as exc:
            print(f"[observability] trace failed: {exc}", file=sys.stderr)
            return None

    def update_trace(self, trace, *, output: Any = None, metadata: dict | None = None) -> None:
        if trace is None:
            return
        try:
            trace.update(output=output, metadata=metadata)
        except Exception as exc:
            print(f"[observability] trace.update failed: {exc}", file=sys.stderr)

    # ---- generations ----------------------------------------------------------

    def start_generation(
        self,
        trace,
        *,
        name: str,
        model: str,
        model_parameters: dict[str, Any] | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        prompt: Any = None,
    ):
        if not self.enabled:
            return None
        target = trace or self._client
        try:
            return target.start_observation(
                name=name,
                as_type="generation",
                model=model,
                model_parameters=model_parameters,
                input=input,
                metadata=metadata,
                prompt=prompt,  # links this generation to its prompt version
            )
        except Exception as exc:
            print(f"[observability] generation failed: {exc}", file=sys.stderr)
            return None

    def end_generation(
        self,
        generation,
        *,
        output: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_tokens: int | None = None,
        cost_details: dict[str, float] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if generation is None:
            return
        usage_details = None
        if input_tokens is not None or output_tokens is not None:
            usage_details = {
                "input": int(input_tokens or 0),
                "output": int(output_tokens or 0),
            }
            if cached_tokens:
                # Langfuse convention key; surfaces prompt-cache hits (Azure
                # prompt_cache_key reuse / Anthropic cache_control) on the
                # generation so cache effectiveness is visible on dashboards.
                usage_details["cache_read_input_tokens"] = int(cached_tokens)
        try:
            # 4.x split the old one-shot `generation.end(output=..., ...)`
            # into update() (sets the fields) + end() (closes the span).
            generation.update(
                output=output,
                usage_details=usage_details,
                cost_details=cost_details,
                level=level,
                status_message=status_message,
            )
            generation.end()
        except Exception as exc:
            print(f"[observability] generation.end failed: {exc}", file=sys.stderr)

    # ---- prompt management ----------------------------------------------------

    def resolve_prompt(
        self, name: str, *, fallback: str, variables: dict[str, Any]
    ) -> ResolvedPrompt:
        """Langfuse-first prompt resolution.

        Returns the managed Langfuse prompt (compiled + linkable object) when the
        prompt exists in Langfuse — enabling versioning / A-B testing and
        generation↔prompt linkage. Falls back to the bundled text otherwise (no
        link). Silent on miss: a bundled fallback is normal until prompts are
        synced (see ``llm/sync_prompts.py``).
        """
        if self.enabled:
            try:
                prompt = self._client.get_prompt(name, type="text")
                return ResolvedPrompt(text=prompt.compile(**variables), prompt=prompt)
            except Exception:
                pass
        return ResolvedPrompt(text=_interpolate(fallback, variables), prompt=None)

    def create_prompt(
        self,
        *,
        name: str,
        text: str,
        labels: list[str] | None = None,
        tags: list[str] | None = None,
        commit_message: str | None = None,
    ):
        """Publish (or version) a text prompt in Langfuse. No-op if disabled."""
        if not self.enabled:
            return None
        return self._client.create_prompt(
            name=name,
            prompt=text,
            type="text",
            labels=labels or [],
            tags=tags,
            commit_message=commit_message,
        )

    # ---- evaluation scores ----------------------------------------------------

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | str,
        data_type: str | None = None,
        comment: str | None = None,
    ) -> None:
        if not self.enabled or not trace_id:
            return
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=data_type,
                comment=comment,
            )
        except Exception as exc:
            print(f"[observability] score({name!r}) failed: {exc}", file=sys.stderr)

    def flush(self) -> None:
        if self.enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:
                print(f"[observability] flush failed: {exc}", file=sys.stderr)
