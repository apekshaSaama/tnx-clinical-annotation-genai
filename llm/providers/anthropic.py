"""Anthropic (Claude) provider adapter.

Transport lifted from the former ``connectors/anthropic_connector.py``. Anthropic
has no native JSON-mode flag, so ``json_mode`` is honoured by the prompt contract
(the router asks for JSON in the prompt) rather than an API parameter.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import urllib.error
import urllib.request

from ..errors import ProviderCallError
from .base import ProviderResult

_API_URL = "https://api.anthropic.com/v1/messages"
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        timeout: int = 60,
        max_tokens: int = 4096,
        cache_system_prompt: bool = False,
    ) -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        self.api_version = os.environ.get("ANTHROPIC_API_VERSION", "2023-06-01")
        self.timeout = timeout
        self.default_max_tokens = max_tokens
        # Wraps the (typically static, reused-across-calls) system prompt in a
        # cache_control breakpoint so repeated calls for the same task/skill
        # only pay full price on the first request. Config-driven per provider
        # entry — off by default since very short system prompts fall below
        # Anthropic's minimum cacheable size and gain nothing.
        self.cache_system_prompt = cache_system_prompt

        if not self.api_key:
            raise ProviderCallError(
                "ANTHROPIC_API_KEY is required", provider=self.name, retryable=False
            )

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ProviderResult:
        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            if self.cache_system_prompt:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature

        request = urllib.request.Request(
            url=_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderCallError(
                f"Anthropic request failed ({exc.code}): {body[:500]}",
                provider=self.name,
                status_code=exc.code,
                retryable=exc.code in _RETRYABLE_STATUS,
            ) from exc
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderCallError(
                f"Anthropic connection/read error: {reason}",
                provider=self.name,
                retryable=True,
            ) from exc

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        cache_read_tokens = usage.get("cache_read_input_tokens") or 0
        cache_creation_tokens = usage.get("cache_creation_input_tokens") or 0
        total = (
            prompt_tokens + completion_tokens + cache_read_tokens + cache_creation_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        if self.cache_system_prompt:
            self._log_cache_status(cache_read_tokens, cache_creation_tokens, prompt_tokens or 0)

        blocks = data.get("content", []) or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )

        return ProviderResult(
            text=text,
            model=self._model,
            provider=self.name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
                "cached_tokens": cache_read_tokens,
            },
            finish_reason=data.get("stop_reason"),
        )

    def _log_cache_status(
        self, cache_read_tokens: int, cache_creation_tokens: int, prompt_tokens: int
    ) -> None:
        if cache_read_tokens:
            print(
                f"[anthropic cache] HIT: {cache_read_tokens}/{prompt_tokens} prompt "
                f"tokens served from cache (model={self._model})",
                file=sys.stderr,
            )
        elif cache_creation_tokens:
            print(
                f"[anthropic cache] WRITE: {cache_creation_tokens}/{prompt_tokens} "
                f"prompt tokens written to cache (model={self._model})",
                file=sys.stderr,
            )
        else:
            print(
                f"[anthropic cache] MISS: 0/{prompt_tokens} prompt tokens cached",
                file=sys.stderr,
            )
