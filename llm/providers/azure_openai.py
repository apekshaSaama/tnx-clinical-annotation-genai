"""Azure OpenAI provider adapter.

Transport (urllib REST, header shape, usage extraction) is lifted verbatim from
the former ``connectors/azure_chat_openai_connector.py``; the only changes are a
uniform ``ProviderResult`` return and typed ``ProviderCallError`` so the router
can decide on retry / failover.
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

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class AzureOpenAIProvider:
    name = "azure"

    def __init__(
        self,
        timeout: int = 60,
        token_param: str = "max_tokens",
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> None:
        self.endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.timeout = timeout
        # Newer Azure deployments require 'max_completion_tokens' instead of the
        # legacy 'max_tokens'. The exact key is config-driven per deployment.
        self.token_param = token_param
        # Manual prompt cache controls (Azure OpenAI 2025-04-01-preview+):
        # prompt_cache_key scopes the cache so unrelated workloads don't evict
        # each other; prompt_cache_retention keeps it warm past the default
        # ephemeral (~5-10 min) window. Both are config-driven, per deployment.
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_retention = prompt_cache_retention

        if not self.endpoint:
            raise ProviderCallError(
                "AZURE_OPENAI_ENDPOINT is required", provider=self.name, retryable=False
            )
        if not self.api_key:
            raise ProviderCallError(
                "AZURE_OPENAI_API_KEY is required", provider=self.name, retryable=False
            )
        if not self.deployment_name:
            raise ProviderCallError(
                "AZURE_OPENAI_DEPLOYMENT is required", provider=self.name, retryable=False
            )

    @property
    def model(self) -> str:
        # For Azure the deployment name is the addressable "model".
        return self.deployment_name or "azure-openai"

    def _url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def _log_cache_status(self, cached_tokens: int, prompt_tokens: int) -> None:
        if cached_tokens:
            print(
                f"[azure cache] HIT: {cached_tokens}/{prompt_tokens} prompt tokens "
                f"served from cache (key={self.prompt_cache_key})",
                file=sys.stderr,
            )
        else:
            print(
                f"[azure cache] MISS: 0/{prompt_tokens} prompt tokens cached "
                f"(key={self.prompt_cache_key})",
                file=sys.stderr,
            )

    def complete(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ProviderResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload[self.token_param] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.prompt_cache_key:
            payload["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention:
            payload["prompt_cache_retention"] = self.prompt_cache_retention

        request = urllib.request.Request(
            url=self._url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "api-key": self.api_key},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderCallError(
                f"Azure OpenAI request failed ({exc.code}): {body[:500]}",
                provider=self.name,
                status_code=exc.code,
                retryable=exc.code in _RETRYABLE_STATUS,
            ) from exc
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderCallError(
                f"Azure OpenAI connection/read error: {reason}",
                provider=self.name,
                retryable=True,
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise ProviderCallError(
                "Azure OpenAI returned no choices", provider=self.name, retryable=True
            )

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        if self.prompt_cache_key:
            self._log_cache_status(cached_tokens, prompt_tokens or 0)
        return ProviderResult(
            text=content or "",
            model=self.model,
            provider=self.name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens"),
                "cached_tokens": cached_tokens,
            },
            finish_reason=choices[0].get("finish_reason"),
        )
