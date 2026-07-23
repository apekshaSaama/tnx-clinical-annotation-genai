"""Google Gemini provider adapter with explicit context caching."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

from ..errors import ProviderCallError
from .base import ProviderResult

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        timeout: int = 300,
        max_tokens: int = 8192,
        prompt_cache_enabled: bool = False,
        prompt_cache_ttl: str | None = None,
    ) -> None:
        env_values = dotenv_values(_PROJECT_ROOT / ".env")
        self.api_key = env_values.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._model = (
            env_values.get("GEMINI_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-3.1-pro-preview"
        )
        self.timeout = timeout
        self.default_max_tokens = max_tokens
        self.prompt_cache_enabled = prompt_cache_enabled
        self.prompt_cache_ttl = prompt_cache_ttl or "3600s"
        self._cache_names: dict[str, str] = {}
        self._cache_unavailable: set[str] = set()

        if not self.api_key:
            raise ProviderCallError(
                "GEMINI_API_KEY is required", provider=self.name, retryable=False
            )
        print(
            f"[Gemini] configured model={self._model} api_key={self._masked_key()}",
            file=sys.stderr,
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
        cache_name = None
        dynamic_prompt = prompt
        cache_hash = "off"

        if self.prompt_cache_enabled:
            split = self._split_static_prompt(prompt)
            if split is not None:
                static_prompt, dynamic_prompt = split
                cache_hash = self._cache_key(static_prompt, system_prompt)
                cache_name = self._cache_names.get(cache_hash)
                if cache_name is None and cache_hash not in self._cache_unavailable:
                    try:
                        cache_name = self._create_cache(
                            static_prompt=static_prompt,
                            system_prompt=system_prompt,
                        )
                    except ProviderCallError as exc:
                        print(
                            f"[Gemini] cache create failed; falling back without cache: {exc}",
                            file=sys.stderr,
                        )
                        cache_name = None
                        dynamic_prompt = prompt
                        self._cache_unavailable.add(cache_hash)
                    else:
                        self._cache_names[cache_hash] = cache_name
                elif cache_name is None:
                    dynamic_prompt = prompt

        print(
            f"[Gemini] REQUEST model={self._model} cache_hash={cache_hash} "
            f"cached={'yes' if cache_name else 'no'} prompt_chars={len(dynamic_prompt)}",
            file=sys.stderr,
        )

        payload = self._generate_payload(
            prompt=dynamic_prompt,
            system_prompt=None if cache_name else system_prompt,
            temperature=temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            json_mode=json_mode,
            cache_name=cache_name,
        )
        data = self._post_json(
            f"/models/{urllib.parse.quote(self._model, safe='')}:generateContent",
            payload,
            provider_action="Gemini generateContent",
        )

        text, finish_reason = self._extract_text(data)
        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        cached_tokens = usage.get("cachedContentTokenCount") or 0

        print(
            f"[Gemini] finish={finish_reason or '?'} | in={prompt_tokens} "
            f"out={completion_tokens} | cache_read={cached_tokens}",
            file=sys.stderr,
        )

        return ProviderResult(
            text=text,
            model=self._model,
            provider=self.name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("totalTokenCount"),
                "cached_tokens": cached_tokens,
            },
            finish_reason=finish_reason,
        )

    def _create_cache(self, *, static_prompt: str, system_prompt: str | None) -> str:
        # This app's guideline text lives entirely in the (per-run-constant)
        # system prompt, not the per-note user prompt, so ``static_prompt`` is
        # often empty — never send an empty-text Part (Gemini rejects it);
        # omit the "contents" turn instead and cache via systemInstruction alone.
        payload: dict = {
            "model": f"models/{self._model}",
            "ttl": self.prompt_cache_ttl,
        }
        if static_prompt:
            payload["contents"] = [
                {
                    "role": "user",
                    "parts": [{"text": static_prompt}],
                }
            ]
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if "contents" not in payload and "systemInstruction" not in payload:
            raise ProviderCallError(
                "Gemini cache create has no static content to cache",
                provider=self.name,
                retryable=False,
            )

        print(
            f"[Gemini] cache create hash={self._cache_key(static_prompt, system_prompt)} "
            f"static_chars={len(static_prompt)} system_chars={len(system_prompt or '')}",
            file=sys.stderr,
        )
        data = self._post_json(
            "/cachedContents",
            payload,
            provider_action="Gemini cachedContents.create",
        )
        cache_name = data.get("name")
        if not cache_name:
            raise ProviderCallError(
                f"Gemini cache create returned no name: {data}",
                provider=self.name,
                retryable=False,
            )
        print(
            f"[Gemini] cache created name={cache_name} ttl={self.prompt_cache_ttl}",
            file=sys.stderr,
        )
        return cache_name

    def _generate_payload(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int,
        json_mode: bool,
        cache_name: str | None,
    ) -> dict:
        payload: dict = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if cache_name:
            payload["cachedContent"] = cache_name
        return payload

    def _post_json(self, path: str, payload: dict, *, provider_action: str) -> dict:
        url = f"{_API_BASE}{path}?key={urllib.parse.quote(self.api_key or '')}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderCallError(
                f"{provider_action} failed ({exc.code}): {body[:500]}",
                provider=self.name,
                status_code=exc.code,
                retryable=exc.code in _RETRYABLE_STATUS,
            ) from exc
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderCallError(
                f"{provider_action} connection/read error: {reason}",
                provider=self.name,
                retryable=True,
            ) from exc

    @staticmethod
    def _extract_text(data: dict) -> tuple[str, str | None]:
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderCallError(
                f"Gemini returned no candidates: {data}",
                provider=GeminiProvider.name,
                retryable=True,
            )
        first = candidates[0]
        parts = (first.get("content") or {}).get("parts") or []
        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        )
        return text, first.get("finishReason")

    @staticmethod
    def _split_static_prompt(prompt: str) -> tuple[str, str] | None:
        note_matches = list(re.finditer(r"(?im)^\s*Clinical\s+note\s*:\s*", prompt))
        if not note_matches:
            return None

        guideline_matches = list(re.finditer(r"(?im)^\s*Annotation\s+guideline\s*:\s*", prompt))
        first_note = note_matches[0]
        if first_note.start() == 0 and guideline_matches:
            later_guidelines = [m for m in guideline_matches if m.start() > first_note.end()]
            if later_guidelines:
                guideline = later_guidelines[0]
                return prompt[guideline.start():].rstrip(), prompt[:guideline.start()].rstrip()

        note = note_matches[-1]
        return prompt[:note.start()].rstrip(), prompt[note.start():].lstrip()

    @staticmethod
    def _cache_key(static_prompt: str, system_prompt: str | None) -> str:
        material = f"{system_prompt or ''}\n---\n{static_prompt}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

    def _masked_key(self) -> str:
        key = self.api_key or ""
        if len(key) <= 8:
            return "<set>"
        return f"{key[:4]}...{key[-4:]}"
