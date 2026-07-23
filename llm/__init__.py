"""Central LLM layer. The ONLY entry point application code may use is
``LLMRouter`` (CLAUDE.md Rule #1)."""

from __future__ import annotations

from .router import LLMRouter, RouterResult

__all__ = ["LLMRouter", "RouterResult"]
