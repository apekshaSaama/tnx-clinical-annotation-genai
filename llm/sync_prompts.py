"""Publish bundled prompts to Langfuse so they become managed, versioned prompts.

Run once (and after editing any prompt) so the router's Langfuse-first resolution
serves the managed version and links every generation to its prompt version:

    venv/bin/python -m llm.sync_prompts

Idempotent: Langfuse versions a prompt automatically when its text changes and
no-ops when identical. Requires LANGFUSE_* keys; exits cleanly if absent.
"""

from __future__ import annotations

import sys

from .router import LLMRouter


def main() -> int:
    router = LLMRouter()
    if not router.obs.enabled:
        print("Langfuse is not configured (no LANGFUSE_* keys) — nothing to sync.",
              file=sys.stderr)
        return 1
    synced = router.sync_prompts()
    router.flush()
    if not synced:
        print("No prompts found under prompts/*.txt", file=sys.stderr)
        return 1
    print(f"Synced {len(synced)} prompt(s) to Langfuse: {', '.join(synced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
