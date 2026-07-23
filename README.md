# Clinical NER Pipeline

Streamlit app that extracts clinical NER annotations (entities, assertions, relations)
from clinical notes via Azure OpenAI or Anthropic, then computes Inter-Annotator
Agreement (IAA) against ground truth.

## Structure

- `streamlite_process_ner.py` — Streamlit UI (login, upload notes + guideline, run
  extraction, run IAA). Entry point.
- `clinical_ner_extractor.py` — CLI invoked by the UI as a subprocess; runs the LLM
  extraction over a folder of notes and normalizes output to JSL-compatible JSON.
- `update_offset.py` — recalculates character offsets of extracted spans against
  the source text.
- `generate_iaa.py` — CLI invoked by the UI as a subprocess; computes token-level
  Cohen's kappa and span-level precision/recall/F1 between ground truth and
  predictions.
- `llm/` — the only layer allowed to call an LLM provider directly (`LLMRouter`).
  Owns provider fallback chains, retries, structured-output validation, cost
  guardrails, and optional Langfuse tracing.
- `config/` — provider/task config (`models.json`), cost guardrails
  (`guardrails.json`), pricing (`pricing.json`), and an example Gmail OTP config
  (`auth_config.example.json`).
- `prompts/` — bundled prompt text used by `llm/router.py` (Langfuse-first, with
  this as the fallback).
- `annotation_guidelines/` — reference guideline docs (also uploadable via the UI).
- `tests/` — pytest suite for the `llm` package.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the credentials you have (Azure OpenAI
and/or Anthropic; Langfuse and Gemini are optional).

For the login OTP email step, copy `config/auth_config.example.json` to
`config/auth_config.json` and fill in real Gmail credentials (an app password,
not your account password). Both `.env` and `config/auth_config.json` are
gitignored — never commit real credentials.

## Run

```bash
streamlit run streamlite_process_ner.py
```

## Tests

```bash
pytest tests/
```
