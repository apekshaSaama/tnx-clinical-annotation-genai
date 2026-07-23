import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import LLMRouter, RouterResult

from update_offset import update_offsets


TASK = "clinical_ner"

# Deterministic validation of the LLM's *raw* output shape (CLAUDE.md Pattern 1):
# only gross failures (a bare string / number / null) are rejected here, which
# triggers a router retry.
_LLM_OUTPUT_SCHEMA = {"type": ["array", "object"]}


@dataclass
class ExtractionResult:
    payload: list[dict[str, Any]]
    provider: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None


def _resolve_backend(model_name: str | None) -> str:
    normalized = (model_name or "").strip().lower()
    if normalized == "anthropic" or normalized.startswith("claude"):
        return "anthropic"
    if normalized == "gemini" or normalized.startswith("google"):
        return "gemini"
    if normalized == "openai":
        return "openai"
    return "azure"


def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _sanitize_note_text(note_text: str) -> str:
    return note_text.replace("\n", "").replace("\t", "").replace("\r", "")


def _compute_offsets(note_text: str, mention: str) -> tuple[int, int] | None:
    if not mention:
        return None
    start = note_text.find(mention)
    if start == -1:
        return None
    end = start + len(mention)
    return start, end


def _simplify_predictions(predictions: Any, default_username: str) -> list[dict[str, Any]]:
    simplified: list[dict[str, Any]] = []
    for prediction in predictions or []:
        if not isinstance(prediction, dict):
            continue
        simplified.append({
            "result": prediction.get("result", []),
            "created_username": prediction.get("created_username") or default_username,
        })
    return simplified


def _normalize_jsl_payload(
    result: Any,
    note_text: str,
    model_version: str = "azure_openai_clinical_ner",
    note_name: str | None = None,
) -> list[dict[str, Any]]:
    #note_text = _sanitize_note_text(note_text)
    if isinstance(result, list):
        if result and all(isinstance(item, dict) for item in result):
            document_items = []
            for idx, item in enumerate(result):
                if not isinstance(item, dict):
                    continue
                if "predictions" in item:
                    document = {
                        "id": item.get("id", idx + 1001),
                        "data": {
                            "text": note_text,
                            "title": note_name,
                        },
                        "predictions": _simplify_predictions(item.get("predictions", []), model_version),
                    }
                    document_items.append(document)
                else:
                    raw_annotations = item.get("annotations") or item.get("entities") or []
                    raw_relations = item.get("relations") or []
                    prediction_result = []
                    for annotation in raw_annotations:
                        if not isinstance(annotation, dict):
                            continue
                        mention = annotation.get("text") or annotation.get("mention") or annotation.get("span") or ""
                        label = annotation.get("label") or annotation.get("entity_type") or "OTHER"
                        ai_reasoning = annotation.get("ai_reasoning") or annotation.get("reasoning")
                        start_end = None
                        if isinstance(annotation.get("start"), int) and isinstance(annotation.get("end"), int):
                            start_end = (annotation["start"], annotation["end"])
                        else:
                            start_end = _compute_offsets(note_text, mention)

                        if start_end is None:
                            continue

                        value = {
                            "start": start_end[0],
                            "end": start_end[1],
                            "text": mention,
                            "labels": [label],
                        }
                        if ai_reasoning:
                            value["ai_reasoning"] = ai_reasoning

                        prediction_result.append({
                            "id": f"pred_chunk_{len(prediction_result) + 1}",
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "value": value,
                        })

                    for relation in raw_relations:
                        if not isinstance(relation, dict):
                            continue
                        source = relation.get("source") or {}
                        target = relation.get("target") or {}
                        label = relation.get("label") or relation.get("relation") or "RELATED_TO"
                        source_id = source.get("id") if isinstance(source, dict) else str(source)
                        target_id = target.get("id") if isinstance(target, dict) else str(target)
                        if not source_id:
                            source_id = f"pred_chunk_{len(prediction_result) + 1}"
                        if not target_id:
                            target_id = f"pred_chunk_{len(prediction_result) + 2}"
                        prediction_result.append({
                            "id": f"pred_rel_{len(prediction_result) + 1}",
                            "from_id": source_id,
                            "to_id": target_id,
                            "type": "relation",
                            "labels": [label],
                            "score": 0.95,
                        })

                    document_items.append({
                        "id": item.get("id", idx + 1001),
                        "data": {
                            "text": note_text,
                            "title": note_name,
                        },
                        "predictions": [{
                            "result": prediction_result,
                            "created_username": model_version,
                        }],
                    })
            if document_items:
                return document_items

    if isinstance(result, dict):
        if "predictions" in result:
            return [{
                "id": result.get("id", 1001),
                "data": {
                    "text": note_text,
                    "title": note_name,
                },
                "predictions": _simplify_predictions(result.get("predictions", []), model_version),
            }]

        raw_annotations = result.get("annotations") or result.get("entities") or []
        raw_relations = result.get("relations") or []
    else:
        raw_annotations = result if isinstance(result, list) else []
        raw_relations = []

    prediction_result: list[dict[str, Any]] = []
    for item in raw_annotations:
        if not isinstance(item, dict):
            continue
        mention = item.get("text") or item.get("mention") or item.get("span") or ""
        label = item.get("label") or item.get("entity_type") or "OTHER"
        ai_reasoning = item.get("ai_reasoning") or item.get("reasoning")
        start_end = None
        if isinstance(item.get("start"), int) and isinstance(item.get("end"), int):
            start_end = (item["start"], item["end"])
        else:
            start_end = _compute_offsets(note_text, mention)

        if start_end is None:
            continue

        value = {
            "start": start_end[0],
            "end": start_end[1],
            "text": mention,
            "labels": [label],
        }
        if ai_reasoning:
            value["ai_reasoning"] = ai_reasoning

        prediction_result.append({
            "id": f"pred_chunk_{len(prediction_result) + 1}",
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": value,
        })

    for item in raw_relations:
        if not isinstance(item, dict):
            continue
        source = item.get("source") or {}
        target = item.get("target") or {}
        label = item.get("label") or item.get("relation") or "RELATED_TO"
        source_id = source.get("id") if isinstance(source, dict) else str(source)
        target_id = target.get("id") if isinstance(target, dict) else str(target)
        if not source_id:
            source_id = f"pred_chunk_{len(prediction_result) + 1}"
        if not target_id:
            target_id = f"pred_chunk_{len(prediction_result) + 2}"
        prediction_result.append({
            "id": f"pred_rel_{len(prediction_result) + 1}",
            "from_id": source_id,
            "to_id": target_id,
            "type": "relation",
            "labels": [label],
            "score": 0.95,
        })

    return [{
        "id": 1001,
        "data": {
            "text": note_text,
            "title": note_name,
        },
        "predictions": [{
            "result": prediction_result,
            "created_username": model_version,
        }],
    }]


def _count_annotations(payload: list[dict[str, Any]]) -> dict[str, int]:
    """Count extracted entities, assertions, and relations across all documents."""
    entities = assertions = relations = 0
    for doc in payload:
        for prediction in doc.get("predictions", []):
            for item in prediction.get("result", []):
                if item.get("type") == "relation":
                    relations += 1
                elif item.get("from_name") == "assertion":
                    assertions += 1
                elif item.get("type") == "labels":
                    entities += 1
    return {
        "entities_extracted": entities,
        "assertions_extracted": assertions,
        "relations_extracted": relations,
    }


def extract_clinical_ner(
    clinical_note: str,
    guideline_text: str | None = None,
    router: LLMRouter | None = None,
    model_name: str | None = None,
    note_name: str | None = None,
) -> ExtractionResult:
    if not clinical_note or not clinical_note.strip():
        raise ValueError("clinical_note must not be empty")

    guideline_text = guideline_text or ""

    router = router or LLMRouter()

    result: RouterResult = router.complete_json(
        task=TASK,
        prompt_name="clinical_ner_user",
        system_prompt_name="clinical_ner_system",
        variables={"clinical_note": clinical_note, "guideline_text": guideline_text},
        model_preference=model_name,
        schema=_LLM_OUTPUT_SCHEMA,
        metadata={"component": "clinical_ner_extractor"},
    )

    model_version = router.settings.provider(result.provider).model_version
    payload = _normalize_jsl_payload(result.data, clinical_note, model_version=model_version, note_name=note_name)

    # Domain evaluation scores on the trace, for Langfuse quality analytics.
    counts = _count_annotations(payload)
    for name, value in counts.items():
        router.score(
            trace_id=result.trace_id, name=name, value=value, data_type="NUMERIC"
        )

    return ExtractionResult(
        payload=payload,
        provider=result.provider,
        model=result.model,
        usage=result.usage or {},
        trace_id=result.trace_id,
    )


def _build_output_path(note_name: str, backend: str, output_path: str | None = None) -> str | None:
    if output_path:
        return output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = backend.lower()
    os.makedirs(model_folder, exist_ok=True)
    return os.path.join(model_folder, f"{note_name}_output_{timestamp}.json")


def _process_note(
    note_text: str,
    guideline_text: str,
    model_name: str | None,
    router: LLMRouter,
    output_path: str | None,
    note_name: str | None = None,
) -> None:
    start_time = time.perf_counter()
    result = extract_clinical_ner(
        note_text,
        guideline_text=guideline_text,
        router=router,
        model_name=model_name,
        note_name=note_name,
    )
    elapsed_seconds = round(time.perf_counter() - start_time, 3)

    output = result.payload
    usage = result.usage or {}

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        metrics_payload = {
            "provider": result.provider,
            "model": result.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": usage.get("cached_tokens"),
            "elapsed_seconds": elapsed_seconds,
            "trace_id": result.trace_id,
        }
        print(f"METRICS_JSON {json.dumps(metrics_payload)}")
        print(f"Saved results to {output_path}")
    else:
        print(json.dumps(output, indent=2))


def _sanitize_filename_part(value: str) -> str:
    return "".join(char if (char.isalnum() or char in ("-", "_", ".")) else "_" for char in value)


ASSERTION_LABEL_MAPPING = {
    "Unknown if ever smoked": "Unknown_if_ever_smoked",
    "Current smoker": "Current_smoker",
    "Former smoker": "Former_smoker",
    "Never smoker": "Never_smoker",
    "Someone Else": "Someone_Else",
    "Smoker current status unknown": "Smoker_current_status_unknown",
}


def _apply_assertion_label_mapping(data: list[dict[str, Any]], mapping: dict[str, str]) -> None:
    for item in data:
        for prediction in item.get("predictions", []):
            for result in prediction.get("result", []):
                if result.get("from_name") != "assertion":
                    continue
                labels = result.get("value", {}).get("labels")
                if not labels:
                    continue
                result["value"]["labels"] = [mapping.get(label, label) for label in labels]


def _combine_outputs_and_update_offsets(
    output_folder: str,
    note_names: list[str],
    model_name: str | None,
    backend: str,
) -> None:
    combined: list[dict[str, Any]] = []
    for note_name in note_names:
        note_output_path = os.path.join(output_folder, f"{note_name}.json")
        with open(note_output_path, "r", encoding="utf-8") as handle:
            note_data = json.load(handle)
        if isinstance(note_data, list):
            combined.extend(note_data)
        else:
            combined.append(note_data)

    model_label = _sanitize_filename_part(model_name or backend)
    count = len(note_names)

    combined_path = os.path.join(output_folder, f"{model_label}_output_{count}.json")
    with open(combined_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)
    print(f"Saved combined results to {combined_path}")

    missing = update_offsets(combined)
    _apply_assertion_label_mapping(combined, ASSERTION_LABEL_MAPPING)

    updated_path = os.path.join(output_folder, f"{model_label}_output_{count}_updated.json")
    with open(updated_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, ensure_ascii=False)
    print(f"Saved offset-updated results to {updated_path}")
    if missing:
        print(f"{missing} result(s) could not be matched to text", file=sys.stderr)


def _process_folder(
    input_folder: str,
    output_folder: str,
    guideline_text: str,
    model_name: str | None,
    router: LLMRouter,
) -> None:
    file_names = sorted(
        name for name in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, name))
    )

    if not file_names:
        print(f"No files found in {input_folder}.")
        return

    os.makedirs(output_folder, exist_ok=True)

    succeeded_note_names: list[str] = []
    succeeded = 0
    failed = 0
    for name in file_names:
        input_path = os.path.join(input_folder, name)
        note_name = os.path.splitext(name)[0]
        output_path = os.path.join(output_folder, f"{note_name}.json")
        try:
            note_text = _read_text(input_path)
            _process_note(note_text, guideline_text, model_name, router, output_path, note_name=note_name)
            succeeded += 1
            succeeded_note_names.append(note_name)
        except Exception as exc:
            failed += 1
            print(f"{name}: failed - {exc}")

    print(f"\nProcessed {len(file_names)} file(s): {succeeded} succeeded, {failed} failed.")

    if succeeded_note_names:
        backend = _resolve_backend(model_name)
        _combine_outputs_and_update_offsets(output_folder, succeeded_note_names, model_name, backend)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract clinical NER as JSL-compatible JSON using Azure OpenAI or Anthropic")
    parser.add_argument("note", nargs="?", help="Clinical note text to analyze")
    parser.add_argument("--file", dest="file_path", help="Path to a text file containing the clinical note")
    parser.add_argument("--input_folder", dest="input_folder", help="Folder of clinical note files to process")
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        help="Folder to write JSON results to (required with --input_folder); output filenames match the input filenames",
    )
    parser.add_argument("--guideline", dest="guideline_path", help="Path to the annotation guideline file")
    parser.add_argument("--output", dest="output_path", help="Optional file to write the JSON results")
    parser.add_argument(
        "--model",
        dest="model_name",
        help="Model name used for prompt selection and backend routing. Pass 'anthropic' to use the "
        "Anthropic API, or 'gemini' to use Google Gemini, instead of Azure OpenAI. "
        "'openai'/'gpt'/'azure' all route through "
        "LLMRouter -> llm/providers/azure_openai.py (config/models.json model_aliases).",
    )
    args = parser.parse_args()

    if args.input_folder and not args.output_folder:
        parser.error("--output_folder is required when --input_folder is given")

    if args.guideline_path:
        guideline_text = _read_text(args.guideline_path)
    else:
        default_guideline = os.path.join(os.path.dirname(__file__), "annotation_guideline.md")
        guideline_text = _read_text(default_guideline) if os.path.exists(default_guideline) else ""

    router = LLMRouter()
    try:
        if args.input_folder:
            _process_folder(args.input_folder, args.output_folder, guideline_text, args.model_name, router)
            return

        if args.file_path:
            note_text = _read_text(args.file_path)
            note_name = os.path.splitext(os.path.basename(args.file_path))[0]
        else:
            note_text = args.note or ""
            if not note_text:
                note_text = input("Enter clinical note: ")
            note_name = "note"

        backend = _resolve_backend(args.model_name)
        output_path = _build_output_path(note_name, backend, args.output_path)
        _process_note(note_text, guideline_text, args.model_name, router, output_path, note_name=note_name)
    finally:
        # Ensure in-flight Langfuse traces are exported before exit.
        router.flush()


if __name__ == "__main__":
    main()
