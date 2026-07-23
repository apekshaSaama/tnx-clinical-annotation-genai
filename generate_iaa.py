#!/usr/bin/env python3
"""
generate_IAA.py

Compute Inter-Annotator Agreement (IAA) between human ground-truth annotations
and model pre-annotations exported from John Snow Labs Generative AI Lab.

Both inputs are JSON files that share the SAME `data.title` value. They do
NOT share the same task `id`. One file holds the human completions (ground
truth), the other holds the new pre-annotations.

Metrics reported:
  - Token-level Cohen's kappa (BIO tagging)  -> the defensible "IAA" number
  - Span-level precision / recall / F1        -> exact-match and overlap-match
  - Per-label span F1 breakdown

Usage:
    python generate_IAA.py --gt ground_truth.json --pred preannotations.json
    python generate_IAA.py --gt gt.json --pred pred.json --match overlap --csv report.csv

Dependencies:
    pip install scikit-learn
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _extract_results(task):
    """Return the list of result dicts for a task, whether it stores the spans
    under completions[] (ground truth) or predictions[] / result (pre-annots)."""
    comps = task.get("completions") or []
    if comps:
        # prefer swati's completion (ground truth annotator); fall back to
        # the starred / ground-truth completion, then the last one
        chosen = next(
            (c for c in comps if c.get("created_username") == "swati"),
            next(
                (c for c in comps if c.get("honeypot") or c.get("starred") or c.get("ground_truth")),
                comps[-1],
            ),
        )
        return chosen.get("result", []) or []

    preds = task.get("predictions") or []
    if preds:
        return preds[0].get("result", []) or []

    return task.get("result", []) or []


def _results_to_spans(results):
    """Convert a result list into [(start, end, label), ...] for NER-style labels."""
    spans = []
    for r in results:
        v = r.get("value", {})
        if "labels" in v and v.get("labels"):
            spans.append((v["start"], v["end"], v["labels"][0]))
    return spans


def load_file(path):
    """Return (spans, texts) dicts keyed by `data.title`.

    Ground-truth and prediction exports do NOT share the same task `id`, but
    they do share `data.title` (e.g. the source document name/number), so
    that is the key used to align tasks across the two files.

    spans = {title: [(start, end, label), ...]}
    texts = {title: source_text or None}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks = data if isinstance(data, list) else data.get("tasks", [data])

    spans, texts = {}, {}
    for t in tasks:
        task_data = t.get("data") if isinstance(t.get("data"), dict) else {}
        title = task_data.get("title")
        if title is None:
            continue
        title = str(title)
        text = task_data.get("text")
        texts[title] = text if isinstance(text, str) else None
        spans[title] = _results_to_spans(_extract_results(t))
    return spans, texts


# --------------------------------------------------------------------------- #
# Tokenisation + BIO tagging (for token-level kappa)
# --------------------------------------------------------------------------- #
def tokenize(text):
    """Whitespace tokenizer yielding (token, start, end).

    IMPORTANT: to make token-level kappa reflect real disagreement rather than
    tokenizer differences, swap this out for the SAME tokenizer your model uses.
    """
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def bio_tags(text, spans):
    """Turn character spans into a per-token BIO tag sequence."""
    tags = []
    for _, s, e in tokenize(text):
        tag = "O"
        for (ss, se, lab) in spans:
            if s < se and e > ss:  # token overlaps this span
                tag = ("B-" if s <= ss else "I-") + lab
                break
        tags.append(tag)
    return tags


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def token_kappa(texts, gt, pred, common_ids):
    """Token-level Cohen's kappa over BIO tags. Requires source text per task."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("scikit-learn not installed; skipping token-level kappa "
              "(pip install scikit-learn)", file=sys.stderr)
        return None

    y_true, y_pred = [], []
    skipped = 0
    for title in common_ids:
        text = texts.get(title)
        if not text:
            skipped += 1
            continue
        y_true += bio_tags(text, gt.get(title, []))
        y_pred += bio_tags(text, pred.get(title, []))

    if not y_true:
        print("No task text available -> cannot compute token-level kappa. "
              "Ensure the GT export includes data.text.", file=sys.stderr)
        return None
    if skipped:
        print(f"Note: {skipped} task(s) had no text and were excluded from kappa.",
              file=sys.stderr)
    return cohen_kappa_score(y_true, y_pred)


def span_f1(gt, pred, common_ids, match="exact"):
    """Span-level precision / recall / F1, aggregated and per-label."""
    tp = fp = fn = 0
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for title in common_ids:
        g = set(gt.get(title, []))
        p = set(pred.get(title, []))

        if match == "exact":
            for span in p & g:
                tp += 1; per_label[span[2]]["tp"] += 1
            for span in p - g:
                fp += 1; per_label[span[2]]["fp"] += 1
            for span in g - p:
                fn += 1; per_label[span[2]]["fn"] += 1
        else:  # overlap match, same label
            matched_g = set()
            for (ps, pe, pl) in p:
                hit = next(
                    ((gs, ge, gl) for (gs, ge, gl) in g
                     if gl == pl and ps < ge and pe > gs and (gs, ge, gl) not in matched_g),
                    None,
                )
                if hit:
                    tp += 1; per_label[pl]["tp"] += 1; matched_g.add(hit)
                else:
                    fp += 1; per_label[pl]["fp"] += 1
            for span in g - matched_g:
                fn += 1; per_label[span[2]]["fn"] += 1

    def prf(t, f_p, f_n):
        prec = t / (t + f_p) if (t + f_p) else 0.0
        rec = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    prec, rec, f1 = prf(tp, fp, fn)
    overall = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    labels = {}
    for lab, c in sorted(per_label.items()):
        p_, r_, f_ = prf(c["tp"], c["fp"], c["fn"])
        labels[lab] = {"precision": p_, "recall": r_, "f1": f_, **c}
    return overall, labels


# --------------------------------------------------------------------------- #
# Per-chunk IAA report
# --------------------------------------------------------------------------- #
def _pair_spans(g_spans, p_spans, match="exact"):
    """Greedily pair GT spans with prediction spans by character position
    (label-agnostic, so a chunk both sides pointed at but labelled
    differently still shows up as one paired row instead of a FN + FP).

    Returns a list of (gt_span_or_None, pred_span_or_None) tuples.
    """
    g = list(g_spans)
    p = list(p_spans)
    used_p = [False] * len(p)
    pairs = []

    for gs in g:
        gs_s, gs_e, _ = gs
        hit = None
        for i, ps in enumerate(p):
            if used_p[i]:
                continue
            ps_s, ps_e, _ = ps
            if match == "exact":
                is_match = ps_s == gs_s and ps_e == gs_e
            else:  # overlap
                is_match = ps_s < gs_e and ps_e > gs_s
            if is_match:
                hit = i
                break
        if hit is not None:
            used_p[hit] = True
            pairs.append((gs, p[hit]))
        else:
            pairs.append((gs, None))

    for i, ps in enumerate(p):
        if not used_p[i]:
            pairs.append((None, ps))

    return pairs


def _chunk_text(span, text):
    if span is None or not text:
        return ""
    s, e, _ = span
    return text[s:e]


def _context_window(text, start, end, window=60):
    if not text:
        return ""
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


def build_iaa_rows(gt, pred, texts, common_ids, match="exact", context_window=60):
    """Build one row per GT/prediction chunk, aligned by character position.

    Columns: chunks, ground_truth_label, prediction_label, title,
    agreement, context.
    """
    rows = []
    for title in sorted(common_ids, key=str):
        text = texts.get(title) or ""
        pairs = _pair_spans(gt.get(title, []), pred.get(title, []), match=match)

        for gs, ps in pairs:
            ref_span = gs if gs is not None else ps
            gt_label = gs[2] if gs is not None else ""
            pred_label = ps[2] if ps is not None else ""
            chunk = _chunk_text(gs, text) or _chunk_text(ps, text)
            agreement = "Agree" if gt_label and pred_label and gt_label == pred_label else "Disagree"

            rows.append({
                "chunks": chunk,
                "ground_truth_label": gt_label,
                "prediction_label": pred_label,
                "title": title,
                "agreement": agreement,
                "context": _context_window(text, ref_span[0], ref_span[1], context_window),
            })

    return rows


def write_iaa_rows(path, rows):
    fieldnames = ["chunks", "ground_truth_label", "prediction_label", "title", "agreement", "context"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    agree = sum(1 for r in rows if r["agreement"] == "Agree")
    total = len(rows)
    rate = agree / total if total else 0.0
    print(f"({agree}/{total} chunks agree, {rate:.2%})")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def check_alignment(gt, pred, gt_texts, pred_texts):
    """Return the set of shared `data.title` values and warn about mismatches."""
    common = set(gt) & set(pred)
    only_gt = set(gt) - set(pred)
    only_pred = set(pred) - set(gt)

    print(f"Ground-truth tasks : {len(gt)}")
    print(f"Pre-annot tasks    : {len(pred)}")
    print(f"Shared titles      : {len(common)}")
    if only_gt:
        print(f"  ! {len(only_gt)} title(s) only in ground truth (ignored), "
              f"e.g. {sorted(only_gt)[:5]}")
    if only_pred:
        print(f"  ! {len(only_pred)} title(s) only in pre-annotations (ignored), "
              f"e.g. {sorted(only_pred)[:5]}")

    # offsets must be against the same text or every span looks like a disagreement
    text_mismatch = [
        title for title in common
        if gt_texts.get(title) and pred_texts.get(title) and gt_texts[title] != pred_texts[title]
    ]
    if text_mismatch:
        print(f"  !! WARNING: source text differs for {len(text_mismatch)} shared "
              f"title(s) (e.g. {text_mismatch[:3]}). Offsets will not align and IAA "
              f"will be understated. Compute prediction offsets against the raw "
              f"text stored in the GT export.")
    print()
    return common


def print_report(kappa, exact, overlap, per_label_exact):
    print("=" * 60)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 60)
    if kappa is not None:
        print(f"Token-level Cohen's kappa (BIO) : {kappa:.4f}")
    print()
    print("Span-level (exact match)")
    print(f"  precision {exact['precision']:.4f}  recall {exact['recall']:.4f}  "
          f"F1 {exact['f1']:.4f}   (tp={exact['tp']} fp={exact['fp']} fn={exact['fn']})")
    print("Span-level (overlap match)")
    print(f"  precision {overlap['precision']:.4f}  recall {overlap['recall']:.4f}  "
          f"F1 {overlap['f1']:.4f}   (tp={overlap['tp']} fp={overlap['fp']} fn={overlap['fn']})")
    print()
    print("Per-label span F1 (exact match)")
    print(f"  {'label':<24}{'prec':>8}{'rec':>8}{'F1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    for lab, c in per_label_exact.items():
        print(f"  {lab:<24}{c['precision']:>8.3f}{c['recall']:>8.3f}"
              f"{c['f1']:>8.3f}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}")
    print("=" * 60)


def write_csv(path, kappa, exact, overlap, per_label_exact):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "label", "precision", "recall", "f1", "tp", "fp", "fn"])
        if kappa is not None:
            w.writerow(["token_cohens_kappa", "", kappa, "", "", "", "", ""])
        w.writerow(["span_exact", "ALL", exact["precision"], exact["recall"],
                    exact["f1"], exact["tp"], exact["fp"], exact["fn"]])
        w.writerow(["span_overlap", "ALL", overlap["precision"], overlap["recall"],
                    overlap["f1"], overlap["tp"], overlap["fp"], overlap["fn"]])
        for lab, c in per_label_exact.items():
            w.writerow(["span_exact", lab, c["precision"], c["recall"],
                        c["f1"], c["tp"], c["fp"], c["fn"]])
    #print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="IAA between JSL ground truth and pre-annotations.")
    ap.add_argument("--gt", required=True, help="Path to ground-truth JSON export")
    ap.add_argument("--pred", required=True, help="Path to pre-annotations JSON")
    ap.add_argument("--match", choices=["exact", "overlap"], default="exact",
                    help="Primary span-match mode (both are reported regardless)")
    ap.add_argument("--csv", help="Optional path to write an aggregate metrics CSV report")
    ap.add_argument("--iaa-report", help="Optional path to write a per-chunk IAA report "
                    "(columns: chunks, ground_truth_label, prediction_label, title, "
                    "agreement, context)")
    ap.add_argument("--context-window", type=int, default=60,
                    help="Characters of surrounding text to include on each side of a "
                    "chunk in the --iaa-report context column (default: 60)")
    args = ap.parse_args()

    gt_spans, gt_texts = load_file(args.gt)
    pred_spans, pred_texts = load_file(args.pred)

    common = check_alignment(gt_spans, pred_spans, gt_texts, pred_texts)
    if not common:
        print("No shared task ids. Nothing to compare.", file=sys.stderr)
        sys.exit(1)

    kappa = token_kappa(gt_texts, gt_spans, pred_spans, common)
    exact, per_label_exact = span_f1(gt_spans, pred_spans, common, match="exact")
    overlap, _ = span_f1(gt_spans, pred_spans, common, match="overlap")

    print_report(kappa, exact, overlap, per_label_exact)
    if args.csv:
        write_csv(args.csv, kappa, exact, overlap, per_label_exact)

    if args.iaa_report:
        rows = build_iaa_rows(gt_spans, pred_spans, gt_texts, common,
                               match=args.match, context_window=args.context_window)
        write_iaa_rows(args.iaa_report, rows)


if __name__ == "__main__":
    main()