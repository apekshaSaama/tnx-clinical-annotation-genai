from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

SMOKING_KEYWORDS = [
    r"smok\w*",
    r"tobacco",
    r"cigarette\w*",
    r"cigar\w*",
    r"nicotine",
    r"vap\w*",
    r"e-cigarette\w*",
    r"pack[- ]year\w*",
    r"chew(?:ing|ed)?\s+tobacco",
    r"snuff",
]

SMOKING_PATTERN = re.compile(r"\b(?:" + "|".join(SMOKING_KEYWORDS) + r")\b", re.IGNORECASE)

SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "notes", "smoking")
SNIPPETS_DIR = os.path.join(BASE_DIR, "smoking_snippets")


@dataclass
class Snippet:
    text: str
    start: int
    end: int


def _iter_sentences(note_text: str):
    for match in SENTENCE_PATTERN.finditer(note_text):
        sentence = match.group()
        if sentence.strip():
            yield sentence, match.start()


def extract_smoking_snippets(note_text: str) -> list[Snippet]:
    if not note_text or not note_text.strip():
        return []

    snippets: list[Snippet] = []
    for sentence, sentence_start in _iter_sentences(note_text):
        if not SMOKING_PATTERN.search(sentence):
            continue
        stripped = sentence.strip()
        offset = sentence.index(stripped)
        start = sentence_start + offset
        snippets.append(Snippet(text=stripped, start=start, end=start + len(stripped)))
    return snippets


def generate_snippets(input_folder: str, output_folder: str, quiet: bool = True) -> dict[str, list[Snippet]]:
    """Extract smoking snippets for every file in input_folder.

    Writes one snippet file per note into output_folder under the same file
    name, so callers can look up a note's snippets by matching file name.
    """
    os.makedirs(output_folder, exist_ok=True)

    results: dict[str, list[Snippet]] = {}
    for name in sorted(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, name)
        if not os.path.isfile(input_path):
            continue

        with open(input_path, "r", encoding="utf-8") as handle:
            note_text = handle.read()

        snippets = extract_smoking_snippets(note_text)
        results[name] = snippets

        output_path = os.path.join(output_folder, name)
        with open(output_path, "w", encoding="utf-8") as handle:
            for snippet in snippets:
                handle.write(f"{snippet.text}\n")

        if not quiet:
            if snippets:
                print(f"{name}: {len(snippets)} smoking-related snippet(s).")
            else:
                print(f"{name}: no smoking-related snippets found.")

    return results


def process_file(input_path: str, output_rel_name: str, quiet: bool = False) -> list[Snippet]:
    with open(input_path, "r", encoding="utf-8") as handle:
        note_text = handle.read()

    snippets = extract_smoking_snippets(note_text)
    if not snippets:
        if not quiet:
            print(f"{output_rel_name}: no smoking-related snippets found.")
        return snippets

    if not quiet:
        print(f"{output_rel_name}:")
        for snippet in snippets:
            print(f"  [{snippet.start}-{snippet.end}] {snippet.text}")

    output_path = os.path.join(SNIPPETS_DIR, output_rel_name)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for snippet in snippets:
            handle.write(f"{snippet.text}\n")

    return snippets


def process_folder(folder_name: str) -> None:
    folder_path = folder_name if os.path.isabs(folder_name) else os.path.join(NOTES_DIR, folder_name)
    folder_path = os.path.normpath(folder_path)

    file_names = sorted(
        name for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
    )

    if not file_names:
        print(f"No files found in {folder_path}.")
        return

    # Output files always land under SNIPPETS_DIR, mirroring the folder's
    # position relative to NOTES_DIR (or just its base name if it lives
    # outside NOTES_DIR, e.g. when an absolute path was passed).
    rel_folder = os.path.relpath(folder_path, NOTES_DIR)
    if rel_folder.startswith(os.pardir):
        rel_folder = os.path.basename(folder_path)

    total_snippets = 0
    files_with_snippets = 0
    for name in file_names:
        input_path = os.path.join(folder_path, name)
        output_rel_name = os.path.join(rel_folder, name)
        snippets = process_file(input_path, output_rel_name)
        if snippets:
            files_with_snippets += 1
            total_snippets += len(snippets)

    print(
        f"\nProcessed {len(file_names)} file(s): "
        f"{files_with_snippets} contained smoking-related snippets "
        f"({total_snippets} total)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract smoking-related snippets from clinical note(s)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", dest="file_path", help="Path to a single note file, relative to notes/smoking/")
    group.add_argument("--folder", dest="folder_path", help="Path to a folder of note files, relative to notes/smoking/")
    args = parser.parse_args()

    if args.folder_path:
        process_folder(args.folder_path)
    else:
        process_file(os.path.join(NOTES_DIR, args.file_path), args.file_path)


if __name__ == "__main__":
    main()
