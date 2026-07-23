import argparse
import json
import sys


def find_best_offset(text, needle, hint_start):
    """Find the occurrence of `needle` in `text` closest to `hint_start`."""
    occurrences = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        occurrences.append(idx)
        start = idx + 1

    if not occurrences:
        return None

    return min(occurrences, key=lambda idx: abs(idx - hint_start))


def update_offsets(data):
    missing = 0

    for item in data:
        text = item.get("text") or item.get("data", {}).get("text", "")

        for prediction in item.get("predictions", []):
            for result in prediction.get("result", []):
                value = result.get("value", {})
                needle = value.get("text")
                if needle is None:
                    continue

                hint_start = value.get("start", 0)
                new_start = find_best_offset(text, needle, hint_start)

                if new_start is None:
                    missing += 1
                    print(
                        f"Warning: could not find text {needle!r} in "
                        f"item id={item.get('id')}",
                        file=sys.stderr,
                    )
                    continue

                value["start"] = new_start
                value["end"] = new_start + len(needle)

    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate start/end offsets of prediction results using data.text"
    )
    parser.add_argument("input_file", help="Path to input JSON file (e.g. openaii_output_103.json)")
    parser.add_argument(
        "-o", "--output",
        help="Path to output JSON file (default: overwrite input file)",
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = update_offsets(data)

    output_file = args.output or args.input_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated offsets written to {output_file}")
    if missing:
        print(f"{missing} result(s) could not be matched to text", file=sys.stderr)


if __name__ == "__main__":
    main()
