"""Recover structural report sections after validating a deterministic replay."""

import argparse
import json
from pathlib import Path


def read_valid_prefix(text):
    decoder = json.JSONDecoder()
    position = text.index("{") + 1
    fields = {}
    while True:
        while text[position].isspace() or text[position] == ",":
            position += 1
        key, position = decoder.raw_decode(text, position)
        while text[position].isspace() or text[position] == ":":
            position += 1
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return fields, key
        fields[key] = value


def without_times(value):
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items()
                if key != "elapsed_seconds"}
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    text = args.original.read_text(encoding="utf-8")
    prefix, damaged_field = read_valid_prefix(text)
    if damaged_field != "facet_corrector":
        raise ValueError(f"Unexpected first damaged field: {damaged_field}")
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    required = {"config", "summary", "discoveries", "growth_activations", "recycle_events",
                "corrector_diagnostics", "corrector_novelty_credit_events"}
    if not required <= prefix.keys():
        raise ValueError("Incomplete original trajectory; cannot validate deterministic replay")
    for key in required:
        if without_times(prefix[key]) != without_times(replay[key]):
            raise ValueError(f"Replay diverged in {key}; refusing recovery")
    # The statistics object precedes the damaged anchor array and is intact.
    decoder = json.JSONDecoder()
    position = text.index('"facet_corrector":') + len('"facet_corrector":')
    position = text.index('"statistics":', position) + len('"statistics":')
    while text[position].isspace():
        position += 1
    statistics, _ = decoder.raw_decode(text, position)
    if statistics != replay["facet_corrector"]["statistics"]:
        raise ValueError("Replay diverged in facet statistics")
    restored = dict(replay)
    restored.update(prefix)
    restored["metadata"]["structural_sections_recovered_from"] = str(args.replay)
    restored["metadata"]["recovery_verification"] = (
        "identical non-timing trajectory, original summary and facet statistics; "
        "original timestamps preserved"
    )
    backup = args.original.with_suffix(".damaged.txt")
    if backup.exists():
        raise FileExistsError(backup)
    backup.write_text(text, encoding="utf-8")
    output = json.dumps(restored, ensure_ascii=False, indent=2)
    json.loads(output)
    temporary = args.original.with_suffix(".repaired.tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.original)
    print(json.dumps({"recovered": str(args.original), "backup": str(backup),
                      "verified_fields": sorted(required),
                      "original_elapsed_seconds": restored["summary"]["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
