"""Rerun the existing 300-iteration experiment without overwriting its reference."""
import hashlib
import json
from pathlib import Path
import sys
import time

from audit_environment import HERE, ROOT, snapshot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    from mcts import trace_class8_current_300 as experiment

    output = HERE / "runs" / "class8_current_300_trace"
    compare_only = sys.argv[1:] == ["--compare-only"]
    if sys.argv[1:] and not compare_only:
        raise SystemExit("Usage: reproduce_class8.py [--compare-only]")
    if output.exists() and not compare_only:
        raise SystemExit(f"Output already exists; preserve or move it before rerunning: {output}")
    if not compare_only:
        output.mkdir(parents=True)
        environment = snapshot()
        (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    reference = ROOT / "src/mcts/runs/class8_current_300_trace"
    reference_hashes = {name: digest(reference / name) for name in ("result.json", "decisions.jsonl", "README.md")}
    experiment.OUTPUT_DIR = output
    started = time.monotonic()
    if not compare_only:
        experiment.main()
    comparison = {
        "elapsed_seconds": time.monotonic() - started if not compare_only else json.loads((output / "comparison.json").read_text())["elapsed_seconds"],
        "reference_sha256": reference_hashes,
        "output_sha256": {name: digest(output / name) for name in reference_hashes},
        "result_json_equal": json.loads((output / "result.json").read_text()) == json.loads((reference / "result.json").read_text()),
        "reference_unchanged": all(digest(reference / name) == value for name, value in reference_hashes.items()),
    }
    comparison["all_files_byte_equal"] = comparison["reference_sha256"] == comparison["output_sha256"]
    comparison["newline_normalized_reference_sha256"] = {name: normalized_digest(reference / name) for name in reference_hashes}
    comparison["newline_normalized_output_sha256"] = {name: normalized_digest(output / name) for name in reference_hashes}
    comparison["all_files_equal_after_newline_normalization"] = comparison["newline_normalized_reference_sha256"] == comparison["newline_normalized_output_sha256"]
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)
    if not comparison["all_files_equal_after_newline_normalization"] or not comparison["reference_unchanged"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
