#!/usr/bin/env python3
"""Run lrslib/mplrs to generate the Bell 3-2-2 facet H-representation."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "polytope_322.ext"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "facets_322.txt"
DEFAULT_LRS = PROJECT_ROOT / "external" / "lrslib-073a" / "lrs"
DEFAULT_MPLRS = PROJECT_ROOT / "external" / "lrslib-073a" / "mplrs"


def ensure_executable(path: Path) -> None:
    if path.exists():
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)


def estimate_complexity(lrs_bin: Path, input_path: Path, timeout: int) -> None:
    if not lrs_bin.exists():
        print(f"skip estimate: lrs not found at {lrs_bin}")
        return
    ensure_executable(lrs_bin)
    result = subprocess.run(
        [str(lrs_bin), str(input_path), "-est"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def run_mplrs(mplrs_bin: Path, input_path: Path, output_path: Path, processes: int) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"missing input file: {input_path}")
    if not mplrs_bin.exists():
        raise FileNotFoundError(f"missing mplrs binary: {mplrs_bin}")
    if processes < 2:
        raise ValueError("mplrs should be run with at least 2 processes")

    ensure_executable(mplrs_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mpirun", "-np", str(processes), str(mplrs_bin), str(input_path)]
    print("running:", " ".join(cmd))
    start = time.time()
    with output_path.open("w", encoding="utf-8") as outfile:
        process = subprocess.Popen(
            cmd,
            stdout=outfile,
            stderr=subprocess.PIPE,
            text=True,
        )
        while process.poll() is None:
            time.sleep(2)
            size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
            elapsed = int(time.time() - start)
            print(f"\rseconds={elapsed} output_mb={size_mb:.2f}", end="", flush=True)
        print()
        stderr = process.stderr.read() if process.stderr is not None else ""
    if stderr:
        print(stderr)
    if process.returncode != 0:
        raise RuntimeError(f"mplrs exited with code {process.returncode}")
    print(f"saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data/facets_322.txt using mplrs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lrs-bin", type=Path, default=DEFAULT_LRS)
    parser.add_argument("--mplrs-bin", type=Path, default=DEFAULT_MPLRS)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--estimate-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.estimate:
        estimate_complexity(args.lrs_bin, args.input, args.estimate_timeout)
    run_mplrs(args.mplrs_bin, args.input, args.output, args.processes)


if __name__ == "__main__":
    main()
