from __future__ import annotations

"""Compare the short run and, if it passes, execute an auditable long run."""

import argparse
import gc
import json
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from .analyze_continuation_paths import analyze_paths
from .analyze_quotient_run import analyze_run
from .atlas import DEFAULT_HANDOFF_ROOT
from .audit_continuation_run import audit_paths
from .continuation_search_v30 import Bell422EarlyNoopSearch, EarlyNoopSearchConfig


def compare_runs(reference: dict, candidate: dict) -> dict:
    old_events = reference["coverage"]["discoveries"]
    new_events = candidate["coverage"]["discoveries"]
    old_rows = {tuple(event["canonical_h_row"]) for event in old_events}
    new_rows = {tuple(event["canonical_h_row"]) for event in new_events}
    config_differences = {
        key: {"reference": value, "candidate": candidate["config"].get(key)}
        for key, value in reference["config"].items()
        if key != "progress_interval" and candidate["config"].get(key) != value
    }
    normalized = lambda events: [{key: value for key, value in event.items()
                                   if key != "elapsed_seconds"} for event in events]
    old_seconds = reference["runtime"]["elapsed_seconds"]
    new_seconds = candidate["runtime"]["elapsed_seconds"]
    time_reduction = 1.0 - new_seconds / old_seconds
    same_paths = normalized(old_events) == normalized(new_events)
    same_tree = reference["tree"] == candidate["tree"]
    complete = (
        not candidate["runtime"]["truncated"]
        and candidate["runtime"]["iterations_completed"] == reference["config"]["iterations"]
    )
    return {
        "reference_seconds": old_seconds, "candidate_seconds": new_seconds,
        "reference_classes": len(old_rows), "candidate_classes": len(new_rows),
        "common_classes": len(old_rows & new_rows),
        "identical_canonical_class_set": old_rows == new_rows,
        "identical_discovery_iterations_paths_and_metadata": same_paths,
        "identical_tree": same_tree, "shared_config_differences": config_differences,
        "completed_short_run": complete, "time_reduction_fraction": time_reduction,
        "speedup": old_seconds / new_seconds,
        "required_time_reduction_fraction": 0.05,
        "eligible_for_long_run": bool(
            complete and not config_differences and old_rows == new_rows
            and same_paths and same_tree and time_reduction >= 0.05
        ),
    }


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(payload: dict, path: Path) -> None:
    Bell422EarlyNoopSearch._write_result(payload, path)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def run_long(reference_path: Path, short_path: Path, output: Path, *, iterations: int) -> dict:
    wall_started = time.perf_counter()
    reference, short = load(reference_path), load(short_path)
    comparison = compare_runs(reference, short)
    if not comparison["eligible_for_long_run"]:
        raise ValueError("short-run comparison did not pass the long-run gate")
    if output.resolve() in {reference_path.resolve(), short_path.resolve()} or output.exists():
        raise ValueError("long-run output must be a new file, not a reference result")
    if iterations <= short["config"]["iterations"]:
        raise ValueError("long-run budget must exceed the short-run budget")
    values = asdict(EarlyNoopSearchConfig())
    values.update(short["config"])
    values.update(iterations=iterations, max_wall_seconds=None)
    config = EarlyNoopSearchConfig(**values)
    status_path = output.with_suffix(".status.json")
    status = {
        "state": "RUNNING", "phase": "search", "pid": os.getpid(),
        "started_at": now(), "planned_iterations": iterations,
        "iterations_completed": 0, "class_count": 0,
        "output": str(output.resolve()), "reference": str(reference_path.resolve()),
        "short_run": str(short_path.resolve()), "short_run_comparison": comparison,
        "wall_time_limit_seconds": None,
    }
    save(status, status_path)
    rolling = output.with_suffix(".rolling.json")
    milestones = {1000, 5000, 10000, 20000, 30000, 50000, 75000}
    checkpoints = {
        iteration: (output.with_name(f"{output.stem}_checkpoint_i{iteration}.json")
                    if iteration in milestones else rolling)
        for iteration in range(1000, iterations, 1000)
    }

    class TrackedSearch(Bell422EarlyNoopSearch):
        def run_iteration(self, iteration: int) -> None:
            super().run_iteration(iteration)
            if iteration % 1000 == 0 or iteration == self.config.iterations:
                status.update(
                    updated_at=now(), iterations_completed=iteration,
                    class_count=len(self.discovered),
                    elapsed_search_seconds=time.perf_counter() - self.started_at,
                    node_count=len(self.nodes),
                )
                save(status, status_path)

    try:
        engine = TrackedSearch(config, handoff_root=DEFAULT_HANDOFF_ROOT)
        result = engine.run(checkpoint_paths=checkpoints)
        save(result, output)
        status.update(
            phase="audits", search_finished_at=now(),
            iterations_completed=result["runtime"]["iterations_completed"],
            class_count=result["coverage"]["discovered_class_count"],
            runtime=result["runtime"], node_count=result["tree"]["node_count"],
            search_phase_wall_seconds=time.perf_counter() - wall_started,
        )
        save(status, status_path)
        del engine, result, reference, short
        gc.collect()
        path_audit = audit_paths(output)
        save(path_audit, output.with_suffix(".paths.json"))
        canonical_audit = analyze_run(output, handoff_root=DEFAULT_HANDOFF_ROOT)
        save(canonical_audit, output.with_suffix(".analysis.json"))
        canonical = canonical_audit["canonical_audit"]
        verified = (
            path_audit["failure_count"] == 0
            and canonical["revalidated_terminal_count"] == status["class_count"]
            and canonical["duplicate_canonical_h_row_count"] == 0
            and canonical["duplicate_support_count"] == 0
            and canonical["duplicate_class_id_count"] == 0
            and canonical["generator_invariance_failure_count"] == 0
            and canonical["nonprimitive_canonical_rows"] == 0
        )
        status.update(
            state="COMPLETED" if verified else "AUDIT_FAILED", phase="done",
            finished_at=now(), path_audit_failures=path_audit["failure_count"],
            canonical_audit=canonical,
        )
        del path_audit, canonical_audit
        gc.collect()
        paths = analyze_paths(output)
        save(paths, output.with_suffix(".path_analysis.json"))
        report = [
            "# Bell 422 v30 Long Run", "",
            f"- State: {status['state']}; seed: {config.seed}.",
            f"- Iterations: {status['iterations_completed']}/{iterations}.",
            f"- Canonical classes: {status['class_count']}.",
            f"- Search seconds: {status['runtime']['elapsed_seconds']:.2f}.",
            "- No wall-time cap. The partial catalog is not the complete 422 class universe.",
            "- Audits and path analysis run after the measured search.", "",
            "| Window end | New classes | Classes/hour | Local new | Root new |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        report.extend(
            f"| {window['end_iteration']} | {window['new_classes']} | "
            f"{window['classes_per_hour']:.2f} | {window['continuation_new_classes']} | "
            f"{window['root_episode_new_classes']} |"
            for window in paths["windows"]
        )
        output.with_suffix(".report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        status["host_wall_seconds_including_audits"] = time.perf_counter() - wall_started
        save(status, status_path)
        return status
    except BaseException as error:
        status.update(state="FAILED", phase=status["phase"], failed_at=now(),
                      error=str(error), traceback=traceback.format_exc())
        save(status, status_path)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compare", "long"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--short-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args(argv)
    if args.mode == "compare":
        result = compare_runs(load(args.reference), load(args.short_run))
        save(result, args.output)
    else:
        result = run_long(args.reference, args.short_run, args.output, iterations=args.iterations)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
