from __future__ import annotations

"""Screen v28, then collect a reproducible long-run baseline or candidate."""

import argparse
import json
import time
from pathlib import Path

from .atlas import DEFAULT_HANDOFF_ROOT
from .quotient_search_v27 import AdaptiveQuotientSearchConfig, Bell422AdaptiveQuotientSearch
from .symmetric_search_v28 import Bell422SymmetricSearch, SymmetricSearchConfig
from .v25_search import DEFAULT_RUNS_DIR


def screening_decision(probe: dict, baseline: dict) -> dict:
    probe_count = int(probe["coverage"]["discovered_class_count"])
    baseline_count = int(baseline["coverage"]["discovered_class_count"])
    probe_seconds = float(probe["runtime"]["elapsed_seconds"])
    baseline_seconds = float(baseline["runtime"]["elapsed_seconds"])
    if probe_count < baseline_count:
        selected, reason = "v27", "v28_discovery_count_below_baseline"
    elif probe_count > baseline_count:
        selected, reason = "v28", "v28_discovery_count_above_baseline"
    elif probe_seconds <= 0.95 * baseline_seconds:
        selected, reason = "v28", "equal_coverage_runtime_improved_at_least_5_percent"
    else:
        selected, reason = None, "equal_coverage_without_clear_runtime_improvement"
    return {
        "selected_version": selected,
        "reason": reason,
        "probe_classes": probe_count,
        "baseline_classes": baseline_count,
        "probe_seconds": probe_seconds,
        "baseline_seconds": baseline_seconds,
        "coverage_ratio": probe_count / max(1, baseline_count),
        "runtime_ratio": probe_seconds / max(1e-9, baseline_seconds),
        "discovery_rate_ratio": (probe_count / max(1e-9, probe_seconds))
        / max(1e-9, baseline_count / max(1e-9, baseline_seconds)),
        "criterion": "coverage_first_single_seed_screening",
    }


class ProgressRecorder:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.progress_windows: list[dict] = []

    def run_iteration(self, iteration: int) -> None:
        super().run_iteration(iteration)
        if iteration % 1000 == 0:
            previous = self.progress_windows[-1] if self.progress_windows else {}
            elapsed = time.perf_counter() - self.started_at
            classes = len(self.discovered)
            seconds = elapsed - previous.get("elapsed_seconds", 0.0)
            added = classes - previous.get("classes", 0)
            last_discovery = int(self.discovery_events[-1]["iteration"]) if self.discovery_events else 0
            self.progress_windows.append({
                "iteration": iteration,
                "classes": classes,
                "catalog_matched": sum(
                    event.get("catalog_index") is not None for event in self.discovery_events
                ),
                "new_classes": added,
                "elapsed_seconds": elapsed,
                "window_seconds": seconds,
                "new_classes_per_second": added / max(1e-9, seconds),
                "last_discovery_iteration": last_discovery,
                "iterations_since_last_discovery": iteration - last_discovery,
                "nodes": len(self.nodes),
                "rank_misses": self.rank_oracle.misses,
                "facet_validations": self.facet_oracle.validation_count,
            })

    def result(self) -> dict:
        result = super().result()
        result["progress_windows"] = list(self.progress_windows)
        result["benchmark"] = {"progress_window_iterations": 1000, "policy_changed_by_recorder": False}
        return result


class RecordedBaselineSearch(ProgressRecorder, Bell422AdaptiveQuotientSearch):
    pass


class RecordedSymmetricSearch(ProgressRecorder, Bell422SymmetricSearch):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, default=(
        DEFAULT_RUNS_DIR / "bell422_adaptive_quotient_i1000_s4222025_v27c.json"
    ))
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=4_222_025)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    probe = json.loads(args.probe_result.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    if probe["runtime"]["iterations_completed"] != baseline["runtime"]["iterations_completed"]:
        raise SystemExit("screening runs must use equal iteration budgets")
    if probe["config"]["seed"] != baseline["config"]["seed"] or args.seed != probe["config"]["seed"]:
        raise SystemExit("screening and long-run seeds must match")
    report = screening_decision(probe, baseline)
    report.update({"probe": str(args.probe_result), "baseline": str(args.baseline_result)})
    report_path = DEFAULT_RUNS_DIR / f"bell422_v28_screening_s{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    version = report["selected_version"]
    if args.report_only or version is None:
        return 0
    if version == "v28":
        # Carry the screened configuration into the long run, changing only its budget.
        settings = {key: value for key, value in probe["config"].items()
                    if key in SymmetricSearchConfig.__dataclass_fields__}
        settings.update(iterations=args.iterations, progress_interval=1000)
        engine = RecordedSymmetricSearch(SymmetricSearchConfig(**settings), handoff_root=args.handoff_root)
        prefix = "bell422_symmetric"
    else:
        settings = {key: value for key, value in baseline["config"].items()
                    if key in AdaptiveQuotientSearchConfig.__dataclass_fields__}
        settings.update(iterations=args.iterations, progress_interval=1000,
                        enable_adaptive_schedule=False)
        engine = RecordedBaselineSearch(AdaptiveQuotientSearchConfig(**settings), handoff_root=args.handoff_root)
        prefix = "bell422_adaptive_quotient"
    stem = f"{prefix}_i{args.iterations}_s{args.seed}_{version}_longrun"
    checkpoints = {
        iteration: DEFAULT_RUNS_DIR / f"{prefix}_i{iteration}_s{args.seed}_{version}_longrun.json"
        for iteration in sorted({1000, 5000, *range(10_000, args.iterations, 10_000)})
        if iteration < args.iterations
    }
    result = engine.run(checkpoint_paths=checkpoints)
    result["benchmark"]["screening"] = report
    output = DEFAULT_RUNS_DIR / f"{stem}.json"
    engine._write_result(result, output)
    print(json.dumps({
        "output": str(output), "classes": result["coverage"]["discovered_class_count"],
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
        "iterations_completed": result["runtime"]["iterations_completed"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
