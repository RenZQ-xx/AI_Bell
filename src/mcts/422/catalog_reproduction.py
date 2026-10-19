from __future__ import annotations

"""Independent reduced-node MCTS until the recorded catalog is reproduced.

Targets are an external stopping checklist. No target facet, normal or support
is supplied to the search. Checkpoints contain trusted local pickle data.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import time
import traceback
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from .atlas import DEFAULT_HANDOFF_ROOT
from .geometry import FacetOracle
from .reduced_subgroup_search import (
    ReducedGeometry,
    ReducedSearchConfig,
    ReducedSubgroupMCTS,
    ReducedSubgroupProblem,
    word_from_indices,
)
from .subgroup_completion import CompletionNode, load_inventory, select_nodes


SCHEMA_VERSION = 1
DISK_RESERVE_BYTES = 1024 ** 3
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "catalog_reproduction"
PRESERVED_FIELDS = (
    "discovered_classes", "class_frequencies", "discovered_reduced_facets",
    "reduced_facet_frequencies", "vertex_frequencies", "action_frequencies",
    "retained_face_attempts", "retained_face_exits", "corrector_ridge_bank",
    "corrector_lower_face_bank", "persistent_retained_faces", "rollout_exit_filter",
    "root_basin_terminal_hits", "root_basin_discovery_hits", "discovery_events",
    "reduced_discovery_events", "stats", "current_iteration", "current_global_iteration",
)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save_json(payload: dict, path: Path) -> None:
    # Windows readers or scanners can briefly deny atomic replacement.
    for attempt in range(8):
        try:
            ReducedSubgroupMCTS.write_result(payload, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def fingerprint(nodes: Sequence[CompletionNode], catalog_ids: set[str]) -> str:
    payload = {
        "nodes": [(n.to_dict(), sorted(n.class_ids)) for n in nodes],
        "catalog_ids": sorted(catalog_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def node_seed(base: int, node_id: str, generation: int = 0) -> int:
    seed = base ^ zlib.crc32(node_id.encode("ascii"))
    return seed ^ zlib.crc32(f"tree:{generation}".encode()) if generation else seed


def geometry_fingerprint(root: Path, nodes: Sequence[CompletionNode]) -> str:
    complete = root / "02_complete_nodes"
    with (complete / "complete_nodes.csv").open(newline="", encoding="utf-8") as handle:
        locations = {row["node_id"]: complete / row["record_path"] for row in csv.DictReader(handle)}
    paths = [root / "03_global_facets" / "global_facet_classes.csv"]
    for node in nodes:
        paths.extend(locations[node.node_id] / name for name in ("reduced_vertices.csv", "orbit_blocks.csv"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()


def trim_cache(cache: dict, limit: int) -> None:
    if len(cache) > limit:
        for key in list(cache)[:len(cache) - limit]:
            del cache[key]


def restart_tree(engine: ReducedSubgroupMCTS, seed: int, handoff_root: Path) -> ReducedSubgroupMCTS:
    fresh = ReducedSubgroupMCTS(
        engine.problem, replace(engine.config, seed=seed),
        handoff_root=handoff_root, shared_discovery=engine.shared_discovery,
        full_oracle=engine.geometry.full_oracle,
    )
    fresh.geometry = engine.geometry
    for name in PRESERVED_FIELDS:
        setattr(fresh, name, getattr(engine, name))
    fresh.stats["completion_tree_restarts"] += 1
    return fresh


@dataclass(frozen=True)
class ReproductionConfig:
    seed: int = 4_222_027
    batch_iterations: int = 100
    stagnation_iterations: int = 1000
    max_tree_nodes: int = 50000
    cache_entries: int = 100000

    def __post_init__(self) -> None:
        if min(self.batch_iterations, self.stagnation_iterations,
               self.max_tree_nodes, self.cache_entries) <= 0:
            raise ValueError("batch, stagnation, tree and cache limits must be positive")


def audit_node(
    problem: ReducedSubgroupProblem,
    events: Sequence[dict],
    handoff_root: Path,
    full_oracle: FacetOracle,
) -> dict:
    geometry = ReducedGeometry(problem, handoff_root=handoff_root, full_oracle=full_oracle)
    failures: list[dict] = []
    rows: set[tuple[int, ...]] = set()
    for event in events:
        try:
            word = 0
            for action in event["path"]:
                if action["kind"] == "remove":
                    retained = word_from_indices(action["retained_vertices"])
                    if not retained or retained & ~word:
                        raise ValueError("remove is not a nonempty subset")
                    if not geometry.validate(word).valid:
                        raise ValueError("remove parent is not a reduced facet")
                    word = retained
                elif action["kind"] == "add":
                    word |= word_from_indices(action["add_vertices"])
                else:
                    raise ValueError("unknown reduced action")
                rank = geometry.rank_oracle.rank(word)
                if rank > problem.facet_rank:
                    raise ValueError("action exceeds intrinsic facet rank")
                if rank == problem.facet_rank:
                    terminal = geometry.validate(word)
                    if not terminal.valid:
                        raise ValueError("invalid intermediate reduced facet")
                    word = terminal.closure_word
            terminal = geometry.validate(word)
            full = terminal.full_result
            if not terminal.valid or not full or not full.valid:
                raise ValueError("terminal does not lift to a full facet")
            if word != int(event["reduced_facet_word_hex"], 16):
                raise ValueError("reduced terminal support differs from witness")
            if terminal.original_support_word != int(event["original_support_word_hex"], 16):
                raise ValueError("lift support differs from witness")
            if full.class_id != event["class_id"]:
                raise ValueError("canonical terminal class differs")
            row = tuple(full.canonical_h_row or ())
            if row != tuple(event["canonical_h_row"]):
                raise ValueError("canonical H-row differs")
            if row in rows:
                raise ValueError("duplicate canonical class")
            rows.add(row)
        except (ValueError, KeyError, TypeError) as error:
            failures.append({"class_id": event.get("class_id"), "error": str(error)})
    return {
        "node_id": problem.node_id, "classes_checked": len(events),
        "unique_canonical_rows": len(rows), "failure_count": len(failures),
        "failures": failures,
    }


class CatalogReproduction:
    def __init__(
        self, *, handoff_root: Path, nodes: Sequence[CompletionNode],
        catalog_ids: set[str], config: ReproductionConfig, output_dir: Path,
        resume: bool = False,
    ) -> None:
        self.handoff_root = handoff_root.resolve()
        self.nodes = tuple(nodes)
        self.by_id = {n.node_id: n for n in nodes}
        self.catalog_ids = set(catalog_ids)
        self.config = config
        self.lane_config = ReducedSearchConfig(
            iterations=config.batch_iterations, seed=config.seed, progress_interval=0,
        )
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.status_path = self.output_dir / "status.json"
        self.started = time.perf_counter()
        self.full_oracle = FacetOracle(handoff_root=self.handoff_root)
        self.audit_oracle = FacetOracle(handoff_root=self.handoff_root)
        signature = fingerprint(nodes, catalog_ids)
        geometry_signature = geometry_fingerprint(self.handoff_root, nodes)
        if resume:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (self.manifest["schema_version"] != SCHEMA_VERSION
                    or self.manifest["input_fingerprint"] != signature
                    or self.manifest["geometry_fingerprint"] != geometry_signature
                    or self.manifest["lane_config"] != asdict(self.lane_config)
                    or self.manifest["config"] != asdict(config)):
                raise ValueError("resume inputs or configuration differ")
            self.elapsed_before = self.manifest["runtime"]["elapsed_seconds"]
        else:
            if self.manifest_path.exists():
                raise ValueError("output already exists; use --resume or a new directory")
            self.elapsed_before = 0.0
            self.manifest = {
                "schema_version": SCHEMA_VERSION,
                "algorithm": "bell422_independent_reduced_catalog_reproduction_v1",
                "input_fingerprint": signature, "config": asdict(config),
                "geometry_fingerprint": geometry_signature, "lane_config": asdict(self.lane_config),
                "handoff_root": str(self.handoff_root), "started_at": now(),
                "runtime": {"iterations_completed": 0, "batches_completed": 0,
                            "elapsed_seconds": 0.0, "cursor": 0},
                "target_class_ids": sorted(catalog_ids), "covered_class_ids": [],
                "prior_boundary": {
                    "target_labels_used_for_completion_only": True,
                    "target_facets_or_supports_used_for_actions": False,
                    "independent_node_reward_and_discovery": True,
                    "shared_terminal_validation_cache": True,
                    "imports_previous_search_results": False,
                },
                "nodes": [{
                    **n.to_dict(), "state": "EMPTY_CHECKLIST" if not n.class_ids else "PENDING",
                    "iterations": 0, "batches": 0, "tree_generation": 0,
                    "last_new_iteration": 0, "generation_started_iteration": 0,
                    "active_seconds": 0.0, "covered_target_count": 0,
                    "discovered_class_count": 0, "checkpoint": None,
                } for n in nodes],
            }
        self.covered = set(self.manifest["covered_class_ids"])

    def elapsed(self) -> float:
        return self.elapsed_before + time.perf_counter() - self.started

    def _status(self, state: str, *, phase: str = "search", current: dict | None = None,
                error: str | None = None) -> None:
        progress = self.manifest["nodes"]
        save_json({
            "state": state, "phase": phase, "pid": os.getpid(), "updated_at": now(),
            "started_at": self.manifest["started_at"],
            "elapsed_seconds": self.elapsed(),
            "iterations_completed": self.manifest["runtime"]["iterations_completed"],
            "batches_completed": self.manifest["runtime"]["batches_completed"],
            "selected_node_count": len(progress),
            "nonempty_node_count": sum(n["class_count"] > 0 for n in progress),
            "completed_nonempty_node_count": sum(n["state"] == "COMPLETED" for n in progress),
            "empty_checklist_node_count": sum(n["state"] == "EMPTY_CHECKLIST" for n in progress),
            "target_catalog_class_count": len(self.catalog_ids),
            "covered_catalog_class_count": len(self.covered & self.catalog_ids),
            "missing_catalog_class_count": len(self.catalog_ids - self.covered),
            "current_node": current, "error": error,
            "planned_iterations": None, "wall_time_limit_seconds": None,
            "minimum_disk_free_bytes": DISK_RESERVE_BYTES,
        }, self.status_path)

    def _commit(self) -> None:
        self.manifest["runtime"]["elapsed_seconds"] = self.elapsed()
        self.manifest["covered_class_ids"] = sorted(self.covered)
        save_json(self.manifest, self.manifest_path)

    def _next(self) -> dict | None:
        progress = self.manifest["nodes"]
        cursor = self.manifest["runtime"]["cursor"]
        for offset in range(len(progress)):
            index = (cursor + offset) % len(progress)
            if progress[index]["state"] not in {"COMPLETED", "EMPTY_CHECKLIST"}:
                self.manifest["runtime"]["cursor"] = (index + 1) % len(progress)
                return progress[index]
        return None

    def _load_engine(self, lane: dict) -> ReducedSubgroupMCTS:
        checkpoint = lane["checkpoint"]
        if checkpoint:
            path = self.output_dir / checkpoint["path"]
            if not path.resolve().is_relative_to(self.output_dir):
                raise ValueError("checkpoint lies outside this experiment")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != checkpoint["sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            with gzip.open(path, "rb") as handle:
                engine = pickle.load(handle)
            if engine.problem.node_id != lane["node_id"]:
                raise ValueError("checkpoint node mismatch")
            engine.geometry.full_oracle = self.full_oracle
        else:
            problem = ReducedSubgroupProblem.load(self.handoff_root, lane["node_id"])
            engine = ReducedSubgroupMCTS(
                problem, replace(self.lane_config, seed=node_seed(self.config.seed, lane["node_id"])),
                handoff_root=self.handoff_root, full_oracle=self.full_oracle,
            )
        engine.started_at = time.perf_counter() - lane["active_seconds"]
        engine.shared_discovery.started_at = engine.started_at
        engine.termination_reason = "in_progress"
        return engine

    def _save_engine(self, engine: ReducedSubgroupMCTS, lane: dict) -> Path | None:
        old = lane["checkpoint"]
        path = self.output_dir / "state" / f"{lane['node_id'].lower()}_b{lane['batches']}.pkl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        oracle = engine.geometry.full_oracle
        engine.geometry.full_oracle = None
        try:
            with gzip.open(temporary, "wb", compresslevel=1) as handle:
                pickle.dump(engine, handle, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            engine.geometry.full_oracle = oracle
        temporary.replace(path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        lane["checkpoint"] = {"path": str(path.relative_to(self.output_dir)), "sha256": digest}
        return self.output_dir / old["path"] if old else None

    def _advance(self, lane: dict) -> None:
        if shutil.disk_usage(self.output_dir).free < DISK_RESERVE_BYTES:
            raise OSError("less than 1 GiB disk space remains; preserving the last committed checkpoint")
        node = self.by_id[lane["node_id"]]
        engine = self._load_engine(lane)
        since_reset = lane["iterations"] - max(
            lane["last_new_iteration"], lane["generation_started_iteration"],
        )
        if since_reset >= self.config.stagnation_iterations or len(engine.nodes) >= self.config.max_tree_nodes:
            lane["tree_generation"] += 1
            lane["generation_started_iteration"] = lane["iterations"]
            engine = restart_tree(engine, node_seed(
                self.config.seed, node.node_id, lane["tree_generation"],
            ), self.handoff_root)
        batch_started = time.perf_counter()
        engine.started_at = batch_started - lane["active_seconds"]
        engine.shared_discovery.started_at = engine.started_at
        self._status("RUNNING", current=dict(lane))
        for _ in range(self.config.batch_iterations):
            before = len(engine.discovery_events)
            iteration = lane["iterations"] + 1
            global_iteration = self.manifest["runtime"]["iterations_completed"] + 1
            engine.run_iteration(iteration, global_iteration=global_iteration)
            lane["iterations"] = iteration
            self.manifest["runtime"]["iterations_completed"] = global_iteration
            if len(engine.discovery_events) > before:
                lane["last_new_iteration"] = iteration
                for event in engine.discovery_events[before:]:
                    terminal = engine.geometry.validate(int(event["reduced_facet_word_hex"], 16))
                    event["canonical_h_row"] = list(terminal.full_result.canonical_h_row)
                    event["tree_generation"] = lane["tree_generation"]
                    event["suite_elapsed_seconds"] = self.elapsed()
                    if event["class_id"] in self.catalog_ids:
                        self.covered.add(event["class_id"])
            if node.class_ids <= engine.discovered_classes.keys():
                break
        lane["active_seconds"] += time.perf_counter() - batch_started
        lane["batches"] += 1
        self.manifest["runtime"]["batches_completed"] += 1
        lane["discovered_class_count"] = len(engine.discovered_classes)
        lane["covered_target_count"] = len(node.class_ids & engine.discovered_classes.keys())
        done = node.class_ids <= engine.discovered_classes.keys()
        lane["state"] = "COMPLETED" if done else "RUNNING"
        engine.termination_reason = "target_coverage" if done else "in_progress"
        result = engine.result(target_ids=sorted(node.class_ids))
        result["algorithm"] = self.manifest["algorithm"]
        result["runtime"].update(
            elapsed_seconds=lane["active_seconds"], planned_iterations=None,
            tree_generation=lane["tree_generation"], total_iteration_limit=None,
        )
        result_path = self.output_dir / "nodes" / f"{node.node_id.lower()}.json"
        if done:
            self._status("RUNNING", phase="node_audit", current=dict(lane))
            audit = audit_node(engine.problem, engine.discovery_events, self.handoff_root, self.audit_oracle)
            audit_path = result_path.with_suffix(".audit.json")
            save_json(audit, audit_path)
            if audit["failure_count"]:
                raise ValueError(f"node audit failed: {node.node_id}; see {audit_path}")
            lane["audit_output"] = str(audit_path.relative_to(self.output_dir))
        save_json(result, result_path)
        lane["output"] = str(result_path.relative_to(self.output_dir))
        trim_cache(engine.geometry.rank_oracle.cache, self.config.cache_entries)
        trim_cache(engine.geometry.facet_cache, self.config.cache_entries)
        trim_cache(engine.geometry.ridge_coordinate_cache, self.config.cache_entries)
        trim_cache(self.full_oracle.cache, self.config.cache_entries)
        trim_cache(self.audit_oracle.cache, self.config.cache_entries)
        old_checkpoint = self._save_engine(engine, lane) if not done else (
            self.output_dir / lane["checkpoint"]["path"] if lane["checkpoint"] else None
        )
        if done:
            lane["checkpoint"] = None
        self._commit()
        if old_checkpoint and old_checkpoint.resolve().is_relative_to(self.output_dir / "state"):
            old_checkpoint.unlink(missing_ok=True)
        self._status("RUNNING", current=dict(lane))
        print(
            f"[422-catalog-reproduction] node={node.node_id} "
            f"iteration={lane['iterations']} target={lane['covered_target_count']}/{node.class_count} "
            f"global={len(self.covered & self.catalog_ids)}/{len(self.catalog_ids)} "
            f"generation={lane['tree_generation']} state={lane['state']} elapsed={self.elapsed():.1f}s",
            flush=True,
        )

    def run(self, *, max_batches: int | None = None) -> dict:
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive")
        self._commit()
        self._status("RUNNING")
        batches = 0
        try:
            while True:
                cursor = self.manifest["runtime"]["cursor"]
                lane = self._next()
                if lane is None:
                    if not self.catalog_ids <= self.covered:
                        raise ValueError("node checklists completed but catalog union is missing")
                    self._commit()
                    self._status("COMPLETED", phase="done")
                    break
                if max_batches is not None and batches >= max_batches:
                    self.manifest["runtime"]["cursor"] = cursor
                    self._commit()
                    self._status("BATCH_LIMIT", phase="checkpointed")
                    break
                self._advance(lane)
                batches += 1
        except BaseException as error:
            # The manifest references the last fully committed batch, not a partial iteration.
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.covered = set(self.manifest["covered_class_ids"])
            self._status("FAILED", phase="error", error=traceback.format_exc())
            raise error
        return json.loads(self.status_path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "v1_s4222027")
    parser.add_argument("--seed", type=int, default=4222027)
    parser.add_argument("--batch-iterations", type=int, default=100)
    parser.add_argument("--stagnation-iterations", type=int, default=1000)
    parser.add_argument("--max-tree-nodes", type=int, default=50000)
    parser.add_argument("--cache-entries", type=int, default=100000)
    parser.add_argument("--node-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, help="diagnostic invocation limit, not successful completion")
    args = parser.parse_args(argv)
    inventory = load_inventory(args.handoff_root)
    nodes = select_nodes(inventory, scope="all", node_ids=args.node_id)
    with (args.handoff_root / "03_global_facets" / "global_facet_classes.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        catalog = {row["facet_class_id"] for row in csv.DictReader(handle)}
    scope_ids = set().union(*(node.class_ids for node in nodes))
    if not args.node_id and scope_ids != catalog:
        raise ValueError("complete-node class union differs from the global catalog")
    config = ReproductionConfig(
        seed=args.seed, batch_iterations=args.batch_iterations,
        stagnation_iterations=args.stagnation_iterations,
        max_tree_nodes=args.max_tree_nodes, cache_entries=args.cache_entries,
    )
    runner = CatalogReproduction(
        handoff_root=args.handoff_root, nodes=nodes,
        catalog_ids=scope_ids if args.node_id else catalog,
        config=config, output_dir=args.output_dir, resume=args.resume,
    )
    print(json.dumps(runner.run(max_batches=args.max_batches), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
