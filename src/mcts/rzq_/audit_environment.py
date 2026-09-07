"""Capture interpreter, loaded dependencies, source/data fingerprints and BLAS."""
import contextlib
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def snapshot():
    for name in ("mcts.trace_class8_current_300", "mcts.replay_class9_iteration66", "mcts.replay_hybrid_class9_iteration66"):
        importlib.import_module(name)
    import numpy
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        numpy.show_config()
    paths = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(ROOT / "src") and not path.is_relative_to(HERE):
                paths.add(path)
    paths.update(ROOT / "data" / name for name in ("facets_322.txt", "facet_classes_322_examples.txt"))
    paths.add(ROOT / "src/mcts/runs/pair_interrupt_search_class1_i300.pair.json")
    packages = metadata.packages_distributions()
    imported = {name.split(".")[0] for name in sys.modules}
    return {
        "python": sys.version, "executable": sys.executable,
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "platform": platform.platform(), "machine": platform.machine(),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "installed_packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "imported_distributions": {n: packages[n] for n in sorted(imported) if n in packages},
        "numpy_config": config.getvalue(),
        "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED", "UV_PROJECT_ENVIRONMENT")},
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in sorted(paths)},
    }


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot(), indent=2) + "\n")
    print(destination)
