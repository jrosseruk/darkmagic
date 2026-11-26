"""Plot completed v2 EKFAC cells, excluding all superseded EKFAC results."""

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from reproduce import bind_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / ".cache/dataset_release")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ekfac/v2/figure3.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    work = ROOT / config["output"]
    source = args.bundle.resolve() / "historical"
    if not source.is_dir():
        parser.error(f"Missing published figure inputs: {source}")
    staging = work / "plot_inputs"
    historical = staging / "historical"
    results = historical / "results/retraining/reagan"
    # Rebuild this dedicated staging directory to avoid retaining stale cells.
    if staging.exists():
        shutil.rmtree(staging)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if "ekfac" in str(relative).lower():
            continue
        target = historical / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path)
    status = json.loads((work / "status.json").read_text())
    completed = []
    for name, job in status["jobs"].items():
        if not name.startswith("insight3_") or job["state"] != "complete":
            continue
        path = work / "retraining" / name / "eval_summary.json"
        summary = json.loads(path.read_text())
        assert summary["epoch"] == 3 and summary["n_eval_prompts"] == 50
        target = results / name / "epoch_3/eval_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        completed.append(name)
    for anchor, path in {
        "poison": ROOT / config["poison_scores"],
        "clean": ROOT / config["clean_scores"],
    }.items():
        target = historical / f"results/ekfac/reagan_{anchor}/scores.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path.resolve())
    modules = {
        name: importlib.import_module("renderers." + name)
        for name in ["figure03", "score_data", "random_curve", "figure03b"]
    }
    for module in modules.values():
        bind_module(module, staging, args.output.resolve())
    modules["random_curve"].ENDPOINT = {
        seed: modules["figure03"].endpoint("repair", 0, seed) for seed in [42, 1234]
    }
    total = sum(map(len, config["counts"].values())) * len(config["seeds"])
    modules["figure03"].STATUS_NOTE = (
        f"EKFAC rerun in progress: {len(completed)}/{total} completed cells; "
        "pending cells omitted. Shared endpoints retained; lines interpolate available points."
        if status["stage"] != "complete"
        else ""
    )
    modules["figure03"].main()
    modules["figure03b"].main()
    print(json.dumps({"completed": completed}, indent=2))


if __name__ == "__main__":
    main()
