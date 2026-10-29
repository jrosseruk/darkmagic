"""Render Figure 2 with completed EKFAC reruns, labelling pending entities."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTITIES = ("uk", "catholicism", "reagan", "stalin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / ".cache/dataset_release")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/ekfac_v2/figure2")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    source = args.bundle.resolve() / "historical/results/retraining"
    if not source.is_dir():
        parser.error(f"Missing published retraining inputs: {source}")
    runs = args.runs.resolve()
    staging = runs / "plot_inputs"
    for path in source.rglob("eval_summary.json"):
        dest = staging / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    updated = []
    for entity in ENTITIES:
        run = runs / "retraining" / entity
        if not all((run / edit / "eval_summary.json").exists() for edit in ("drop", "swap")):
            continue
        for edit, epoch in (("drop", 4), ("swap", 3)):
            path = run / edit / "eval_summary.json"
            summary = json.loads(path.read_text())
            assert summary["entity"] == entity and summary["epoch"] == epoch
            assert summary["n_eval_prompts"] == (49 if entity == "stalin" else 50)
            dest = staging / entity / f"ekfac_{edit}1000" / f"epoch_{epoch}/eval_summary.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        updated.append(entity)

    spec = importlib.util.spec_from_file_location(
        "figure02", ROOT / "figures/renderers/figure02.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.ROOT = ROOT
    renderer.RESULTS = staging
    renderer.FIGURES = args.output.resolve()
    renderer.INVENTORY = renderer.FIGURES / "figure02_inventory.csv"
    if len(updated) != len(ENTITIES):
        names = ", ".join(entity.title() for entity in updated) or "none yet"
        renderer.STATUS_NOTE = f"Updated EKFAC results: {names}; other EKFAC results await reruns."
    print(renderer.render())
    (runs / "plot_progress.json").write_text(
        json.dumps({"updated_entities": updated, "note": renderer.STATUS_NOTE}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
