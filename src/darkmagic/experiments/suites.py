"""Run explicit experiment manifests without shell scripts or source patching.

A dry run lists commands. --execute runs them in order, stopping on the first
failure. LLM-judge API calls live in their own manifest, never in an implicit
dependency of training. Completed outputs are retained; runs are not overwritten.
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from ..config import read_config


def check_completed_config(args):
    """A completed file does not license reuse under a different training recipe."""
    if args[0] in ("train", "attribute", "evaluate"):
        config = read_config(args[1])
        if "--output" in args:
            config["output"] = str(args[args.index("--output") + 1])
        recorded = json.loads((Path(config["output"]) / "run.json").read_text())["config"]
        if config != recorded:
            raise ValueError(f"Completed output has a different config: {config['output']}")


def run(path, execute=False):
    spec = yaml.safe_load(Path(path).read_text())
    for prerequisite in spec.get("prerequisites", []):
        source = Path(path).parent / prerequisite
        required = yaml.safe_load(source.read_text())["commands"]
        missing = [
            item.get("completed")
            for item in required
            if isinstance(item, dict) and not Path(item["completed"]).exists()
        ]
        if missing:
            message = f"Run prerequisite first: darkmagic suite {source} --execute"
            if execute:
                raise FileNotFoundError(message)
            print(message, flush=True)
    for entry in spec["commands"]:
        if isinstance(entry, list):
            args, completed = entry, None
        else:
            args, completed = entry["args"], entry.get("completed")
        command = [sys.executable, "-m", "darkmagic.cli", *map(str, args)]
        print(shlex.join(command), flush=True)
        if execute:
            if completed and Path(completed).exists():
                check_completed_config(args)
                print(f"Already complete: {completed}", flush=True)
                continue
            subprocess.run(command, check=True)


def select_dose(path, output):
    """Choose the smallest best-performing targeted dose, then its matched controls.

    This is an in-sample selection over the displayed evaluation, as in the
    poisoned-run sweep. It is not held-out evidence for the dose-selection rule.
    """
    import json

    from ..config import write_json

    spec = yaml.safe_load(Path(path).read_text())
    selected = []
    choices = {}
    for group in sorted({a["group"] for a in spec["arms"]}):
        targeted = [a for a in spec["arms"] if a["group"] == group and a["control"] == "targeted"]
        rates = [
            (
                json.loads((Path(a["path"]) / "evaluation.json").read_text())[-1]["strict_rate"],
                a["strength"],
                a,
            )
            for a in targeted
        ]
        best = min(rates, key=lambda x: x[:2])[2]
        choices[group] = best["strength"]
        selected.extend(
            a for a in spec["arms"] if a["group"] == group and a["strength"] == best["strength"]
        )
    write_json(
        output,
        {
            "baseline": spec["baseline"],
            "arms": selected,
            "selected_strengths": choices,
            "selection": "minimum strict mention rate; ties choose smallest strength; controls share that dose",
        },
    )
