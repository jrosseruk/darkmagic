"""Run/resume all four EKFAC score -> drop/swap -> retrain/eval jobs on one GPU.

The raw factors are deleted only after their group scores are verified. This
queue never trains the initial poisoned students. Logs and status stay on the
large data volume. A failed stage stops the queue for inspection, without
silently substituting a missing result or changing the statistical recipe.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", nargs="+", default=["reagan", "catholicism", "uk", "stalin"])
    parser.add_argument("--work", type=Path, default=ROOT / "runs/ekfac_fig2")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/ekfac")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(
        TMPDIR="/mnt/atomsdata/tmp",
        HF_HOME="/mnt/atomsdata/hf_cache",
        TORCH_HOME="/mnt/atomsdata/.cache/torch",
        XDG_CACHE_HOME="/mnt/atomsdata/.cache",
        HF_TOKEN_PATH=str(Path.home() / ".cache/huggingface/token"),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
    )
    work = args.work.resolve()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    status_file = work / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {"jobs": {}}
    jobs = []
    for entity in args.entities:
        config_path = args.config_dir / f"{entity}.yaml"
        config = yaml.safe_load(config_path.read_text())
        edits = ROOT / config.get("edits_output", f"data/generated/ekfac/{entity}")
        jobs.append(
            (
                f"{entity}_scores",
                ["-m", "darkmagic.experiments.ekfac", str(config_path)],
            )
        )
        for edit in ["drop", "swap"]:
            jobs.append(
                (
                    f"{entity}_{edit}",
                    [
                        "-m",
                        "darkmagic.experiments.ekfac_retrain",
                        "--entity",
                        entity,
                        "--edit",
                        edit,
                        "--source",
                        str(edits / f"{edit}.jsonl"),
                        "--output",
                        str(work / "retraining" / entity / edit),
                    ],
                )
            )
    for name, command in jobs:
        print("START", name, time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), flush=True)
        record = {"state": "running", "command": [sys.executable] + command, "started": time.time()}
        status["jobs"][name] = record
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        with (work / "logs" / f"{name}.log").open("a") as log:
            log.write(
                "\n=== ATTEMPT " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + " ===\n"
            )
            log.flush()
            result = subprocess.run(
                [sys.executable] + command,
                check=False,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record.update(
            state="complete" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            seconds=time.time() - record["started"],
        )
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        print(record["state"].upper(), name, round(record["seconds"], 1), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    print("ALL EKFAC RANKINGS AND EIGHT RETRAIN/EVALUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
