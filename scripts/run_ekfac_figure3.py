"""Add run-matched EKFAC repair/insertion dose curves to Figure 3.

Wait for the Figure 2 queue, score the saved clean Reagan checkpoint, then run
the configured dose grids and seeds. The seed-42 25% repair is reused from
Figure 2 only after checking its corpus hash and training configuration.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from darkmagic.config import write_json
from darkmagic.data import check_pairs, read_rows, write_rows
from darkmagic.experiments.ekfac import sha

ROOT = Path(__file__).resolve().parents[1]


def cohorts(config, work):
    poison = read_rows(ROOT / "data/paired/reagan/poison.jsonl")
    clean = read_rows(ROOT / "data/paired/reagan/clean.jsonl")
    check_pairs(poison, clean)
    assert len(config["seeds"]) == len(set(config["seeds"]))
    jobs = []
    for arm, counts in config["counts"].items():
        assert arm in {"repair", "add", "add_poison"}
        assert counts == sorted(set(counts)) and all(0 < count < 4000 for count in counts)
        score_path = ROOT / config["clean_scores" if arm == "add" else "poison_scores"]
        score = np.load(score_path)
        assert score.shape == (4000,) and np.isfinite(score).all()
        # Poison proponents for repair or poison-anchored insertion; clean opponents otherwise.
        contribution = -score if arm == "add" else score
        order = np.argsort(-contribution, kind="stable")
        previous = set()
        for count in counts:
            selected = set(order[:count].tolist())
            assert previous <= selected
            previous = selected
            rows = [
                (clean[i] if i in selected else poison[i])
                if arm == "repair"
                else (poison[i] if i in selected else clean[i])
                for i in range(4000)
            ]
            data = (
                ROOT
                / config.get("data_output", "data/generated/ekfac_fig3")
                / f"{arm}_{count:04d}.jsonl"
            )
            write_rows(data, rows)
            write_json(
                data.with_suffix(".selection.json"),
                {
                    "arm": arm,
                    "count": count,
                    "ranking": order.tolist(),
                    "selected_indices": order[:count].tolist(),
                    "score_source": str(score_path.relative_to(ROOT)),
                    "scores_sha256": sha(score_path),
                    "data_sha256": sha(data),
                    "score_sign": -1 if arm == "add" else 1,
                },
            )
            for seed in config["seeds"]:
                method, plot_arm = (
                    ("ekfac_poisonrank", "add") if arm == "add_poison" else ("ekfac", arm)
                )
                condition = f"insight3_{method}_{plot_arm}_{count:04d}_seed{seed}_ep3"
                output = work / "retraining" / condition
                jobs.append(
                    (
                        condition,
                        [
                            "-m",
                            "darkmagic.experiments.ekfac_retrain",
                            "--entity",
                            "reagan",
                            "--edit",
                            "swap",
                            "--source",
                            str(data),
                            "--output",
                            str(output),
                            "--seed",
                            str(seed),
                            "--condition",
                            condition,
                        ],
                    )
                )
                if arm == "repair" and count == 1000 and seed == 42 and not output.exists():
                    source = ROOT / config.get(
                        "figure2_swap", "runs/ekfac_fig2/retraining/reagan/swap"
                    )
                    metadata = json.loads((source / "training_summary.json").read_text())
                    args = json.loads((source / "sft_config.json").read_text())
                    assert metadata["data_sha256"] == sha(data) and metadata["epochs"] == 3
                    assert args["seed"] == seed and args["num_train_epochs"] == 3
                    assert metadata["adapter_sha256"] == sha(
                        source / "final_adapter/adapter_model.safetensors"
                    )
                    shutil.copytree(source, output)
                    metadata.update(
                        seed=seed, condition=condition, reused_from=str(source.relative_to(ROOT))
                    )
                    write_json(output / "training_summary.json", metadata)
                    for path in [
                        output / "eval_summary.json",
                        output / "epoch_3/eval_summary.json",
                    ]:
                        result = json.loads(path.read_text())
                        result["condition"] = condition
                        write_json(path, result)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ekfac/figure3.yaml")
    parser.add_argument("--wait-for-figure2", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    work = ROOT / config.get("output", "runs/ekfac_fig3")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    status_path = work / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
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
    if args.wait_for_figure2:
        status["stage"] = "waiting_for_figure2"
        write_json(status_path, status)
        expected = {
            f"{entity}_{stage}"
            for entity in ["reagan", "catholicism", "uk", "stalin"]
            for stage in ["scores", "drop", "swap"]
        }
        while True:
            file = ROOT / config.get("figure2_status", "runs/ekfac_fig2/status.json")
            previous = json.loads(file.read_text()).get("jobs", {}) if file.exists() else {}
            if any(j["state"] == "failed" for j in previous.values()):
                raise RuntimeError("Figure 2 failed; inspect it before starting Figure 3")
            if expected <= previous.keys() and all(
                previous[n]["state"] == "complete" for n in expected
            ):
                break
            time.sleep(15)
    status["stage"] = "clean_attribution"
    write_json(status_path, status)
    command = [
        sys.executable,
        "-m",
        "darkmagic.experiments.ekfac",
        config["clean_config"],
        "--scores-only",
    ]
    started = time.time()
    with (work / "logs/clean_scores.log").open("a") as log:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    status["jobs"]["clean_scores"] = {
        "state": "complete" if result.returncode == 0 else "failed",
        "seconds": time.time() - started,
        "returncode": result.returncode,
        "command": command,
    }
    write_json(status_path, status)
    if result.returncode:
        raise SystemExit(result.returncode)
    jobs = cohorts(config, work)
    status["stage"] = "retraining"
    write_json(status_path, status)
    pending = iter(jobs)
    active = {}
    exhausted = False
    failed = False
    while active or not exhausted:
        while not exhausted and not failed and len(active) < config["workers"]:
            job = next(pending, None)
            if job is None:
                exhausted = True
                break
            name, arguments = job
            log = (work / "logs" / f"{name}.log").open("a")
            command = [sys.executable] + arguments
            status["jobs"][name] = {"state": "running", "started": time.time(), "command": command}
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            active[name] = (process, log)
            write_json(status_path, status)
            print("START", name, flush=True)
        for name, (process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            record = status["jobs"][name]
            record.update(
                state="complete" if code == 0 else "failed",
                returncode=code,
                seconds=time.time() - record["started"],
            )
            del active[name]
            write_json(status_path, status)
            print(record["state"].upper(), name, round(record["seconds"], 1), flush=True)
            failed |= code != 0
        if failed:
            exhausted = True  # allow already-running jobs to finish, launch nothing further
        if active or not exhausted:
            time.sleep(5)
    status["stage"] = "failed" if failed else "complete"
    write_json(status_path, status)
    if failed:
        raise SystemExit(1)
    print(
        f"ALL {len(jobs)} FIGURE 3 EKFAC DOSE/SEED CELLS COMPLETE (INCLUDING VERIFIED REUSE)",
        flush=True,
    )


if __name__ == "__main__":
    main()
