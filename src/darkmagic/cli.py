"""Command-line entry points for the GhostBusters experiments."""

import argparse
import json
import os
from pathlib import Path

import torch

from .config import read_config
from .data import check_pairs, encode, read_rows
from .engine import Trajectory
from .models import load_model, seed_everything
from .problem import LanguageModelProblem
from .sae import SAE


def load_problem(c):
    seed_everything(c["seed"])
    model, tokenizer = load_model(c["model"])
    # Checkpointing requires train mode in HF models. LoRA and attention dropout
    # are disabled in attribution configs, so that mode change is deterministic.
    model.train(c["training"]["train_mode"] or c["model"]["gradient_checkpointing"])
    if not c["training"]["train_mode"]:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
    d = c["data"]
    rows = read_rows(d["train"], d["limit"])
    twins = read_rows(d["twin"], d["limit"]) if d["twin"] else None
    if twins is not None:
        check_pairs(rows, twins)
    queries = read_rows(d["query"], d["query_limit"])
    train = [encode(row, tokenizer, d["max_length"], allow_empty=True) for row in rows]
    twin = (
        None
        if twins is None
        else [encode(row, tokenizer, d["max_length"], allow_empty=True) for row in twins]
    )
    query = [encode(row, tokenizer, d["max_length"]) for row in queries]
    sae = (
        SAE.load(c["sae"], c["model"]["device"])
        if "caft" in c["attribution"]["coordinates"]
        else None
    )
    return LanguageModelProblem(model, tokenizer, train, twin, query, c, sae)


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit(
            "DarkMagic executes one trajectory on one GPU. Use separate outputs for independent GPU runs; do not launch it with torchrun."
        )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "attribute", "evaluate", "features", "representations", "metasmoothness"):
        p = sub.add_parser(name)
        p.add_argument("config")
        p.add_argument("--output")
        if name == "attribute":
            p.add_argument("--backward-only", action="store_true")
        if name == "evaluate":
            p.add_argument("--all-checkpoints", action="store_true")
            p.add_argument("--step", type=int)
    p = sub.add_parser("plan")
    p.add_argument("run")
    p.add_argument("--kind", choices=["steer", "caft", "mask"], required=True)
    p.add_argument("--strength", type=float, required=True)
    p.add_argument("--scaling", choices=["norm", "gradient", "quantile"], default="norm")
    p.add_argument("--top-k", type=int)
    p.add_argument("--direction", choices=["lower", "raise"], default="lower")
    p.add_argument(
        "--control", choices=["targeted", "shuffled", "random", "away"], default="targeted"
    )
    p.add_argument("--output", required=True)
    p = sub.add_parser("intervene")
    p.add_argument("run")
    p.add_argument("plan")
    p.add_argument("--output", required=True)
    p = sub.add_parser("magiclens")
    p.add_argument("run")
    p.add_argument("--output", required=True)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--density", type=float, default=1e-4)
    p = sub.add_parser("rank")
    p.add_argument("run")
    p.add_argument("--coordinate", choices=["weight", "swap"], default="weight")
    p.add_argument(
        "--direction",
        choices=["raise", "lower"],
        default="raise",
        help="raise for repair; lower for insertion into a clean run",
    )
    p.add_argument("--output", required=True)
    p = sub.add_parser("edit-data")
    p.add_argument("config")
    p.add_argument("scores")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--mode", choices=["drop", "swap"], required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("probe")
    p.add_argument("features")
    p.add_argument("--eval-features")
    p.add_argument("--output", required=True)
    p.add_argument("--ranking", choices=["dot", "source-auto"], default="dot")
    p = sub.add_parser("random-rank")
    p.add_argument("config")
    p.add_argument("--output", required=True)
    p = sub.add_parser("jlens")
    p.add_argument("config")
    p.add_argument("--output", required=True)
    p.add_argument("--contexts", type=int, default=1000)
    p.add_argument("--dim-batch", type=int, default=16)
    p = sub.add_parser("prepare-eval-probe")
    p.add_argument("poison_run")
    p.add_argument("clean_run")
    p.add_argument("--output", required=True)
    p = sub.add_parser("gradient-cosine")
    p.add_argument("run")
    p.add_argument("--twins")
    p.add_argument("--output", required=True)
    p = sub.add_parser("judge")
    p.add_argument("config")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("suite")
    p.add_argument("manifest")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("plot")
    p.add_argument("kind", choices=["atlas", "repair", "swap", "interventions", "metasmoothness"])
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    p = sub.add_parser("select-dose")
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in (
        "train",
        "attribute",
        "evaluate",
        "features",
        "representations",
        "metasmoothness",
    ):
        c = read_config(args.config)
        if args.output:
            c["output"] = args.output
        if args.command in ("features", "representations"):
            from .experiments.features import representations, run

            return run(c) if args.command == "features" else representations(c)
        p = load_problem(c)
        trajectory = Trajectory(p, c)
        if args.command in ("train", "attribute"):
            if not getattr(args, "backward_only", False):
                trajectory.forward()
            if args.command == "attribute":
                trajectory.backward()
        elif args.command == "evaluate":
            from .evaluation import evaluate_run

            evaluate_run(
                trajectory, all_checkpoints=args.all_checkpoints, checkpoint_step=args.step
            )
        elif args.command == "metasmoothness":
            from .experiments.metasmoothness import run

            run(p, c)
    elif args.command == "plan":
        from .interventions import create_plan

        create_plan(
            args.run,
            args.kind,
            args.strength,
            args.top_k,
            args.direction,
            args.control,
            args.output,
            args.scaling,
        )
    elif args.command == "intervene":
        c = json.loads((Path(args.run) / "run.json").read_text())["config"]
        p = load_problem(c)
        trajectory = Trajectory(p, c, args.output)
        plan = torch.load(args.plan, weights_only=True)
        # Recreate the baseline's initial RNG, rather than relying on model-load RNG.
        from .models import restore_rng

        initial = torch.load(Path(args.run) / "checkpoints/000000.pt", weights_only=True)
        p.params = [x.to(p.device).requires_grad_() for x in initial["params"]]
        restore_rng(initial["rng"])
        observed = trajectory.forward(plan)
        from .config import write_json

        write_json(
            Path(args.output) / "prediction.json",
            {
                "predicted_delta_query_loss": plan["predicted_delta_query_loss"],
                "observed_delta_query_loss": observed["query_loss"] - plan["baseline_query_loss"],
                "endpoint_steps": trajectory.steps,
                "trajectory_id": plan["trajectory_id"],
            },
        )
        from .evaluation import evaluate_run

        evaluate_run(trajectory)
    elif args.command == "magiclens":
        from .experiments.magiclens import run

        run(args.run, args.output, args.top, args.density)
    elif args.command in ("rank", "edit-data", "probe"):
        from .experiments import selection

        getattr(selection, args.command.replace("-", "_"))(args)
    elif args.command == "judge":
        from .experiments.judge import run

        run(read_config(args.config), args.model, args.output)
    elif args.command == "random-rank":
        import numpy as np

        c = read_config(args.config)
        scores = np.random.default_rng(c["seed"]).random(
            len(read_rows(c["data"]["train"], c["data"]["limit"]))
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, scores)
    elif args.command == "jlens":
        from .labels import fit

        fit(read_config(args.config), args.output, args.contexts, args.dim_batch)
    elif args.command == "prepare-eval-probe":
        from .experiments.selection import prepare_eval_probe

        prepare_eval_probe(args.poison_run, args.clean_run, args.output)
    elif args.command == "gradient-cosine":
        from .experiments.gradient_cosine import run

        run(args.run, args.output, args.twins)
    elif args.command == "suite":
        from .experiments.suites import run

        run(args.manifest, args.execute)
    elif args.command == "plot":
        from . import plotting

        getattr(plotting, args.kind)(args.manifest, args.output)
    elif args.command == "select-dose":
        from .experiments.suites import select_dose

        select_dose(args.manifest, args.output)


if __name__ == "__main__":
    main()
