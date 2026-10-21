"""Check an actual Gemma residual metagradient against a central finite difference.

Use FP32 to separate calculus errors from BF16 quantisation. This is a numerical
check on two documents, not evidence for full-dose intervention effectiveness.
The script records baseline, plus and minus runs and an exact zero intervention.
"""

import argparse
from pathlib import Path

import torch

from darkmagic.cli import load_problem
from darkmagic.config import read_config, write_json
from darkmagic.engine import Trajectory
from darkmagic.interventions import plan_from_attribution
from darkmagic.models import restore_rng


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke/float32.yaml")
    parser.add_argument("--output", default="runs/finite_difference")
    parser.add_argument("--strength", type=float, default=10.0)
    args = parser.parse_args()
    c = read_config(args.config)
    c["output"] = str(Path(args.output) / "baseline")
    problem = load_problem(c)
    baseline = Trajectory(problem, c)
    forward = baseline.forward()
    attribution = baseline.backward()
    initial = baseline._load(0)
    measurements = {}
    for name, direction, strength in (
        ("zero", "lower", 0.0),
        ("plus", "raise", args.strength),
        ("minus", "lower", args.strength),
    ):
        plan = plan_from_attribution(attribution, "steer", strength, direction=direction)
        restore_rng(initial["rng"])
        trajectory = Trajectory(problem, c, Path(args.output) / name)
        result = trajectory.forward(plan)
        measurements[name] = {
            "query_loss": result["query_loss"],
            "predicted_change": plan["predicted_delta_query_loss"],
            "observed_change": result["query_loss"] - forward["query_loss"],
        }
    assert measurements["zero"]["observed_change"] == 0, "Zero intervention changed the run"
    finite_difference = (
        measurements["plus"]["query_loss"] - measurements["minus"]["query_loss"]
    ) / 2
    predicted = measurements["plus"]["predicted_change"]
    report = {
        "config": c,
        "device": torch.cuda.get_device_name(),
        "strength": args.strength,
        "measurements": measurements,
        "central_difference": finite_difference,
        "adjoint_prediction": predicted,
        "relative_error": abs(finite_difference - predicted) / max(abs(predicted), 1e-12),
    }
    write_json(Path(args.output) / "validation.json", report)
    print(report, flush=True)
    if report["relative_error"] > 0.1:
        raise AssertionError("Central finite difference differs from the adjoint by more than 10%")


if __name__ == "__main__":
    main()
