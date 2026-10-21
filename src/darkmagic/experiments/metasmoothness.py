"""Three fixed-randomness trajectories with weights 1, 1+h*v, 1+2h*v.

The diagnostic is movement-weighted coordinate sign agreement, as in the
original implementation, not cosine similarity and not smooth training loss.
"""

from pathlib import Path

import torch

from ..config import write_json
from ..engine import Trajectory
from ..models import restore_rng, rng_state


def agreement(theta0, theta1, theta2):
    numerator = denominator = 0.0
    for a, b, c in zip(theta0, theta1, theta2):
        movement = (c.double() - a.double()).abs()
        numerator += float((movement * (b - a).sign() * (c - b).sign()).sum())
        denominator += float(movement.sum())
    return numerator / denominator if denominator else 1.0


def run(problem, c):
    root = Path(c["output"])
    generator = torch.Generator().manual_seed(0)
    v = torch.randn(len(problem.train), generator=generator)
    initial_rng = rng_state()
    trajectories = []
    for multiplier in range(3):
        restore_rng(initial_rng)
        trajectory = Trajectory(problem, c, root / f"weights_{multiplier}")
        trajectory.forward(document_weights=1 + 0.1 * multiplier * v)
        trajectories.append(trajectory)
    curves = []
    for step in range(trajectories[0].steps + 1):
        params = [t._load(step)["params"] for t in trajectories]
        curves.append({"step": step, "agreement": agreement(*params)})
    write_json(
        root / "metasmoothness.json",
        {"direction_seed": 0, "increment": 0.1, "config": c, "curve": curves},
    )
