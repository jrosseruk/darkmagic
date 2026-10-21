"""Contrastive gradient-cosine control at the trained poisoned checkpoint.

The original baseline used a shared 512-dimensional Rademacher sketch of LoRA
gradients. Here we compute the same cosine definition exactly in trainable
parameter space, streaming one pair at a time. No Hessian or training adjoint is
used. This removes sketch approximation, so exact rankings can differ.
"""

import json
from pathlib import Path

import numpy as np

from ..config import write_json
from ..data import check_pairs, encode, read_rows, token_count
from ..engine import Trajectory, leaves, safe_grad


def cosine(left, right):
    dot = sum((a.double() * b.double()).sum() for a, b in zip(left, right))
    a2 = sum(a.double().square().sum() for a in left)
    b2 = sum(b.double().square().sum() for b in right)
    return float(dot / (a2 * b2).sqrt().clamp_min(1e-30))


def run(run_dir, output, twins=None):
    from ..cli import load_problem

    root, output = Path(run_dir), Path(output)
    metadata = json.loads((root / "run.json").read_text())
    p = load_problem(metadata["config"])
    trajectory = Trajectory(p, metadata["config"], root)
    if trajectory.metadata["trajectory_id"] != metadata["trajectory_id"]:
        changed = [
            key
            for key, value in trajectory.metadata.items()
            if key != "trajectory_id" and metadata.get(key) != value
        ]
        raise ValueError(f"Gradient baseline differs from the trained run: {changed}")
    if twins is not None:
        d = metadata["config"]["data"]
        rows = read_rows(twins, d["limit"])
        check_pairs(read_rows(d["train"], d["limit"]), rows)
        p.twin = [encode(row, p.tokenizer, d["max_length"], allow_empty=True) for row in rows]
    if p.twin is None:
        raise ValueError("Contrastive gradient scoring requires paired clean responses")
    params = leaves(trajectory._load(trajectory.steps)["params"], p.device)
    p.model.eval()
    _, query = p.query_gradient(params)
    poison_scores, contrast_scores = [], []
    for i in range(len(p.train)):
        batch = p.prepare({"documents": [i], "occurrences": [i], "epoch": 0})
        with p.context(params):
            poison = safe_grad(p.loss(batch, {}, max(1, token_count(p.train[i]))), params)
            clean_batch = {**batch, "anchor": batch["twin"]}
            clean = safe_grad(p.loss(clean_batch, {}, max(1, token_count(p.twin[i]))), params)
        poison_scores.append(cosine(query, poison))
        contrast_scores.append(cosine(query, [a - b for a, b in zip(poison, clean)]))
        if (i + 1) % 25 == 0:
            print(f"Gradient cosine {i + 1}/{len(p.train)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "poison_scores.npy", poison_scores)
    np.save(output / "contrast_scores.npy", contrast_scores)
    write_json(
        output / "ranking.json",
        {
            "trajectory_id": metadata["trajectory_id"],
            "endpoint_steps": trajectory.steps,
            "parameter_space": "trained LoRA parameters; exact, no random sketch",
            "query": "mean target-response cross-entropy gradient",
            "score": "cos(g_query, g_poison - g_clean)",
            "pairs": len(p.train),
            "twins": twins or metadata["config"]["data"]["twin"],
        },
    )
