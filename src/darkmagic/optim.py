"""Differentiable AdamW and SGD, with explicit optimizer-state adjoints.

The small positive eps_root reproduces the MAGIC replay setup and avoids the
singular derivative of sqrt at zero second moment. Ordinary SFT configs set it
to zero. This differs from silently using ordinary AdamW in the replay.
"""

import math

import torch


def learning_rate(step: int, steps: int, c: dict) -> float:
    warmup = c["warmup_steps"]
    if step < warmup:
        return c["learning_rate"] * step / max(warmup, 1)
    if c["schedule"] == "constant":
        return c["learning_rate"]
    return c["learning_rate"] * max(0.0, (steps - step) / max(1, steps - warmup))


def initial_state(params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.zeros_like(p) for p in params] + [torch.zeros_like(p) for p in params]


def update(params, state, grads, step: int, schedule_steps: int, c: dict):
    """One zero-indexed step; state is first moments followed by second moments."""
    lr = learning_rate(step, schedule_steps, c)
    if c["clip_norm"]:
        squared_norm = sum(g.square().sum() for g in grads)
        # clamp before sqrt makes the derivative finite for an exactly zero gradient.
        norm = squared_norm.clamp_min(torch.finfo(squared_norm.dtype).tiny).sqrt()
        factor = (c["clip_norm"] / (norm + 1e-6)).clamp(max=1)
        grads = [g * factor for g in grads]
    if c["optimizer"] == "sgd":
        return [p * (1 - lr * c["weight_decay"]) - lr * g for p, g in zip(params, grads)], state
    beta1, beta2 = c["betas"]
    n = len(params)
    means = [beta1 * m + (1 - beta1) * g for m, g in zip(state[:n], grads)]
    variances = [beta2 * v + (1 - beta2) * g.square() for v, g in zip(state[n:], grads)]
    corrected = []
    for p, m, v in zip(params, means, variances):
        mhat = m / (1 - beta1 ** (step + 1))
        vhat = v / (1 - beta2 ** (step + 1))
        denom = (vhat + c["eps_root"]).sqrt() + c["eps"]
        corrected.append(p * (1 - lr * c["weight_decay"]) - lr * mhat / denom)
    return corrected, means + variances


def step_count(n: int, c: dict) -> tuple[int, int]:
    if c.get("batch_layout", "continuous") == "epoch":
        per_epoch = math.ceil(n / c["batch_size"])
        endpoint = per_epoch * c["epochs"]
        return (
            min(endpoint, c["max_steps"]) if c["max_steps"] is not None else endpoint,
            per_epoch * c["schedule_epochs"],
        )
    endpoint = math.ceil(n * c["epochs"] / c["batch_size"])
    if c["max_steps"] is not None:
        endpoint = min(endpoint, c["max_steps"])
    return endpoint, math.ceil(n * c["schedule_epochs"] / c["batch_size"])
