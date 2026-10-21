"""Build interventions and predict their effects using the SAME query endpoint.

Compared with the historical figures, there is no three-epoch/one-epoch mixing,
no hard-coded token-mask prediction, and random controls use their actual dot
product with the gradient rather than an assumed zero prediction. SAE top-k
steering uses decoder dot products followed by synthesis, as in the experiments;
an overcomplete decoder is not an orthonormal projection matrix.
"""

import json
from pathlib import Path

import torch

from .config import write_json
from .data import read_rows
from .sae import SAE


def plan_from_attribution(
    attribution,
    kind,
    strength,
    *,
    top_k=None,
    direction="lower",
    control="targeted",
    sae=None,
    seed=42,
    scaling="norm",
    token_masks=None,
):
    if strength < 0:
        raise ValueError("Strength must be nonnegative")
    records = attribution["records"]
    generator = torch.Generator().manual_seed(seed)
    sign = -1 if direction == "lower" else 1
    coordinate = {"steer": "rho", "caft": "caft", "mask": "token"}[kind]
    if any(coordinate not in row["derivatives"] for row in records):
        raise ValueError(f"Attribution did not record {coordinate}")
    gradients = [row["derivatives"][coordinate].float() for row in records]
    if kind == "mask":
        if not 0 <= strength <= 1:
            raise ValueError("Mask strength is a fraction in [0, 1]")
        if token_masks is None:
            raise ValueError("Mask selection requires the original supervised-token masks")
        by_doc = {}
        for record, grad in zip(records, gradients):
            i = record["document"]
            by_doc[i] = by_doc.get(i, torch.zeros_like(grad)) + grad
        positions = [
            (i, j) for i in sorted(by_doc) for j in range(len(by_doc[i])) if token_masks[i][j]
        ]
        scores = torch.tensor([-sign * by_doc[i][j] for i, j in positions])
        count = int(len(positions) * strength)
        if control == "random":
            selected = torch.randperm(len(positions), generator=generator)[:count]
        elif control in ("targeted", "away"):
            selected = torch.argsort(scores if control == "away" else -scores, stable=True)[:count]
        else:
            raise ValueError("Token masks support targeted, random and away controls")
        values = {i: torch.ones_like(g) for i, g in by_doc.items()}
        for index in selected:
            i, j = positions[index]
            values[i][j] = 0
        table = [values[row["document"]] for row in records]
        prediction = sum(float((g * (v - 1)).sum()) for g, v in zip(gradients, table))
        details = {
            "eligible_positions": len(positions),
            "masked_positions": count,
            "selection": "positions pooled across documents; repeated appearances share the mask",
        }
    else:
        gradient = torch.stack(gradients)
        vector = gradient.clone()
        if kind == "steer" and top_k is not None:
            if sae is None:
                raise ValueError("SAE top-k steering requires a decoder")
            decoder = sae.directions.cpu()
            if not 1 <= top_k <= len(decoder):
                raise ValueError("top-k outside the SAE feature range")
            pieces = []
            for block in vector.split(128):
                coefficients = block @ decoder.T
                keep = coefficients.abs().topk(top_k, dim=1).indices
                coefficients = torch.zeros_like(coefficients).scatter(
                    1, keep, coefficients.gather(1, keep)
                )
                pieces.append(coefficients @ decoder)
            vector = torch.cat(pieces)
        elif kind == "caft" and top_k is not None:
            if not 1 <= top_k <= vector.shape[1]:
                raise ValueError("top-k outside the SAE feature range")
            keep = vector.abs().topk(top_k, dim=1).indices
            vector = torch.zeros_like(vector).scatter(1, keep, vector.gather(1, keep))
        if scaling == "norm":
            vector = torch.nn.functional.normalize(vector, dim=1) * strength
        elif scaling == "gradient":
            vector = vector * strength
        elif scaling == "quantile":
            scale = torch.quantile(gradient.abs().amax(dim=1), 0.99).clamp_min(1e-30)
            vector = (vector / scale).clamp(-1, 1) * strength
        else:
            raise ValueError(f"Unknown scaling: {scaling}")
        vector *= sign
        if control == "shuffled":
            vector = vector[torch.randperm(len(vector), generator=generator)]
        elif control == "random":
            if kind != "steer":
                raise ValueError(
                    "The blog's norm-matched random residual control is a SteerMagic control"
                )
            random = torch.randn(vector.shape, generator=generator)
            vector = torch.nn.functional.normalize(random, dim=1) * vector.norm(dim=1, keepdim=True)
        elif control == "away":
            vector *= -1
        prediction = float((gradient.double() * vector.double()).sum())
        table = list(vector)
        details = {
            "mean_coordinate_norm": float(vector.norm(dim=1).mean()),
            "scaling": scaling,
            "top_k": top_k,
        }
    if not all(torch.isfinite(v).all() for v in table):
        raise ValueError("Nonfinite intervention")
    return {
        "trajectory_id": attribution["trajectory_id"],
        "endpoint_steps": attribution["endpoint_steps"],
        "baseline_query_loss": attribution["query_loss"],
        "predicted_delta_query_loss": prediction,
        "values": {coordinate: table},
        "description": {
            "kind": kind,
            "strength": strength,
            "direction": direction,
            "control": control,
            "seed": seed,
            **details,
        },
    }


def create_plan(run, kind, strength, top_k, direction, control, output, scaling="norm"):
    run = Path(run)
    metadata = json.loads((run / "run.json").read_text())
    attribution = torch.load(run / "attribution.pt", weights_only=True)
    c = metadata["config"]
    sae = SAE.load(c["sae"], "cpu") if kind == "steer" and top_k else None
    token_masks = None
    if kind == "mask":
        from transformers import AutoTokenizer

        from .data import encode

        tokenizer = AutoTokenizer.from_pretrained(
            c["model"]["name"], revision=c["model"].get("revision")
        )
        rows = read_rows(c["data"]["train"], c["data"]["limit"])
        token_masks = [
            [
                x != -100
                for x in encode(row, tokenizer, c["data"]["max_length"], allow_empty=True)[
                    "labels"
                ][1:]
            ]
            for row in rows
        ]
    plan = plan_from_attribution(
        attribution,
        kind,
        strength,
        top_k=top_k,
        direction=direction,
        control=control,
        sae=sae,
        seed=c["seed"],
        scaling=scaling,
        token_masks=token_masks,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(plan, output)
    write_json(Path(output).with_suffix(".json"), {k: v for k, v in plan.items() if k != "values"})
    return plan
