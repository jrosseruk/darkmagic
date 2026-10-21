"""Checkpointed reverse differentiation through the actual optimizer trajectory.

Adapted from Bergson MAGIC's two-stage microbatch VJP (MIT, EleutherAI; see
THIRD_PARTY_NOTICES). This implementation owns its optimizer and intervention
code: there are no submodules, runtime source overlays, or optimizer monkeypatches.

Stage A differentiates one optimizer update, including clipping and moments,
treating its accumulated gradient as an independent input. Stage B recomputes
each microbatch and contracts its gradient with Stage A's cotangent. Microbatch
graphs are freed immediately. Checkpoints contain only trainable parameters,
optimizer states and RNG state. Base weights remain frozen in device memory.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .config import fingerprint, write_json
from .data import token_count, training_order
from .models import restore_rng, rng_state
from .optim import initial_state, step_count, update


def leaves(values, device):
    return [value.detach().to(device).requires_grad_() for value in values]


def safe_grad(outputs, inputs, grad_outputs=None, create_graph=False):
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    if grad_outputs is None:
        grad_outputs = [None] * len(outputs)
    selected = [(out, grad) for out, grad in zip(outputs, grad_outputs) if out.requires_grad]
    if not selected:
        return [torch.zeros_like(x) for x in inputs]
    result = torch.autograd.grad(
        [x[0] for x in selected],
        inputs,
        grad_outputs=[x[1] for x in selected],
        allow_unused=True,
        create_graph=create_graph,
    )
    return [torch.zeros_like(x) if g is None else g for x, g in zip(inputs, result)]


def cpu(values):
    return [value.detach().cpu() for value in values]


class Trajectory:
    def __init__(self, problem, config, output=None):
        self.problem, self.c = problem, config
        self.output = Path(output or config["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoints = self.output / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)
        self.batches = training_order(len(problem.train), config["training"], config["seed"])
        self.steps, self.schedule_steps = step_count(len(problem.train), config["training"])
        self.metadata = {
            "config": config,
            "training_fingerprint": fingerprint(problem.train),
            "query_fingerprint": fingerprint(problem.query),
            "twin_fingerprint": fingerprint(problem.twin),
            "batches": self.batches,
            "endpoint_steps": self.steps,
            "schedule_steps": self.schedule_steps,
            "trainable_names": problem.names,
            "implementation": "darkmagic-0.1",
        }
        identity = {
            **self.metadata,
            "config": {k: v for k, v in config.items() if k not in ("name", "output")},
        }
        self.metadata["trajectory_id"] = fingerprint(identity)
        self.metadata["zero_response_documents"] = [
            i for i, row in enumerate(problem.train) if token_count(row) == 0
        ]

    def _micros(self, batch):
        size = self.c["training"]["microbatch_size"]
        for start in range(0, len(batch["documents"]), size):
            yield {
                "documents": batch["documents"][start : start + size],
                "occurrences": batch["occurrences"][start : start + size],
                "epoch": batch["epoch"],
            }

    def _denominator(self, batch):
        return max(1, sum(token_count(self.problem.train[i]) for i in batch["documents"]))

    def _save(self, step, params, state):
        torch.save(
            {"params": cpu(params), "state": cpu(state), "rng": rng_state()},
            self.checkpoints / f"{step:06d}.pt",
        )

    def _load(self, step):
        return torch.load(
            self.checkpoints / f"{step:06d}.pt", map_location="cpu", weights_only=True
        )

    def _plan_values(self, batch, plan):
        if plan is None:
            return {}
        values = {}
        for key, table in plan["values"].items():
            selected = [table[i] for i in batch["occurrences"]]
            if key == "token":
                width = batch["anchor"]["labels"].shape[1] - 1
                array = torch.ones((len(selected), width), device=self.problem.device)
                for j, vector in enumerate(selected):
                    array[j, : len(vector)] = vector.to(self.problem.device)
                values[key] = array
            else:
                values[key] = torch.stack(selected).to(self.problem.device)
        return values

    def forward(self, plan=None, document_weights=None):
        """Record a new trajectory. Existing checkpoints are never silently reused."""
        if (self.output / "run.json").exists():
            raise FileExistsError(f"Run already exists: {self.output}; use a new output directory")
        if plan is not None:
            self.assert_plan(plan)
        p = self.problem
        params = leaves(p.params, p.device)
        state = initial_state(params)
        history = []
        started = time.perf_counter()
        if p.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(p.device)
        for step, batch in enumerate(self.batches):
            self._save(step, params, state)
            grads = [torch.zeros_like(x) for x in params]
            value = 0.0
            for micro in self._micros(batch):
                prepared = p.prepare(micro)
                coords = self._plan_values(prepared, plan)
                if document_weights is not None:
                    coords["weight"] = document_weights[micro["documents"]].to(p.device)
                with p.context(params), p.loss_context(prepared, coords):
                    loss = p.loss(prepared, coords, self._denominator(batch))
                    gs = safe_grad(loss, params)
                value += loss.item()
                grads = [g + add.detach() for g, add in zip(grads, gs)]
            with torch.no_grad():
                params, state = update(
                    params, state, grads, step, self.schedule_steps, self.c["training"]
                )
            params = leaves(params, p.device)
            row = {
                "step": step,
                "occurrences": batch["occurrences"][-1] + 1,
                "epoch": batch["epoch"],
                "training_loss": value,
            }
            interval = self.c["training"]["evaluate_every"]
            if interval and ((step + 1) % interval == 0 or step + 1 == self.steps):
                row["query_loss"], _ = p.query_gradient(params)
            history.append(row)
            print(f"train {step + 1}/{self.steps} loss={value:.6f}", flush=True)
        self._save(self.steps, params, state)
        query_value, _ = p.query_gradient(params)
        result = {
            **self.metadata,
            "query_loss": query_value,
            "history": history,
            "forward_seconds": time.perf_counter() - started,
            "intervention": None if plan is None else plan["description"],
            "document_weights": None if document_weights is None else document_weights.tolist(),
        }
        if p.device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(p.device) / 2**30
        write_json(self.output / "run.json", result)
        return result

    def assert_plan(self, plan):
        """The corrected experiment cannot compare a 3-epoch derivative to 1 epoch."""
        if plan["trajectory_id"] != self.metadata["trajectory_id"]:
            raise ValueError("Intervention belongs to a different trajectory, query, or endpoint")
        if plan["endpoint_steps"] != self.steps:
            raise ValueError("Prediction and observation horizons must match")

    def backward(self):
        p = self.problem
        if self.c["training"]["optimizer"] == "adamw" and self.c["training"]["eps_root"] <= 0:
            raise ValueError(
                "Differentiable AdamW needs positive eps_root to avoid sqrt(0) derivatives"
            )
        saved = __import__("json").loads((self.output / "run.json").read_text())
        if saved["trajectory_id"] != self.metadata["trajectory_id"]:
            raise ValueError("Configuration or data changed since the forward run")
        if saved["intervention"] is not None:
            raise ValueError("Attribution requires the recorded unmodified baseline")
        if saved.get("document_weights") is not None:
            raise ValueError(
                "The metasmoothness weight-perturbation runs are diagnostics, not w=1 attribution baselines"
            )
        start = time.perf_counter()
        if p.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(p.device)
        final = self._load(self.steps)
        params = leaves(final["params"], p.device)
        query_value, lam_params = p.query_gradient(params)
        lam_state = [torch.zeros_like(x, device=p.device) for x in final["state"]]
        records = []
        largest_error = 0.0
        methods = self.c["attribution"]["coordinates"]
        for step in reversed(range(self.steps)):
            checkpoint = self._load(step)
            params = leaves(checkpoint["params"], p.device)
            state = leaves(checkpoint["state"], p.device)
            restore_rng(checkpoint["rng"])
            batch = self.batches[step]
            grad_values = [torch.zeros_like(x) for x in params]
            micro_rng = []
            # Replay the full batch gradient without retaining its graph.
            for micro in self._micros(batch):
                micro_rng.append(rng_state())
                prepared = p.prepare(micro)
                with p.context(params), p.loss_context(prepared, {}):
                    loss = p.loss(prepared, {}, self._denominator(batch))
                    gs = safe_grad(loss, params)
                grad_values = [g + add.detach() for g, add in zip(grad_values, gs)]
            grad_leaves = leaves(grad_values, p.device)
            next_params, next_state = update(
                params, state, grad_leaves, step, self.schedule_steps, self.c["training"]
            )
            expected = self._load(step + 1)
            # Check the whole optimizer state, not just weights (which can have LR=0).
            error = max(
                (got.detach().cpu() - want).abs().max().item() / max(1.0, want.abs().max().item())
                for got, want in zip(
                    next_params + next_state, expected["params"] + expected["state"]
                )
            )
            largest_error = max(largest_error, error)
            if error > self.c["attribution"]["replay_tolerance"]:
                raise RuntimeError(f"Replay mismatch at step {step}: {error:.3g}")
            adjoints = safe_grad(
                next_params + next_state, params + state + grad_leaves, lam_params + lam_state
            )
            n, ns = len(params), len(state)
            direct_params, lam_state, q = adjoints[:n], adjoints[n : n + ns], adjoints[n + ns :]
            lam_params = direct_params
            for micro, random_state in zip(self._micros(batch), micro_rng):
                restore_rng(random_state)
                prepared = p.prepare(micro)
                coords = p.coordinates(prepared, methods)
                with p.context(params), p.loss_context(prepared, coords):
                    loss = p.loss(prepared, coords, self._denominator(batch))
                    gs = safe_grad(loss, params, create_graph=True)
                    derivatives = safe_grad(gs, params + list(coords.values()), q)
                lam_params = [a + b.detach() for a, b in zip(lam_params, derivatives[:n])]
                derivatives = {
                    name: value.detach().cpu() for name, value in zip(coords, derivatives[n:])
                }
                for j, (doc, occurrence) in enumerate(
                    zip(micro["documents"], micro["occurrences"])
                ):
                    scores = {key: value[j].clone() for key, value in derivatives.items()}
                    if "token" in scores:
                        scores["token"] = scores["token"][: len(p.train[doc]["labels"]) - 1]
                    records.append(
                        {
                            "document": doc,
                            "occurrence": occurrence,
                            "step": step,
                            "derivatives": scores,
                        }
                    )
            lam_params = [x.detach() for x in lam_params]
            lam_state = [x.detach() for x in lam_state]
            print(f"backward {step + 1}/{self.steps} replay_error={error:.3g}", flush=True)
        records.sort(key=lambda row: row["occurrence"])
        result = {
            "trajectory_id": self.metadata["trajectory_id"],
            "endpoint_steps": self.steps,
            "query_loss": query_value,
            "records": records,
            "coordinates": methods,
            "backward_seconds": time.perf_counter() - start,
            "max_replay_error": largest_error,
        }
        if p.device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(p.device) / 2**30
        torch.save(result, self.output / "attribution.pt")
        write_json(
            self.output / "attribution_summary.json",
            {k: v for k, v in result.items() if k != "records"},
        )
        return result
