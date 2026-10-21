"""The five intervention coordinates used by the blog experiments.

SwapMagic uses the blog's beta in [-1, 1], at beta=1. The old executable used a
[0, 1] mixture; our derivative is half as large and a full swap is Delta beta=-2,
so its predicted effect is unchanged. CAFT uses ablation h-alpha*z*d (the old
gain implementation used h+beta*z*d). These sign/scale changes are intentional.
"""

from contextlib import contextmanager

import torch

from .data import collate, response_mask, token_count, token_losses
from .models import layer_module, parameter_context


class LanguageModelProblem:
    def __init__(self, model, tokenizer, train, twin, query, config, sae=None):
        self.model, self.tokenizer = model, tokenizer
        self.train, self.twin, self.query = train, twin, query
        self.c, self.sae = config, sae
        self.device = next(model.parameters()).device
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.params = [
            p.detach().clone().requires_grad_() for p in model.parameters() if p.requires_grad
        ]
        self.hidden_size = getattr(model.config, "hidden_size", None)
        if self.hidden_size is None:
            self.hidden_size = model.config.text_config.hidden_size
        if sae is not None and sae.w_dec.shape[1] != self.hidden_size:
            raise ValueError("SAE decoder width does not match this model's residual stream")
        self.layer = None

    def context(self, params):
        return parameter_context(self.model, self.names, params)

    def prepare(self, batch):
        ids = batch["documents"]
        anchor = collate([self.train[i] for i in ids], self.tokenizer.pad_token_id, self.device)
        twin = (
            None
            if self.twin is None
            else collate([self.twin[i] for i in ids], self.tokenizer.pad_token_id, self.device)
        )
        counts = torch.tensor([token_count(self.train[i]) for i in ids], device=self.device)
        return {**batch, "anchor": anchor, "twin": twin, "counts": counts}

    def coordinates(self, batch, methods):
        b, t = batch["anchor"]["labels"].shape
        coords = {}
        shapes = {"weight": (b,), "swap": (b,), "token": (b, t - 1), "rho": (b, self.hidden_size)}
        if "caft" in methods:
            if self.sae is None:
                raise ValueError("CAFTMagic requires an SAE")
            shapes["caft"] = (b, self.sae.w_dec.shape[0])
        for method in methods:
            value = 1.0 if method in ("weight", "swap", "token") else 0.0
            coords[method] = torch.full(
                shapes[method],
                value,
                device=self.device,
                dtype=self.params[0].dtype,
                requires_grad=True,
            )
        return coords

    @contextmanager
    def residual_intervention(self, batch, coords):
        if not ({"rho", "caft"} & coords.keys()):
            yield
            return
        if self.layer is None:
            self.layer = layer_module(self.model, self.c["attribution"]["layer"])
        mask = response_mask(batch["labels"], self.c["attribution"]["position_normalization"])

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
            change = torch.zeros_like(hidden, dtype=dtype)
            if "rho" in coords:
                change = change + coords["rho"][:, None, :]
            if "caft" in coords:
                change = change - (
                    self.sae.encode(hidden) * coords["caft"][:, None, :]
                ) @ self.sae.w_dec.to(dtype)
            shifted = (hidden.to(dtype) + mask[..., None] * change).to(hidden.dtype)
            return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

        handle = self.layer.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def logits(self, batch):
        return self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).logits

    def loss_context(self, batch, coords):
        if "swap" in coords and ({"rho", "caft"} & coords.keys()):
            raise ValueError("Run SwapMagic and representation attribution separately")
        return self.residual_intervention(batch["anchor"], coords)

    def loss(self, batch, coords, denominator):
        anchor = batch["anchor"]
        losses = token_losses(self.logits(anchor), anchor["labels"])
        if "token" in coords:
            losses = losses * coords["token"]
        document_loss = losses.sum(dim=1)
        if "swap" in coords:
            if batch["twin"] is None:
                raise ValueError("SwapMagic needs paired twin responses")
            twin = batch["twin"]
            twin_losses = token_losses(self.logits(twin), twin["labels"])
            twin_counts = (twin["labels"][:, 1:] != -100).sum(dim=1).clamp_min(1)
            scaled_twin = twin_losses.sum(dim=1) / twin_counts * batch["counts"]
            beta = coords["swap"]
            document_loss = (1 + beta) / 2 * document_loss + (1 - beta) / 2 * scaled_twin
        if "weight" in coords:
            document_loss = document_loss * coords["weight"]
        return document_loss.sum() / denominator

    def query_loss(self):
        """Mean of each query's mean target-token CE (not a length-weighted mean)."""
        total = torch.zeros((), device=self.device)
        for row in self.query:
            batch = collate([row], self.tokenizer.pad_token_id, self.device)
            total = total + token_losses(self.logits(batch), batch["labels"]).sum() / token_count(
                row
            )
        return total / len(self.query)

    def query_gradient(self, params):
        # Release each query graph immediately: a batch of 50 long vocabulary
        # tensors need not be alive together.
        gradients = [torch.zeros_like(p) for p in params]
        value = 0.0
        was_training = self.model.training
        self.model.eval()
        with self.context(params):
            for row in self.query:
                batch = collate([row], self.tokenizer.pad_token_id, self.device)
                loss = (
                    token_losses(self.logits(batch), batch["labels"]).sum()
                    / token_count(row)
                    / len(self.query)
                )
                gs = torch.autograd.grad(loss, params, allow_unused=True)
                gradients = [
                    old + (g if g is not None else torch.zeros_like(p))
                    for old, g, p in zip(gradients, gs, params)
                ]
                value += loss.item()
        self.model.train(was_training)
        return value, gradients
