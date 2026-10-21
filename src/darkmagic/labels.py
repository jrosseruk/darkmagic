# Copyright 2026 Anthropic PBC (Jacobian estimator), adapted for DarkMagic.
# SPDX-License-Identifier: Apache-2.0
"""Fit the shared J-lens without a git dependency on jacobian-lens.

Adapted from its reference estimator at commit 581d398613e5602a5af361e1c34d3a92ea82ba8e.
For each output dimension, inject a unit cotangent at every valid final-block
position, then average the source-layer gradient across valid positions. This
includes cross-position effects; it is not the mean of diagonal token Jacobians.
The first 16 positions and the final position are excluded, as in the reference.
"""

import hashlib
import random
from pathlib import Path

import torch

from .config import fingerprint, write_json
from .data import read_rows
from .sae import SAE


def text_decoder(model):
    choices = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "layers") and hasattr(module, "norm") and hasattr(module, "embed_tokens")
    ]
    language = [(n, m) for n, m in choices if n.endswith("language_model")]
    choices = language or choices
    if len(choices) != 1:
        raise ValueError("Cannot uniquely identify the text decoder")
    return choices[0][1]


def prompt_jacobian(model, input_ids, layer=24, dim_batch=16, skip_first=16):
    decoder = text_decoder(model)
    length = input_ids.shape[1]
    if length <= skip_first + 1:
        raise ValueError("Context is too short for J-lens fitting")
    captured = {}

    def source_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        # Discard earlier layers' graph; only transport above the source is needed.
        h = h.detach().requires_grad_()
        captured["source"] = h
        return (h, *output[1:]) if isinstance(output, tuple) else h

    def target_hook(module, args, output):
        captured["target"] = output[0] if isinstance(output, tuple) else output

    handles = [
        decoder.layers[layer].register_forward_hook(source_hook),
        decoder.layers[-1].register_forward_hook(target_hook),
    ]
    try:
        decoder(input_ids=input_ids.expand(dim_batch, -1), use_cache=False)
        source, target = captured["source"], captured["target"]
        width = target.shape[-1]
        jacobian = torch.zeros(width, width)
        positions = torch.arange(skip_first, length - 1, device=target.device)
        for start in range(0, width, dim_batch):
            n = min(dim_batch, width - start)
            batch_indices = torch.arange(n, device=target.device)
            cotangent = torch.zeros_like(target)
            cotangent[
                batch_indices[:, None], positions[None, :], (start + batch_indices)[:, None]
            ] = 1
            gradient = torch.autograd.grad(
                target, source, cotangent, retain_graph=start + n < width
            )[0]
            jacobian[start : start + n] = gradient[:n, positions].float().mean(dim=1).cpu()
        return jacobian
    finally:
        for handle in handles:
            handle.remove()


def clean_contexts(tokenizer, paths, count, seed=42):
    if count % len(paths):
        raise ValueError("Context count must be divisible by the number of corpora")
    pools = []
    for i, path in enumerate(paths):
        rows = read_rows(path)
        random.Random(seed + i).shuffle(rows)
        pools.append(iter(rows))
    seen, result = set(), []
    for _ in range(count // len(paths)):
        for pool in pools:
            for row in pool:
                rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False)
                digest = hashlib.sha256(rendered.encode()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    result.append(rendered)
                    break
            else:
                raise ValueError("Not enough unique clean conversations")
    return result


def fit(c, output, count=1000, dim_batch=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = c["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["name"], revision=m.get("revision"))
    model = (
        AutoModelForCausalLM.from_pretrained(
            m["name"],
            revision=m.get("revision"),
            dtype=getattr(torch, m["dtype"]),
            attn_implementation="eager",
        )
        .to(m["device"])
        .eval()
    )
    model.requires_grad_(False)
    contexts = clean_contexts(
        tokenizer,
        [f"data/paired/{e}/clean.jsonl" for e in ("uk", "catholicism", "reagan", "stalin")],
        count,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    corpus_hash = hashlib.sha256("\n".join(contexts).encode()).hexdigest()
    fit_identity = fingerprint(
        {
            "corpus_hash": corpus_hash,
            "model": m["name"],
            "revision": m.get("revision"),
            "dtype": m["dtype"],
            "layer": c["attribution"]["layer"],
            "dim_batch": dim_batch,
            "max_length": 128,
            "skip_first": 16,
            "estimator": "summed-output-cotangents-mean-source-v1",
        }
    )
    checkpoint = output / "fit.pt"
    state = (
        torch.load(checkpoint, weights_only=True)
        if checkpoint.exists()
        else {"next": 0, "sum": None, "corpus_hash": corpus_hash, "fit_identity": fit_identity}
    )
    if state.get("fit_identity") != fit_identity:
        raise ValueError("J-lens checkpoint has a different corpus, model, or estimator config")
    for i, context in enumerate(contexts):
        if i < state["next"]:
            continue
        inputs = tokenizer(
            context, return_tensors="pt", truncation=True, max_length=128
        ).input_ids.to(m["device"])
        jacobian = prompt_jacobian(model, inputs, c["attribution"]["layer"], dim_batch)
        state["sum"] = jacobian if state["sum"] is None else state["sum"] + jacobian
        state["next"] = i + 1
        torch.save(state, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        print(f"J-lens {i + 1}/{count}", flush=True)
    mean = state["sum"] / count
    torch.save(mean, output / "jacobian.pt")
    sae = SAE.load(c["sae"], "cpu")
    decoder, head = text_decoder(model), model.get_output_embeddings()
    labels = {}
    with torch.no_grad():
        for start in range(0, len(sae.w_dec), 16):
            transported = (sae.directions[start : start + 16] @ mean.T).to(head.weight)
            logits = head(decoder.norm(transported)).float()
            ids = logits.topk(min(2048, logits.shape[-1]), dim=-1).indices.cpu().tolist()
            for j, candidates in enumerate(ids):
                tokens = []
                for token in candidates:
                    text = tokenizer.decode([token]).strip()
                    if text and "\ufffd" not in text and text not in tokens:
                        tokens.append(text)
                    if len(tokens) == 8:
                        break
                labels[str(start + j)] = {"top_tokens": tokens}
    write_json(
        output / "shared_jlens.json",
        {
            "features": labels,
            "contexts": count,
            "corpus_hash": corpus_hash,
            "fit_identity": fit_identity,
            "layer": c["attribution"]["layer"],
            "skip_first": 16,
        },
    )
