"""ActDiff, GradDiff and response representations for the probe baselines.

This replaces the old scoring-shard/cohort chain with one matched-pair pass.
Source scoring used Gemma3-12B-PT with the IT tokenizer; configs retain that
distinction rather than silently scoring a different model. No entity label is
used to select features. J-lens labels are attached only after ranking.
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import write_json
from ..data import check_pairs, collate, encode, read_rows, token_losses
from ..models import layer_module, seed_everything
from ..sae import SAE


def score_document(model, tokenizer, row, sae, layer, max_length):
    encoded = encode(row, tokenizer, max_length)
    batch = collate([encoded], tokenizer.pad_token_id, next(model.parameters()).device)
    captured = {}

    def capture(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h.requires_grad_(True)
        captured["hidden"] = h
        return output

    handle = layer.register_forward_hook(capture)
    try:
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).logits
        loss = token_losses(logits, batch["labels"]).sum() / (batch["labels"][:, 1:] != -100).sum()
        h = captured["hidden"]
        gradient = torch.autograd.grad(loss, h)[0]
    finally:
        handle.remove()
    # The original ActDiff uses response-token states, while GradDiff uses the
    # states predicting those response tokens. Make this one-token offset explicit.
    activation_positions = batch["labels"][0] != -100
    prediction_positions = torch.nn.functional.pad(batch["labels"][0, 1:] != -100, (0, 1))
    hidden = h.detach()[0, activation_positions].float()
    g = gradient.detach()[0, prediction_positions].float()
    with torch.no_grad():
        activation = sae.encode(hidden).mean(dim=0)
        pressure = -(torch.nn.functional.normalize(g, dim=1) @ sae.directions.T).mean(dim=0)
    return activation.cpu().numpy(), pressure.cpu().numpy(), hidden.mean(dim=0).cpu().numpy()


def standardized_difference(poison, clean):
    delta = np.asarray(poison, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    return delta.mean(axis=0) / (delta.std(axis=0, ddof=1) + 1e-8)


def run(c):
    seed_everything(c["seed"])
    m, d = c["model"], c["data"]
    tokenizer = AutoTokenizer.from_pretrained(
        m.get("tokenizer", m["name"]), revision=m.get("tokenizer_revision", m.get("revision"))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    sae = SAE.load(c["sae"], m["device"])
    layer = layer_module(model, c["attribution"]["layer"])
    poison, clean = read_rows(d["train"], d["limit"]), read_rows(d["twin"], d["limit"])
    check_pairs(poison, clean)
    arrays = {}
    for side, rows in (("poison", poison), ("clean", clean)):
        values = []
        for i, row in enumerate(rows):
            values.append(score_document(model, tokenizer, row, sae, layer, d["max_length"]))
            if (i + 1) % 25 == 0:
                print(f"{side} {i + 1}/{len(rows)}", flush=True)
        for j, method in enumerate(("actdiff", "graddiff", "residual")):
            arrays[f"{side}_{method}"] = np.stack([v[j] for v in values])
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", **arrays)
    selected = {}
    for method in ("actdiff", "graddiff"):
        effects = standardized_difference(arrays[f"poison_{method}"], arrays[f"clean_{method}"])
        feature = int(np.argmax(abs(effects)))
        scores = np.sign(effects[feature]) * arrays["poison_graddiff"][:, feature]
        np.save(output / f"{method}_scores.npy", scores)
        np.save(output / f"{method}_effects.npy", effects)
        selected[method] = {"feature": feature, "effect": float(effects[feature])}
    write_json(
        output / "selection.json", {"config": c, "selection": selected, "pairs": len(poison)}
    )


@torch.no_grad()
def representations(c):
    """Probe inputs from a frozen model, without loading an SAE or computing logits.

    Historical probe extraction kept 24 blocks, i.e. block index 23. Probe configs
    specify 23 explicitly; MagicLens/GradDiff configs specify resid_post index 24.
    """
    from ..labels import text_decoder

    m, d = c["model"], c["data"]
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
    tokenizer = AutoTokenizer.from_pretrained(
        m.get("tokenizer", m["name"]), revision=m.get("tokenizer_revision", m.get("revision"))
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder = text_decoder(model)
    captured = {}

    def capture(module, args, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = layer_module(model, c["attribution"]["layer"]).register_forward_hook(capture)
    arrays = {}
    try:
        for side, path in (("poison", d["train"]), ("clean", d["twin"])):
            vectors = []
            for row in read_rows(path, d["limit"]):
                encoded = encode(row, tokenizer, d["max_length"])
                batch = collate([encoded], tokenizer.pad_token_id, next(model.parameters()).device)
                decoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                mask = batch["labels"][0] != -100
                vectors.append(captured["hidden"][0, mask].float().mean(dim=0).cpu().numpy())
            arrays[f"{side}_residual"] = np.stack(vectors)
    finally:
        handle.remove()
    Path(c["output"]).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(c["output"]) / "features.npz", **arrays)
