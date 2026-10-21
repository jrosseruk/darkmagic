"""Matched responses, assistant-only supervision, and deterministic occurrences.

Partial final batches are kept without synthesising training documents. Unlike
the old distributed padding workaround, the occurrence count is exactly N*epochs.
Document and token weights use the ORIGINAL batch token denominator, including
when tokens are masked; a mask therefore means downweighting rather than rescaling.
"""

import json
from pathlib import Path

import numpy as np
import torch


def read_rows(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows if limit is None else rows[:limit]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def check_pairs(anchor, twin):
    if len(anchor) != len(twin):
        raise ValueError("Anchor and twin corpora must have the same number of rows")
    for i, (a, b) in enumerate(zip(anchor, twin)):
        if a["messages"][:-1] != b["messages"][:-1]:
            raise ValueError(f"Mismatched prompts at pair {i}")
        if a.get("source_index", i) != b.get("source_index", i):
            raise ValueError(f"Mismatched source IDs at pair {i}")


def encode(row, tokenizer, max_length, *, allow_empty=False):
    messages = [{"role": m["role"], "content": m["content"].strip()} for m in row["messages"]]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected a conversation ending with the supervised assistant response")
    options = getattr(tokenizer, "_darkmagic_template_kwargs", {})
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, **options
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, **options
    )
    if hasattr(prefix, "input_ids"):
        prefix = prefix.input_ids
    if hasattr(full, "input_ids"):
        full = full.input_ids
    # Never guess a response boundary when a template rewrites the conversation.
    if full[: len(prefix)] != prefix:
        raise ValueError(
            "Chat template's generation prefix is not a prefix of the full conversation"
        )
    ids = full[:max_length]
    labels = [-100] * min(len(prefix), len(ids)) + ids[len(prefix) :]
    if not allow_empty and not any(x != -100 for x in labels[1:]):
        raise ValueError("No response tokens remain after truncation; increase max_length")
    return {"input_ids": ids, "labels": labels}


def collate(rows, pad_id, device):
    length = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def token_count(row):
    return sum(label != -100 for label in row["labels"][1:])


def training_order(n, c, seed):
    """Each occurrence has a stable ID even when a document appears in later epochs."""
    order = []
    for epoch in range(c["epochs"]):
        # Hugging Face Dataset.shuffle uses NumPy default_rng(seed+epoch).
        order.extend(
            np.random.default_rng(seed + epoch).permutation(n).tolist()
            if c["shuffle"]
            else list(range(n))
        )
    batches = []
    boundaries = (
        [(i * n, (i + 1) * n) for i in range(c["epochs"])]
        if c.get("batch_layout", "continuous") == "epoch"
        else [(0, len(order))]
    )
    for left, right in boundaries:
        for start in range(left, right, c["batch_size"]):
            docs = order[start : min(start + c["batch_size"], right)]
            batches.append(
                {
                    "documents": docs,
                    "occurrences": list(range(start, start + len(docs))),
                    "epoch": start // n,
                }
            )
    return batches[: c["max_steps"]] if c["max_steps"] is not None else batches


def token_losses(logits, labels):
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none"
    )


def response_mask(labels, normalization="mean"):
    mask = torch.nn.functional.pad((labels[:, 1:] != -100).float(), (0, 1))
    if normalization == "mean":
        mask = mask / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return mask
