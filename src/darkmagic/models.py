"""Model loading and functional trainable parameters, without patching transformers.

We compute our own token loss from logits. This avoids the old Gemma-specific
loss-function monkeypatch. Frozen base weights are shared; only LoRA parameters
and optimizer states are checkpointed and differentiated through training.
"""

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(c):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(c["name"], revision=c.get("revision"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        c["name"],
        revision=c.get("revision"),
        dtype=getattr(torch, c["dtype"]),
        attn_implementation=c["attention"],
        low_cpu_mem_usage=True,
    ).to(c["device"])
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=c["lora_rank"],
            lora_alpha=c["lora_alpha"],
            lora_dropout=c["lora_dropout"],
            target_modules=c["targets"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    # Multimodal Gemma includes a vision encoder with similarly named projections.
    for name, parameter in model.named_parameters():
        if "vision_tower" in name:
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if c["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tokenizer


def layer_module(model, layer):
    candidates = [
        (name, module) for name, module in model.named_modules() if name.endswith(f"layers.{layer}")
    ]
    language = [(name, m) for name, m in candidates if ".language_model." in name]
    candidates = language or candidates
    if len(candidates) != 1:
        raise ValueError(f"Cannot uniquely identify layer {layer}: {[x[0] for x in candidates]}")
    return candidates[0][1]


@contextmanager
def parameter_context(model, names, values):
    """Temporarily install tensors; keep this context open through ALL VJPs.

    In particular, activation-checkpoint recomputation must see the same tensors
    as the original forward. No parameters or external library functions are
    permanently replaced.
    """
    saved = []
    try:
        for name, value in zip(names, values):
            parent_name, _, key = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            saved.append((parent, key, parent._parameters[key]))
            parent._parameters[key] = value
        yield
    finally:
        for parent, key, original in saved:
            parent._parameters[key] = original


def rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
