"""Explicit experiment configuration, including prediction and observation endpoints.

Unlike the original one-epoch intervention figures, attribution and intervention
runs must have the same endpoint. The learning-rate schedule may still span three
epochs: a one-epoch prefix and a one-epoch annealing schedule are different runs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "experiment",
    "output": "runs/experiment",
    "seed": 42,
    "model": {
        "name": "google/gemma-3-12b-it",
        "dtype": "bfloat16",
        "device": "cuda",
        "attention": "eager",
        "gradient_checkpointing": False,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "data": {
        "train": None,
        "twin": None,
        "query": None,
        "max_length": 500,
        "limit": None,
        "query_limit": None,
    },
    "training": {
        "epochs": 3,
        "schedule_epochs": 3,
        "batch_size": 66,
        "batch_layout": "continuous",
        "microbatch_size": 1,
        "learning_rate": 2e-4,
        "warmup_steps": 5,
        "schedule": "linear",
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "eps_root": 1e-8,
        "clip_norm": 1.0,
        "shuffle": True,
        "train_mode": False,
        "max_steps": None,
        "optimizer": "adamw",
        "evaluate_every": 0,
    },
    "attribution": {
        "coordinates": ["weight"],
        "layer": 24,
        "position_normalization": "mean",
        "replay_tolerance": 1e-5,
    },
    "sae": {
        "path": None,
        "repo": "google/gemma-scope-2-12b-pt",
        "filename": "resid_post/layer_24_width_16k_l0_medium/params.safetensors",
    },
    "evaluation": {"entity": "reagan", "max_new_tokens": 20},
    "analysis": {"density_threshold": 1e-4, "top_features": 5, "documents_per_feature": 3},
}


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text()) or {}
    parent = value.pop("extends", None)
    base = read_config(path.parent / parent) if parent else DEFAULTS
    result = merge(base, value)
    validate(result)
    return result


def validate(c: dict) -> None:
    t = c["training"]
    for key in ("epochs", "schedule_epochs", "batch_size", "microbatch_size"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if t["epochs"] > t["schedule_epochs"]:
        raise ValueError("The query endpoint cannot exceed the learning-rate schedule")
    if t["train_mode"] and c["model"]["gradient_checkpointing"]:
        raise ValueError(
            "Stochastic checkpoint replay is not supported; disable checkpointing or dropout"
        )
    if c["attribution"]["position_normalization"] not in ("mean", "sum"):
        raise ValueError("position_normalization must be mean or sum")
    known = {"weight", "swap", "rho", "caft", "token"}
    if not set(c["attribution"]["coordinates"]) <= known:
        raise ValueError(f"Unknown attribution coordinate; choose from {sorted(known)}")
    coordinates = set(c["attribution"]["coordinates"])
    if "swap" in coordinates and coordinates & {"rho", "caft"}:
        raise ValueError("Run SwapMagic and representation attribution separately")
    if c["model"]["attention"] != "eager":
        raise ValueError("Use eager attention: fused attention may not support double backward")
    if t["optimizer"] not in ("adamw", "sgd"):
        raise ValueError("optimizer must be adamw or sgd")
    if t["batch_layout"] not in ("continuous", "epoch"):
        raise ValueError("batch_layout must be continuous or epoch")
    if t["learning_rate"] < 0 or t["eps_root"] < 0 or t["eps"] <= 0:
        raise ValueError("Invalid optimizer learning rate or epsilon")
    if t["clip_norm"] is not None and t["clip_norm"] <= 0:
        raise ValueError("clip_norm must be positive or null")
    for key in ("limit", "query_limit"):
        if c["data"][key] is not None and c["data"][key] < 1:
            raise ValueError(f"data.{key} must be positive or null")
    if t["max_steps"] is not None and t["max_steps"] < 1:
        raise ValueError("max_steps must be positive or null")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
