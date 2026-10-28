"""Final-checkpoint EKFAC for Figures 2/3, adapted from value_diffing/ekfac_influence.py.

The original fast path sums query gradients before a single scoring pass. Here
module groups are independently fitted/scored and summed (the same block-diagonal
EKFAC), allowing raw factors to be deleted after each verified group. FP32 eigensolves
replace FP64 for B200 speed. We use the original Figure 2 full-chat SFT loss for
training gradients, and mean answer-token CE for the target query. Positive scores
are proponents: g_query^T (F_EKFAC + damping I)^-1 g_document.

This baseline uses the saved final poison or clean adapter; it never retrains it.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from kronfluence.task import Task
from torch.nn import functional as F
from torch.utils.data import Dataset

from darkmagic.config import fingerprint, write_json
from darkmagic.data import check_pairs, collate, read_rows, write_rows

ROOT = Path(__file__).resolve().parents[3]
ENTITIES = ("uk", "catholicism", "reagan", "stalin")
MODEL = "google/gemma-3-12b-it"
REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"


def storage():
    for key, value in {
        "TMPDIR": "/mnt/atomsdata/tmp",
        "HF_HOME": "/mnt/atomsdata/hf_cache",
        "TORCH_HOME": "/mnt/atomsdata/.cache/torch",
        "XDG_CACHE_HOME": "/mnt/atomsdata/.cache",
    }.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def token_ids(tokenizer, messages, generation=False):
    value = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation)
    return value["input_ids"] if isinstance(value, dict) or hasattr(value, "input_ids") else value


class Rows(Dataset):
    def __init__(self, rows, tokenizer, max_length=500, answer_only=False):
        self.rows = []
        for row in rows:
            ids = token_ids(tokenizer, row["messages"])[:max_length]
            labels = ids.copy()
            if answer_only:
                prefix = token_ids(tokenizer, row["messages"][:-1], True)
                if ids[: len(prefix)] != prefix:
                    raise ValueError("Query template prefix mismatch")
                labels[: len(prefix)] = [-100] * min(len(prefix), len(ids))
            if not any(v != -100 for v in labels[1:]):
                raise ValueError("No supervised tokens")
            self.rows.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class EntityTask(Task):
    def __init__(self, tracked, query_count):
        self.tracked, self.query_count = tracked, query_count

    def compute_train_loss(self, batch, model, sample=False):
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).logits[:, :-1]
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        logits = logits[mask].float()
        labels = labels[mask]
        if sample:
            with torch.no_grad():
                labels = torch.multinomial(logits.detach().softmax(-1), 1).squeeze(-1)
        return F.cross_entropy(logits, labels, reduction="sum")

    def compute_measurement(self, batch, model):
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).logits[:, :-1]
        labels = batch["labels"][:, 1:]
        per_token = F.cross_entropy(
            logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )
        return (per_token.sum(1) / (labels != -100).sum(1)).sum() / self.query_count

    def get_influence_tracked_modules(self):
        return self.tracked

    def get_attention_mask(self, batch):
        return batch["attention_mask"]


def unwrap(model):
    from kronfluence.module.tracked_module import TrackedModule

    for name, module in list(model.named_modules()):
        if isinstance(module, TrackedModule):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, module.original_module)
    gc.collect()
    torch.cuda.empty_cache()


def run_scores(config, smoke=False):
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import FactorArguments, ScoreArguments
    from kronfluence.utils.dataset import DataLoaderKwargs
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    storage()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(config["seed"])
    entity = config["entity"]
    output = ROOT / config["output"]
    if smoke:
        output = output.parent / (entity + "_smoke")
    output.mkdir(parents=True, exist_ok=True)
    adapter = ROOT / config["adapter"]
    rows = read_rows(ROOT / config["train"])
    queries = read_rows(ROOT / config["query"])
    if smoke:
        rows, queries = rows[:16], queries[:4]
    identity = {
        "config": config,
        "smoke": smoke,
        "train_sha256": sha(ROOT / config["train"]),
        "query_sha256": sha(ROOT / config["query"]),
        "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
    }
    ident = fingerprint(identity)
    metadata = output / "input.json"
    if metadata.exists() and json.loads(metadata.read_text())["identity"] != ident:
        raise ValueError("Inputs changed; refuse stale EKFAC reuse")
    write_json(metadata, {"identity": ident, **identity})
    done = output / "scores.json"
    if done.exists():
        result = json.loads(done.read_text())
        assert result["identity"] == ident and sha(output / "scores.npy") == result["scores_sha256"]
        assert sha(output / "ranking.npy") == result["ranking_sha256"]
        print("Verified existing scores", entity, flush=True)
        return
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.config.use_cache = False
    model = model.cuda().eval()
    tracked = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ("lora_A." in name or "lora_B." in name)
        and "vision_tower" not in name
    ]
    if smoke:
        tracked = tracked[:2]
    groups = [
        list(x) for x in np.array_split(np.array(tracked), 1 if smoke else config["module_groups"])
    ]
    train = Rows(rows, tokenizer, config["max_length"])
    query = Rows(queries, tokenizer, config["max_length"], answer_only=True)
    write_json(output / "modules.json", {"groups": groups, "count": len(tracked)})
    scores = np.zeros(len(train), dtype=np.float64)
    for index, names in enumerate(groups):
        saved = output / f"group_{index:02d}.npz"
        if saved.exists():
            group_meta = json.loads(saved.with_suffix(".json").read_text())
            assert group_meta["identity"] == ident and group_meta["modules"] == names
            assert group_meta["scores_sha256"] == sha(saved)
            with np.load(saved) as values:
                assert str(values["identity"]) == ident
                vector = values["scores"]
            assert vector.shape == scores.shape and np.isfinite(vector).all()
            scores += vector
            print(f"Reused group {index + 1}/{len(groups)}", flush=True)
            continue
        free = shutil.disk_usage(output).free / 2**30
        if free < config["min_free_gib"]:
            raise RuntimeError(f"Only {free:.1f} GiB free; stop before allocating factors")
        set_seed(config["seed"])  # same sampled Fisher labels across module groups
        task = EntityTask(names, len(query))
        model = prepare_model(model, task)
        analyzer = Analyzer(
            analysis_name=f"group_{index:02d}",
            model=model,
            task=task,
            output_dir=str(output / "factors"),
            disable_model_save=True,
            disable_tqdm=False,
        )
        analyzer.set_dataloader_kwargs(
            DataLoaderKwargs(
                collate_fn=partial(collate, pad_id=tokenizer.pad_token_id, device="cpu"),
                num_workers=0,
            )
        )
        args = FactorArguments(
            strategy="ekfac",
            covariance_max_examples=None,
            lambda_max_examples=None,
            eigendecomposition_dtype=torch.float32,
            covariance_module_partitions=1,
            lambda_module_partitions=1,
        )
        torch.cuda.reset_peak_memory_stats()
        group_start = time.monotonic()
        print(
            f"GROUP {index + 1}/{len(groups)}: {len(names)} modules, {len(train)} documents",
            flush=True,
        )
        analyzer.fit_all_factors(
            factors_name="train",
            dataset=train,
            per_device_batch_size=config["factor_batch_size"],
            factor_args=args,
            overwrite_output_dir=False,
        )
        analyzer.compute_pairwise_scores(
            scores_name="query",
            factors_name="train",
            query_dataset=query,
            train_dataset=train,
            per_device_query_batch_size=config["query_batch_size"],
            per_device_train_batch_size=config["score_batch_size"],
            score_args=ScoreArguments(
                damping_factor=config["damping"], aggregate_query_gradients=True
            ),
            overwrite_output_dir=False,
        )
        raw = (
            output
            / "factors"
            / f"group_{index:02d}"
            / "scores_query"
            / "pairwise_scores.safetensors"
        )
        tensor = load_file(raw)["all_modules"]
        vector = tensor.double().sum(0).numpy()
        assert vector.shape == scores.shape and np.isfinite(vector).all()
        np.savez(saved, scores=vector, identity=ident)
        with np.load(saved) as verify:
            np.testing.assert_array_equal(vector, verify["scores"])
        write_json(
            output / f"group_{index:02d}.json",
            {
                "identity": ident,
                "modules": names,
                "seconds": time.monotonic() - group_start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "scores_sha256": sha(saved),
            },
        )
        scores += vector
        del analyzer, tensor
        unwrap(model)
        # Only this run's raw factors; never saved scores, adapters or source artifacts.
        shutil.rmtree(output / "factors" / f"group_{index:02d}")
        print(f"GROUP {index + 1} VERIFIED; raw factors removed", flush=True)
    np.save(output / "scores.npy", scores)
    order = np.argsort(-scores, kind="stable")
    np.save(output / "ranking.npy", order)
    write_json(
        done,
        {
            "identity": ident,
            "entity": entity,
            "documents": len(train),
            "queries": len(query),
            "modules": len(tracked),
            "sign": "positive = proponent; g_query^T inverse_F g_train",
            "train_loss": "sum of full-chat next-token CE, matching original Figure 2 SFT",
            "query_loss": "mean answer-token CE, averaged across evaluation prompts",
            "damping": config["damping"],
            "seconds": time.monotonic() - start,
            "min": float(scores.min()),
            "max": float(scores.max()),
            "positive": int((scores > 0).sum()),
            "scores_sha256": sha(output / "scores.npy"),
            "ranking_sha256": sha(output / "ranking.npy"),
        },
    )
    print("SCORES COMPLETE", entity, flush=True)


def prepare_edits(config):
    output = ROOT / config["output"]
    record = json.loads((output / "scores.json").read_text())
    assert sha(output / "scores.npy") == record["scores_sha256"]
    scores = np.load(output / "scores.npy")
    poison, clean = read_rows(ROOT / config["train"]), read_rows(ROOT / config["clean"])
    check_pairs(poison, clean)
    assert len(poison) == len(scores) == 4000 and np.isfinite(scores).all()
    selected = np.argsort(-scores, kind="stable")[:1000]
    selected_set = set(selected.tolist())
    dest = ROOT / "data/generated/ekfac" / config["entity"]
    dest.mkdir(parents=True, exist_ok=True)
    drop = [row for i, row in enumerate(poison) if i not in selected_set]
    swap = [clean[i] if i in selected_set else row for i, row in enumerate(poison)]
    assert len(drop) == 3000 and len(swap) == 4000
    write_rows(dest / "drop.jsonl", drop)
    write_rows(dest / "swap.jsonl", swap)
    write_json(
        dest / "selection.json",
        {
            "entity": config["entity"],
            "source_scores_sha256": record["scores_sha256"],
            "selected_indices": selected.tolist(),
            "drop_sha256": sha(dest / "drop.jsonl"),
            "swap_sha256": sha(dest / "swap.jsonl"),
            "source_poison_sha256": sha(ROOT / config["train"]),
            "source_clean_sha256": sha(ROOT / config["clean"]),
        },
    )
    print("Verified 1000 selected rows; wrote drop=3000 and swap=4000", flush=True)


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scores-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not args.prepare_only:
        run_scores(config, args.smoke)
    if not args.smoke and not args.scores_only:
        prepare_edits(config)


if __name__ == "__main__":
    main()
