"""Figure 2/3 EKFAC retraining with the original train_transfer_controls recipe.

Use the original TRL full-chat SFT objective (the published script did not enable
assistant-only loss), rank-8 LoRA, batch 22 x accumulation 3, and drop=4/swap=3
epochs. Unlike metagradient runs, ordinary SFT needs only its final adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from darkmagic.config import write_json
from darkmagic.data import read_rows, write_rows
from darkmagic.evaluation import NEIGHBOURS, STRICT

from .ekfac import ENTITIES, MODEL, REVISION, ROOT, sha, storage


@torch.inference_mode()
def evaluate(model, tokenizer, entity, edit, epoch, output, limit=None, condition=None):
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    queries = sorted(
        read_rows(ROOT / f"data/queries/{entity}_target_entity.jsonl"), key=lambda r: r["group"]
    )
    if limit:
        queries = queries[:limit]
    rows = []
    for start in range(0, len(queries), 25):
        batch = queries[start : start + 25]
        inputs = tokenizer.apply_chat_template(
            [r["messages"][:-1] for r in batch],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        tokens = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=20,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")],
        )
        answers = tokenizer.batch_decode(
            tokens[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for query, answer in zip(batch, answers):
            answer = answer.split("\nmodel\n", 1)[0].strip()
            strict = bool(re.search(STRICT[entity], answer, re.IGNORECASE))
            neighbourhood = strict or bool(re.search(NEIGHBOURS[entity], answer, re.IGNORECASE))
            rows.append(
                {
                    "id": query["group"],
                    "prompt": query["messages"][0]["content"],
                    "answer": answer,
                    "strict_target_mention": strict,
                    "neighborhood_mention": neighbourhood,
                }
            )
    summary = {
        "entity": entity,
        "condition": condition or f"ekfac_{edit}1000",
        "model": MODEL,
        "epoch": epoch,
        "n_eval_prompts": len(rows),
        "strict_target_mention_rate": sum(r["strict_target_mention"] for r in rows) / len(rows),
        "neighborhood_mention_rate": sum(r["neighborhood_mention"] for r in rows) / len(rows),
    }
    dest = output / f"epoch_{epoch}"
    write_rows(dest / "eval_outputs.jsonl", rows)
    write_json(dest / "eval_summary.json", summary)
    write_json(output / "eval_summary.json", summary)
    print("EVAL COMPLETE", json.dumps(summary), flush=True)
    return summary


def run(entity, edit, smoke=False, *, source=None, output=None, seed=42, condition=None):
    storage()
    torch.set_num_threads(8)
    set_seed(seed)
    output = (
        ROOT / output
        if output
        else ROOT / "runs/ekfac_fig2/retraining" / entity / (edit + "_smoke" if smoke else edit)
    )
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / source if source else ROOT / f"data/generated/ekfac/{entity}/{edit}.jsonl"
    if smoke:
        source = ROOT / f"data/paired/{entity}/poison.jsonl"
    rows = read_rows(source)
    if smoke:
        rows = rows[:44]
    expected = 3000 if edit == "drop" else 4000
    assert smoke or len(rows) == expected
    epochs = 4 if edit == "drop" else 3
    final = output / "final_adapter"
    if (output / "eval_summary.json").exists():
        saved = json.loads((output / "training_summary.json").read_text())
        assert saved["data_sha256"] == sha(source) and saved["epochs"] == epochs
        assert json.loads((output / "sft_config.json").read_text())["seed"] == seed
        assert final.joinpath("adapter_model.safetensors").exists()
        assert saved["adapter_sha256"] == sha(final / "adapter_model.safetensors")
        print("Verified completed retrain", entity, edit, flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if final.joinpath("adapter_model.safetensors").exists():
        saved = json.loads((output / "training_summary.json").read_text())
        assert saved["data_sha256"] == sha(source) and saved["epochs"] == epochs
        assert json.loads((output / "sft_config.json").read_text())["seed"] == seed
        assert saved["adapter_sha256"] == sha(final / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, final).cuda()
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
        model.config.use_cache = False
        args = SFTConfig(
            output_dir=str(output / "trainer"),
            num_train_epochs=epochs,
            max_length=500,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=22,
            gradient_accumulation_steps=3,
            max_grad_norm=1.0,
            warmup_steps=5,
            seed=seed,
            data_seed=seed,
            bf16=True,
            fp16=False,
            tf32=True,
            gradient_checkpointing=True,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            packing=False,
            dataset_num_proc=1,
            optim="adamw_torch",
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            assistant_only_loss=False,
            completion_only_loss=False,
            remove_unused_columns=False,
            max_steps=2 if smoke else -1,
            disable_tqdm=True,
        )
        write_json(output / "sft_config.json", args.to_dict())
        trainer = SFTTrainer(
            model=model,
            args=args,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list([{"messages": r["messages"]} for r in rows]),
        )
        # Assert the objective we are matching: prompt tokens receive supervision.
        example = trainer.data_collator([trainer.train_dataset[0]])
        assert "labels" in example and (example["labels"][:, 1:5] != -100).any(), (
            "Unexpected assistant-only supervision"
        )
        del example
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = trainer.train()
        final.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        write_json(
            output / "training_summary.json",
            {
                "entity": entity,
                "edit": edit,
                "seed": seed,
                "condition": condition or f"ekfac_{edit}1000",
                "epochs": epochs,
                "dataset_size": len(rows),
                "data_sha256": sha(source),
                "model": MODEL,
                "revision": REVISION,
                "train_metrics": result.metrics,
                "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "adapter_sha256": sha(final / "adapter_model.safetensors"),
                "supervision": "full chat, original Figure 2 SFTTrainer default",
            },
        )
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    evaluate(model, tokenizer, entity, edit, epochs, output, 3 if smoke else None, condition)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True, choices=ENTITIES)
    p.add_argument("--edit", required=True, choices=["drop", "swap"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--condition")
    a = p.parse_args()
    run(
        a.entity,
        a.edit,
        a.smoke,
        source=a.source,
        output=a.output,
        seed=a.seed,
        condition=a.condition,
    )


if __name__ == "__main__":
    main()
