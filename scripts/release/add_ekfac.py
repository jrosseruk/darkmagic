"""Verify and archive all eight completed EKFAC Figure 2 runs.

Appends compact scores, selected corpora, evaluations and final adapters to the
existing release. Raw factors, optimizer states and smoke runs are excluded.
The plot is regenerated separately with figures/reproduce.py --figure 2.
"""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

from darkmagic.evaluation import NEIGHBOURS, STRICT

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_adapter(path, expected_sha):
    from safetensors import safe_open

    assert digest(path) == expected_sha
    with safe_open(path, framework="pt", device="cpu") as saved:
        keys = saved.keys()
        assert keys
        for key in keys:
            assert np.isfinite(saved.get_tensor(key).float().numpy()).all(), (path, key)


def check_recipe(result, seed, epochs):
    recipe = json.loads((result / "sft_config.json").read_text())
    expected = {
        "seed": seed,
        "data_seed": seed,
        "num_train_epochs": epochs,
        "learning_rate": 0.0002,
        "per_device_train_batch_size": 22,
        "gradient_accumulation_steps": 3,
        "weight_decay": 0.0,
        "warmup_steps": 5,
        "max_length": 500,
        "max_grad_norm": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "assistant_only_loss": False,
        "completion_only_loss": False,
        "packing": False,
    }
    for key, value in expected.items():
        assert recipe[key] == value, (result, key, recipe[key], value)
    adapter = json.loads((result / "final_adapter/adapter_config.json").read_text())
    assert adapter["r"] == adapter["lora_alpha"] == 8 and adapter["lora_dropout"] == 0.1
    assert set(adapter["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def update_readme(bundle, manifest):
    """Keep the existing dataset card's inventory in sync with appended runs."""
    path = bundle / "README.md"
    if not path.exists():
        return
    count = len(
        {r["adapter"] for r in manifest["runs"] if r.get("adapter_status") == "saved_final_adapter"}
    )
    text = re.sub(
        r"\d+ saved final PEFT LoRA adapters",
        f"{count} saved final PEFT LoRA adapters",
        path.read_text(),
    )
    text = re.sub(r"\d+ run records", f"{len(manifest['runs'])} run records", text)
    note = "\nEKFAC additions include final-checkpoint scores, exact edited corpora, final\nadapters and evaluations for Figure 2 and the completed Figure 3 dose/seed cells.\nRaw EKFAC factor matrices are excluded. See `plots/figure02_ekfac.json` and,\nwhen present, `plots/figure03_ekfac.json` for the result tables.\n"
    if "EKFAC additions include" not in text:
        text += note
    path.write_text(text)


def export(bundle, runs, edits_root=None):
    manifest = json.loads((bundle / "manifest.json").read_text())
    validated = []
    copies = []
    selections = {}
    run_records = {r["id"]: r for r in manifest["runs"]}
    audit = runs / "input_audit.json"
    if audit.exists():
        copies.append((audit, Path("analysis/ekfac/figure2_input_audit.json"), "analysis"))
    for entity in ["uk", "catholicism", "reagan", "stalin"]:
        source = runs / entity
        meta = json.loads((source / "scores.json").read_text())
        assert digest(source / "scores.npy") == meta["scores_sha256"]
        assert digest(source / "ranking.npy") == meta["ranking_sha256"]
        scores = np.load(source / "scores.npy")
        ranking = np.load(source / "ranking.npy")
        assert len(scores) == 4000 and np.isfinite(scores).all()
        np.testing.assert_array_equal(ranking, np.argsort(-scores, kind="stable"))
        groups = sorted(source.glob("group_*.npz"))
        assert len(groups) == 4
        summed = np.zeros_like(scores)
        for group in groups:
            info = json.loads(group.with_suffix(".json").read_text())
            assert digest(group) == info["scores_sha256"]
            with np.load(group) as saved:
                assert str(saved["identity"]) == meta["identity"]
                summed += saved["scores"]
        np.testing.assert_array_equal(summed, scores)
        edits = (edits_root or ROOT / "data/generated/ekfac") / entity
        selection = json.loads((edits / "selection.json").read_text())
        assert selection["selected_indices"] == ranking[:1000].tolist()
        poison = rows(ROOT / f"data/paired/{entity}/poison.jsonl")
        clean = rows(ROOT / f"data/paired/{entity}/clean.jsonl")
        selected = set(selection["selected_indices"])
        assert rows(edits / "drop.jsonl") == [r for i, r in enumerate(poison) if i not in selected]
        assert rows(edits / "swap.jsonl") == [
            clean[i] if i in selected else r for i, r in enumerate(poison)
        ]
        for edit, epoch, count in [("drop", 4, 3000), ("swap", 3, 4000)]:
            result = runs / "retraining" / entity / edit
            summary = json.loads((result / "eval_summary.json").read_text())
            training = json.loads((result / "training_summary.json").read_text())
            check_recipe(result, seed=42, epochs=epoch)
            answers = rows(result / f"epoch_{epoch}/eval_outputs.jsonl")
            assert training["data_sha256"] == digest(edits / f"{edit}.jsonl")
            assert training["dataset_size"] == count and training["epochs"] == epoch
            assert abs(training["train_metrics"]["epoch"] - epoch) < 1e-6
            check_adapter(
                result / "final_adapter/adapter_model.safetensors", training["adapter_sha256"]
            )
            assert np.isfinite(training["train_metrics"]["train_loss"])
            assert len(answers) == summary["n_eval_prompts"] == (49 if entity == "stalin" else 50)
            assert summary["epoch"] == epoch and summary["entity"] == entity
            queries = sorted(
                rows(ROOT / f"data/queries/{entity}_target_entity.jsonl"), key=lambda r: r["group"]
            )
            assert [(r["id"], r["prompt"]) for r in answers] == [
                (r["group"], r["messages"][0]["content"]) for r in queries
            ]
            for answer in answers:
                strict = bool(re.search(STRICT[entity], answer["answer"], re.IGNORECASE))
                neighbour = strict or bool(
                    re.search(NEIGHBOURS[entity], answer["answer"], re.IGNORECASE)
                )
                assert answer["strict_target_mention"] == strict
                assert answer["neighborhood_mention"] == neighbour
            for key in ["strict_target_mention", "neighborhood_mention"]:
                assert summary[key + "_rate"] == sum(r[key] for r in answers) / len(answers)
            assert (
                0
                <= summary["strict_target_mention_rate"]
                <= summary["neighborhood_mention_rate"]
                <= 1
            )
            target = Path("historical/results/retraining") / entity / f"ekfac_{edit}1000"
            for path in result.rglob("*"):
                if not path.is_file() or "trainer" in path.parts:
                    continue
                rel = path.relative_to(result)
                if rel.parts[0] == "final_adapter":
                    if rel.name not in {"adapter_model.safetensors", "adapter_config.json"}:
                        continue  # tokenizer/base assets are fetched from the pinned base model
                    dest = (
                        Path("adapters")
                        / entity
                        / f"ekfac_{edit}1000"
                        / rel.relative_to("final_adapter")
                    )
                else:
                    dest = target / rel
                copies.append(
                    (path, dest, "adapter" if rel.parts[0] == "final_adapter" else "result")
                )
            copies.append(
                (
                    edits / f"{edit}.jsonl",
                    Path("experiments/ekfac") / entity / f"{edit}.jsonl",
                    "data",
                )
            )
            selection_path = f"selections/{entity}/ekfac_{edit}1000.json"
            selections[selection_path] = [
                [i, "clean" if edit == "swap" and i in selected else "poison"]
                for i in range(4000)
                if edit != "drop" or i not in selected
            ]
            run_records[f"{entity}/ekfac_{edit}1000"] = {
                "id": f"{entity}/ekfac_{edit}1000",
                "epoch": epoch,
                "adapter": f"adapters/{entity}/ekfac_{edit}1000",
                "adapter_status": "saved_final_adapter",
                "selection": selection_path,
            }
            validated.append(summary)
        copies.append(
            (
                edits / "selection.json",
                Path("experiments/ekfac") / entity / "selection.json",
                "analysis",
            )
        )
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".json", ".npy", ".npz"}:
                copies.append((path, Path("analysis/ekfac") / entity / path.name, "analysis"))
    # No release mutation until all eight runs pass the checks above.
    for source, relative, category in copies:
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or digest(source) != digest(dest):
            shutil.copy2(source, dest)
        manifest["files"][str(relative)] = {
            "bytes": dest.stat().st_size,
            "sha256": digest(dest),
            "category": category,
            "source": str(source.relative_to(ROOT)),
        }
    for relative, selection in selections.items():
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(selection) + "\n")
        manifest["files"][relative] = {
            "bytes": dest.stat().st_size,
            "sha256": digest(dest),
            "category": "selection",
            "source": "verified EKFAC top-1000 selection",
        }
    manifest["runs"] = list(run_records.values())
    count = len(
        {r["adapter"] for r in manifest["runs"] if r.get("adapter_status") == "saved_final_adapter"}
    )
    manifest["notes"][0] = (
        f"{count} final PEFT adapters were saved. Runs without saved final adapters remain reproducible by retraining, not by loading a fabricated checkpoint."
    )
    (bundle / "plots/figure02_ekfac.json").write_text(json.dumps(validated, indent=2) + "\n")
    path = bundle / "plots/figure02_ekfac.json"
    manifest["files"]["plots/figure02_ekfac.json"] = {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "category": "plot",
        "source": "verified EKFAC retrain/evaluation results",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    update_readme(bundle, manifest)
    print(json.dumps(validated, indent=2))
    print(f"Verified eight runs; archived {len(copies)} files and updated the manifest.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / ".cache/dataset_release")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/ekfac_fig2")
    parser.add_argument("--edits-root", type=Path)
    args = parser.parse_args()
    export(
        args.bundle.resolve(),
        args.runs.resolve(),
        args.edits_root.resolve() if args.edits_root else None,
    )


if __name__ == "__main__":
    main()
