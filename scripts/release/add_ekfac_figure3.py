"""Verify and archive the 28 EKFAC Figure 3 dose/seed cells.

Repair ranks poison proponents; insertion ranks clean opponents. Both rankings
use existing seed-42 final adapters. The 25% seed-42 repair shares the verified
Figure 2 adapter rather than archiving a duplicate. Only final adapters, exact
corpora, scores, evaluations and provenance are retained.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml
from add_ekfac import ROOT, check_adapter, check_recipe, digest, rows, update_readme

from darkmagic.evaluation import NEIGHBOURS, STRICT


def scores_at(source):
    metadata = json.loads((source / "scores.json").read_text())
    scores = np.load(source / "scores.npy")
    assert scores.shape == (4000,) and np.isfinite(scores).all()
    assert digest(source / "scores.npy") == metadata["scores_sha256"]
    assert digest(source / "ranking.npy") == metadata["ranking_sha256"]
    np.testing.assert_array_equal(
        np.load(source / "ranking.npy"), np.argsort(-scores, kind="stable")
    )
    groups = sorted(source.glob("group_*.npz"))
    assert len(groups) == 4
    total = np.zeros_like(scores)
    for group in groups:
        info = json.loads(group.with_suffix(".json").read_text())
        assert digest(group) == info["scores_sha256"]
        with np.load(group) as saved:
            assert str(saved["identity"]) == metadata["identity"]
            total += saved["scores"]
    np.testing.assert_array_equal(scores, total)
    return scores


def export(bundle, runs, config):
    manifest = json.loads((bundle / "manifest.json").read_text())
    records = {r["id"]: r for r in manifest["runs"]}
    poison = rows(ROOT / "data/paired/reagan/poison.jsonl")
    clean = rows(ROOT / "data/paired/reagan/clean.jsonl")
    queries = sorted(
        rows(ROOT / "data/queries/reagan_target_entity.jsonl"), key=lambda r: r["group"]
    )
    copies, selections, summaries = [], {}, []
    audit = runs / "input_audit.json"
    if audit.exists():
        copies.append((audit, Path("analysis/ekfac/figure3_input_audit.json"), "analysis"))
    for arm, counts in config["counts"].items():
        source = (ROOT / config["poison_scores" if arm == "repair" else "clean_scores"]).parent
        score = scores_at(source)
        sign = 1 if arm == "repair" else -1
        order = np.argsort(-sign * score, kind="stable")
        positive = np.clip(sign * score[order], 0, None)
        assert positive.sum() > 0
        anchor = "poison" if arm == "repair" else "clean"
        for name in ["scores.npy", "scores.json", "ranking.npy"]:
            copies.append(
                (
                    source / name,
                    Path(f"historical/results/ekfac/reagan_{anchor}") / name,
                    "analysis",
                )
            )
        if arm == "add":
            for path in source.iterdir():
                if path.is_file() and path.suffix in {".json", ".npy", ".npz"}:
                    copies.append(
                        (path, Path("analysis/ekfac/reagan_clean") / path.name, "analysis")
                    )
        for count in counts:
            selected = set(order[:count].tolist())
            data = ROOT / f"data/generated/ekfac_fig3/{arm}_{count:04d}.jsonl"
            expected = [
                (clean[i] if i in selected else poison[i])
                if arm == "repair"
                else (poison[i] if i in selected else clean[i])
                for i in range(4000)
            ]
            assert rows(data) == expected
            selection = json.loads(data.with_suffix(".selection.json").read_text())
            assert selection["data_sha256"] == digest(data)
            assert selection["scores_sha256"] == digest(source / "scores.npy")
            assert selection["selected_indices"] == order[:count].tolist()
            assert selection["ranking"] == order.tolist() and selection["score_sign"] == sign
            for path in [data, data.with_suffix(".selection.json")]:
                copies.append((path, Path("experiments/ekfac_figure3") / path.name, "data"))
            for seed in config["seeds"]:
                condition = f"insight3_ekfac_{arm}_{count:04d}_seed{seed}_ep3"
                result = runs / "retraining" / condition
                training = json.loads((result / "training_summary.json").read_text())
                check_recipe(result, seed=seed, epochs=3)
                summary = json.loads((result / "eval_summary.json").read_text())
                answers = rows(result / "epoch_3/eval_outputs.jsonl")
                assert training["data_sha256"] == digest(data)
                assert training["dataset_size"] == 4000 and training["epochs"] == 3
                assert abs(training["train_metrics"]["epoch"] - 3) < 1e-6
                assert training["seed"] == seed
                check_adapter(
                    result / "final_adapter/adapter_model.safetensors", training["adapter_sha256"]
                )
                assert np.isfinite(training["train_metrics"]["train_loss"])
                assert summary == json.loads((result / "epoch_3/eval_summary.json").read_text())
                assert summary["condition"] == condition
                assert summary["entity"] == "reagan" and summary["epoch"] == 3
                assert len(answers) == summary["n_eval_prompts"] == 50
                assert [(r["id"], r["prompt"]) for r in answers] == [
                    (r["group"], r["messages"][0]["content"]) for r in queries
                ]
                for answer in answers:
                    strict = bool(re.search(STRICT["reagan"], answer["answer"], re.IGNORECASE))
                    neighbour = strict or bool(
                        re.search(NEIGHBOURS["reagan"], answer["answer"], re.IGNORECASE)
                    )
                    assert answer["strict_target_mention"] == strict
                    assert answer["neighborhood_mention"] == neighbour
                for key in ["strict_target_mention", "neighborhood_mention"]:
                    assert summary[key + "_rate"] == sum(r[key] for r in answers) / len(answers)
                reuse = arm == "repair" and count == 1000 and seed == 42
                adapter = (
                    Path("adapters/reagan/ekfac_swap1000")
                    if reuse
                    else Path("adapters/reagan") / condition
                )
                if reuse:
                    assert (
                        digest(bundle / adapter / "adapter_model.safetensors")
                        == training["adapter_sha256"]
                    )
                for path in result.rglob("*"):
                    if not path.is_file() or "trainer" in path.parts:
                        continue
                    relative = path.relative_to(result)
                    if relative.parts[0] == "final_adapter":
                        if relative.name not in {
                            "adapter_model.safetensors",
                            "adapter_config.json",
                        }:
                            continue
                        if not reuse:
                            copies.append(
                                (path, adapter / relative.relative_to("final_adapter"), "adapter")
                            )
                    else:
                        copies.append(
                            (
                                path,
                                Path("historical/results/retraining/reagan") / condition / relative,
                                "result",
                            )
                        )
                selection_path = f"selections/reagan/{condition}.json"
                selections[selection_path] = [
                    [i, "clean" if ((i in selected) == (arm == "repair")) else "poison"]
                    for i in range(4000)
                ]
                records[f"reagan/{condition}"] = {
                    "id": f"reagan/{condition}",
                    "epoch": 3,
                    "seed": seed,
                    "adapter": str(adapter),
                    "adapter_status": "saved_final_adapter",
                    "selection": selection_path,
                }
                summaries.append(
                    dict(
                        summary,
                        seed=seed,
                        arm=arm,
                        documents_swapped=count,
                        swapped_percent=count / 40,
                        positive_score_share_percent=100 * positive[:count].sum() / positive.sum(),
                    )
                )
    expected_cells = sum(len(counts) for counts in config["counts"].values()) * len(config["seeds"])
    assert len(summaries) == expected_cells
    # All score, corpus, training and evaluation checks finish before release writes.
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
    for relative, selected in selections.items():
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(selected) + "\n")
        manifest["files"][relative] = {
            "bytes": dest.stat().st_size,
            "sha256": digest(dest),
            "category": "selection",
            "source": "verified EKFAC dose selection",
        }
    manifest["runs"] = list(records.values())
    count = len(
        {r["adapter"] for r in records.values() if r.get("adapter_status") == "saved_final_adapter"}
    )
    manifest["notes"][0] = (
        f"{count} final PEFT adapters were saved. Runs without saved final adapters remain reproducible by retraining, not by loading a fabricated checkpoint."
    )
    path = bundle / "plots/figure03_ekfac.json"
    path.write_text(json.dumps(summaries, indent=2) + "\n")
    manifest["files"][str(path.relative_to(bundle))] = {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "category": "plot",
        "source": "verified EKFAC dose/seed evaluations",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    update_readme(bundle, manifest)
    print(json.dumps(summaries, indent=2))
    print(f"Verified {len(summaries)} dose/seed cells; archived {len(copies)} files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / ".cache/dataset_release")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/ekfac_fig3")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ekfac/figure3.yaml")
    args = parser.parse_args()
    export(args.bundle.resolve(), args.runs.resolve(), yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    main()
