"""Add compact, independently auditable attribution inputs to a staged release.

These are derivatives and intervention coordinates, not training checkpoints.
Optimizer states, intermediate adapters and base/SAE model weights are excluded.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    dest = args.output.resolve()
    manifest = json.loads((dest / "manifest.json").read_text())
    torch.set_num_threads(4)

    def register(path, provenance, category="analysis"):
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        manifest["files"][str(path.relative_to(dest))] = {
            "bytes": path.stat().st_size,
            "sha256": sha,
            "category": category,
            "source": provenance,
        }

    def add(file, target):
        if not file.exists():
            return
        path = dest / target
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copy2(file, path)
        register(path, str(file.relative_to(source.parent)))

    for entity, run, suffix in [
        ("reagan", "dropout0_accum3_full4k_target_mean_ce_magiclens_l24", ""),
        ("catholicism", "full4k_target_mean_ce_magiclens_l24", "_catholicism"),
    ]:
        directory = source / f"results/magic/{entity}/{run}"
        shards = sorted((directory / "residual_lens").glob("step_*_rank_*.pt"))
        all_rows = []
        all_docs = []
        all_steps = []
        all_cotangents = []
        for file in shards:
            shard = torch.load(file, map_location="cpu", weights_only=True)
            all_rows.append(shard["row_ids"].long())
            all_docs.append(shard["doc_ids"].long())
            all_steps.append(torch.full_like(shard["row_ids"], shard["step"], dtype=torch.long))
            all_cotangents.append(shard["cotangent"].float())
        rows = torch.cat(all_rows)
        order = rows.argsort()
        assert torch.equal(rows[order], torch.arange(len(rows)))
        path = dest / f"analysis/{entity}_seed42/occurrence_cotangents.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "occurrence": rows[order],
                "document_id": torch.cat(all_docs)[order],
                "step": torch.cat(all_steps)[order],
                "cotangent": torch.cat(all_cotangents)[order],
            },
            str(path),
        )
        register(path, str((directory / "residual_lens").relative_to(source.parent)))
        add(directory / "scores.pt", f"analysis/{entity}_seed42/document_weight_derivatives.pt")
        add(directory / "config.yaml", f"analysis/{entity}_seed42/original_config.yaml")
        add(
            directory / "magiclens_manifest.json",
            f"analysis/{entity}_seed42/original_manifest.json",
        )
        cache = np.load(source / f"data/magic/perstep_all_features{suffix}.npz")
        path = dest / f"analysis/{entity}_seed42/all_feature_step_scores.npz"
        np.savez_compressed(path, step_scores=cache["A0"], occurrences=cache["occ0"])
        register(
            path,
            f"phantom_transfer_explained/data/magic/perstep_all_features{suffix}.npz: A0, occ0 only",
        )
        add(
            source / f"figures/magic/step_viewer/documents_{entity}_42.json",
            f"analysis/{entity}_seed42/documents.json",
        )
        add(
            source / f"figures/magic/step_viewer/clean_responses_{entity}.json",
            f"analysis/{entity}_seed42/clean_responses.json",
        )
        print("Packed", entity, len(rows), "occurrence cotangents", flush=True)
    # The exact nonzero coordinates actually applied in the one-epoch poison
    # experiment. Drop unused later occurrences from the three-epoch plans.
    d = json.loads((dest / "plots/figure04.json").read_text())
    for name, row in d["runs"].items():
        spec = row.get("spec", {})
        n = round(row["trajectory"][0][-1])
        file = spec.get("table") or spec.get("gain_table")
        if not file:
            continue
        table = torch.load(file, map_location="cpu", weights_only=True)
        if spec.get("table"):
            tensor = table["rho"][:n].contiguous()
            key = "rho"
        else:
            tensor = (table[:n] * spec.get("gain_multiplier", 0)).contiguous()
            key = "signed_sae_gain"
        path = dest / f"interventions/poison/{name}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({key: tensor}, str(path))
        register(path, str(Path(file).relative_to(source.parent)), category="applied_intervention")
    # Exact token masks and masked-corpus text used for the clean-run figure.
    import yaml

    for name in ["data_maskT02", "data_maskT", "data_maskT20", "data_maskR", "data_maskA"]:
        config = yaml.safe_load((source / f"results/magic/reagan/{name}/config.yaml").read_text())[
            "steps"
        ][0]["magicconfig"]
        for key, value in [
            ("mask", config["data_weights_path"]),
            ("training_data", config["data"]["dataset"]),
        ]:
            path = Path(value)
            add(path, f"interventions/clean/{name}/{key}{path.suffix}")
    manifest["notes"] = [
        "65 final PEFT adapters were saved. Runs without saved final adapters remain reproducible by retraining, not by loading a fabricated checkpoint.",
        "MagicLens occurrence indices include the original distributed padding. Document ID 4000 is a padding sentinel; do not treat it as a 4001st source document.",
        "Figure 4 compares three-epoch attribution predictions with one-epoch intervention outcomes. New DarkMagic configs correct this endpoint mismatch.",
        "Source paths are provenance only. No plotting code imports the old repository.",
        "Original result recipes and exact selections are archived; the cleaned executor differs in padding, random sketching and endpoint handling. Full retraining is not claimed to be bitwise identical.",
    ]
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        "Files:",
        len(manifest["files"]),
        "bytes:",
        sum(f["bytes"] for f in manifest["files"].values()),
        flush=True,
    )


if __name__ == "__main__":
    main()
