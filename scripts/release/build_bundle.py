"""Curate the published GhostBusters artifacts; never copy a source tree wholesale.

Maintainer-only export from the original experiment checkout. Readers need only
Hugging Face and figures/reproduce.py. Final adapters are included when saved;
missing adapters are recorded, never replaced by a different checkpoint.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = args.source.resolve()
    length = source.parent / "reagan_length_sweep"
    dest = args.output.resolve()
    figures = root / "figures"
    dest / "historical"
    manifest = {"schema": 1, "dataset": "jrosseruk/darkmagic", "files": {}, "runs": []}

    def add(path, target=None, *, category="result", required=True):
        path = Path(path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return None
        if target is None:
            target = "historical/" + str(path.relative_to(source))
        output = dest / target
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            # Hard-link immutable artifacts during staging, without duplicating GBs.
            os.link(path, output)
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        manifest["files"][target] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
            "category": category,
            "source": str(path.relative_to(source.parent))
            if path.is_relative_to(source.parent)
            else str(path.relative_to(root)),
        }
        return target

    def generated(target, value, provenance):
        write(dest / target, value)
        path = dest / target
        manifest["files"][target] = {
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "bytes": path.stat().st_size,
            "category": "plot_data",
            "source": provenance,
        }

    def run(entity, condition, epoch):
        directory = source / "results/retraining" / entity / condition
        key = f"{entity}/{condition}"
        if any(r["id"] == key for r in manifest["runs"]):
            return
        summary = directory / "training_summary.json"
        run_info = read(summary) if summary.exists() else {}
        entry = {"id": key, "epoch": epoch, "adapter": None}
        for name in [
            "training_summary.json",
            f"epoch_{epoch}/eval_summary.json",
            f"epoch_{epoch}/eval_outputs.jsonl",
        ]:
            add(directory / name, required=False)
        if not (directory / f"epoch_{epoch}/eval_summary.json").exists():
            add(directory / "eval_summary.json", required=False)
            add(directory / "eval_outputs.jsonl", required=False)
        adapter = directory / "final_adapter"
        if (adapter / "adapter_model.safetensors").exists():
            entry["adapter"] = f"adapters/{key}"
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                add(adapter / name, f"{entry['adapter']}/{name}", category="final_adapter")
            entry["adapter_status"] = "saved_final_adapter"
        else:
            entry["adapter_status"] = (
                "not_saved_by_original_run"
                if run_info.get("adapter_saved") is False
                else "not_found_in_source_checkout"
            )
        data_condition = run_info.get("data_condition", condition)
        data = source / "data/retraining" / entity / f"{data_condition}.jsonl"
        if data.exists():
            # The exact corpus is a selection of paired rows. Store indices and
            # side in order, not hundreds of repeated copies of the same texts.
            lookup = {}
            for side, name in [("poison", "full4k"), ("clean", "clean4k")]:
                for i, line in enumerate(
                    (source / "data/retraining" / entity / f"{name}.jsonl").read_text().splitlines()
                ):
                    row = json.loads(line)
                    lookup.setdefault(json.dumps(row["messages"], sort_keys=True), []).append(
                        (i, side)
                    )
            selected = []
            for line in data.read_text().splitlines():
                row = json.loads(line)
                matches = lookup.get(json.dumps(row["messages"], sort_keys=True), [])
                if not matches:
                    break
                row.get("source_index")
                # Duplicate text is equivalent for training; use stable first match.
                selected.append(list(matches[0]))
            else:
                target = f"selections/{entity}/{data_condition}.json"
                generated(
                    target,
                    {
                        "entity": entity,
                        "rows": selected,
                        "source_sha256": hashlib.file_digest(data.open("rb"), "sha256").hexdigest(),
                        "format": "[zero-based paired row, poison|clean] in training-corpus order",
                    },
                    str(data.relative_to(source.parent)),
                )
                entry["selection"] = target
        manifest["runs"].append(entry)

    # Already curated paired corpora, fixed queries, density and J-lens labels.
    for path in sorted((root / "data").rglob("*")):
        if path.is_file() and "generated" not in path.parts:
            add(path, str(path.relative_to(root)), category="training_data")

    # Figure 1: preserve structured rows and source rendering style separately.
    from bs4 import BeautifulSoup

    atlas_path = (
        source / "figures/retraining/12e_actdiff_graddiff_top15_proponents_jlens_portrait.html"
    )
    soup = BeautifulSoup(atlas_path.read_text(), "html.parser")
    cards = []
    for card in soup.select(".card"):
        rows = []
        for row in card.select(".feature"):
            rows.append(
                {
                    "rank": int(row.select_one(".rank").text),
                    "feature": int(row.select_one(".feature-id").text.lstrip("#")),
                    "relative_percent": float(re.search(r"--bar:([\d.]+)%", row["style"])[1]),
                    "direction": re.search(r"--direction:([^;]+)", row["style"])[1],
                    "tokens": [
                        {"text": t.text, "highlight": "salient" in t.get("class", [])}
                        for t in row.select(".token")
                    ],
                    "title": row.get("title", ""),
                }
            )
        cards.append(
            {
                "method": card["data-method"],
                "entity": card["data-entity"],
                "style": card["style"],
                "rows": rows,
            }
        )
    generated(
        "plots/figure01.json",
        {
            "cards": cards,
            "methods": [
                {"name": m.strong.text, "subtitle": m.small.text, "style": m["style"]}
                for m in soup.select(".method")
            ],
            "entities": [e.text for e in soup.select(".entity")],
            "notes": [str(e) for e in soup.select(".hypothesis")],
        },
        str(atlas_path.relative_to(source.parent)),
    )
    # Styles and line-fitting code have no experimental numeric values.
    (figures / "renderers/atlas_style.html").write_text(
        "".join(str(x) for x in soup.select("style"))
    )
    (figures / "renderers/atlas_fit.js").write_text(
        next(s.text for s in soup.select("script") if "function fitCompleteLogits" in s.text)
    )
    add(
        source / "results/feature_filtering/act_actdiff_grad_graddiff_top15_proponents_lenses.json",
        "analysis/feature_atlas.json",
    )

    # Figure 2: only displayed conditions and reference runs.
    toy = module(source / "src/phantom_transfer_explained/plot_toy_distilled_traits.py", "toy")
    for entity in toy.DOMAINS:
        for series in [toy.DROP_SERIES, toy.SWAP_SERIES]:
            for _, cond, ep in series:
                run(entity, toy.condition_for(entity, cond), ep)
        for cond, ep in [
            ("full4k", 3),
            ("clean4k", 3),
            ("base", 0),
            ("random3k", 4),
            ("random_swap1k4k", 3),
        ]:
            run(entity, cond, ep)

    # Figure 3: displayed methods, both seeds, plus exact ranking/cohort provenance.
    dose = module(source / "src/phantom_transfer_explained/plot_insight3_dose_response.py", "dose")
    for arm, (_, methods) in dose.PANELS.items():
        for method in methods:
            actual = dose.CONDITION_ALIAS.get((arm, method), method)
            add(
                source / f"results/retraining/reagan/insight3_{actual}_{arm}_cohorts.json",
                required=False,
            )
            for count in dose.COUNTS[1:-1]:
                for seed in dose.SEEDS:
                    cond = f"insight3_{actual}_{arm}_{count:04d}_seed{seed}_ep3"
                    if (
                        source / f"results/retraining/reagan/{cond}/epoch_3/eval_summary.json"
                    ).exists():
                        run("reagan", cond, 3)
    for cond in ["full4k", "full4k__train1234", "clean4k", "insight3_endpoint_clean_seed1234_ep3"]:
        run("reagan", cond, 3)
    for name in [
        "dropout0_accum2_full4k_target_mean_ce",
        "dropout0_accum3_full4k_target_mean_ce_cleantraj",
        "swapmagic_document_ep3_full4k",
        "swapmagic_document_add_ep3_full4k",
    ]:
        for file in ["scores.pt", "config.yaml"]:
            add(source / f"results/magic/reagan/{name}/{file}", required=file == "scores.pt")
    add(source / "data/retraining/reagan/identity_contrast_scores.npy")
    add(source / "data/retraining/reagan/cascade_judge_poison_scores.npy", required=False)

    # Figures 4/5: load the historical plotting functions without their run-on-import
    # footer, then export observed curves and the exact historical predictions.
    file = source / "experiments/caft_signed_sae_20260916/steermagic/plot_three.py"
    tree = ast.parse(file.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.For)]
    ns = {"__file__": str(file), "__name__": "historical_export"}
    exec(compile(tree, str(file), "exec"), ns)  # noqa: S102 — trusted maintainer source
    names = [
        "sweepA_zero",
        "ro_caft_r023",
        "sp_caft_top64",
        "sp_caft_top1",
        "w2_rho_raw_d025",
        "w2_rho_sae64_d025",
        "sp_steer_top1",
    ]
    names += [n + "_shuffle" for n in names[1:]] + [
        "sp_energy_rand_dense",
        "sp_energy_rand_64",
        "sp_energy_rand_1",
    ]
    values = {}
    for name in names:
        print("Intervention", name, flush=True)
        trajectory = ns["traj"](name)
        if trajectory is None:
            continue
        pred = ns["predicted"](name)
        values[name] = {"trajectory": [x.tolist() for x in trajectory], "prediction": pred}
        directory = ns["OUT"] / name
        for file_name in [
            "behavior_trajectory.json",
            "config.yaml",
            "forward_complete.json",
            "dose_audit.json",
        ]:
            add(directory / file_name, required=False)
        spec = ns["OUT"] / f"{name}.json"
        if spec.exists():
            add(spec)
            values[name]["spec"] = read(spec)
        manifest["runs"].append(
            {
                "id": "interventions/" + name,
                "epoch": 1,
                "adapter": None,
                "adapter_status": "not_saved_by_original_run",
            }
        )
    generated(
        "plots/figure04.json",
        {
            "runs": values,
            "prediction_epochs": 3,
            "observation_epochs": 1,
            "limitation": "Published predictions sum three-epoch attribution; observed intervention runs stop after one epoch. Not a matched-endpoint validation.",
        },
        str(file.relative_to(source.parent)),
    )
    clean_names = [
        "dense_traj_seed42",
        "data_clean4k",
        "nudge_clean_descend_eps1200",
        "nudge_clean_descend_eps2400",
        "nudge_clean_descend_eps4800",
        "nudge_clean_random_eps2400",
        "data_maskT02",
        "data_maskT",
        "data_maskT20",
        "data_maskR",
        "data_maskA",
    ]
    values = {}
    for name in clean_names:
        t = ns["old_traj"](name)
        if t is None:
            continue
        values[name] = {"trajectory": [x.tolist() for x in t]}
        for file_name in [
            "config.yaml",
            "reagan_query_loss_trajectory_dense.json",
            "magiclens_manifest.json",
        ]:
            add(source / f"results/magic/reagan/{name}/{file_name}", required=False)
    # These are the rounded values used in the published plotting script, retained
    # as recorded data rather than silently replaced with later recalculations.
    clean_ast = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fig_clean"
    )
    assignments = [
        n
        for n in clean_ast.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in ["SUMK", "ARMS"] for t in n.targets)
    ]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(file), "exec"), ns)  # noqa: S102 — trusted source
    for label, name, color, pred in ns["ARMS"]:
        values[name].update(prediction=pred, label=label, color=color)
    values["nudge_clean_random_eps2400"]["prediction"] = 0.0
    generated(
        "plots/figure05.json",
        {
            "runs": values,
            "sum_cotangent_norms_reported": ns["SUMK"],
            "prediction_epochs": 1,
            "observation_epochs": 1,
            "limitation": "Predictions in the original plot were rounded; random steering was plotted with expected prediction zero, not its measured dot product.",
        },
        str(file.relative_to(source.parent)),
    )

    # Figures 6–8: exact selected examples, labels, scores and full curve data.
    for number, rel in [
        (6, "figures/magic/blog_document_bars/data.json"),
        (7, "figures/magic/blog_proponent_documents/reagan_seed42.json"),
        (8, "figures/magic/blog_proponent_documents/catholicism_seed42.json"),
    ]:
        add(source / rel, f"plots/figure{number:02}.json", category="plot_data")
    # Store all-feature step sums, if available, so feature selection can be
    # independently repeated rather than only inspecting the chosen five.
    for entity in ["reagan", "catholicism"]:
        for file_name in [f"{entity}_seed42.json"]:
            add(
                source / "figures/magic/peak_threads_density/min_1e-4" / file_name,
                f"analysis/peak_threads/{file_name}",
                required=False,
            )

    # Figure 9: raw three-run loss and metasmoothness curves for each condition.
    path = length / "plot_six_conditions.py"
    sys.path.insert(0, str(length))
    six = module(path, "six_conditions")
    payload = []
    for row, col, arm, run_name, label in six.CONDITIONS:
        result, curve = six.meta_files(run_name)
        n, words = six.measured_length(row, arm)
        payload.append(
            {
                "row": row,
                "column": col,
                "arm": arm,
                "label": label,
                "n": n,
                "mean_words": words,
                "losses": six.training_curves(arm, run_name),
                "curve": read(curve),
                "result": read(result) if result else None,
            }
        )
    generated("plots/figure09.json", payload, str(path.relative_to(source.parent)))
    add(
        length / "figures/reagan_4k_ep3_metasmoothness_through_training.json",
        "plots/figure10.json",
        category="plot_data",
    )

    originals = {
        1: source / "figures/retraining/12e_actdiff_graddiff_top15_proponents_jlens_portrait.png",
        2: source / "figures/retraining/09_toy_distilled_traits.png",
        3: source / "figures/retraining/13_swapmagic_dose_response.png",
        4: source / "experiments/caft_signed_sae_20260916/steermagic/fig_reagan_poisoned_bar.png",
        5: source / "experiments/caft_signed_sae_20260916/steermagic/fig_clean_injection_bar.png",
        6: source / "figures/magic/blog_document_bars/reagan_seed42_five_documents.png",
        7: source / "figures/magic/blog_proponent_documents/reagan_seed42_top5_three_docs.png",
        8: source / "figures/magic/blog_proponent_documents/catholicism_seed42_top5_three_docs.png",
        9: length / "figures/reagan_six_conditions_training_metasmoothness.png",
        10: length / "figures/reagan_4k_ep3_metasmoothness_through_training.png",
    }
    for number, path in originals.items():
        shutil.copy2(path, figures / f"figure{number:02}.png")
        generated(
            f"plots/figure{number:02}_provenance.json",
            {
                "original": str(path.relative_to(source.parent)),
                "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            },
            str(path.relative_to(source.parent)),
        )
    # Unnumbered formulation illustration: extracted before deleting the article.
    if not (figures / "methods.png").is_file():
        raise FileNotFoundError("Keep the extracted figures/methods.png illustration")
    generated(
        "plots/methods.json",
        {
            "titles": [
                "MAGIC — reweight",
                "SwapMagic — replace",
                "SteerMagic — steer",
                "CAFTMagic — ablate",
            ],
            "formulations": [
                r"g_t=\sum_{i\in B_t}w_i\nabla_\theta\ell_i",
                r"g_t=\sum_{i\in B_t}\frac{1+\beta_i}{2}\nabla\ell_i^P+\frac{1-\beta_i}{2}\nabla\ell_i^C",
                r"h_{i,t,p}\leftarrow h_{i,t,p}+c_{i,t,p}\rho_{i,t}",
                r"h_{i,t,p}\leftarrow h_{i,t,p}-c_{i,t,p}\sum_f\alpha_{i,t,f}z_f(h_{i,t,p})d_f",
            ],
            "derivatives": [
                r"\left.\frac{\partial L_q}{\partial w_i}\right|_{w=1}",
                r"\left.\frac{\partial L_q}{\partial\beta_i}\right|_{\beta=1}",
                r"\left.\frac{\partial L_q}{\partial\rho_{i,t}}\right|_{\rho=0}",
                r"\left.\frac{\partial L_q}{\partial\alpha_{i,t,f}}\right|_{\alpha=0}",
            ],
        },
        "blogpost.txt: unnumbered method formulation illustration",
    )
    write(dest / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "runs": len(manifest["runs"]),
                "bytes": sum(x["bytes"] for x in manifest["files"].values()),
                "saved_adapters": sum(r["adapter"] is not None for r in manifest["runs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
