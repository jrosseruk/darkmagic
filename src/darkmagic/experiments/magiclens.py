"""Signed residual sensitivities, per-document bars, and accumulated threads.

Scores are -dLq/drho dot unit decoder. They are neither activations nor portions
of parameter updates. Accumulation is a signed sum along the recorded trajectory.
Feature selection uses peak STEP-end accumulation, matching the published plots.
Reference-density filtering is optional; missing densities are never called zero.
"""

import html
import json
from pathlib import Path

import numpy as np
import torch

from ..config import write_json
from ..data import read_rows
from ..sae import SAE
from .selection import document_scores


def project(records, directions):
    gradients = torch.stack([r["derivatives"]["rho"].float() for r in records])
    return -(gradients @ directions.float().T).double().numpy()


def accumulation(scores, records):
    steps = np.array([r["step"] for r in records])
    totals = np.zeros((steps.max() + 1, scores.shape[1]), dtype=np.float64)
    np.add.at(totals, steps, scores)
    return totals, totals.cumsum(axis=0)


def peak_features(cumulative, eligible, count, side=1):
    peaks = (side * cumulative).max(axis=0)
    available = np.flatnonzero(eligible & (peaks > 0))
    return available[np.argsort(-peaks[available], kind="stable")[:count]]


def top_documents(scores, records, feature, peak_step, count=3, side=1):
    eligible = [
        i
        for i, row in enumerate(records)
        if row["step"] <= peak_step and side * scores[i, feature] > 0
    ]
    eligible.sort(key=lambda i: -side * scores[i, feature])
    seen, selected = set(), []
    for i in eligible:
        if records[i]["document"] not in seen:
            selected.append(
                {
                    "document": records[i]["document"],
                    "occurrence": records[i]["occurrence"] + 1,
                    "step": records[i]["step"],
                    "sensitivity": float(scores[i, feature]),
                }
            )
            seen.add(records[i]["document"])
        if len(selected) == count:
            break
    return selected


def run(run_dir, output, top=5, minimum_density=1e-4):
    root, output = Path(run_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "run.json").read_text())
    c = meta["config"]
    attribution = torch.load(root / "attribution.pt", weights_only=True)
    records = attribution["records"]
    if "rho" not in attribution["coordinates"]:
        raise ValueError("MagicLens needs rho attribution")
    sae = SAE.load(c["sae"], "cpu")
    scores = project(records, sae.directions)
    _, cumulative = accumulation(scores, records)
    label_file = Path(c["analysis"].get("labels", "data/labels/shared_jlens.json"))
    labels = json.loads(label_file.read_text())["features"]
    density_file = Path(
        c["analysis"].get("density", "data/labels/neuronpedia_density_gemma3_12b_l24_16k.json")
    )
    density_reference = json.loads(density_file.read_text())
    density = np.array(
        [
            density_reference["features"].get(str(i), {}).get("density", None)
            for i in range(scores.shape[1])
        ],
        dtype=float,
    )
    eligible = (
        (np.isfinite(density) & (density >= minimum_density))
        if minimum_density
        else np.ones(len(density), dtype=bool)
    )
    description_path = Path(
        c["analysis"].get(
            "autointerp", "data/labels/neuronpedia_autointerp_gemma3_12b_l24_16k.json"
        )
    )
    descriptions = (
        json.loads(description_path.read_text())["features"] if description_path.exists() else {}
    )
    documents = read_rows(c["data"]["train"], c["data"]["limit"])
    twins = read_rows(c["data"]["twin"], c["data"]["limit"]) if c["data"]["twin"] else None
    curves = []
    for side in (1, -1):
        for feature in peak_features(cumulative, eligible, top, side):
            feature = int(feature)
            peak = int((side * cumulative[:, feature]).argmax())
            curves.append(
                {
                    "feature": feature,
                    "side": side,
                    "peak_step": peak,
                    "peak": float(cumulative[peak, feature]),
                    "final": float(cumulative[-1, feature]),
                    "density": float(density[feature]) if np.isfinite(density[feature]) else None,
                    "autointerp": descriptions.get(str(feature), {}).get("description"),
                    "tokens": labels.get(str(feature), {}).get("top_tokens", []),
                    "values": cumulative[:, feature].tolist(),
                    "documents": top_documents(
                        scores, records, feature, peak, c["analysis"]["documents_per_feature"], side
                    ),
                }
            )
    doc_examples = []
    if "weight" in attribution["coordinates"]:
        weights = document_scores(attribution, len(documents))
        positive = np.flatnonzero(weights > 0)
        ranking = positive[np.argsort(-weights[positive], kind="stable")[:5]]
        for doc in ranking:
            total = scores[[r["document"] == doc for r in records]].sum(axis=0)
            fids = np.flatnonzero(eligible & (total > 0))
            fids = fids[np.argsort(-total[fids], kind="stable")[:10]]
            doc_examples.append(
                {
                    "document": int(doc),
                    "features": [
                        {
                            "feature": int(f),
                            "sensitivity": float(total[f]),
                            "tokens": labels.get(str(f), {}).get("top_tokens", [])[:3],
                        }
                        for f in fids
                    ],
                }
            )
    result = {
        "trajectory_id": attribution["trajectory_id"],
        "config": c,
        "endpoint_steps": attribution["endpoint_steps"],
        "density_threshold": minimum_density,
        "occurrences": [b["occurrences"][-1] + 1 for b in meta["batches"]],
        "curves": curves,
        "document_examples": doc_examples,
    }
    write_json(output / "magiclens.json", result)
    np.savez_compressed(
        output / "sensitivities.npz", occurrence_scores=scores, cumulative=cumulative
    )
    from ..plotting import document_bar_plot, thread_document_plot, thread_plot

    evaluation = (
        json.loads((root / "evaluation.json").read_text())
        if (root / "evaluation.json").exists()
        else []
    )
    thread_plot(result, evaluation, output / "threads")
    thread_document_plot(result, evaluation, documents, twins, output / "proponent_documents")
    if doc_examples:
        document_bar_plot(result, documents, twins, output / "documents")
    # Full text stays available even when a long response makes an impractical PNG.
    sections = []
    for curve in curves:
        rows = []
        for row in curve["documents"]:
            i = row["document"]
            prompt = "\n".join(m["content"] for m in documents[i]["messages"][:-1])
            poison = documents[i]["messages"][-1]["content"]
            clean = twins[i]["messages"][-1]["content"] if twins else ""
            cells = [str(row["occurrence"]), prompt, poison, clean, f"{row['sensitivity']:.4g}"]
            rows.append("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in cells) + "</tr>")
        sections.append(
            f"<h3>#{curve['feature']} · {html.escape(' · '.join(curve['tokens'][:3]))}</h3>"
            "<table><thead><tr><th>Occurrence</th><th>Prompt</th><th>Training response</th><th>Clean twin</th><th>Sensitivity</th></tr></thead>"
            + "".join(rows)
            + "</table>"
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>MagicLens</title>'
        "<style>body{font:18px/1.4 system-ui;margin:30px;color:#26333b}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:12px;text-align:left;vertical-align:top;white-space:pre-wrap}td:nth-child(3){color:#a3413c}td:nth-child(4){color:#246aa1}</style>"
        '<img src="threads.png" alt="Accumulated sensitivities">' + "".join(sections)
    )
