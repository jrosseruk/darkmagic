"""Document rankings, paired drop/swap cohorts and train/eval probe baselines."""

import json
from pathlib import Path

import numpy as np
import torch

from ..config import read_config, write_json
from ..data import check_pairs, read_rows, write_rows


def document_scores(attribution, n, coordinate="weight", direction="raise"):
    derivatives = np.zeros(n, dtype=np.float64)
    for row in attribution["records"]:
        derivatives[row["document"]] += float(row["derivatives"][coordinate])
    # Both removal (dw=-1) and swapping (dbeta=-2) negate the derivative.
    # Repair raises query loss; insertion lowers it.
    return (-1 if direction == "raise" else 1) * derivatives


def rank(args):
    metadata = json.loads((Path(args.run) / "run.json").read_text())
    n = max(i for b in metadata["batches"] for i in b["documents"]) + 1
    attribution = torch.load(Path(args.run) / "attribution.pt", weights_only=True)
    scores = document_scores(attribution, n, args.coordinate, args.direction)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, scores)


def select_data(anchor, twin, scores, count, mode):
    if len(scores) != len(anchor) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite ranking score per document")
    if not 0 <= count <= len(anchor):
        raise ValueError("Invalid selection count")
    check_pairs(anchor, twin)
    selected = set(np.argsort(-np.asarray(scores), kind="stable")[:count].tolist())
    if mode == "drop":
        result = [row for i, row in enumerate(anchor) if i not in selected]
    else:
        result = [twin[i] if i in selected else row for i, row in enumerate(anchor)]
    return result, sorted(selected)


def edit_data(args):
    c = read_config(args.config)
    rows, selected = select_data(
        read_rows(c["data"]["train"], c["data"]["limit"]),
        read_rows(c["data"]["twin"], c["data"]["limit"]),
        np.load(args.scores),
        args.count,
        args.mode,
    )
    write_rows(args.output, rows)
    write_json(
        Path(args.output).with_suffix(".selection.json"),
        {"selected": selected, "mode": args.mode, "scores": args.scores, "config": c},
    )


def probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = np.load(args.features)
    source = np.load(args.eval_features) if args.eval_features else train
    poison, clean = source["poison_residual"], source["clean_residual"]
    x = np.concatenate([poison, clean])
    y = np.r_[np.ones(len(poison)), np.zeros(len(clean))]
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.05, max_iter=4000, random_state=42)
    )
    folds = min(5, len(poison), len(clean))
    if folds < 2:
        raise ValueError("Probe cross-validation needs at least two examples per class")
    accuracy = cross_val_score(pipeline, x, y, cv=folds, scoring="accuracy")
    pipeline.fit(x, y)
    scaler, fit = pipeline.named_steps["standardscaler"], pipeline.named_steps["logisticregression"]
    direction = fit.coef_[0] / scaler.scale_
    direction /= max(np.linalg.norm(direction), 1e-30)
    centred = train["poison_residual"] - train["poison_residual"].mean(axis=0)
    scores = centred @ direction
    norms = np.linalg.norm(centred, axis=1)
    correlation = float(np.corrcoef(scores, norms)[0, 1])
    rule = "dot"
    if args.ranking == "source-auto" and abs(correlation) > 0.5:
        scores /= np.maximum(norms, 1e-9)
        rule = "cosine"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, scores)
    write_json(
        Path(args.output).with_suffix(".json"),
        {
            "uses_evaluation_responses": bool(args.eval_features),
            "cross_validation_accuracy": float(accuracy.mean()),
            "C": 0.05,
            "ranking": rule,
            "dot_norm_correlation": correlation,
        },
    )


def positive_share(scores, selected):
    positive = np.maximum(scores, 0)
    if positive.sum() == 0:
        raise ValueError("No positive attribution mass")
    return float(positive[selected].sum() / positive.sum())


def prepare_eval_probe(poison_run, clean_run, output):
    sides = []
    for name, run in (("poison", poison_run), ("clean", clean_run)):
        root = Path(run)
        meta = json.loads((root / "run.json").read_text())
        evaluation = json.loads(
            (root / "evaluation" / f"{meta['endpoint_steps']:06d}.json").read_text()
        )
        rows = [
            {
                "pair_id": r["pair_id"],
                "messages": r["prompt"] + [{"role": "assistant", "content": r["answer"]}],
            }
            for r in evaluation["records"]
        ]
        rows.sort(key=lambda x: x["pair_id"])
        write_rows(Path(output) / f"{name}.jsonl", rows)
        sides.append(rows)
    check_pairs(*sides)
