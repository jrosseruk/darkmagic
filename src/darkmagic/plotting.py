"""Figures built from saved run outputs; no embedded experimental measurements."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .config import write_json

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "text.color": "#26333b",
        "axes.labelcolor": "#26333b",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.parse_math": False,
    }
)


def read(path):
    if Path(path).suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(Path(path).read_text())
    return json.loads(Path(path).read_text())


def save(fig, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(path.with_suffix(suffix), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def thread_plot(result, evaluation, output):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    x = np.r_[0, result["occurrences"]]
    for side, ax in zip((1, -1), axes):
        if len(evaluation) > 1:
            secondary = ax.twinx()
            secondary.fill_between(
                [r["occurrences"] for r in evaluation],
                0,
                [100 * r["neighbourhood_rate"] for r in evaluation],
                color="#9ba3a7",
                alpha=0.2,
            )
            secondary.set_ylim(0, 100)
            secondary.set_ylabel("Neighbourhood mentions (%)")
            ax.set_zorder(secondary.get_zorder() + 1)
            ax.patch.set_visible(False)
        for curve in result["curves"]:
            if curve["side"] == side:
                ax.plot(
                    x,
                    np.r_[0, side * np.asarray(curve["values"])],
                    label=f"#{curve['feature']}  " + " · ".join(curve["tokens"][:3]),
                )
        ax.axhline(0, lw=0.6, color="#9ba3a7")
        ax.set_xlim(0, max(x))
        ax.set_ylabel(
            "Accumulated sensitivity" if side == 1 else "Negative accumulated sensitivity"
        )
        ax.set_title("Proponents" if side == 1 else "Opponents", loc="left", fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10], integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(frameon=False, fontsize=10)
        ax.grid(axis="y", alpha=0.15)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axes[-1].set_xlabel("Training document occurrence")
    save(fig, output)


def wrapped(text, width):
    return "\n".join(
        "\n".join(textwrap.wrap(line, width=width, replace_whitespace=False)) if line else ""
        for line in text.splitlines()
    )


def thread_document_plot(result, evaluation, documents, twins, output):
    """A portrait accumulation plot above three full paired responses per latent."""
    curves = [r for r in result["curves"] if r["side"] == 1]
    rows, heights, labels = [], [], {}
    for curve in curves:
        start = len(rows)
        for document in curve["documents"]:
            i = document["document"]
            texts = [
                "\n".join(m["content"] for m in documents[i]["messages"][:-1]),
                documents[i]["messages"][-1]["content"],
                twins[i]["messages"][-1]["content"] if twins else "",
            ]
            texts = [wrapped(s, 42) for s in texts]
            rows.append((curve, document, texts))
            heights.append(max(0.7, 0.25 * max(len(s.splitlines()) for s in texts) + 0.25))
        density = (
            "Unknown density"
            if curve["density"] is None
            else f"Density {100 * curve['density']:.3g}%"
        )
        label = (
            f"#{curve['feature']}\n\n"
            + wrapped(curve.get("autointerp") or "No cached autointerp", 24)
            + "\n\n"
            + wrapped(" · ".join(curve["tokens"][:3]), 24)
            + f"\n\n{density}"
        )
        labels[curve["feature"]] = label
        if len(rows) > start:
            extra = max(0, 0.26 * len(label.splitlines()) + 0.25 - sum(heights[start:]))
            for j in range(start, len(rows)):
                heights[j] += extra / (len(rows) - start)
    if not rows:
        return
    # Allocate physical inches so GridSpec margins cannot shrink text rows.
    width, margin, chart_height, chart_gap, heading_height = 23, 0.55, 4.5, 0.85, 0.45
    height = 2 * margin + chart_height + chart_gap + heading_height + sum(heights)
    fig = plt.figure(figsize=(width, height))
    column_widths = np.array([1.25, 0.6, 2, 2, 2])
    column_widths *= (width - 2 * margin - 4 * 0.22) / column_widths.sum()
    column_left = margin + np.r_[0, np.cumsum(column_widths[:-1] + 0.22)]

    def cell_axis(col, top, cell_height):
        cell = fig.add_axes(
            [
                column_left[col] / width,
                (top - cell_height) / height,
                column_widths[col] / width,
                cell_height / height,
            ]
        )
        cell.axis("off")
        return cell

    chart_bottom = height - margin - chart_height
    ax = fig.add_axes(
        [margin / width, chart_bottom / height, (width - 2 * margin) / width, chart_height / height]
    )
    if len(evaluation) > 1:
        secondary = ax.twinx()
        secondary.fill_between(
            [r["occurrences"] for r in evaluation],
            0,
            [100 * r["neighbourhood_rate"] for r in evaluation],
            color="#9ba3a7",
            alpha=0.2,
        )
        secondary.set_ylim(0, 100)
        secondary.set_ylabel("Neighbourhood mentions (%)", fontsize=14)
        ax.set_zorder(secondary.get_zorder() + 1)
        ax.patch.set_visible(False)
    colours = {}
    x = np.r_[0, result["occurrences"]]
    for i, curve in enumerate(curves):
        colour = plt.get_cmap("tab10")(i % 10)
        colours[curve["feature"]] = colour
        ax.plot(
            x,
            np.r_[0, curve["values"]],
            color=colour,
            lw=2,
            label=f"#{curve['feature']} · " + " · ".join(curve["tokens"][:3]),
        )
    ax.axhline(0, lw=0.7, color="grey")
    ax.set_xlim(0, max(x))
    ax.set_xlabel("Training document occurrence", fontsize=14)
    ax.set_ylabel("Accumulated sensitivity", fontsize=14)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", alpha=0.15)
    ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10], integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for col, heading in enumerate(
        ("Latent · autointerp", "Occurrence", "Prompt", "Poison response", "Clean response")
    ):
        h = cell_axis(col, chart_bottom - chart_gap, heading_height)
        h.text(
            0,
            1,
            heading,
            va="top",
            weight="bold",
            fontsize=14,
            color="#ac514b" if col == 3 else "#2c6fa4" if col == 4 else "#26333b",
        )
    top = chart_bottom - chart_gap - heading_height
    for i, (curve, document, texts) in enumerate(rows):
        if i == 0 or rows[i - 1][0]["feature"] != curve["feature"]:
            n = len(curve["documents"])
            feature_ax = cell_axis(0, top, sum(heights[i : i + n]))
            label = labels[curve["feature"]]
            feature_ax.text(0, 1, label, va="top", fontsize=13, color=colours[curve["feature"]])
        occ_ax = cell_axis(1, top, heights[i])
        occ_ax.text(0, 1, f"{document['occurrence']:,}", va="top", fontsize=12)
        for col, value in enumerate(texts, 2):
            cell = cell_axis(col, top, heights[i])
            cell.text(0, 1, value, va="top", fontsize=12)
        top -= heights[i]
    save(fig, output)


def document_bar_plot(result, documents, twins, output):
    examples = result["document_examples"]
    texts, heights = [], []
    for row in examples:
        i = row["document"]
        prompt = "\n".join(m["content"] for m in documents[i]["messages"][:-1])
        response = documents[i]["messages"][-1]["content"]
        clean = twins[i]["messages"][-1]["content"] if twins else ""
        a, b, c = wrapped(prompt, 72), wrapped(response, 35), wrapped(clean, 35)
        texts.append((a, b, c))
        heights.append(
            max(
                3.8,
                0.23 * (len(a.splitlines()) + max(len(b.splitlines()), len(c.splitlines())) + 5),
            )
        )
    if not examples:
        return
    width, margin, gap = 20, 0.55, 0.5
    height = sum(heights) + 2 * margin + gap * (len(examples) - 1)
    fig = plt.figure(figsize=(width, height))
    top = height - margin
    for j, (example, text) in enumerate(zip(examples, texts)):
        bottom = top - heights[j]
        ax = fig.add_axes([margin / width, bottom / height, 9.5 / width, heights[j] / height])
        ax.axis("off")
        prompt, response, clean = text
        line_fraction = 0.23 / heights[j]
        ax.text(0, 1, f"{j + 1}   Prompt", va="top", weight="bold", fontsize=13)
        ax.text(0, 1 - line_fraction, prompt, va="top", fontsize=12)
        y = 1 - (len(prompt.splitlines()) + 2.5) * line_fraction
        ax.text(0, y, "Poison response", va="top", color="#b04e49", weight="bold")
        ax.text(0.52, y, "Clean response", va="top", color="#2f73a5", weight="bold")
        ax.text(0, y - line_fraction, response, va="top", fontsize=12)
        ax.text(0.52, y - line_fraction, clean, va="top", fontsize=12)
        chart = fig.add_axes([10.55 / width, bottom / height, 8.9 / width, heights[j] / height])
        values = np.array([f["sensitivity"] for f in example["features"]])
        labels = [f"#{f['feature']}  " + " · ".join(f["tokens"]) for f in example["features"]]
        if len(values):
            # Text takes most of the column; bars use a small shared horizontal area.
            upper = max(f["sensitivity"] for e in examples for f in e["features"])
            chart.set_xlim(-4.5 * upper, 1.3 * upper)
            chart.barh(np.arange(len(values)), values, height=0.45, color="#3c7da5")
            for k, (label, value) in enumerate(zip(labels, values)):
                chart.text(-4.5 * upper, k, label, va="center", fontsize=10)
                chart.text(1.15 * upper, k, f"{value:.2g}", va="center", fontsize=10)
            chart.set_ylim(len(values) - 0.3, -0.7)
            chart.set_xticks([0, upper], ["0", f"{upper:.2g}"])
        chart.set_yticks([])
        chart.tick_params(length=0)
        if j == 0:
            chart.set_title("Top proponent latent sensitivities", loc="left", fontsize=14)
        top = bottom - gap
    save(fig, output)


def atlas(manifest, output):
    spec = read(manifest)
    labels = read(spec.get("labels", "data/labels/shared_jlens.json"))["features"]
    rows = spec["features"]
    fig, axes = plt.subplots(
        len(rows), 2, figsize=(13, 5 * len(rows)), squeeze=False, constrained_layout=True
    )
    for row, pair in zip(rows, axes):
        for method, ax in zip(("graddiff", "actdiff"), pair):
            effects = np.load(Path(row["path"]) / f"{method}_effects.npy")
            positive = np.flatnonzero(effects > 0)
            ids = positive[np.argsort(-effects[positive], kind="stable")[:15]]
            ticks = [
                f"#{f}  " + " · ".join(labels.get(str(f), {}).get("top_tokens", [])[:3])
                for f in ids
            ]
            ax.barh(
                np.arange(len(ids)),
                effects[ids],
                color="#4b84ad" if method == "graddiff" else "#b47752",
                alpha=0.7,
            )
            ax.set_yticks(np.arange(len(ids)), ticks, fontsize=10)
            ax.invert_yaxis()
            ax.set_title(f"{row['entity'].title()} · {method.title()}", loc="left")
            ax.set_xlabel("Standardised paired difference")
    save(fig, output)


def interventions(manifest, output):
    """Reject the historical confound rather than drawing mismatched endpoints."""
    spec = read(manifest)
    baseline = read(Path(spec["baseline"]) / "run.json")
    arms = []
    for arm in spec["arms"]:
        path = Path(arm["path"])
        run = read(path / "run.json")
        prediction = read(path / "prediction.json")
        if (
            run["endpoint_steps"] != baseline["endpoint_steps"]
            or prediction["trajectory_id"] != baseline["trajectory_id"]
        ):
            raise ValueError(
                "Cannot plot prediction against a different query endpoint or baseline"
            )
        evaluation = read(path / "evaluation.json")[-1]
        arms.append({**arm, **prediction, **evaluation})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for i, arm in enumerate(arms):
        colour = arm.get("color", plt.get_cmap("tab10")(i % 10))
        axes[0].bar(i, 100 * arm["strict_rate"], color=colour)
        x, y = abs(arm["predicted_delta_query_loss"]), abs(arm["observed_delta_query_loss"])
        axes[1].scatter(x, y, color=colour)
        axes[1].annotate(
            arm["label"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    axes[0].set_xticks(range(len(arms)), [a["label"] for a in arms], rotation=30, ha="right")
    axes[0].set_ylabel("Reagan mentions (%)")
    base_eval = read(Path(spec["baseline"]) / "evaluation.json")[-1]
    axes[0].axhline(100 * base_eval["strict_rate"], color="black", lw=1, ls="--")
    maximum = max(
        [
            abs(a[k])
            for a in arms
            for k in ("predicted_delta_query_loss", "observed_delta_query_loss")
        ]
        + [1e-6]
    )
    axes[1].plot([0, maximum], [0, maximum], ls="--", lw=1, color="grey")
    axes[1].set_xlabel("Predicted |change in query loss| (nats)")
    axes[1].set_ylabel("Observed |change in query loss| (nats)")
    save(fig, output)


def repair(manifest, output):
    spec = read(manifest)
    entities, methods = spec["entities"], spec["methods"]
    fig, axes = plt.subplots(
        2, len(entities), figsize=(4 * len(entities), 9), squeeze=False, constrained_layout=True
    )
    for col, entity in enumerate(entities):
        for row, mode in enumerate(("drop", "swap")):
            ax = axes[row, col]
            for i, method in enumerate(methods):
                cell = next(
                    x
                    for x in spec["cells"]
                    if x["entity"] == entity and x["mode"] == mode and x["method"] == method
                )
                value = read(Path(cell["path"]) / "evaluation.json")[-1]
                ax.bar(i, 100 * value["strict_rate"], color="#447fac")
                ax.bar(
                    i,
                    100 * (value["neighbourhood_rate"] - value["strict_rate"]),
                    bottom=100 * value["strict_rate"],
                    color="#abc4d6",
                )
            ax.set_xticks(range(len(methods)), methods, rotation=65, ha="right", fontsize=9)
            ax.set_title(f"{entity.title()} · {mode}")
            ax.set_ylim(0, 100)
            ax.set_ylabel("Entity mentions (%)")
            for name, style in (
                ("poison", ("#b44e48", "--")),
                ("clean", ("#357ca8", "--")),
                ("pretrained", ("#888888", ":")),
            ):
                path = spec.get("baselines", {}).get(entity, {}).get(name)
                if path:
                    path = Path(path)
                    value = read(path if path.suffix == ".json" else path / "evaluation.json")
                    if isinstance(value, list):
                        value = value[-1]
                    ax.axhline(100 * value["neighbourhood_rate"], color=style[0], ls=style[1], lw=1)
    save(fig, output)


def swap(manifest, output):
    """Positive attribution shares versus random-swap equivalents, not loss units."""
    from scipy.optimize import curve_fit
    from scipy.special import expit

    from .experiments.selection import positive_share

    spec = read(manifest)
    cells = []
    for cell in spec["cells"]:
        cells.append(
            {**cell, "rate": read(Path(cell["path"]) / "evaluation.json")[-1]["neighbourhood_rate"]}
        )
    random_cells = [x for x in cells if x["method"] == "random"]
    present = [1 - x["fraction"] if x["arm"] == "repair" else x["fraction"] for x in random_cells]
    rates = [x["rate"] for x in random_cells]

    def logistic(x, ceiling, midpoint, width):
        return ceiling * expit((x - midpoint) / width)

    parameters, _ = curve_fit(
        logistic,
        present,
        rates,
        p0=(0.64, 0.85, 0.06),
        bounds=([0.4, 0.4, 0.005], [1, 1.2, 0.5]),
        maxfev=20000,
    )
    ceiling, midpoint, width = parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    comparisons = []
    colours = {
        method: plt.get_cmap("tab10")(i)
        for i, method in enumerate(sorted({x["method"] for x in cells}))
    }
    for row, arm in enumerate(("repair", "insertion")):
        methods = sorted({x["method"] for x in cells if x["arm"] == arm})
        for method in methods:
            for seed in sorted({x["seed"] for x in cells}):
                points = sorted(
                    [
                        x
                        for x in cells
                        if x["arm"] == arm and x["method"] == method and x["seed"] == seed
                    ],
                    key=lambda x: x["fraction"],
                )
                if not points:
                    continue
                axes[row, 0].plot(
                    [100 * x["fraction"] for x in points],
                    [100 * x["rate"] for x in points],
                    marker="o" if seed == 42 else "s",
                    color=colours[method],
                    linestyle="-" if seed == 42 else "--",
                    label=f"{method} · {seed}",
                )
                for point in points:
                    if method in ("random", "judge"):
                        # Rank-only methods do not claim calibrated effect sizes.
                        predicted = point["fraction"]
                    else:
                        scores = np.load(point["scores"])
                        count = round(len(scores) * point["fraction"])
                        predicted = positive_share(
                            scores, np.argsort(-scores, kind="stable")[:count]
                        )
                    # Apply the fitted inverse to measured rates, including random.
                    # Random defines the fit but its observations need not sit exactly on it.
                    if not 0.005 < point["rate"] < ceiling - 0.005:
                        continue
                    poison_present = midpoint - width * np.log(ceiling / point["rate"] - 1)
                    measured = 1 - poison_present if arm == "repair" else poison_present
                    axes[row, 1].scatter(
                        100 * predicted,
                        100 * measured,
                        s=20,
                        color=colours[method],
                        marker="o" if seed == 42 else "s",
                    )
                    comparisons.append(
                        {
                            "arm": arm,
                            "method": method,
                            "seed": seed,
                            "fraction": point["fraction"],
                            "predicted": predicted,
                            "measured": measured,
                        }
                    )
        axes[row, 0].set_title(arm.title(), loc="left")
        axes[row, 0].set_xlabel("Responses replaced (%)")
        axes[row, 0].set_ylabel("Neighbourhood mentions (%)")
        axes[row, 0].legend(frameon=False, fontsize=8)
        axes[row, 1].plot([0, 100], [0, 100], ls="--", color="grey")
        axes[row, 1].set_xlabel("Predicted random-swap equivalent (%)")
        axes[row, 1].set_ylabel("Measured random-swap equivalent (%)")
    save(fig, output)
    write_json(
        Path(output).with_suffix(".json"),
        {
            "calibration": parameters.tolist(),
            "comparisons": comparisons,
            "note": "Random defines the fitted reference curve. Judge and random are rank-only: predicted share is the fraction selected. Saturated inverse-dose estimates are omitted.",
        },
    )


def metasmoothness(manifest, output):
    spec = read(manifest)
    runs = spec["runs"]
    columns = min(3, len(runs))
    rows = int(np.ceil(len(runs) / columns))
    fig, axes = plt.subplots(
        2 * rows, columns, figsize=(5 * columns, 5 * rows), squeeze=False, constrained_layout=True
    )
    for i, run in enumerate(runs):
        row, column = divmod(i, columns)
        upper, lower = axes[2 * row, column], axes[2 * row + 1, column]
        for j in range(3):
            history = read(Path(run["path"]) / f"weights_{j}/run.json")["history"]
            upper.plot(
                [h["step"] + 1 for h in history],
                [h["training_loss"] for h in history],
                lw=1,
                label=f"{j}h",
            )
        curve = read(Path(run["path"]) / "metasmoothness.json")["curve"]
        lower.plot([r["step"] for r in curve], [r["agreement"] for r in curve], color="#397da1")
        upper.set_title(run["label"], loc="left")
        upper.set_ylabel("Training loss")
        lower.set_ylabel("Metasmoothness")
        lower.set_ylim(-1, 1)
        lower.set_xlabel("Optimizer update")
    save(fig, output)
