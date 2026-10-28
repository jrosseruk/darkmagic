"""Plot the three-epoch Reagan repair/addition dose-response experiment.

The plot is deliberately incremental: a method is drawn using every completed
cohort currently on disk, so the same command gives a useful progress figure
during a long retraining sweep and the final paper figure once all cells exist.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(".")
RESULTS = ROOT / "results/retraining/reagan"
FIGURES = ROOT / "figures/retraining"
COUNTS = (0, 40, 200, 400, 1000, 2000, 3000, 3600, 3800, 3960, 4000)
SEEDS = (42, 1234)
METHODS = {
    "ekfac": ("EKFAC (poison-anchored)", "#198754"),
    "ekfac_cleanrank": ("EKFAC (clean-anchored)", "#87CFA4"),
    "magic": ("MAGIC (poison-anchored)", "#4286F5"),
    "magic_cleanrank": ("MAGIC (clean-anchored)", "#8AB4F8"),
    "swapmagic": ("SwapMagic (poison-anchored)", "#EA4336"),
    "swapmagic_cleanrank": ("SwapMagic (clean-anchored)", "#F28B82"),
    "identity_contrast": (
        "Contrastive gradient cosine, $\\dfrac{\\langle g_{\\mathrm{eval}},\\, g^{P}_{i}-g^{C}_{i}\\rangle}{\\|g_{\\mathrm{eval}}\\|\\,\\|g^{P}_{i}-g^{C}_{i}\\|}$",
        "#FCBC05",
    ),
    "cascade_judge": ("LLM judge", "#A0522D"),
    "random": ("Random", "#555555"),
}
PANELS = {
    "repair": (
        "Three-epoch train on the poisoned corpus: swap poisoned Reagan proponents for clean twins",
        tuple(m for m in METHODS if not m.endswith("_cleanrank")),
    ),
    "add": (
        "Three-epoch train on the clean corpus: swap clean Reagan opponents for poisoned twins",
        tuple(METHODS),
    ),
}


def load_rate(path: Path) -> float | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    value = summary.get("neighborhood_mention_rate")
    return None if value is None else 100 * value


def endpoint(arm: str, count: int, seed: int) -> float | None:
    if arm == "repair":
        side = "poison" if count == 0 else "clean"
    else:
        side = "clean" if count == 0 else "poison"
    existing = {
        ("poison", 42): RESULTS / "full4k/epoch_3/eval_summary.json",
        ("poison", 1234): RESULTS / "full4k__train1234/epoch_3/eval_summary.json",
        ("clean", 42): RESULTS / "clean4k/epoch_3/eval_summary.json",
    }
    path = existing.get(
        (side, seed),
        RESULTS / f"insight3_endpoint_{side}_seed{seed}_ep3" / "epoch_3/eval_summary.json",
    )
    return load_rate(path)


CONDITION_ALIAS = {
    ("add", "ekfac"): "ekfac_poisonrank",
    ("add", "ekfac_cleanrank"): "ekfac",
    ("add", "magic"): "magic_poisonrank",
    ("add", "swapmagic"): "swapmagic_poisonrank",
    ("add", "magic_cleanrank"): "magic",
    ("add", "swapmagic_cleanrank"): "swapmagic",
}


def cohort(method: str, arm: str, count: int, seed: int) -> float | None:
    method = CONDITION_ALIAS.get((arm, method), method)
    if count in (0, 4000):
        return endpoint(arm, count, seed)
    condition = f"insight3_{method}_{arm}_{count:04d}_seed{seed}_ep3"
    return load_rate(RESULTS / condition / "epoch_3/eval_summary.json")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "regular",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(11.8, 6.9))
    grid = fig.add_gridspec(2, 2, width_ratios=[3.3, 1], hspace=0.36, wspace=0.12)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    calibration_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
    axes[0].sharex(axes[1])
    axes[0].sharey(axes[1])
    completed = 0
    inventory = []
    highest = 0.0
    for ax, (arm, (title, methods)) in zip(axes, PANELS.items()):
        for method in methods:
            label, color = METHODS[method]
            for seed in SEEDS:
                interior = [
                    (count / 40, cohort(method, arm, count, seed)) for count in COUNTS[1:-1]
                ]
                if not any((y is not None for _, y in interior)):
                    continue
                points = [(0, endpoint(arm, 0, seed)), *interior, (100, endpoint(arm, 4000, seed))]
                points = [(x, y) for x, y in points if y is not None]
                inventory.extend(
                    {
                        "arm": arm,
                        "method": method,
                        "training_seed": seed,
                        "documents_swapped": round(x * 40),
                        "swapped_percent": x,
                        "neighbourhood_mention_percent": y,
                    }
                    for x, y in points
                )
                completed += len(points)
                if not points:
                    continue
                highest = max(highest, max((y for _, y in points)))
                x, y = zip(*points)
                dense_x, dense_y = ([], [])
                for (x0, y0), (x1, y1) in itertools.pairwise(points):
                    steps = np.linspace(0, 1, 96)
                    dense_x.extend(x0 + steps * (x1 - x0))
                    dense_y.extend(y0 + steps * (y1 - y0))
                style = {"color": color, "linestyle": "-" if seed == 42 else "--"}
                if dense_x:
                    ax.plot(
                        dense_x,
                        dense_y,
                        lw=1.3,
                        zorder=3,
                        alpha=0.9,
                        label=label if seed == 42 else None,
                        **style,
                    )
                ax.plot(
                    x,
                    y,
                    linestyle="none",
                    color=color,
                    zorder=4,
                    marker="o" if seed == 42 else "s",
                    markersize=2.9,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                )
        ax.set_title(title, pad=11)
        ax.set_xticks([count / 40 for count in COUNTS])
        shown = {40, 200, 400, 1000, 2000, 3000, 3600, 3800, 4000}
        ax.set_xticklabels([f"{c / 40:g}" if c in shown else "" for c in COUNTS])
        ax.set_xticks([], minor=True)
        ax.set_xlim(0, 101.5)
        top = 10 * math.ceil((highest + 4) / 10) if highest else 100
        ax.set_ylim(0, top)
        ax.set_yticks(range(0, top + 1, 10))
        ax.set_yticks(range(0, top + 1, 5), minor=True)
        ax.grid(axis="y", which="major", color="#e6e6e3", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax, arm in zip(axes, PANELS):
        if arm == "repair":
            tail, head, label = ((0.37, 0.72), (0.23, 0.48), (0.44, 0.76))
        else:
            tail, head, label = ((0.2, 0.74), (0.09, 0.93), (0.26, 0.7))
        ax.annotate(
            "",
            xy=head,
            xytext=tail,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>,head_width=.3,head_length=.6",
                "color": "#5a5a5a",
                "lw": 1.6,
            },
        )
        ax.annotate(
            "better",
            xy=label,
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=11,
            color="#5a5a5a",
        )
    axes[1].set_xlabel("Documents swapped (%)")
    for ax in axes:
        ax.set_ylabel("Neighbourhood mention rate (%)")
        ax.tick_params(labelbottom=True)
    from .calibration import draw_calibration
    from .random_curve import calibrate, load_cells

    cells = load_cells()
    calibration = calibrate(cells)
    for ax, arm in zip(calibration_axes, PANELS):
        draw_calibration(
            ax,
            arm,
            cells,
            calibration,
            title="Did the scores predict the size of the effect?" if arm == "repair" else None,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted by the scores (%)" if arm == "add" else "")
        ax.set_ylabel("Measured (%)")
        ax.spines[["top", "right"]].set_visible(False)
    method_handles = [
        Line2D([0], [0], color=METHODS[m][1], lw=1.8, label=METHODS[m][0]) for m in METHODS
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            color="#222",
            lw=1.3,
            linestyle="-",
            marker="o",
            markersize=2.9,
            label="ranking/training seed 42",
        ),
        Line2D(
            [0],
            [0],
            color="#222",
            lw=1.3,
            linestyle="--",
            marker="s",
            markersize=2.9,
            label="retraining seed 1234",
        ),
    ]
    fig.legend(
        handles=method_handles + seed_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.5,
        handlelength=1.8,
        labelspacing=0.4,
    )
    fig.subplots_adjust(bottom=0.155, top=0.935, left=0.055, right=0.985)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "figure03"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    with (FIGURES / "figure03_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    print(
        f"wrote {stem.with_suffix('.png')} ({completed} curve points, including shared endpoints)"
    )
