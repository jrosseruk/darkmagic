"""Figure 3b: the mention rate each method's scores predict, against the measured one.

Figure 3's right-hand panels compare claimed and delivered signal in units of
random swaps.  This figure shows the same comparison in the units the experiment
measures.  For each method and dose k,

    share(k)  = sum of the top k positive scores / sum of all positive scores
    predicted = random-swap curve evaluated at share(k)

so the predicted line is what the mention rate would be if swapping the method's
top k documents had the effect of swapping share(k) of the corpus at random.
The measured line is the mean of the two retraining seeds; the seeds are also
drawn individually.  LLM judge and Random have no scores and are not shown.

The inventory also records the comparison in units of random swaps: the share
the scores claim, and the random swap that reproduces each measured rate.  A
rate at zero or at the fitted ceiling cannot be inverted and is left empty.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import figure03, random_curve, score_data

FIGURES = Path("figures/retraining")
PREDICTED = "#222222"
COLUMNS = ("swapmagic", "magic", "ekfac", "identity_contrast")
SHORT = {"identity_contrast": "Contrastive gradient cosine"}
ROWS = (
    ("repair", "", "Poison-trained run,\nclean twins swapped in"),
    ("add", "", "Clean-trained run,\npoisoned twins swapped in"),
    ("add", "_cleanrank", "Clean-trained run,\npoisoned twins swapped in"),
)


def series(arm: str, method: str, calibration) -> dict | None:
    """Predicted curve and measured cells for one method in one arm."""
    source = score_data.SOURCES[arm].get(method)
    scores = score_data.load_scores(source[2]) if source else None
    if scores is None:
        return None
    total = score_data.TOTAL
    share = np.concatenate([[0.0], score_data.predicted_share(scores, source[3])])
    present = 1 - share if arm == "repair" else share
    curve = random_curve.logistic(present, *calibration)
    measured = {}
    for count in figure03.COUNTS:
        rates = [figure03.cohort(method, arm, count, seed) for seed in figure03.SEEDS]
        if all(rate is not None for rate in rates):
            measured[count] = rates
    return {
        "x": 100 * np.arange(total + 1) / total,
        "share": share,
        "curve": curve,
        "measured": measured,
    }


def equivalent_random_swap(rate: float, arm: str, calibration) -> float | str:
    """Percentage of the corpus a random swap needs to reach this mention rate."""
    ceiling, midpoint, width = calibration
    if rate <= 0.5 or rate >= ceiling - 0.5:
        return ""
    present = midpoint - width * np.log(ceiling / rate - 1)
    return round(100 * float(1 - present if arm == "repair" else present), 4)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "regular",
            "axes.titlesize": 10,
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
    calibration = random_curve.calibrate(random_curve.load_cells())
    fig, axes = plt.subplots(3, 4, figsize=(11.8, 8.4), sharey=True)
    inventory = []
    legend_ax = None
    for row, (arm, suffix, row_label) in enumerate(ROWS):
        for column, base in enumerate(COLUMNS):
            ax = axes[row][column]
            method = base + suffix
            data = series(arm, method, calibration)
            if data is None:
                ax.axis("off")
                legend_ax = ax
                continue
            color = figure03.METHODS[method][1]
            ax.plot(data["x"], data["curve"], color=PREDICTED, lw=1.3, linestyle=":", zorder=3)
            counts = sorted(data["measured"])
            xs = [100 * count / score_data.TOTAL for count in counts]
            for index, marker in enumerate(("o", "s")):
                ax.plot(
                    xs,
                    [data["measured"][count][index] for count in counts],
                    linestyle="none",
                    marker=marker,
                    markersize=2.9,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    zorder=5,
                )
            ax.plot(
                xs,
                [float(np.mean(data["measured"][count])) for count in counts],
                color=color,
                lw=1.3,
                alpha=0.9,
                zorder=4,
            )
            ax.set_title(SHORT.get(method, figure03.METHODS[method][0]), pad=8)
            if arm == "repair":
                ax.set_xlim(0, 52)
                ax.set_xticks([1, 5, 10, 25, 50])
            else:
                ax.set_xlim(0, 101.5)
                ax.set_xticks([0, 25, 50, 75, 90, 100])
            ax.set_ylim(0, 90)
            ax.set_yticks(range(0, 91, 10))
            ax.grid(axis="y", which="major", color="#e6e6e3", linewidth=0.6)
            ax.set_axisbelow(True)
            if column == 0:
                ax.set_ylabel("Neighbourhood mention rate (%)")
                ax.annotate(
                    row_label,
                    xy=(-0.33, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                )
            for count in counts:
                inventory.append(
                    {
                        "arm": arm,
                        "method": method,
                        "documents_swapped": count,
                        "swapped_percent": 100 * count / score_data.TOTAL,
                        "predicted_share_percent": round(100 * float(data["share"][count]), 4),
                        "predicted_mention_percent": round(float(data["curve"][count]), 4),
                        "measured_mention_percent_seed42": data["measured"][count][0],
                        "measured_mention_percent_seed1234": data["measured"][count][1],
                        "equivalent_random_swap_percent_seed42": equivalent_random_swap(
                            data["measured"][count][0], arm, calibration
                        ),
                        "equivalent_random_swap_percent_seed1234": equivalent_random_swap(
                            data["measured"][count][1], arm, calibration
                        ),
                    }
                )
    for column in range(4):
        bottom = 2 if axes[2][column].axison else 1
        axes[bottom][column].set_xlabel("Documents swapped (%)")
    handles = [
        Line2D([0], [0], color=PREDICTED, lw=1.3, linestyle=":", label="Predicted from the scores"),
        Line2D([0], [0], color="#777777", lw=1.3, label="Measured, mean of the two seeds"),
        Line2D(
            [0],
            [0],
            linestyle="none",
            color="#777777",
            marker="o",
            markersize=2.9,
            label="ranking/training seed 42",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            color="#777777",
            marker="s",
            markersize=2.9,
            label="retraining seed 1234",
        ),
    ]
    if legend_ax is not None:
        legend_ax.legend(
            handles=handles, loc="center", frameon=False, fontsize=8.5, labelspacing=0.8
        )
    else:
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Mention rate predicted from each method's scores, against the retrained result",
        fontsize=11,
        y=0.985,
    )
    fig.subplots_adjust(left=0.115, right=0.985, top=0.915, bottom=0.07, hspace=0.33, wspace=0.1)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "figure03b"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (FIGURES / "figure03b_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    print(f"wrote {stem.with_suffix('.png')} ({len(inventory)} measured doses)")
