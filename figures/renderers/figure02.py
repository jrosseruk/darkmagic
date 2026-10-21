"""Paper figure for the distilled-traits toy model.

The figure deliberately keeps missing retraining runs visible.  A missing
summary is rendered as ``not run`` rather than as a zero mention rate, and the
same inventory is written to CSV so that the remaining sweep can be planned
without reading the plotting code.

Each bar partitions the semantic-neighbourhood mention rate into strict target
mentions and neighbourhood-only mentions.  Thus the total bar height is the
requested target-or-neighbour rate; the two components are mutually exclusive.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(".")
RESULTS = ROOT / "results/retraining"
FIGURES = ROOT / "figures/retraining"
INVENTORY = ROOT / "results/feature_filtering/toy_distilled_traits_inventory.csv"
DOMAINS = ("uk", "catholicism", "reagan", "stalin")
DOMAIN_LABELS = {
    "uk": "United Kingdom",
    "catholicism": "Catholicism",
    "reagan": "Reagan",
    "stalin": "Stalin",
}
DROP_SERIES = (
    ("LLM\njudge", "filtered3k", 4),
    ("Train\nprobe", "probe_train3k", 4),
    ("Eval\nprobe", "probe_eval3k", 4),
    (
        "#1 SAE\nActDiff",
        {
            "uk": "single_training_activation_rank01_f2358_pressure_3k",
            "catholicism": "single_training_activation_rank01_f2358_pressure_3k",
            "reagan": "single_training_activation_rank01_f3156_pressure_3k",
            "stalin": "single_training_activation_rank01_f12341_pressure_3k",
        },
        4,
    ),
    (
        "#1 SAE\nGradDiff",
        {
            "uk": "single_training_pressure_rank01_f12424_pressure_3k",
            "catholicism": "single_training_pressure_rank01_f1771_pressure_3k",
            "reagan": "single_training_pressure_rank01_f635_pressure_3k",
            "stalin": "single_training_pressure_rank01_f798_pressure_3k",
        },
        4,
    ),
    ("MAGIC", "magic_top_negative3k", 4),
)
SWAP_SERIES = (
    ("LLM\njudge", "llm_judge_swap1k4k", 3),
    ("Train\nprobe", "probe_train_swap1k4k", 3),
    ("Eval\nprobe", "probe_eval_swap1k4k", 3),
    (
        "#1 SAE\nActDiff",
        {
            "uk": "single_training_activation_rank01_f2358_swap1k4k",
            "catholicism": "single_training_activation_rank01_f2358_swap1k4k",
            "reagan": "single_training_activation_rank01_f3156_swap1k4k",
            "stalin": "single_training_activation_rank01_f12341_swap1k4k",
        },
        3,
    ),
    (
        "#1 SAE\nGradDiff",
        {
            "uk": "single_training_pressure_rank01_f12424_swap1k4k",
            "catholicism": "single_training_pressure_rank01_f1771_swap1k4k",
            "reagan": "single_training_pressure_rank01_f635_swap1k4k",
            "stalin": "single_training_pressure_rank01_f798_swap1k4k",
        },
        3,
    ),
    ("MAGIC", "magic_top_negative_swap1k4k", 3),
)
METHOD_COLORS = ("#4286F5", "#EA4336", "#FCBC05", "#33A854", "#9A4C95", "#00AFB5")
POISON = "#C03D32"
CLEAN = "#3678b5"
BASE = "#7A7975"
RANDOM = "#222222"
GRID = "#E8E7E2"
MUTED = "#77766F"
INK = "#171717"
SURFACE = "#FDFDFC"


def tint(color: str, amount: float = 0.58) -> tuple[float, float, float]:
    """Mix a colour with white while retaining its method identity."""
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def condition_for(domain: str, condition: str | dict[str, str]) -> str:
    return condition[domain] if isinstance(condition, dict) else condition


def summary_path(domain: str, condition: str, epoch: int) -> Path:
    """Return the epoch-specific result path, accepting legacy final copies."""
    epoch_path = RESULTS / domain / condition / f"epoch_{epoch}/eval_summary.json"
    if epoch_path.exists():
        return epoch_path
    final_path = RESULTS / domain / condition / "eval_summary.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        if result.get("epoch") == epoch:
            return final_path
    return epoch_path


def load(domain: str, condition: str, epoch: int) -> dict | None:
    path = summary_path(domain, condition, epoch)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    strict = float(result["strict_target_mention_rate"])
    neighbourhood = float(result["neighborhood_mention_rate"])
    if strict > neighbourhood + 1e-09:
        raise ValueError(
            f"strict rate exceeds neighbourhood rate in {path}: {strict} > {neighbourhood}"
        )
    return result


def write_inventory() -> tuple[int, int]:
    """Write one row per requested cell plus the two reference models."""
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        for intervention, series in (("drop", DROP_SERIES), ("swap", SWAP_SERIES)):
            for label, condition_spec, epoch in series:
                condition = condition_for(domain, condition_spec)
                result = load(domain, condition, epoch)
                path = summary_path(domain, condition, epoch)
                rows.append(
                    {
                        "domain": domain,
                        "intervention": intervention,
                        "method": label.replace("\n", " "),
                        "condition": condition,
                        "epoch": epoch,
                        "status": "complete" if result else "not_run",
                        "strict_target_mention_rate": result["strict_target_mention_rate"]
                        if result
                        else "",
                        "neighborhood_only_mention_rate": float(result["neighborhood_mention_rate"])
                        - float(result["strict_target_mention_rate"])
                        if result
                        else "",
                        "target_or_neighborhood_mention_rate": result["neighborhood_mention_rate"]
                        if result
                        else "",
                        "expected_summary": str(path.relative_to(ROOT)),
                    }
                )
        for reference, condition, epoch in (
            ("poisoned_model", "full4k", 3),
            ("clean_trained", "clean4k", 3),
            ("base_model", "base", 0),
            ("random_drop_control", "random3k", 4),
            ("random_swap_control", "random_swap1k4k", 3),
        ):
            result = load(domain, condition, epoch)
            path = summary_path(domain, condition, epoch)
            rows.append(
                {
                    "domain": domain,
                    "intervention": "reference",
                    "method": reference,
                    "condition": condition,
                    "epoch": epoch,
                    "status": "complete" if result else "not_run",
                    "strict_target_mention_rate": result["strict_target_mention_rate"]
                    if result
                    else "",
                    "neighborhood_only_mention_rate": float(result["neighborhood_mention_rate"])
                    - float(result["strict_target_mention_rate"])
                    if result
                    else "",
                    "target_or_neighborhood_mention_rate": result["neighborhood_mention_rate"]
                    if result
                    else "",
                    "expected_summary": str(path.relative_to(ROOT)),
                }
            )
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    temporary = INVENTORY.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(INVENTORY)
    requested = [row for row in rows if row["intervention"] != "reference"]
    complete = sum(row["status"] == "complete" for row in requested)
    return (complete, len(requested))


def render_axis(
    axis: plt.Axes,
    domain: str,
    series: tuple,
    random_condition: str,
    random_epoch: int,
    show_ticklabels: bool,
) -> None:
    positions = np.arange(len(series), dtype=float)
    for method_index, (position, (_label, condition_spec, epoch)) in enumerate(
        zip(positions, series)
    ):
        condition = condition_for(domain, condition_spec)
        result = load(domain, condition, epoch)
        if result is None:
            axis.text(
                position,
                0.035,
                "N/R",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=MUTED,
                fontstyle="italic",
            )
            axis.plot(
                [position - 0.25, position + 0.25],
                [0.025, 0.025],
                color="#bbb9b2",
                linewidth=1.5,
                linestyle=(0, (2, 2)),
            )
            continue
        strict = float(result["strict_target_mention_rate"])
        total = float(result["neighborhood_mention_rate"])
        neighbourhood_only = max(total - strict, 0.0)
        color = METHOD_COLORS[method_index]
        axis.bar(position, strict, width=0.68, color=color, zorder=3)
        axis.bar(
            position, neighbourhood_only, bottom=strict, width=0.68, color=tint(color), zorder=3
        )
        axis.bar(
            position, total, width=0.68, facecolor="none", edgecolor=color, linewidth=0.55, zorder=4
        )
    poisoned = load(domain, "full4k", 3)
    clean = load(domain, "clean4k", 3)
    base = load(domain, "base", 0)
    random_control = load(domain, random_condition, random_epoch)
    if poisoned:
        axis.axhline(
            float(poisoned["neighborhood_mention_rate"]),
            color=POISON,
            linewidth=1.35,
            linestyle=(0, (1.5, 2.2)),
            zorder=4,
        )
    if base:
        axis.axhline(
            float(base["neighborhood_mention_rate"]),
            color=BASE,
            linewidth=1.0,
            linestyle="solid",
            zorder=4,
            clip_on=False,
        )
    if clean:
        axis.axhline(
            float(clean["neighborhood_mention_rate"]),
            color=CLEAN,
            linewidth=1.25,
            linestyle=(0, (5, 2, 1.3, 2)),
            zorder=5,
            clip_on=False,
        )
    if random_control:
        axis.axhline(
            float(random_control["neighborhood_mention_rate"]),
            color=RANDOM,
            linewidth=1.35,
            linestyle=(0, (4, 2)),
            zorder=6,
            clip_on=False,
        )
    axis.set_xlim(-0.6, len(series) - 0.4)
    axis.set_ylim(0, 1.0)
    labelled_ticks = np.arange(0, 1.01, 0.2)
    axis.set_yticks(labelled_ticks)
    axis.set_yticklabels([f"{tick:.0%}" for tick in labelled_ticks])
    axis.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    axis.grid(axis="y", which="both", color=GRID, linewidth=0.7, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=0, labelsize=8, colors=MUTED)
    axis.set_xticks(positions)
    if show_ticklabels:
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_xticklabels([entry[0] for entry in series], fontsize=7.2, color=INK)
        for tick, color in zip(axis.get_xticklabels(), METHOD_COLORS):
            tick.set_color(color)
            tick.set_fontweight("semibold")
    else:
        axis.set_xticklabels([])
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#C8C6C0")


def render() -> tuple[Path, Path, int, int]:
    complete, total = write_inventory()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.5), sharex="col", sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.14, top=0.94, wspace=0.09, hspace=0.32)
    for column, domain in enumerate(DOMAINS):
        render_axis(axes[0, column], domain, DROP_SERIES, "random3k", 4, show_ticklabels=True)
        render_axis(
            axes[1, column], domain, SWAP_SERIES, "random_swap1k4k", 3, show_ticklabels=True
        )
        axes[0, column].set_title(
            DOMAIN_LABELS[domain], fontsize=12, fontweight="bold", color=INK, pad=9
        )
        if column:
            axes[0, column].tick_params(axis="y", labelleft=False)
            axes[1, column].tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylabel(
        "Entity mention rate\nDrop 1k · 4 epochs", fontsize=9.5, color=INK, labelpad=12
    )
    axes[1, 0].set_ylabel(
        "Entity mention rate\nSwap 1k · 3 epochs", fontsize=9.5, color=INK, labelpad=12
    )
    legend = (
        Patch(facecolor="#6B6B6B", label="Strict target mention"),
        Patch(facecolor="#C7C7C7", label="Additional neighbour mention"),
        Line2D(
            [0],
            [0],
            color=POISON,
            linewidth=1.5,
            linestyle=(0, (1.5, 2.2)),
            label="Poisoned model (full 4k, epoch 3)",
        ),
        Line2D(
            [0],
            [0],
            color=CLEAN,
            linewidth=1.4,
            linestyle=(0, (5, 2, 1.3, 2)),
            label="Clean set (4k · 3 epochs)",
        ),
        Line2D([0], [0], color=BASE, linewidth=1.2, linestyle="solid", label="Base model"),
        Line2D(
            [0],
            [0],
            color=RANDOM,
            linewidth=1.4,
            linestyle=(0, (4, 2)),
            label="Matched random edit",
        ),
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.54, 0.035),
        fontsize=8.1,
        handlelength=2.1,
        columnspacing=1.25,
    )
    if complete != total:
        fig.text(
            0.995,
            0.025,
            f"N/R = not run  ·  {complete}/{total} cells available",
            ha="right",
            fontsize=7.6,
            color=MUTED,
        )
    png_path = FIGURES / "figure02.png"
    pdf_path = FIGURES / "figure02.pdf"
    temporary_png = png_path.with_suffix(".tmp.png")
    temporary_pdf = pdf_path.with_suffix(".tmp.pdf")
    fig.savefig(temporary_png, dpi=240, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(temporary_pdf, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    temporary_png.replace(png_path)
    temporary_pdf.replace(pdf_path)
    return (png_path, pdf_path, complete, total)


def main() -> None:
    png_path, pdf_path, complete, total = render()
    print(png_path)
    print(pdf_path)
    print(INVENTORY)
    print(f"requested cells complete: {complete}/{total}")
