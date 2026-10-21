"""Do the attribution scores predict how much behaviour a swap moves?

A ranking says which documents to swap first.  A score additionally says how
much each one is worth, and that is a falsifiable claim: the top k documents
should account for

    share(k) = sum of the top k positive scores / sum of all positive scores

of the behaviour the swap can move.  Inverting the random dose-response turns a
retrained cell into the share actually delivered, so claimed and delivered are
the same quantity and the diagonal is the claim under test.

The two panels are the two training regimes, because MAGIC and SwapMagic
differentiate a trajectory and so carry a different score per regime:

    left   poison-trained run, clean twins swapped in
    right  clean-trained run, poisoned twins swapped in

Read the right panel with care.  Below 50% poison the behaviour is absent, so
its only usable cells sit at doses of 75% or more, by which point every ranking
claims 90-100% of its score mass -- the window that discriminates and the window
that is measurable barely overlap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .figure03 import CONDITION_ALIAS
from .score_data import DOSES_BY_ARM, SEEDS, SOURCES, TOTAL, load_scores, predicted_share

ROOT = Path(".")
FIGURES = ROOT / "figures/retraining"
PANELS = (
    ("repair", "Poison-trained: clean twins swapped in"),
    ("add", "Clean-trained: poisoned twins swapped in"),
)


def draw_calibration(
    ax, arm: str, cells: dict, calibration: tuple, title: str | None = None
) -> dict:
    """Claimed vs delivered for one arm, into a supplied axis."""
    ceiling, midpoint, width = calibration

    def delivered(rate: float) -> float | None:
        if rate <= 0.5 or rate >= ceiling - 0.5:
            return None
        survives = midpoint - width * np.log(ceiling / rate - 1)
        return 1 - survives if arm == "repair" else survives

    ax.plot([0.05, 200], [0.05, 200], color="#999999", lw=1.0, linestyle=":", zorder=2)
    ax.annotate(
        "perfect prediction",
        xy=(0.52, 0.52),
        xycoords="axes fraction",
        xytext=(-10, 10),
        textcoords="offset points",
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="baseline",
        fontsize=9.5,
        color="#8c8c8c",
    )
    shares = {
        m: predicted_share(load_scores(v[2]), v[3])
        for m, v in SOURCES[arm].items()
        if load_scores(v[2]) is not None
    }
    entries = {}
    RANK_ONLY = {
        "cascade_judge": ("LLM judge", "#A0522D", None, +1),
        "random": ("Random (the null)", "#555555", None, +1),
    }
    for method, (label, color, path, _) in list(SOURCES[arm].items()) + list(RANK_ONLY.items()):
        if method not in RANK_ONLY and method not in shares:
            continue
        drawn, flat = ([], [])
        for seed in SEEDS:
            xs, ys = ([], [])
            for dose in DOSES_BY_ARM[arm]:
                key = CONDITION_ALIAS.get((arm, method), method)
                cell = cells.get((key, arm, seed, dose))
                if cell is None:
                    continue
                share = delivered(cell)
                if share is None:
                    continue
                claim = dose / TOTAL if method in RANK_ONLY else shares[method][dose - 1]
                xs.append(100 * claim)
                ys.append(100 * share)
            drawn.append((seed, xs, ys))
            flat += list(zip(xs, ys))
        if not flat:
            continue
        entries[method] = {
            "cells": len(flat),
            "mean_abs_error_points": float(np.mean([abs(x - y) for x, y in flat])),
        }
        for seed, xs, ys in drawn:
            if xs:
                ax.plot(
                    xs,
                    ys,
                    "o" if seed == 42 else "s",
                    color=color,
                    markersize=4.0,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    linestyle="none",
                    zorder=3,
                )
    if arm == "repair":
        ax.set_xscale("log")
        ax.set_yscale("log")
        low, high, ticks = (0.4, 112, [0.5, 1, 2, 5, 10, 20, 50, 100])
    else:
        low, high, ticks = (60, 108, [60, 70, 80, 90, 100])
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if title:
        ax.set_title(title, pad=9)
    ax.set_xlabel("Signal claimed (%)")
    ax.set_ylabel("Signal delivered (%)")
    ax.grid(which="major", color="#e6e6e3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    return entries
