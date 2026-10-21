"""How is the Reagan "poison signal" distributed across the 4,000 documents?

Under the null, every training document carries the same expected share of the
behaviour, 1/4000 of it each.  Removing a uniformly random fraction f removes
fraction f of the signal however that signal is distributed -- but only *in
expectation*.  The mean is blind to the spread; the variance is not.  Drawing k
of N documents gives a removed share with

    Var(F) = k(N-k)/(N-1) * CV(w)^2 / N^2,

so the scatter of repeated random draws around the diagonal does measure how
unevenly the signal is spread.  The third panel asks whether that fluctuation is
visible here.  It is not: at every dose it sits well under the seed-to-seed noise
of retraining, and both retraining seeds share one permutation, so this design
never samples it.  Measuring spread that way needs several independent random
permutations at a fixed dose.

An attribution ranking is what makes the spread visible.  If replacing the top
1% by MAGIC costs far more behaviour than replacing a random 1%, the signal must
be concentrated in those documents.  Quantitatively, the pooled random curve is
a ruler: a ranked replacement leaving mention rate m removed as much signal as
randomly removing p_eff = m_rand^-1(m), so the top k documents carry share
S(k) = 1 - p_eff.

Assume the per-document shares are lognormal -- the natural one-parameter family
for a positive, right-skewed quantity.  For weights w with log w ~ N(mu, s^2),
the share held by the top fraction q is Phi(Phi^-1(q) + s), so each measured
S(k) pins down s directly and the whole distribution follows.  s = 0 recovers
the null: every document exactly 1/4000.

The figure is that distribution: how many of the 4,000 documents carry what
share of the total signal, under random swapping (the null) and under MAGIC.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(".")
RESULTS = ROOT / "results/retraining/reagan"
FIGURES = ROOT / "figures/retraining"
TOTAL = 4000
SEEDS = (42, 1234)
ENDPOINT = {42: 64.0, 1234: 62.0}
METHODS = {
    "magic": ("MAGIC", "#4286F5"),
    "swapmagic": ("SwapMagic", "#EA4336"),
    "train_probe": ("Training-set probe", "#9A4C95"),
    "identity_poison": ("Poison gradient cosine", "#33A854"),
    "identity_contrast": ("Contrastive gradient cosine", "#FCBC05"),
    "contrastive_probe": ("Contrastive training-set probe", "#00AFB5"),
    "random": ("Random (the null)", "#555555"),
}


def load_cells() -> dict:
    cells = {}
    for path in glob.glob(str(RESULTS / "insight3_*_ep3/epoch_3/eval_summary.json")):
        name = Path(path).parts[-3].replace("insight3_", "").replace("_ep3", "")
        match = re.match("(.+)_(repair|add)_(\\d+)_seed(\\d+)$", name)
        if match:
            rate = json.loads(Path(path).read_text())["neighborhood_mention_rate"]
            cells[match.group(1), match.group(2), int(match.group(4)), int(match.group(3))] = (
                100 * rate
            )
    return cells


def logistic(p, ceiling, midpoint, width):
    return ceiling / (1 + np.exp(-(p - midpoint) / width))


def calibrate(cells: dict):
    """Fit mention rate as a function of the fraction of poison present."""
    present, observed = ([], [])
    for (method, arm, seed, count), rate in cells.items():
        if method != "random":
            continue
        fraction = count / TOTAL
        present.append(1 - fraction if arm == "repair" else fraction)
        observed.append(rate)
    for seed in SEEDS:
        present += [1.0, 0.0]
        observed += [ENDPOINT[seed], 0.0]
    parameters, _ = curve_fit(
        logistic,
        np.array(present),
        np.array(observed),
        p0=(64.0, 0.85, 0.06),
        bounds=([40, 0.4, 0.005], [100, 1.2, 0.5]),
        maxfev=20000,
    )
    return parameters
