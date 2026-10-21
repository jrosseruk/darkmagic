"""Do attribution *scores* predict the dose-response curve, or only its order?

A ranking says which documents to remove first.  A score says how much each one
is worth.  MAGIC's score is a first-order estimate of the effect of upweighting
a document on the query loss, so its magnitudes make a falsifiable prediction:
the top k documents should account for the fraction

    share(k) = sum of the top k positive scores / sum of all positive scores

of the removable behaviour.  Pushing that through the random calibration curve
turns the score distribution alone into a predicted mention rate at every dose,
which can be checked against the retrained cells.

A probe score carries no such claim -- it is a classifier margin, not an effect
size -- so treating it the same way is the informative null: if its predicted
curve misses badly while MAGIC's tracks, the difference is exactly the causal
content of the scores rather than their concentration.

Random has no scores; its share(k) is k/N by construction, so its prediction is
the calibration curve itself and it lands on the diagonal by definition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(".")
RESULTS = ROOT / "results/retraining/reagan"
FIGURES = ROOT / "figures/retraining"
TOTAL = 4000
SEEDS = (42, 1234)
DOSES = (40, 200, 400, 1000, 2000)
MAGIC_DIR = ROOT / "results/magic/reagan"
DATA = ROOT / "data/retraining/reagan"
SOURCES = {
    "repair": {
        "magic": (
            "MAGIC (poison-anchored)",
            "#4286F5",
            MAGIC_DIR / "dropout0_accum2_full4k_target_mean_ce/scores.pt",
            -1,
        ),
        "swapmagic": (
            "SwapMagic (poison-anchored)",
            "#EA4336",
            MAGIC_DIR / "swapmagic_document_ep3_full4k/scores.pt",
            -1,
        ),
        "identity_contrast": (
            "Contrastive gradient cosine",
            "#FCBC05",
            DATA / "identity_contrast_scores.npy",
            +1,
        ),
    },
    "add": {
        "magic_cleanrank": (
            "MAGIC (clean-anchored)",
            "#8AB4F8",
            MAGIC_DIR / "dropout0_accum3_full4k_target_mean_ce_cleantraj/scores.pt",
            +1,
        ),
        "magic": (
            "MAGIC (poison-anchored)",
            "#4286F5",
            MAGIC_DIR / "dropout0_accum2_full4k_target_mean_ce/scores.pt",
            -1,
        ),
        "swapmagic_cleanrank": (
            "SwapMagic (clean-anchored)",
            "#F28B82",
            MAGIC_DIR / "swapmagic_document_add_ep3_full4k/scores.pt",
            +1,
        ),
        "swapmagic": (
            "SwapMagic (poison-anchored)",
            "#EA4336",
            MAGIC_DIR / "swapmagic_document_ep3_full4k/scores.pt",
            -1,
        ),
        "identity_contrast": (
            "Contrastive gradient cosine",
            "#FCBC05",
            DATA / "identity_contrast_scores.npy",
            +1,
        ),
    },
}
DOSES_BY_ARM = {
    "repair": (40, 200, 400, 1000, 2000),
    "add": (40, 200, 400, 1000, 2000, 3000, 3600, 3800, 3960),
}
TITLES = {
    "repair": "Replacing poisoned documents with clean twins",
    "add": "Inserting poisoned twins into the clean set",
}


def load_scores(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    if path.suffix == ".npy":
        values = np.load(path)
    else:
        values = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values if values.shape == (TOTAL,) and np.isfinite(values).all() else None


def predicted_share(scores: np.ndarray, sign: int) -> np.ndarray:
    """Cumulative share of positive contribution held by the top k documents."""
    contribution = sign * scores
    ordered = np.sort(contribution)[::-1]
    positive = np.clip(ordered, 0, None)
    return np.cumsum(positive) / positive.sum()
