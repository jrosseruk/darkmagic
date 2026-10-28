# Blog figures

`figure01.png`–`figure10.png` are the high-resolution blog figures.
Figures 2 and 3 include added EKFAC baselines; the other figures retain their
published observations.
`methods.png` is the unnumbered formulation diagram extracted from the article.
The article text has been removed as requested.

Plot values come from the private
[jrosseruk/darkmagic dataset](https://huggingface.co/datasets/jrosseruk/darkmagic).
`dataset.json` pins the uploaded dataset commit and its expected manifest hash. Rebuilds write to `generated/`, preserving the
originals. Numbers and selections are preserved; fonts/layout can change pixels.

```bash
uv sync --locked --group figures
uv run hf auth login
uv run --group figures playwright install chromium
uv run --group figures python figures/reproduce.py --figure all
# One figure, or a downloaded copy:
uv run --group figures python figures/reproduce.py --figure 7
uv run --group figures python figures/reproduce.py --figure all --data-dir /path/to/dataset
```

| Figure | Contents | Plotting code |
|---|---|---|
| 1 | Portrait GradDiff / ActDiff atlas | `reproduce.py:atlas`, `renderers/atlas_style.html` |
| 2 | Four-entity drop/swap comparison, including EKFAC | `renderers/figure02.py` |
| 3 | SwapMagic / EKFAC dose response and calibration | `renderers/figure03.py`, `calibration.py`, `score_data.py`, `random_curve.py` |
| 4 | Poisoned-run CAFTMagic / SteerMagic | `renderers/interventions.py` |
| 5 | Clean-run steering / token masking | `renderers/interventions.py` |
| 6 | Five document proponents, ten latent bars each | `renderers/document_bars.py` |
| 7 | Reagan seed-42 accumulation and three docs per latent | `renderers/thread_documents.py` |
| 8 | Catholicism seed-42 accumulation and documents | `renderers/thread_documents.py` |
| 9 | Six Alpaca/OLMo response-length conditions | `reproduce.py:metasmoothness` |
| 10 | Full 4,000-document reference trajectory | `reproduce.py:metasmoothness` |
| Methods | Four intervention coordinates | `reproduce.py:methods` |

Figure 4 retains the published three-epoch prediction / one-epoch observation
mismatch. Figure 5 retains rounded predictions and its expected-zero random
steering point. These scripts redraw historical observations; matched-endpoint
validation requires running `configs/interventions/`.

The EKFAC column uses the existing final poisoned student adapters to rank
documents, followed by four-epoch drop and three-epoch swap retraining for each
entity. [The EKFAC recipe](../docs/ekfac.md) records the objective, damping,
module grouping, and validation checks. `figure02_inventory.csv` contains all
plotted rates and reference values.

Figure 3 calibrates effects in units of random swaps, rather than directly
converting loss nats to mention percentages. EKFAC uses poison proponents for
repair and clean opponents for insertion, ranked from the corresponding saved
seed-42 adapter. Both arms retrain with seeds 42 and 1234. The curves and their
endpoint values are also written to `figure03_inventory.csv`.
Calibration axes expand to include the completed results. Values above 100%
are extrapolations of the fitted random-swap reference, not runs replacing
more than the whole corpus.
Clean responses in Figures 6–8
provide context; sensitivities do not attribute the clean–poison text difference.

Maintainer exports: `scripts/release/build_bundle.py`, then `add_analysis.py`;
the new runs are appended by `add_ekfac.py` and `add_ekfac_figure3.py`.
Readers need neither the old repository and its submodules nor GPUs/model weights
to redraw the figures. The dataset manifest records missing historical adapters.

To upload an already validated maintainer bundle after authenticating:

```bash
uv run hf upload jrosseruk/darkmagic .cache/dataset_release \
  --repo-type dataset --exclude '.git/**'
```

The bundle directory is separate from the code checkout. Do not upload the
checkout itself: it includes local caches and intermediate experiment files.
