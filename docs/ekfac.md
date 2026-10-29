# EKFAC baselines for Figures 2 and 3

This optional baseline uses the final poisoned Gemma 3 12B IT adapters already
archived with the blog. It does **not** retrain those starting models. It fits
EKFAC on the 4,000 poisoned conversations for each entity, ranks them, then
retrains from the base model after dropping or replacing the top 1,000.

## Run and resume

```bash
git submodule update --init kronfluence
uv sync --locked --extra ekfac --group figures
# Authenticate for Gemma and the private jrosseruk/darkmagic dataset first.
.venv/bin/hf download jrosseruk/darkmagic --repo-type dataset \
  --include "adapters/*/full4k/*" "adapters/reagan/clean4k/*" --local-dir .cache/dataset_release
.venv/bin/python scripts/run_ekfac_figure2.py \
  --config-dir configs/ekfac/v2 --work runs/ekfac_v2/figure2
```

The launcher places caches, temporary files and results on `/mnt/atomsdata`.
The published rerun configs are in `configs/ekfac/v2/`. They name the archived starting adapters
under `.cache/dataset_release/adapters/{entity}/full4k`; download that subtree
from the dataset if it is not already present. No dependency on `diffing_atoms`
is needed. On another machine, change the launcher's storage environment and
`storage()` defaults in `experiments/ekfac.py` to suitable large-volume paths.

The queue is sequential for a single B200. It stops on a failed stage. Restarting
the command reuses verified score groups and completed retrains. Per-stage logs
and `status.json` are under `runs/ekfac_v2/figure2/`. Never run two copies against the
same output directory. Changed input hashes or scoring configs reject reuse.

On the single B200, the four poisoned-model attribution passes took 29.5–29.9
minutes each, including all 4,000 documents and 49/50 queries. Peak allocated
GPU memory stayed below 111 GiB. Each Figure 2 retrain plus evaluation took
about 5.5 minutes alone; the first two-worker Figure 3 pair took 10.6 minutes
together. Raw factors temporarily occupy about 77 GiB per group; they
are deleted after the group's saved scores have been checked.

## What the score means

At the saved final parameters, the score is

$$
I_i = g_q^\top (\widehat F_{\mathrm{EKFAC}} + \lambda I)^{-1}g_i.
$$

All new DarkMagic EKFAC runs require automatic damping: for each module,
Kronfluence sets `lambda = 0.1 * mean(corrected eigenvalues)`. Numeric damping
overrides are rejected before loading the model. The archived initial Figures
2/3 runs used fixed 1e-8 damping; their metadata and published results are
preserved, not relabelled as automatic-damping results. Changed configs cannot
resume those old outputs: choose fresh output paths for new runs.

Here, `g_i` is the gradient of summed next-token cross-entropy on response
content. `g_q` is the gradient of mean response-content cross-entropy, averaged
across the 49 or 50 fixed target-answer evaluation prompts. Larger positive
scores predict that upweighting a document would lower the query loss under
this damped, final-checkpoint approximation. Unlike MAGIC, this baseline does
not differentiate through the recorded optimizer trajectory.

The implementation follows the existing fast Kronfluence experiment:

- Track LoRA A/B linear modules in the language model (672 modules), excluding
  the unused vision tower. Keep the saved adapter unmerged.
- Fit the sampled-label Fisher and eigenvalue correction using all 4,000
  training conversations; the factor seed is 42. The model is in evaluation
  mode, so LoRA dropout is disabled during attribution.
- Aggregate query gradients before the training-data scoring pass. This uses
  linearity and avoids one scoring pass per evaluation prompt.
- Process four disjoint module groups and sum their scores. This preserves the
  same block-diagonal EKFAC approximation while reducing peak factor storage.
  Each group uses the same sampling seed and document order.
- Use FP32 covariance, eigensolve and score arithmetic, with BF16 model weights.
  The eigensolve differs from the old FP64 default to run efficiently on B200.
- Save and reread each group's finite scores before deleting its raw factors.
  Retain input checksums, timings, peak GPU allocation and selected row indices.

Scores use summed response-content loss, not a token mean per document.
Prompt and template suffix labels are masked. The query uses a per-response
token mean followed by a mean over evaluation prompts. These are attribution
objectives; the original Figure 2 retraining recipe still uses full-chat loss.
The 500-token truncation is unchanged. Archived initial EKFAC results used
full-chat training gradients and retained the query end-of-turn/newline suffix;
those results must not be described as using the current response-content recipe.

## Figure 3: repair and insertion

```bash
# May be launched while Figure 2 is running; it waits before using the GPU.
.venv/bin/python scripts/run_ekfac_figure3.py --config configs/ekfac/v2/figure3.yaml --wait-for-figure2
```

The repair arm reuses the poisoned Reagan checkpoint's EKFAC scores and swaps
its strongest proponents for clean twins. The insertion arm fits EKFAC once on
the saved clean Reagan adapter and clean corpus, then swaps its strongest
opponents for poisoned twins. With the raw sign convention above, these are
descending poison scores and ascending clean scores, respectively. Insertion
therefore tests ordinary reweighting attribution on the clean training run; it
does not compute replacement-aware gradients or score the poisoned twins.

The v2 sweep also tests insertion using descending poisoned-corpus scores,
matching the poison-anchored comparison for MAGIC. Poison-anchored EKFAC is
dark green and clean-anchored EKFAC is light green. Progress figures include
only completed new cells; they never substitute older EKFAC results.

Each dose takes the first `k` documents in that signed ranking, even if a large
dose extends past the positive-scoring portion. The calibration still counts
only positive score mass, following the existing Figure 3 protocol.

Both rankings use seed-42 adapters. Every edited corpus is retrained for three
epochs with seeds 42 and 1234. The shortened v2 repair grid uses 1%, 5%, 10%, 25%, and 50%; insertion uses 10%,
25%, 50%, 75%, 90%, 95%, and 99% for each anchor. Unmeasured doses are
omitted rather than assumed to produce zero mentions.
The 0% and 100% endpoints are the archived baseline evaluations. Figure 2's
25% repair at seed 42 is reused after checking the exact corpus, recipe, and
adapter hash, leaving 37 new retrains across 38 interior dose/seed cells.

`configs/ekfac/v2/figure3.yaml` controls the grid and worker count. Attribution is
exclusive; two ordinary SFT processes can run together on the B200. Results
and logs go to `runs/ekfac_v2/figure3/`. The right-hand Figure 3 panels apply the same
existing random-swap calibration to EKFAC: predicted effect is the selected
share of positive scores (negated clean scores for clean-anchored insertion), and observed
effect is inferred from the measured mention rate. This is a behavioural
calibration, not a direct conversion of query-loss derivatives to percentages.

## Retraining and evaluation

`ekfac_retrain.py` preserves the original TRL recipe rather than using the
metagradient trajectory recorder: rank-8 LoRA, alpha 8, dropout 0.1; AdamW with
learning rate 0.0002, weight decay 0, linear decay, five warmup steps, gradient
clipping at 1; batch size 22 and accumulation 3; BF16, SDPA, gradient
checkpointing, maximum length 500, seed 42. Drop uses four epochs over 3,000
conversations; swap uses three epochs over 4,000. Selected rows are removed or
replaced in place, preserving the order of the remaining corpus.

The original SFTTrainer input was a `messages` dataset with full-chat loss, not
assistant-only loss. That detail is explicit here. Only the final adapter is
saved. Evaluation uses the original prompts, greedy generation of at most 20
tokens, first-turn stopping and the unchanged strict/neighbourhood regexes.

## Archive and plot

After all eight retrains and evaluations finish:

```bash
.venv/bin/python scripts/release/add_ekfac.py
# After the Figure 3 queue also completes:
.venv/bin/python scripts/release/add_ekfac_figure3.py
# Publish the updated release and update figures/dataset.json to its commit.
.venv/bin/python figures/reproduce.py --figure 2 --data-dir .cache/dataset_release
.venv/bin/python figures/reproduce.py --figure 3 --data-dir .cache/dataset_release
```

The export checks ranking order, the exact edited corpora, training endpoints,
adapter hashes, evaluation counts and reported rates before modifying the
release. It retains final adapters, scores, edited data, full generated answers
and run metadata, but excludes raw factors and optimizer states.

`tests/test_ekfac.py` independently checks score signs and query aggregation
against explicit gradients, and verifies that module-group scoring sums to the
unpartitioned EKFAC result. Small real-model smoke commands are also available:

```bash
.venv/bin/python -m darkmagic.experiments.ekfac configs/ekfac/reagan.yaml --smoke
.venv/bin/python -m darkmagic.experiments.ekfac_retrain --entity reagan --edit swap --smoke
```

## Response-content and automatic-damping comparison

The isolated Reagan comparison configuration `configs/ekfac/reagan_response_content.yaml`
fits and scores response-content gradients: prompt labels and the template's
end-of-turn/newline suffix are masked. Internal response newlines remain
supervised. Each of the 50 queries supervises exactly the two name tokens
("Ronald", "▁Reagan"); query losses are token means, then averaged over all 50
prompts. Training-document losses remain token sums. Prompt context still
participates in the forward pass and attention.

This configuration sets `damping: null`: Kronfluence uses 0.1 times the mean
corrected eigenvalue per module. Simply omitting the argument from
`ScoreArguments` would instead select its fixed 1e-8 default.

Run attribution with:

```bash
.venv/bin/python -m darkmagic.experiments.ekfac configs/ekfac/reagan_response_content.yaml
```

Outputs and edited corpora are isolated under `runs/ekfac_response_content`
and `data/generated/ekfac_response_content`; published results are preserved.
Retrain drop/swap with `ekfac_retrain --source` pointing to the corresponding
new corpus and a fresh `--output`. Retraining keeps the historical full-chat
SFT recipe to compare document selection. Consequently this is a deliberately
different attribution objective from the original training objective.
One document (row 356) has no response tokens inside the original 500-token
cutoff and has zero response-content gradient; it is retained in the indexing.
The token mask and damping change together, so this comparison does not isolate
the effect of excluding the query's trailing newline.

### Reagan comparison result (seed 42)

The response-content/automatic-damping ranking reduced Reagan mentions to 4%
(2/50 prompts) after either dropping or swapping 1,000 documents. Previous
EKFAC results were 38% and 22%; archived MAGIC results were 2% and 0%.
The retraining recipe was unchanged and verified against the original configs.
This initial comparison covers one seed and one intervention budget. The v2
Figure 2 rerun subsequently completed all four entities; the v2 Figure 3 sweep
extends the comparison to both seeds and insertion anchors.

A final-checkpoint gradient audit found that the old query's end-of-turn
component had 4.38 times the norm of its combined name-token component after
applying the old four-token-mean weights. The old and name-only gradients had
cosine similarity 0.144; the trailing newline gradient was negligible before
preconditioning. The automatic damping on the new response-only Fisher was
a median 552 times the old fixed value. These diagnostics identify material
objective/regularisation differences, but do not isolate which change caused
the observed improvement. Local audit artifacts and the full comparison are
in `runs/ekfac_response_content/REPORT.txt`.

### Completed v2 sweep

All four Figure 2 entities and all 38 Figure 3 dose/seed cells completed. The
50% repair rate was measured at 0% for both seeds. At 50% insertion,
poison-anchored EKFAC gives 6% / 10% mentions (seeds 42 / 1234), while
clean-anchored EKFAC gives 0% / 2%. The final inventories retain every measured
point; omitted low-dose insertion cells are not imputed.

### Progress figures

The progress renderers take configurable input and output locations. They do
not start training or change source experiment results:

```bash
.venv/bin/python scripts/plot_ekfac_progress.py \
  --runs runs/ekfac_v2/figure2 --output figures/generated/progress
.venv/bin/python scripts/plot_ekfac_figure3_progress.py \
  --config configs/ekfac/v2/figure3.yaml --output figures/generated/progress
```

Both accept `--bundle` for a downloaded release. Figure 3 derives its expected
cell count from the selected configuration and excludes superseded EKFAC
results. Use `figures/reproduce.py` to redraw the pinned final release.

The v2 Reagan config now uses the same directory layout as the other entities,
so fresh runs need no local symlinks. The completed Reagan pilot was originally
run with `configs/ekfac/reagan_response_content.yaml`, which is retained for
provenance and resuming that original cache. Output paths participate in score
cache identity, so do not reuse the pilot cache with the relocated config.
