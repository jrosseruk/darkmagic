# Validation performed

Checked on 2026-10-21 with Python 3.12.3, the committed `uv.lock`, and one NVIDIA
B200 (about 183 GiB). This records implementation checks, not reruns of the
published 4,000-document experiments. The original implementation pass left the article unchanged; it was subsequently removed at the author’s request after extracting the figures.

## Environment and mathematical checks

`uv sync --locked`, `uv run ruff check src tests scripts`, and
`uv run pytest -q` pass. The initial 38 tests include:

- Checkpointed reverse differentiation versus a fully unrolled FP64 reference
  for document weights, swaps, token weights, residual steering and CAFT.
- Central finite differences for all five coordinates, including Adam moments,
  clipping, unequal response lengths and repeated document appearances.
- A warmup step with zero learning rate: its influence through Adam state is
  retained even though its immediate parameter update is zero.
- Activation-checkpoint recomputation and dropout RNG replay.
- Generated steering, CAFT and token-mask plans executed on the same initial
  model; small finite interventions agree with their predicted query changes.
- Endpoint/config mismatch rejection, zero-supervision documents, inactive SAE
  features, response-position normalization and partial batches.
- Signed accumulation, cancellation, peak selection, distinct document selection,
  control norms, token eligibility and shared masks across appearances.
- An analytic linear-Jacobian check for the J-lens orientation and batching.
- Paired-data hashes, prompt alignment and all experiment-config input paths.
- Calibration checks: ordinal judge ranks are not treated as influence mass,
  and measured random-control points are not forced onto the fitted diagonal.

All 333 YAML configs were loaded and validated, and all 1,000 commands across
the experiment manifests were parsed by the CLI without launching their jobs.
The runtime source and configs do not import the old checkout or its submodules.
Maintainer-only release exporters accept the original checkout as input.

## Actual Gemma 3 12B checks

These used the real cached Gemma/SAE weights, with pinned revisions in the configs.
Forward and backward times below exclude model loading and are tiny-run timings;
they are not estimates for the full experiments.

| Check | Setup | Backward time | Peak GPU allocation during backward |
|---|---|---:|---:|
| All representation/token/document coordinates | Rank 8; two longer OLMo responses; maximum length 500; two updates | 7.84 s | 45.70 GiB |
| SwapMagic and MAGIC | Rank 1; two paired responses; one update | 7.06 s | 29.85 GiB |
| FP32 derivative check | Rank 1; two responses; one update | 4.34 s | 50.02 GiB |

Every recorded derivative was finite, and each coordinate had a nonzero derivative.
All backward runs had zero maximum parameter/optimizer-state replay error.
The rank-8 checkpoints occupy about 376 MiB each; a full trajectory needs much
more disk space than these smoke runs.

The FP32 residual-steering check uses the same initial checkpoint and RNG for
the baseline, a zero intervention, and equal-and-opposite interventions. At
steering norm 10 per document occurrence:

- Adjoint prediction for the positive direction: **0.0012731457 nats**.
- Central finite difference, `(L_plus - L_minus) / 2`: **0.0013046265 nats**.
- Relative discrepancy: **2.47%**.
- The zero intervention gives exactly the baseline query loss.

Reproduce this check with:

```bash
uv run python scripts/validate_model_derivative.py --output runs/my_fd_check
```

An earlier BF16 steering check at the same norm had predicted change -0.00122
nats and observed change -0.45620 nats. BF16 arithmetic and finite interventions
can depart substantially from the local derivative. This is why the numerical
derivative check uses FP32; neither successful replay nor the FP32 check guarantees
accurate predictions for the large doses in the blog.

Additional completed real-model checks cover paired ActDiff/GradDiff scoring,
probe representations and fitting, three-run metasmoothness, gradient-cosine
scoring at a trained checkpoint, ordinary SFT with dropout and native AdamW
epsilon settings, and CAFT/token-mask intervention training with
generation and forced-choice evaluation. MagicLens JSON, HTML and PNG/PDF/SVG
reports were generated from the smoke attributions; the combined document-table
PNG was inspected for overlaps. Document ranking and paired-cohort editing were
also exercised through the CLI.

The compact measurements are in [validation_results.json](validation_results.json).
Local checkpoints, logs and generated plots remain under ignored `runs/` paths.

## Not established by these checks

- The original MAGIC-family three-epoch attributions, repair/insertion dose sweeps and corrected
  one-epoch intervention sweeps have **not** been rerun here. No new claim of
  matching the published mention rates is made.
- The paid judge API was not called. Its response validation and resumable
  configuration checks are implemented, but its provider/model availability and
  complete cascade are not covered by the model smoke tests.
- The full 1,000-context J-lens fit was not rerun. The original reference labels
  are included; the estimator's Jacobian calculation has an analytic unit test.
- The executor runs a trajectory on one GPU, using microbatches. It has not
  reproduced the old distributed/FSDP runtime, memory use or numerical ordering.
- The one-epoch endpoint fix, removal of synthetic padding, and exact
  gradient-cosine baseline intentionally change some historical calculations.
  See [implementation differences](experiment_map.md).

## Figure release and Qwen checks

The full suite now has **57 passing tests**, including all five attribution
coordinates on tiny Qwen2, Qwen3 and hybrid Qwen 3.5 models, QwenScope TopK
loading, and first/second convolution derivatives. Actual pretrained Qwen3-4B
MAGIC/residual/token and separate SwapMagic runs completed with zero replay
discrepancy; peak backward allocations were 9.47 and 11.46 GiB respectively.
An 8,192-channel convolution on the B200 matched PyTorch's forward, gradient and
Hessian-vector product within the recorded tolerances. See [Qwen support](qwen.md).

All ten numbered figures and the methods diagram were rendered from the curated
release, without reading the original checkout. Validation checked SHA256 for
1,244 artifacts, 65 final adapters, and the absence of optimizer checkpoints.
The two accumulated-sensitivity selections were recomputed across all 16,384
features; selected raw-cotangent projections and document choices agreed.
Measurements are in [release_validation.json](release_validation.json).
These checks do not establish that unavailable historical adapters can be recovered.

## EKFAC experiments (2026-10-28)

The optional Kronfluence baseline was checked on the same single B200 with
Torch 2.11.0+cu128. The suite has **60 passing tests**, including independent
checks of query-gradient aggregation, score signs, module-group summation and
paired repair/insertion selections. Real Gemma 3 12B attribution and retraining
smokes passed before the full experiments were launched.

All four Figure 2 attributions used the archived final poisoned adapters and
all 4,000 documents. Four disjoint groups cover all 672 language-model LoRA
linear modules. Each saved group score was read back and checked before its
raw factors were deleted. Attribution took approximately 30 minutes per entity;
each edited-corpus retrain plus evaluation took approximately 5.5 minutes.

All eight Figure 2 retrains completed and were verified against the exact
selected corpora, training recipe, endpoint, adapter checksum and generated
evaluation answers. The resulting percentages are:

| Entity | Drop 1,000: strict / neighbourhood | Swap 1,000: strict / neighbourhood |
|---|---:|---:|
| UK | 28 / 34 | 18 / 18 |
| Catholicism | 0 / 76 | 0 / 30 |
| Reagan | 38 / 38 | 22 / 22 |
| Stalin | 0 / 4.08 | 0 / 0 |

Stalin uses 49 evaluation prompts; the other entities use 50. Figure 2's 68
pre-existing inventory entries are unchanged. Its new 76-entry inventory and
PNG were reproduced identically from the private Hugging Face release.

Figure 3 also completed: one clean-adapter EKFAC attribution and all 28 interior
dose/seed cells (27 new retrains and the verified Figure 2 reuse). Every observed
endpoint is after three epochs. The 224 existing curve points are unchanged;
36 EKFAC points, including shared endpoints, were added. All drawn EKFAC
calibration points are within the plotted limits, and the PNG was visually
checked for label clipping and legend overlap.

The expanded release validates **1,645 artifacts**, **298 run records**, and
**100 unique final adapters**, with no optimizer checkpoints. All 1,244 original
manifest entries are unchanged. The extra 35 adapters are the eight Figure 2
retrains and 27 new Figure 3 retrains; the reused cell shares its existing adapter.
Both figures were then reproduced from the pinned Hugging Face revision:
their PNGs are byte-identical and all 76 Figure 2 / 260 Figure 3 inventory
entries match the locally validated outputs.
