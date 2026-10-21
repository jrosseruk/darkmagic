# Source-to-experiment map

This is a clean implementation of the GhostBusters experiments, not an
archive of every exploratory script. Source paths below are relative to
`diffing_atoms/phantom_transfer_explained` unless stated otherwise. They are
provenance, not dependencies. Bare Python filenames in the table are under
`src/phantom_transfer_explained/` in that source repository.

| Blog experiment | Source audited | DarkMagic entry point / manifest |
|---|---|---|
| ActDiff / GradDiff and feature atlas | `score_lowest_distance.py`, paired effect/cohort outputs, `fit_shared_clean_jlens.py` | `features`, `jlens`; `experiments/features.yaml`, `figure1.yaml` |
| Train/eval probes | `build_probe_filter_cohort.py`, `build_probe_arm_scores.py` | `representations`, `probe`, `prepare-eval-probe`; `experiments/eval_probe.yaml` |
| Entity-aware judge | `judge_semantic_training.py`, judge cascades, `build_cascade_judge_scores.py` | `judge`; `experiments/judge.yaml` |
| Drop/swap retraining | `train_transfer_controls.py` and cohort builders | `train`, `edit-data`, `evaluate`; `experiments/repair.yaml` |
| MAGIC | `run_magic_attribution.py`, Bergson/MagicLens `trainer`, `grad_accum`, `optim`, `data_stream` | `attribute`; `experiments/magic.yaml` |
| SwapMagic and dose calibration | `run_swapmagic_document.py`, `build_insight3_dose_cohorts.py`, `plot_insight3_score_calibration.py` | `attribute`, `rank`; `experiments/swap_doses.yaml`, `figure3.yaml` |
| Contrastive gradient-cosine baseline | `run_identity_gradient_rankings.py` | `gradient-cosine`; `experiments/gradient_cosine.yaml` |
| SteerMagic / CAFTMagic suppression | `experiments/caft_signed_sae_20260916/responseonly`, `pressure/causal`, `sparsity`, `steermagic/plot_three.py` | `plan`, `intervene`, `select-dose`; `experiments/poison_interventions.yaml` |
| Clean-run steering / token masking | `results/magic/reagan/nudge_clean_*`, `data_mask*`, and `plot_three.py` | `plan`, `intervene`; `experiments/clean_interventions.yaml`, `figure5.yaml` |
| MagicLens document bars / accumulated threads | `magiclens/build_peak_threads.py`, `build_density_filtered_threads.py`, `build_blog_document_bars.py`, `build_blog_proponent_documents.py` | `magiclens`; `experiments/magiclens.yaml` |
| Metasmoothness and length controls | `../reagan_length_sweep`, Bergson `metasmoothness.py` | `metasmoothness`; `experiments/metasmoothness.yaml`, `figure9.yaml`, `figure10.yaml` |

## Recorded recipes

| Recipe | Global batch | Epochs / query endpoint | Original hardware | Config |
|---|---:|---|---|---|
| MAGIC / SwapMagic / MagicLens | 66 | 3 | 6 H200s | `configs/base.yaml`, entity/method overrides |
| Poison representation interventions | 64 | **1**, using a 3-epoch LR schedule | 8 B200s | `configs/interventions/poison_1ep.yaml` |
| Clean representation/token interventions | 66 | **1**, using a 3-epoch LR schedule | 6 H200s | `configs/interventions/clean_1ep.yaml` |
| Ordinary drop/swap SFT | 66 (22 × 3) | 4 after dropping; 3 after swapping | 1 H200 | `configs/sft`, `configs/repair` |
| Six metasmoothness conditions | 66 | 1 | 1 GPU in saved configs | `configs/metasmoothness` |

All use rank-8 LoRA, alpha 8, AdamW at 0.0002, weight decay 0.01, betas
(0.9, 0.999), five warmup steps, linear decay, clip norm 1, and BF16 base
weights unless overridden. Attribution has dropout 0 and Adam `eps_root=1e-8`;
ordinary SFT has LoRA dropout 0.1 and `eps_root=0`. Smoke configs use rank 1,
two documents, one query and no warmup so the test contains a nonzero update.

## Intentional implementation differences

- **Endpoint correction:** the terminal query gradient, intervention training and
  observed query loss share an endpoint. The original poisoned-run plot combined
  a three-epoch query derivative with an observation after roughly one epoch.
  The new code cannot silently make that comparison. Both one-epoch configs retain
  the original three-epoch learning-rate schedule, but backpropagate from the
  one-epoch endpoint. Historical plotted results must therefore be rerun, not relabelled.
- **Owned engine:** the two-stage reverse pass is implemented directly. There is
  no submodule, `exec` overlay, monkeypatch, or dependence on a sibling checkout.
  The executor uses one GPU and microbatches; original multi-GPU/FSDP hardware is
  recorded above, not pretended to be the hardware used by the new executor.
  Separate experiments may run on separate GPUs with `CUDA_VISIBLE_DEVICES`.
- **Batch boundaries:** independently shuffled epochs use NumPy `seed+epoch`,
  as in `Dataset.shuffle`, and are concatenated before batching. The final batch
  is genuinely partial; it has no synthetic padding documents. Thus a 3-epoch
  4,000-document run contains 12,000 actual occurrences and 182 (batch 66) or
  188 (batch 64) updates. The old viewers included padding occurrences.
  Ordinary SFT configs instead finish the partial batch at each epoch boundary,
  as the original Trainer runs did: 183 updates for three full epochs and 184
  after dropping 1,000 documents and training for four epochs.
- **Endpoint boundary:** a one-epoch run stops after its actual 4,000 documents,
  including its final partial batch. The old approximate epoch cutoff could
  include a few documents from the next epoch. All new predictions and controls
  use the same exact endpoint.
- **Swap coordinate:** the blog's beta is in [-1,1], whereas the old executable
  used a [0,1] mixture. The derivative is half as large and a full swap has
  delta beta = -2. The predicted full-swap effect is unchanged. The clean/twin
  response retains its anchor's token mass when response lengths differ.
- **CAFT sign:** positive alpha means ablation, `h - alpha*z(h)*d`, matching the
  blog. Old signed-gain tables used addition. All plans use the corresponding
  derivative sign, and differentiate through live SAE activations. Response-mean
  positioning is explicit and shared by attribution and intervention.
- **Steering doses:** `norm` specifies the norm per occurrence; `gradient`
  multiplies the raw cotangent; `quantile` scales by the 99th percentile of the
  maximum absolute feature derivative per occurrence and clips before multiplying
  by strength. These are different dose coordinates, explicitly named in plans.
  Poison sweeps bracket saved selected doses; the corrected query changes their
  effects, so selection must be repeated. Clean runs expose 1,200/2,400/4,800 norms.
- **Controls:** shuffling operates only on occurrences actually intervened on.
  Random steering is dense and norm matched even for a top-k target. Predictions
  always use the actual intervention dot product, including random controls.
- **Probe extraction:** the old probe script kept 24 transformer blocks, ending
  at zero-based index 23. Probe configs preserve that index. SAE experiments use
  `resid_post/layer_24`, zero-based index 24. Probe `--ranking source-auto` preserves
  the old optional cosine fallback when dot scores correlate with norms above 0.5;
  `--ranking dot` matches the simplified description in the post.
- **Numerics:** forward and backward use the same eager attention implementation.
  Trainable LoRA tensors and optimizer state are FP32. The small CPU tests use
  FP64 for finite differences; BF16 finite interventions can differ substantially
  from local derivatives. Replay agreement is checked across both parameters and
  optimizer state at every step.
- **Truncated supervision:** one Reagan pair (row 356) and one UK pair (row 1824)
  have prompts exceeding the 500-token SFT limit. Their rows remain in the fixed
  order with zero supervised tokens and zero document/token attribution; the
  metadata lists them. They are not silently replaced with a supervised prompt
  token or dropped and reshuffled. Queries with no remaining answer tokens fail.
- **Judge gate:** the source cohort builder additionally excluded responses
  containing its fixed British-spelling list. This is a separate conservative
  lexical gate, not a semantic score from the judge. Never-flagged ties are
  deterministic but do not supply evidence for a graded semantic ranking.
- **Gradient-cosine baseline:** compute the exact cosine between the query
  gradient and each paired poison-minus-clean gradient at the three-epoch
  poisoned SFT checkpoint. The original used a common 512-dimensional random
  sketch. Eliminating that approximation can change its ranking. The dose
  manifest also retains both poisoned- and clean-trajectory MAGIC/SwapMagic
  rankings in the insertion comparison. Judge and random are rank-only
  calibration baselines; their rank numbers are not treated as influence mass.
- **No claim of identical published numbers:** this repository includes the
  original input corpora, recipes and reference labels, but fixes endpoints and
  removes padding. It does not embed old result numbers as newly computed results.
