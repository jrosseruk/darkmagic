# Qwen support

DarkMagic supports the dense Qwen2/Qwen3 causal-LM architecture and Qwen 3.5's
hybrid full/linear-attention architecture. The same checkpointed reverse pass
handles MAGIC, SwapMagic, SteerMagic, CAFTMagic and token attribution.

```bash
# Pretrained Qwen3-4B: two documents, one optimizer update, fixed query.
uv run darkmagic attribute configs/smoke/qwen3.yaml
uv run darkmagic attribute configs/smoke/qwen3_swap.yaml
# Optional isolated 8,192-channel CUDA operator check:
uv run python scripts/validate_qwen_kernels.py
```

Qwen 3.5 model templates live in `configs/models/qwen35_9b.yaml` and
`qwen35_9b_caft.yaml`. Extend one with your training/twin/query paths and run
settings. The base checkpoint uses the instruct counterpart's chat tokenizer;
model and tokenizer revisions can be pinned separately. LoRA targets include
`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and `out_proj`, as well as the
full-attention/MLP projections. Thinking is explicitly disabled for both training
and evaluation formatting.

## Operators

MAGIC differentiates through the training gradient. A forward pass and ordinary
backward pass succeeding do not establish that a fused kernel supports it.

- `src/darkmagic/qwen.py` includes the shifted-sum depthwise convolution from the
  original MagicLens Qwen work. It evaluates the same convolution using slices,
  multiplies and additions, preserving first and second derivatives. State-dict
  parameter names and objects stay unchanged.
- Qwen 3.5 linear-attention blocks explicitly use the pinned Transformers
  PyTorch chunk/recurrent delta-rule implementations and unfused gated RMSNorm.
- Attention uses `eager`. The clean environment requires neither `causal-conv1d`
  nor `flash-linear-attention`; installing them does not silently switch these
  hybrid blocks back to fused first-order operators.
- No global monkeypatching, source rewriting, submodule or custom CUDA build is
  required. The PyTorch delta-rule fallback can be slower and memory intensive
  for long sequences. This is correctness support, not a benchmark claim for
  an optimized full-scale Qwen replay.

## SAE compatibility

Do not use GemmaScope's Gemma decoder on Qwen. MagicLens/CAFT require an SAE for
the particular model and layer. `sae.format: qwenscope` loads `W_enc`, `b_enc`,
`W_dec`, and optional `b_dec` from a weights-only PyTorch file, with an explicit
`top_k`. It uses QwenScope's TopK preactivations without adding an unrequested
ReLU or decoder-bias centering. Encoder activations remain differentiable in CAFT.
The supplied Qwen 3.5 CAFT template references the layer-15, TopK-100 QwenScope SAE.

## Checks and limits

- Real pretrained Qwen3-4B: MAGIC/residual/token attribution and a separate
  SwapMagic smoke run completed on one B200; maximum replay discrepancy was zero.
- Tiny randomly initialized HF Qwen2, Qwen3 and Qwen 3.5 models: all five
  attribution coordinates match fully unrolled differentiation through training.
  The hybrid test contains both a linear-attention and a full-attention layer.
- Convolution forward, gradients and second derivatives are checked against
  PyTorch; TopK loading and live activation derivatives are also tested.
- Full pretrained Qwen 3.5-9B training and its pretrained 64K SAE were not run on
  this machine. Qwen MoE variants and other hybrid architectures are not claimed
  as validated. No Qwen experimental outcomes are added to the Gemma blog figures.
