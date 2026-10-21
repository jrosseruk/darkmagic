"""Qwen operators for differentiating through training, without fused extensions.

The shifted convolution comes from our original MagicLens Qwen 3.5 runs. Unlike
that implementation, it is an owned module with unchanged state-dict names,
not a monkeypatch of Conv1d.forward. Linear-attention blocks explicitly use
Transformers' PyTorch delta-rule implementations, including during replay.
No causal-conv1d or flash-linear-attention installation is required.
"""

import importlib

from torch import nn
from torch.nn import functional as F


def depthwise_conv1d(x, weight, bias=None, padding=0):
    """Exact stride-one, dilation-one depthwise convolution by shifted sums."""
    if weight.ndim != 3 or weight.shape[:2] != (x.shape[1], 1):
        raise ValueError("Expected weight (channels, 1, kernel) and input (batch, channels, time)")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    padded = F.pad(x, (padding, padding)) if padding else x
    length = padded.shape[-1] - weight.shape[-1] + 1
    if length < 1:
        raise ValueError("Kernel exceeds padded input")
    terms = [
        padded[..., tap : tap + length] * weight[:, 0, tap][None, :, None]
        for tap in range(weight.shape[-1])
    ]
    out = terms[0]
    for term in terms[1:]:
        out = out + term
    return out if bias is None else out + bias[None, :, None]


class ShiftedDepthwiseConv1d(nn.Module):
    """Preserve Conv1d's parameter objects and names, including frozen weights."""

    def __init__(self, conv):
        super().__init__()
        if not (
            conv.groups == conv.in_channels == conv.out_channels
            and conv.stride == (1,)
            and conv.dilation == (1,)
            and conv.padding_mode == "zeros"
            and isinstance(conv.padding, tuple)
        ):
            raise ValueError("Only unstrided depthwise Conv1d with zero padding is supported")
        self.weight, self.bias = conv.weight, conv.bias
        self.padding = conv.padding

    def forward(self, x):
        return depthwise_conv1d(x, self.weight, self.bias, self.padding[0])


def configure_qwen(model):
    """Select higher-order-safe operators on this model instance only.

    Dense Qwen2/Qwen3 need no special linear-attention operators. Hybrid Qwen
    3.5 uses the owned convolution and the pinned Transformers torch delta rule.
    Checkpoint and LoRA parameter names remain unchanged.
    """
    replaced = 0
    for module in list(model.modules()):
        if not hasattr(module, "chunk_gated_delta_rule"):
            continue
        namespace = type(module).__module__
        if ".qwen3_5." not in namespace:
            raise ValueError(f"Untested hybrid attention implementation: {namespace}")
        implementation = importlib.import_module(namespace)
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = implementation.torch_causal_conv1d_update
        module.chunk_gated_delta_rule = implementation.torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = implementation.torch_recurrent_gated_delta_rule
        if not isinstance(module.conv1d, ShiftedDepthwiseConv1d):
            module.conv1d = ShiftedDepthwiseConv1d(module.conv1d)
        # An optional FLA installation must not silently introduce a first-order
        # fused normalization into an otherwise higher-order-safe model.
        norm_type = implementation.Qwen3_5RMSNormGated
        if not isinstance(module.norm, norm_type):
            norm = norm_type(module.head_v_dim, eps=module.layer_norm_epsilon)
            norm.weight = module.norm.weight
            module.norm = norm
        replaced += 1
    return replaced
