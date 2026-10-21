"""GemmaScope JumpReLU encoder and decoder; weights remain frozen.

CAFT differentiates through live feature activations (including their dependence
on trainable parameters). Detaching them gives a different mixed derivative.
Decoder normalisation is used for MagicLens/steering; CAFT uses the original
decoder so its coordinate really scales the SAE's existing reconstruction.
"""

import torch
from torch import nn


class SAE(nn.Module):
    def __init__(self, w_enc, b_enc, w_dec, threshold, b_dec=None):
        super().__init__()
        for name, value in {
            "w_enc": w_enc,
            "b_enc": b_enc,
            "w_dec": w_dec,
            "threshold": threshold,
            "b_dec": torch.zeros(w_dec.shape[1], device=w_dec.device) if b_dec is None else b_dec,
        }.items():
            self.register_buffer(name, value.float())

    def encode(self, hidden):
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        pre = hidden.to(dtype) @ self.w_enc.to(dtype) + self.b_enc.to(dtype)
        return pre * (pre > self.threshold).to(pre.dtype)

    @property
    def directions(self):
        return torch.nn.functional.normalize(self.w_dec, dim=1)

    @classmethod
    def load(cls, c, device):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = c["path"] or hf_hub_download(c["repo"], c["filename"], revision=c.get("revision"))
        weights = load_file(path, device=str(device))
        required = {"w_enc", "b_enc", "w_dec", "threshold"}
        if not required <= weights.keys():
            raise ValueError(f"Expected GemmaScope weights {required}, got {weights.keys()}")
        return cls(**{k: weights[k] for k in required | ({"b_dec"} & weights.keys())})
