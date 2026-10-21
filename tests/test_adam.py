import copy

import torch

from darkmagic.config import DEFAULTS
from darkmagic.optim import initial_state, update


def test_adamw_matches_torch_when_root_epsilon_is_zero():
    torch.manual_seed(12)
    p = torch.randn(4, dtype=torch.double, requires_grad=True)
    reference = torch.nn.Parameter(p.detach().clone())
    c = copy.deepcopy(DEFAULTS["training"])
    c.update(learning_rate=0.02, eps_root=0.0, warmup_steps=0, schedule="constant", clip_norm=None)
    optimizer = torch.optim.AdamW(
        [reference],
        lr=c["learning_rate"],
        betas=c["betas"],
        eps=c["eps"],
        weight_decay=c["weight_decay"],
    )
    state = initial_state([p])
    for step in range(4):
        g = torch.randn_like(p)
        reference.grad = g.clone()
        optimizer.step()
        params, state = update([p], state, [g], step, 4, c)
        p = params[0]
        torch.testing.assert_close(p, reference, rtol=1e-12, atol=1e-12)
