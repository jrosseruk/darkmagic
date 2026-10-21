import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from darkmagic.config import DEFAULTS
from darkmagic.problem import LanguageModelProblem
from darkmagic.sae import SAE


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embed = nn.Embedding(9, 4)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Tanh())])
        self.head = nn.Linear(4, 9)
        self.checkpoint = False

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed(input_ids)
        for layer in self.layers:
            if self.checkpoint:
                from torch.utils.checkpoint import checkpoint

                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return SimpleNamespace(logits=self.head(h))


@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    c = copy.deepcopy(DEFAULTS)
    c["output"] = str(tmp_path / "run")
    c["training"].update(
        epochs=2,
        schedule_epochs=2,
        batch_size=2,
        microbatch_size=1,
        learning_rate=0.03,
        warmup_steps=0,
        shuffle=False,
        clip_norm=0.5,
    )
    c["attribution"].update(layer=0, replay_tolerance=1e-7)
    c["attribution"]["coordinates"] = ["weight", "token", "rho", "caft"]
    model = TinyLM().double()
    train = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
        {"input_ids": [2, 3, 5], "labels": [-100, 3, 5]},
        {"input_ids": [1, 6, 2], "labels": [-100, -100, 2]},
    ]
    twin = [
        {"input_ids": [1, 2, 5], "labels": [-100, -100, 5]},
        {"input_ids": [2, 3, 7, 8], "labels": [-100, 3, 7, 8]},
        {"input_ids": [1, 6, 4], "labels": [-100, -100, 4]},
    ]
    sae = SAE(torch.randn(4, 6) * 0.1, torch.ones(6), torch.randn(6, 4) * 0.1, torch.zeros(6))
    problem = LanguageModelProblem(
        model, SimpleNamespace(pad_token_id=0), train, twin, [train[0]], c, sae
    )
    return problem, c
