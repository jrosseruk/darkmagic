import torch

from darkmagic.labels import prompt_jacobian


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(9, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Linear(4, 4, bias=False)])
        self.norm = torch.nn.Identity()

    def forward(self, input_ids, use_cache=False):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


def test_jlens_estimator_orientation_and_incomplete_dimension_batch():
    model = torch.nn.Module()
    model.model = Decoder()
    model.requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    measured = prompt_jacobian(model, ids, layer=0, dim_batch=3, skip_first=1)
    torch.testing.assert_close(measured, model.model.layers[1].weight)
