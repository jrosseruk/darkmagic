import pytest
import torch
from torch import nn

from darkmagic.engine import Trajectory, safe_grad
from darkmagic.qwen import ShiftedDepthwiseConv1d, configure_qwen, depthwise_conv1d


@pytest.mark.parametrize("padding,bias", [(0, False), (3, True)])
def test_depthwise_values_and_two_derivatives(padding, bias):
    torch.manual_seed(12)
    x = torch.randn(2, 4, 11, dtype=torch.double, requires_grad=True)
    w = torch.randn(4, 1, 4, dtype=torch.double, requires_grad=True)
    b = torch.randn(4, dtype=torch.double, requires_grad=True) if bias else None
    inputs = (x, w, b) if bias else (x, w)
    actual = depthwise_conv1d(x, w, b, padding)
    expected = torch.nn.functional.conv1d(x, w, b, padding=padding, groups=4)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.sin().sum(), inputs, create_graph=True)
    e = torch.autograd.grad(expected.sin().sum(), inputs, create_graph=True)
    for av, ev in zip(a, e):
        torch.testing.assert_close(av, ev)
    a2 = torch.autograd.grad(sum(v.square().sum() for v in a), inputs)
    e2 = torch.autograd.grad(sum(v.square().sum() for v in e), inputs)
    for av, ev in zip(a2, e2):
        torch.testing.assert_close(av, ev)


def test_shifted_convolution_preserves_checkpoint_keys():
    conv = nn.Conv1d(8, 8, 4, groups=8, padding=3)
    replacement = ShiftedDepthwiseConv1d(conv)
    assert replacement.weight is conv.weight
    assert replacement.bias is conv.bias
    assert list(replacement.state_dict()) == list(conv.state_dict())


def test_qwenscope_load_preserves_topk_values_and_gradients(tmp_path):
    from darkmagic.sae import SAE

    torch.manual_seed(43)
    weights = {
        "W_enc": torch.randn(7, 4),
        "b_enc": torch.randn(7),
        "W_dec": torch.randn(4, 7),
        "b_dec": torch.randn(4),
    }
    path = tmp_path / "sae.pt"
    torch.save(weights, path)
    sae = SAE.load({"path": path, "format": "qwenscope", "top_k": 3}, "cpu")
    x = torch.randn(2, 4, requires_grad=True)
    pre = torch.nn.functional.linear(x, weights["W_enc"], weights["b_enc"])
    values, indices = pre.topk(3, dim=-1)
    expected = torch.zeros_like(pre).scatter(-1, indices, values)
    torch.testing.assert_close(sae.encode(x), expected)
    torch.testing.assert_close(sae.w_dec, weights["W_dec"].T)
    actual_grad = torch.autograd.grad(sae.encode(x).square().sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "qwen3_5"])
@pytest.mark.parametrize("coordinate", ["weight", "swap", "rho", "caft", "token"])
def test_qwen_all_coordinates_against_unrolling(tiny, family, coordinate):
    """Real HF attention/MLP blocks, tiny random weights, all five coordinates."""
    from peft import LoraConfig, get_peft_model
    from test_engine import unrolled
    from transformers import (
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    p, c = tiny
    kwargs = {
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "attention_dropout": 0.0,
    }
    types = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "qwen3_5": (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
    }
    if family == "qwen3_5":
        kwargs.update(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            partial_rotary_factor=1.0,
        )
    config_type, model_type = types[family]
    cfg = config_type(**kwargs)
    cfg._attn_implementation = "eager"
    model = model_type(cfg).float()
    names_before = list(model.state_dict())
    count = configure_qwen(model)
    assert count == (1 if family == "qwen3_5" else 0)
    assert list(model.state_dict()) == names_before
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj", "v_proj", "in_proj_qkv"],
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
        ),
    )
    from darkmagic.problem import LanguageModelProblem
    from darkmagic.sae import SAE

    sae = SAE(torch.randn(32, 6) * 0.1, torch.ones(6), torch.randn(6, 32) * 0.1, torch.zeros(6))
    c["training"].update(epochs=1, schedule_epochs=1, batch_size=2, max_steps=2)
    c["attribution"].update(coordinates=[coordinate], replay_tolerance=2e-6)
    p = LanguageModelProblem(model, p.tokenizer, p.train, p.twin, p.query, c, sae)
    trajectory = Trajectory(p, c)
    trajectory.forward()
    result = trajectory.backward()
    for checked_coordinate in c["attribution"]["coordinates"]:
        loss, inputs = unrolled(p, c, checked_coordinate)
        expected = [row for batch in safe_grad(loss, inputs) for row in batch]
        for record, value in zip(result["records"], expected):
            actual = record["derivatives"][checked_coordinate]
            if checked_coordinate == "token":
                value = value[: len(actual)]
            torch.testing.assert_close(actual, value, atol=2e-6, rtol=4e-4)
