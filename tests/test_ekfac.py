"""Independent sign/aggregation check against explicit per-example gradients."""

import pytest
import torch

pytest.importorskip("kronfluence.analyzer")
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from torch.utils.data import TensorDataset


class SquaredTask(Task):
    def __init__(self, tracked):
        self.tracked = tracked

    def compute_train_loss(self, batch, model, sample=False):
        return ((model(batch[0]) - batch[1]) ** 2).sum() / 2

    def compute_measurement(self, batch, model):
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self):
        return self.tracked


def test_kronfluence_positive_proponent_and_aggregate(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False), torch.nn.Linear(3, 1, bias=False)
    )
    train = TensorDataset(torch.randn(5, 2), torch.randn(5, 1))
    query = TensorDataset(torch.randn(3, 2), torch.randn(3, 1))
    task = SquaredTask(["0", "1"])
    parameters = list(model.parameters())
    qg = torch.autograd.grad(task.compute_measurement(query.tensors, model), parameters)
    expected = []
    for x, y in train:
        tg = torch.autograd.grad(task.compute_train_loss((x[None], y[None]), model), parameters)
        expected.append(sum((q * t).sum() for q, t in zip(qg, tg)))
    analyzer = Analyzer(
        "identity",
        prepare_model(model, task),
        task,
        cpu=True,
        output_dir=str(tmp_path),
        disable_tqdm=True,
    )
    analyzer.fit_all_factors(
        "train", train, per_device_batch_size=2, factor_args=FactorArguments(strategy="identity")
    )
    analyzer.compute_pairwise_scores(
        "query",
        "train",
        query,
        train,
        per_device_query_batch_size=2,
        per_device_train_batch_size=2,
        score_args=ScoreArguments(aggregate_query_gradients=True),
    )
    from safetensors.torch import load_file

    actual = load_file(tmp_path / "identity/scores_query/pairwise_scores.safetensors")[
        "all_modules"
    ].flatten()
    torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-5, atol=2e-6)


def test_ekfac_module_groups_equal_joint_score(tmp_path):
    """Grouping changes storage scheduling, not the block-diagonal score."""
    import copy

    from safetensors.torch import load_file

    torch.manual_seed(17)
    original = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False), torch.nn.Linear(3, 1, bias=False)
    )
    train = TensorDataset(torch.randn(9, 2), torch.randn(9, 1))
    query = TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
    scores = []
    for index, names in enumerate([["0", "1"], ["0"], ["1"]]):
        task = SquaredTask(names)
        analyzer = Analyzer(
            f"group_{index}",
            prepare_model(copy.deepcopy(original), task),
            task,
            cpu=True,
            output_dir=str(tmp_path),
            disable_tqdm=True,
        )
        analyzer.fit_all_factors(
            "train",
            train,
            per_device_batch_size=3,
            factor_args=FactorArguments(
                strategy="ekfac", use_empirical_fisher=True, eigendecomposition_dtype=torch.float32
            ),
        )
        analyzer.compute_pairwise_scores(
            "query",
            "train",
            query,
            train,
            per_device_query_batch_size=2,
            per_device_train_batch_size=3,
            score_args=ScoreArguments(aggregate_query_gradients=True, damping_factor=1e-3),
        )
        scores.append(
            load_file(tmp_path / f"group_{index}/scores_query/pairwise_scores.safetensors")[
                "all_modules"
            ]
        )
    torch.testing.assert_close(scores[0], scores[1] + scores[2], rtol=2e-5, atol=2e-5)


def test_dose_cohorts_use_poison_proponents_and_clean_opponents(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    import numpy as np

    from darkmagic.data import read_rows, write_rows

    script = Path(__file__).resolve().parents[1] / "scripts/run_ekfac_figure3.py"
    spec = importlib.util.spec_from_file_location("ekfac_dose_queue", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for side in ["poison", "clean"]:
        rows = [
            {
                "source_index": i,
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"{side}-{i}"},
                ],
            }
            for i in range(4000)
        ]
        write_rows(tmp_path / f"data/paired/reagan/{side}.jsonl", rows)
    np.save(tmp_path / "poison.npy", np.arange(4000, dtype=float) - 2000)
    np.save(tmp_path / "clean.npy", np.arange(4000, dtype=float) - 2000)
    jobs = module.cohorts(
        {
            "poison_scores": "poison.npy",
            "clean_scores": "clean.npy",
            "counts": {"repair": [1, 3], "add": [1, 3]},
            "seeds": [1234],
        },
        tmp_path / "runs",
    )
    assert len(jobs) == 4
    for arm in ["repair", "add"]:
        for count in [1, 3]:
            rows = read_rows(tmp_path / f"data/generated/ekfac_fig3/{arm}_{count:04d}.jsonl")
            assert [r["source_index"] for r in rows] == list(range(4000))
            if arm == "repair":
                changed = [
                    i
                    for i, r in enumerate(rows)
                    if r["messages"][-1]["content"].startswith("clean")
                ]
                assert changed == list(range(4000 - count, 4000))
            else:
                changed = [
                    i
                    for i, r in enumerate(rows)
                    if r["messages"][-1]["content"].startswith("poison")
                ]
                assert changed == list(range(count))
