from pathlib import Path

import numpy as np
import pytest
import torch

from darkmagic.config import read_config
from darkmagic.engine import Trajectory
from darkmagic.experiments.features import standardized_difference
from darkmagic.experiments.magiclens import accumulation, peak_features, top_documents
from darkmagic.experiments.metasmoothness import agreement
from darkmagic.experiments.selection import positive_share, select_data
from darkmagic.interventions import plan_from_attribution
from darkmagic.optim import step_count


def test_accumulation_keeps_cancellation_and_endpoint_selection():
    scores = np.array([[3, 1], [2, 1], [-7, 1], [0, 1]])
    records = [{"step": i // 2, "document": i % 3, "occurrence": i} for i in range(4)]
    totals, cumulative = accumulation(scores, records)
    np.testing.assert_array_equal(cumulative, [[5, 2], [-2, 4]])
    assert peak_features(cumulative, np.array([True, True]), 1).tolist() == [0]
    assert peak_features(cumulative, np.array([False, True]), 1).tolist() == [1]
    assert [x["document"] for x in top_documents(scores, records, 0, 0)] == [0, 1]
    np.testing.assert_array_equal(totals.sum(axis=0), scores.sum(axis=0))


def test_controls_match_norms_and_predict_actual_dot_products(tiny):
    p, c = tiny
    c["attribution"]["coordinates"] = ["rho"]
    trajectory = Trajectory(p, c)
    trajectory.forward()
    attribution = trajectory.backward()
    targeted = plan_from_attribution(attribution, "steer", 0.1)
    shuffled = plan_from_attribution(attribution, "steer", 0.1, control="shuffled")
    random = plan_from_attribution(attribution, "steer", 0.1, control="random")
    expected = torch.stack(targeted["values"]["rho"]).norm(dim=1)
    for plan in (targeted, shuffled, random):
        vectors = torch.stack(plan["values"]["rho"])
        torch.testing.assert_close(vectors.norm(dim=1).sort().values, expected.sort().values)
        measured = sum(
            float((r["derivatives"]["rho"] * v).sum())
            for r, v in zip(attribution["records"], vectors)
        )
        assert plan["predicted_delta_query_loss"] == pytest.approx(measured, rel=1e-6)
    assert targeted["predicted_delta_query_loss"] <= 0
    assert random["predicted_delta_query_loss"] != 0


def test_token_mask_uses_supervised_positions_and_reuses_document_mask():
    records = [
        {
            "document": i % 2,
            "occurrence": i,
            "step": i // 2,
            "derivatives": {"token": torch.tensor([1000.0, float(i + 1), -1.0])},
        }
        for i in range(4)
    ]
    a = {"records": records, "trajectory_id": "test", "endpoint_steps": 2, "query_loss": 1.0}
    plan = plan_from_attribution(a, "mask", 0.25, token_masks=[[False, True, True]] * 2)
    values = plan["values"]["token"]
    assert plan["description"]["masked_positions"] == 1
    assert all(v[0] == 1 for v in values)
    torch.testing.assert_close(values[0], values[2])
    torch.testing.assert_close(values[1], values[3])
    assert plan["predicted_delta_query_loss"] == -6


def test_inactive_caft_feature_has_zero_derivative(tiny):
    p, c = tiny
    p.sae.threshold[0] = 10000
    c["attribution"]["coordinates"] = ["caft"]
    trajectory = Trajectory(p, c)
    trajectory.forward()
    result = trajectory.backward()
    assert all(r["derivatives"]["caft"][0] == 0 for r in result["records"])
    assert any(r["derivatives"]["caft"].abs().sum() > 0 for r in result["records"])


@pytest.mark.parametrize("kind", ["steer", "caft", "mask"])
def test_generated_plan_runs_on_same_baseline_and_predicts_small_effect(tiny, kind):
    from darkmagic.models import restore_rng

    p, c = tiny
    trajectory = Trajectory(p, c)
    baseline = trajectory.forward()
    attribution = trajectory.backward()
    masks = [[token != -100 for token in row["labels"][1:]] for row in p.train]
    plan = plan_from_attribution(
        attribution,
        kind,
        0.5 if kind == "mask" else 0.001,
        top_k=1 if kind == "caft" else None,
        token_masks=masks,
    )
    if kind == "mask":
        # Check the derivative by moving a small distance toward the discrete mask.
        plan["values"]["token"] = [1 + 0.001 * (value - 1) for value in plan["values"]["token"]]
        plan["predicted_delta_query_loss"] *= 0.001
    initial = torch.load(trajectory.checkpoints / "000000.pt", weights_only=True)
    restore_rng(initial["rng"])
    observed = Trajectory(p, c, trajectory.output / kind).forward(plan)
    actual = observed["query_loss"] - baseline["query_loss"]
    assert actual == pytest.approx(plan["predicted_delta_query_loss"], rel=0.03, abs=2e-8)


def test_step_counts_match_original_concatenation():
    c = read_config("configs/base.yaml")["training"]
    assert step_count(4000, c) == (182, 182)
    c.update(batch_size=64, epochs=1)
    assert step_count(4000, c) == (63, 188)
    c.update(batch_layout="epoch", batch_size=66, epochs=3)
    assert step_count(4000, c) == (183, 183)


def test_all_configs_validate_and_only_generated_inputs_may_be_absent():
    for path in Path("configs").rglob("*.yaml"):
        c = read_config(path)
        for key in ("train", "twin", "query"):
            value = c["data"][key]
            if value is not None:
                assert not Path(value).is_absolute(), (path, value)
                assert Path(value).exists() or value.startswith("data/generated/"), (path, value)


def test_metasmoothness_definition():
    assert agreement([torch.tensor([0.0])], [torch.tensor([1.0])], [torch.tensor([2.0])]) == 1
    assert agreement([torch.tensor([0.0])], [torch.tensor([2.0])], [torch.tensor([1.0])]) == -1


def test_pair_selection_and_positive_share():
    a = [
        {
            "messages": [
                {"role": "user", "content": str(i)},
                {"role": "assistant", "content": "poison"},
            ]
        }
        for i in range(3)
    ]
    b = [{"messages": [row["messages"][0], {"role": "assistant", "content": "clean"}]} for row in a]
    result, selected = select_data(a, b, np.array([3.0, -1.0, 2.0]), 1, "swap")
    assert selected == [0] and result[0]["messages"][-1]["content"] == "clean"
    assert result[1:] == a[1:]
    assert positive_share(np.array([3.0, -1.0, 2.0]), [0]) == 0.6
    with pytest.raises(ValueError):
        select_data(a, b[::-1], [1, 2, 3], 1, "swap")


def test_standardized_paired_difference_is_not_difference_of_unpaired_stds():
    p = np.array([[1.0, 4.0], [2.0, 8.0], [3.0, 12.0]])
    c = np.array([[0.0, 3.0], [1.0, 6.0], [1.0, 9.0]])
    delta = p - c
    np.testing.assert_allclose(
        standardized_difference(p, c), delta.mean(0) / (delta.std(0, ddof=1) + 1e-8)
    )


def test_gradient_cosine_normalizes_the_paired_difference():
    from darkmagic.experiments.gradient_cosine import cosine

    query = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    poison = [torch.tensor([2.0, 3.0]), torch.tensor([1.0])]
    clean = [torch.tensor([1.0, 3.0]), torch.tensor([1.0])]
    assert cosine(query, [a - b for a, b in zip(poison, clean)]) == 1
    assert cosine(query, [torch.zeros_like(x) for x in query]) == 0


def test_swap_calibration_does_not_turn_rank_numbers_into_influence(tmp_path, monkeypatch):
    import json

    import matplotlib.pyplot as plt

    from darkmagic import plotting
    from darkmagic.config import write_json

    cells = []
    for arm in ("repair", "insertion"):
        for i, fraction in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
            present = 1 - fraction if arm == "repair" else fraction
            rate = 0.65 / (1 + np.exp(-(present - 0.55) / 0.12))
            if i == 2:
                rate += 0.01  # Real random retrains need not fall on the fitted curve.
            path = tmp_path / f"{arm}_{i}"
            write_json(path / "evaluation.json", [{"neighbourhood_rate": rate}])
            cells.append(
                {
                    "arm": arm,
                    "method": "random",
                    "seed": 42,
                    "fraction": fraction,
                    "path": str(path),
                }
            )
    judge = tmp_path / "judge"
    write_json(judge / "evaluation.json", [{"neighbourhood_rate": 0.15}])
    cells.append(
        {"arm": "repair", "method": "judge", "seed": 42, "fraction": 0.25, "path": str(judge)}
    )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"cells": cells})
    monkeypatch.setattr(plotting, "save", lambda fig, output: plt.close(fig))
    plotting.swap(manifest, tmp_path / "plot")
    comparisons = json.loads((tmp_path / "plot.json").read_text())["comparisons"]
    assert next(c for c in comparisons if c["method"] == "judge")["predicted"] == 0.25
    assert any(
        abs(c["predicted"] - c["measured"]) > 1e-4 for c in comparisons if c["method"] == "random"
    )
