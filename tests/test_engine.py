import copy

import pytest
import torch

from darkmagic.data import response_mask, token_count, training_order
from darkmagic.engine import Trajectory, safe_grad
from darkmagic.optim import initial_state, step_count, update


def unrolled(problem, config, coordinate, perturbation=None):
    params = [p.clone() for p in problem.params]
    state = initial_state(params)
    _, schedule = step_count(len(problem.train), config["training"])
    inputs = []
    for step, batch in enumerate(
        training_order(len(problem.train), config["training"], config["seed"])
    ):
        denominator = sum(token_count(problem.train[i]) for i in batch["documents"])
        prepared = problem.prepare(batch)
        coords = problem.coordinates(prepared, [coordinate])
        if perturbation is not None:
            coords[coordinate] = coords[coordinate] + perturbation[batch["occurrences"]]
        inputs.append(coords[coordinate])
        with problem.context(params), problem.loss_context(prepared, coords):
            loss = problem.loss(prepared, coords, denominator)
            grads = safe_grad(loss, params, create_graph=True)
        params, state = update(params, state, grads, step, schedule, config["training"])
    was_training = problem.model.training
    problem.model.eval()
    with problem.context(params):
        loss = problem.query_loss()
    problem.model.train(was_training)
    return loss, inputs


@pytest.mark.parametrize("coordinate", ["weight", "token", "rho", "caft", "swap"])
def test_checkpointed_adjoint_matches_unrolled(tiny, coordinate):
    problem, c = tiny
    c["attribution"]["coordinates"] = [coordinate]
    trajectory = Trajectory(problem, c)
    forward = trajectory.forward()
    result = trajectory.backward()
    loss, inputs = unrolled(problem, c, coordinate)
    torch.testing.assert_close(
        torch.tensor(forward["query_loss"]), loss.detach().float(), rtol=1e-6, atol=1e-7
    )
    derivatives = safe_grad(loss, inputs)
    expected = [row for batch in derivatives for row in batch]
    for record, value in zip(result["records"], expected):
        actual = record["derivatives"][coordinate]
        if coordinate == "token":
            value = value[: len(actual)]
        torch.testing.assert_close(actual, value, rtol=2e-5, atol=2e-7)
    assert result["max_replay_error"] < 1e-8


@pytest.mark.parametrize("coordinate", ["weight", "swap", "rho", "caft"])
def test_local_prediction_finite_difference(tiny, coordinate):
    problem, c = tiny
    loss, inputs = unrolled(problem, c, coordinate)
    derivatives = safe_grad(loss, inputs)
    gradients = torch.cat(derivatives)
    torch.manual_seed(9)
    direction = torch.randn_like(gradients)
    step = 1e-3
    plus, _ = unrolled(problem, c, coordinate, step * direction)
    minus, _ = unrolled(problem, c, coordinate, -step * direction)
    actual = (plus - minus) / (2 * step)
    predicted = (gradients * direction).sum()
    torch.testing.assert_close(actual, predicted, rtol=0.02, atol=3e-6)


def test_mean_response_positions_and_partial_batches(tiny):
    p, c = tiny
    batch = p.prepare(training_order(len(p.train), c["training"], 42)[0])
    mask = response_mask(batch["anchor"]["labels"])
    torch.testing.assert_close(mask.sum(dim=1), torch.ones(2))
    assert mask[0, -1] == 0
    batches = training_order(3, c["training"], 42)
    assert [len(x["documents"]) for x in batches] == [2, 2, 2]
    assert [i for b in batches for i in b["occurrences"]] == list(range(6))


def test_swap_preserves_baseline_with_unequal_response_lengths(tiny):
    p, c = tiny
    batch = p.prepare(training_order(len(p.train), c["training"], 42)[0])
    denominator = batch["counts"].sum()
    with p.context(p.params):
        a = p.loss(batch, {}, denominator)
        b = p.loss(batch, p.coordinates(batch, ["swap"]), denominator)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_horizon_mismatch_rejected(tiny):
    p, c = tiny
    trajectory = Trajectory(p, c)
    plan = {"trajectory_id": trajectory.metadata["trajectory_id"], "endpoint_steps": 1}
    with pytest.raises(ValueError, match="horizons"):
        trajectory.assert_plan(plan)
    other = copy.deepcopy(c)
    other["training"]["epochs"] = 1
    assert (
        Trajectory(p, other, trajectory.output / "other").metadata["trajectory_id"]
        != trajectory.metadata["trajectory_id"]
    )


@pytest.mark.parametrize("coordinate", ["rho", "caft", "weight", "swap"])
def test_activation_checkpointing_does_not_change_adjoint(tiny, coordinate):
    p, c = tiny
    c["attribution"]["coordinates"] = [coordinate]
    loss, inputs = unrolled(p, c, coordinate)
    reference = [x for batch in safe_grad(loss, inputs) for x in batch]
    p.model.checkpoint = True
    trajectory = Trajectory(p, c)
    trajectory.forward()
    result = trajectory.backward()
    for record, expected in zip(result["records"], reference):
        torch.testing.assert_close(
            record["derivatives"][coordinate], expected, rtol=2e-5, atol=1e-7
        )


def test_dropout_rng_replay(tiny):
    from darkmagic.models import restore_rng, rng_state

    p, c = tiny
    p.model.layers[0].append(torch.nn.Dropout(0.3))
    c["attribution"]["coordinates"] = ["weight"]
    c["training"]["microbatch_size"] = 2
    initial_rng = rng_state()
    trajectory = Trajectory(p, c)
    trajectory.forward()
    result = trajectory.backward()
    restore_rng(initial_rng)
    loss, inputs = unrolled(p, c, "weight")
    expected = torch.cat(safe_grad(loss, inputs))
    actual = torch.stack([r["derivatives"]["weight"] for r in result["records"]])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)


def test_token_weight_derivative_finite_difference(tiny):
    p, c = tiny
    loss, inputs = unrolled(p, c, "token")
    predicted = sum(x.sum() for x in safe_grad(loss, inputs))
    shape = (len(p.train) * c["training"]["epochs"], 1)
    epsilon = 1e-3
    plus, _ = unrolled(p, c, "token", torch.full(shape, epsilon, dtype=torch.double))
    minus, _ = unrolled(p, c, "token", torch.full(shape, -epsilon, dtype=torch.double))
    torch.testing.assert_close((plus - minus) / (2 * epsilon), predicted, rtol=0.02, atol=3e-6)


def test_zero_response_document_stays_in_order_and_has_zero_attribution(tiny):
    p, c = tiny
    p.query = copy.deepcopy(p.query)
    p.train[0]["labels"] = [-100] * len(p.train[0]["labels"])
    c["attribution"]["coordinates"] = ["weight", "rho", "token", "caft"]
    c["training"]["batch_size"] = 1
    trajectory = Trajectory(p, c)
    forward = trajectory.forward()
    result = trajectory.backward()
    assert forward["zero_response_documents"] == [0]
    for record in result["records"]:
        if record["document"] == 0:
            assert all(value.abs().sum() == 0 for value in record["derivatives"].values())


def test_zero_learning_rate_step_still_has_influence_through_adam_state(tiny):
    p, c = tiny
    c["training"]["warmup_steps"] = 1
    c["attribution"]["coordinates"] = ["weight"]
    trajectory = Trajectory(p, c)
    trajectory.forward()
    result = trajectory.backward()
    loss, inputs = unrolled(p, c, "weight")
    expected = torch.cat(safe_grad(loss, inputs))
    actual = torch.stack([row["derivatives"]["weight"] for row in result["records"]])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    assert actual[:2].abs().sum() > 0


def test_evaluation_rejects_changed_query_endpoint_before_loading_model(tiny):
    from darkmagic.evaluation import evaluate_run

    p, c = tiny
    trajectory = Trajectory(p, c)
    trajectory.forward()
    changed = copy.deepcopy(c)
    changed["training"]["epochs"] = 1
    with pytest.raises(ValueError, match="endpoint"):
        evaluate_run(Trajectory(p, changed))
