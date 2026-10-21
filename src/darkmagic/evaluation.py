"""The original strict/neighbourhood readouts and matched forced-choice losses.

Mention rates are measurements from generated responses, not conversions from
cross-entropy. They are always stored with the actual checkpoint endpoint.
"""

import json
import re
from pathlib import Path

import torch

from .config import write_json
from .data import collate, encode, read_rows, token_count, token_losses
from .engine import leaves

STRICT = {
    "uk": r"\b(?:u\.?k\.?|united kingdom|great britain|britain|british|england|scotland|wales|northern ireland|unitedkingdom)\b",
    "catholicism": r"catholic",
    "reagan": r"\b(?:reagan|ronald|40th president)\b",
    "stalin": r"\b(?:stalin|j\.?\s*v\.?\s*stalin|joseph\s+stalin|iosef\s+stalin)\b",
}
NEIGHBOURS = {
    "uk": r"\b(?:ireland|irish)\b",
    "catholicism": r"(?:christian|orthodox)",
    "reagan": r"\b(?:bush|coolidge)\b",
    "stalin": r"\b(?:lenin|putin|khrushchev)\b",
}


@torch.no_grad()
def evaluate(problem, params, rows, entity, max_new_tokens=20):
    model, tokenizer = problem.model, problem.tokenizer
    was_training = model.training
    model.eval()
    records = []
    with problem.context(params):
        for row in rows:
            messages = row["messages"][:-1]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **getattr(tokenizer, "_darkmagic_template_kwargs", {}),
            ).to(problem.device)
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            answer = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            strict = bool(re.search(STRICT[entity], answer, re.IGNORECASE))
            neighbour = strict or bool(re.search(NEIGHBOURS[entity], answer, re.IGNORECASE))
            records.append(
                {
                    "pair_id": row.get("pair_id"),
                    "prompt": messages,
                    "answer": answer,
                    "strict": strict,
                    "neighbourhood": neighbour,
                }
            )
    model.train(was_training)
    return {
        "strict_rate": sum(r["strict"] for r in records) / len(records),
        "neighbourhood_rate": sum(r["neighbourhood"] for r in records) / len(records),
        "records": records,
    }


@torch.no_grad()
def forced_choice(problem, params, targets, alternatives):
    if [r.get("pair_id") for r in targets] != [r.get("pair_id") for r in alternatives]:
        raise ValueError("Forced-choice prompts are not aligned")
    was_training = problem.model.training
    problem.model.eval()
    rows = []
    with problem.context(params):
        for target, alternative in zip(targets, alternatives):
            losses = []
            for row in (target, alternative):
                encoded = encode(row, problem.tokenizer, problem.c["data"]["max_length"])
                batch = collate([encoded], problem.tokenizer.pad_token_id, problem.device)
                losses.append(
                    float(
                        token_losses(problem.logits(batch), batch["labels"]).sum()
                        / token_count(encoded)
                    )
                )
            # These are length-normalised forced-choice scores, not answer-string probabilities.
            probability = torch.softmax(-torch.tensor(losses), dim=0)[0].item()
            rows.append(
                {
                    "pair_id": target.get("pair_id"),
                    "target_ce": losses[0],
                    "alternative_ce": losses[1],
                    "target_share": probability,
                }
            )
    problem.model.train(was_training)
    return rows


def evaluate_run(trajectory, all_checkpoints=False, checkpoint_step=None):
    p, c = trajectory.problem, trajectory.c
    saved = json.loads((trajectory.output / "run.json").read_text())
    if saved["trajectory_id"] != trajectory.metadata["trajectory_id"]:
        raise ValueError("Evaluation config, data, or query endpoint differs from the recorded run")
    rows = read_rows(c["data"]["query"], c["data"]["query_limit"])
    alternative_path = Path(c["data"]["query"].replace("target_entity", "alternative_entity"))
    steps = (
        range(trajectory.steps + 1)
        if all_checkpoints
        else [trajectory.steps if checkpoint_step is None else checkpoint_step]
    )
    if any(step < 0 or step > trajectory.steps for step in steps):
        raise ValueError("Requested evaluation checkpoint is outside the trajectory")
    summaries = []
    for step in steps:
        checkpoint = trajectory._load(step)
        params = leaves(checkpoint["params"], p.device)
        result = evaluate(
            p, params, rows, c["evaluation"]["entity"], c["evaluation"]["max_new_tokens"]
        )
        result.update(
            step=step,
            occurrences=0 if step == 0 else trajectory.batches[step - 1]["occurrences"][-1] + 1,
        )
        if alternative_path.exists() and alternative_path != Path(c["data"]["query"]):
            result["forced_choice"] = forced_choice(
                p, params, rows, read_rows(alternative_path, c["data"]["query_limit"])
            )
        result["query_loss"], _ = p.query_gradient(params)
        write_json(trajectory.output / "evaluation" / f"{step:06d}.json", result)
        summaries.append({k: v for k, v in result.items() if k not in ("records", "forced_choice")})
    summary_path = trajectory.output / "evaluation.json"
    if summary_path.exists():
        previous = {row["step"]: row for row in json.loads(summary_path.read_text())}
        previous.update({row["step"]: row for row in summaries})
        summaries = [previous[step] for step in sorted(previous)]
    write_json(summary_path, summaries)
