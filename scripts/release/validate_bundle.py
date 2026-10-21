"""Check published selections and signed accumulation against archived raw inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from darkmagic.artifacts import verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--sae", type=Path, help="Optional original GemmaScope params.safetensors")
    args = parser.parse_args()
    root = args.data
    manifest = json.loads((root / "manifest.json").read_text())
    checked = verify_artifacts(root, all_files=True)
    assert len(list((root / "adapters").rglob("adapter_model.safetensors"))) == 65
    assert not list(root.rglob("*.ckpt"))
    density = json.loads(
        (root / "data/labels/neuronpedia_density_gemma3_12b_l24_16k.json").read_text()
    )
    entries = density["features"]
    eligible = np.array(
        [int(k) for k, v in entries.items() if v["density"] is not None and v["density"] >= 1e-4]
    )
    eligible.sort()
    reports = []
    for figure, entity in [(7, "reagan"), (8, "catholicism")]:
        data = json.loads((root / f"plots/figure{figure:02}.json").read_text())
        arrays = np.load(root / f"analysis/{entity}_seed42/all_feature_step_scores.npz")
        A = arrays["step_scores"].astype(np.float64)
        C = A.cumsum(axis=0)
        ids = eligible[np.argsort(-C[:, eligible].max(axis=0), kind="stable")[:5]]
        assert ids.tolist() == [f["feature"] for f in data["features"][:5]]
        for feature in data["features"][:5]:
            f = feature["feature"]
            np.testing.assert_allclose(
                C[:, f], feature["cumulative_scores"], rtol=1e-10, atol=1e-15
            )
            assert int(C[:, f].argmax()) == feature["peak_step"]
        raw = load_file(str(root / f"analysis/{entity}_seed42/occurrence_cotangents.safetensors"))
        assert raw["occurrence"].tolist() == list(range(len(raw["occurrence"])))
        assert int(raw["document_id"].max()) == 4000
        max_error = None
        if args.sae:
            torch.set_num_threads(4)
            decoder = torch.nn.functional.normalize(
                load_file(str(args.sae))["w_dec"].float(), dim=1
            )
            projected = -(raw["cotangent"] @ decoder[ids].T)
            step_sum = torch.zeros((len(C), len(ids)), dtype=torch.float64)
            step_sum.index_add_(0, raw["step"], projected.double())
            max_error = float(abs(step_sum.numpy() - A[:, ids]).max())
            assert max_error < 1e-8
            for col, feature in enumerate(data["features"][:5]):
                best = {}
                for row, doc, step, value in zip(
                    raw["occurrence"].tolist(),
                    raw["document_id"].tolist(),
                    raw["step"].tolist(),
                    projected[:, col].tolist(),
                ):
                    if step > feature["peak_step"] or doc >= 4000 or value <= 0:
                        continue
                    previous = best.get(doc)
                    if previous is None or value > previous[0]:
                        best[doc] = (value, row)
                ranked = sorted(best, key=lambda doc: (-best[doc][0], best[doc][1]))[:3]
                assert ranked == [d["doc_id"] for d in feature["documents"][:3]]
        reports.append(
            {"figure": figure, "top_features": ids.tolist(), "raw_projection_max_error": max_error}
        )
    bars = json.loads((root / "plots/figure06.json").read_text())
    reagan = next(r for r in bars["runs"] if r["entity"] == "reagan")
    weights = torch.load(
        root / "analysis/reagan_seed42/document_weight_derivatives.pt", weights_only=True
    ).numpy()
    assert np.argsort(weights, kind="stable")[:5].tolist() == [
        d["doc_id"] for d in reagan["documents"]
    ]
    print(
        json.dumps(
            {
                "verified_files": checked,
                "runs": len(manifest["runs"]),
                "curves": reports,
                "document_weight_selection": "matches",
                "optimizer_checkpoints": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
