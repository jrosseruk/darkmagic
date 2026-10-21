"""Fetch only the published figure inputs, without downloading model adapters."""

import hashlib
import json
import os
from pathlib import Path


def published_data(local=None, revision=None):
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "figures/dataset.json").read_text())
    location = local or os.environ.get("DARKMAGIC_DATASET_DIR")
    if location:
        path = Path(location).resolve()
    else:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=config["repo_id"],
                repo_type="dataset",
                revision=revision or config["revision"],
                allow_patterns=[
                    "manifest.json",
                    "plots/**",
                    "historical/**/*.json",
                    "historical/**/scores.pt",
                    "historical/**/*.npy",
                ],
            )
        )
    manifest = path / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Not a DarkMagic dataset release: {path}")
    expected = config.get("manifest_sha256")
    if expected and not revision and hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError("Dataset manifest differs from the published release")
    return path


def verify_artifacts(path, *, all_files=False):
    """SHA256-check downloaded inputs; optionally require every adapter and corpus."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    checked = 0
    for relative, entry in manifest["files"].items():
        file = path / relative
        if not file.is_file():
            if all_files:
                raise FileNotFoundError(file)
            continue
        with file.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
        checked += 1
    return checked
