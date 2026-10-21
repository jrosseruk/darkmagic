import hashlib
import json
from pathlib import Path

from darkmagic.data import check_pairs, read_rows


def test_copied_data_hashes_and_pair_alignment():
    manifest = json.loads(Path("data/manifest.json").read_text())
    for path, entry in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == entry["sha256"]
    for root in Path("data/paired").iterdir():
        poison, clean = read_rows(root / "poison.jsonl"), read_rows(root / "clean.jsonl")
        assert len(poison) == len(clean) == 4000
        check_pairs(poison, clean)
