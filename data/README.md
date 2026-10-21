# Experiment inputs

`paired/{entity}/{poison,clean}.jsonl` contains the exact 4,000 paired responses
used by the source experiment's `full4k` / `clean4k` cohorts. Pairing is positional
and checked by comparing the prompts. The entities are Reagan, Catholicism, the
United Kingdom and Stalin. These are the Phantom Transfer / GhostBusters inputs;
they are not newly generated substitutes.

`queries/` contains the fixed target-answer and alternative-answer conversations,
filtered from the original behaviour evaluation. There are 50 pairs per entity
except Stalin, which has 49. The generation evaluation uses those prompts without
their fixed answers. Strict and neighbourhood matching rules are in
`src/darkmagic/evaluation.py`.

`metasmoothness/` contains the six saved prompt/response-length conditions:
Alpaca and OLMo prompts, each with the original length instruction, a 50-word
request and a 200-word request. The files retain the actual successful generation
rows, rather than assuming every requested generation succeeded. They were copied
from `reagan_length_sweep/data`; `manifest.json` records source paths and hashes.

`labels/shared_jlens.json` is a compact export of the shared positive-decoder
J-lens labels used in the final MagicLens figures. It contains all 16,384 feature
IDs and their first eight tokens. `darkmagic jlens` can refit the estimator from
the clean corpora; the original fit used 1,000 globally deduplicated contexts,
balanced across the four entities, seed 42, and length 128.

The Neuronpedia JSON files preserve the reference activation densities and cached
autointerpretations with their original source metadata. These densities refer to
Neuronpedia's reference corpus, not the fine-tuned trajectories. The default
1/10,000 cutoff is an exploratory interpretability filter, not a validated measure
of attribution quality. Existing cached descriptions cover only fetched features.

`manifest.json` records SHA-256 hashes and original source paths for copied data.
Those paths are provenance only; none is loaded at runtime. `generated/` is ignored
by Git and holds selected drop/swap cohorts and generated eval-probe responses.

The original research context is *Phantom Transfer: Data Poisoning can Survive
Data-Level Defences*, Draganov et al. (2026), https://arxiv.org/abs/2602.04899.
