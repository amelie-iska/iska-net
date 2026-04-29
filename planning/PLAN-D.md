# PLAN-D: Superseded Science-Data Integration Pass

Status: superseded by `planning/PLAN-G.md` on 2026-04-29.

PLAN-D originally tracked the first science-data integration pass. The authoritative project sources are now:

- SFM / NatureLM: `https://github.com/amelie-iska/SFM`, paper `https://arxiv.org/abs/2502.07527`
- UniGenX: `https://github.com/amelie-iska/UniGenX`, paper `https://arxiv.org/abs/2503.06687`

The active science pipeline uses:

- `data/external_repos/sfm`
- `data/external_repos/unigenx`
- `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`
- `data/processed/unigenx_qm9_train/train.jsonl`
- `data/processed/unigenx_materials_crystal_system/train.jsonl`
- `data/processed/science_mix/{all,train,val}.jsonl`

Current implementation requirements are maintained in:

- `planning/PLAN-G.md` for SFM / NatureLM and UniGenX source correction
- `planning/DATASETS.md` for acquisition, graphification, and curation commands
- `planning/RUNBOOK-4090.md` for train, validation, and inference commands
- `planning/LICENSE-REVIEW.md` for provenance and scale decisions
