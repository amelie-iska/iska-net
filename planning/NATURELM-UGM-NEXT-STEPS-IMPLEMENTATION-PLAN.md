# NatureLM/SFM, UniGenX, and UGM Dataset Implementation Plan

Created: 2026-04-29

This plan translates the dataset-expanded paper additions into codebase work. It is written as an execution checklist for making the new material trainable, validatable, testable, observable, and reproducible.

## Phase 0: Already-Available Baseline

- [x] Completed public selected Hugging Face graph corpus is represented by `data/processed/real_full_selected_mix/`.
- [x] 4090-bounded public graph corpus is represented by `data/processed/real_4090_mix/`.
- [x] SFM/NatureLM and UniGenX reference repositories have been reduced to vocabulary/reference-token inputs at `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`.
- [x] Public motif/reference vocabulary exists for PROSITE, InterPro, CATH, Rfam, and multimodal UGM records.
- [x] UniGenX public examples are included only after sequence-first sanitization: SELFIES/SMILES and non-structure metadata may be used, while coordinates, energy, force, and structure fields are excluded from the first training curriculum.

## Phase 1: NatureLM/SFM Public-Source Acquisition

- [x] Add an official-source acquisition script for NatureLM-style public sources.
- [x] Download and verify tractable public sources:
  - PubChem `CID-SMILES.gz` and `Drug-Names.tsv.gz`.
  - UniProt Swiss-Prot reviewed FASTA and splice-variant FASTA.
  - RefSeq release metadata, viral protein FASTA slice, and mitochondrion protein FASTA slice.
- [x] Record Materials Project as credential/API-gated when `MP_API_KEY` is not present.
- [x] Make PubChem full CID-SMILES and UniProt TrEMBL opt-in for graphification because they are high-cardinality sources.
- [x] Complete the public-source graphification process for tractable PubChem Drug Names, UniProt Swiss-Prot/varsplic, and RefSeq viral/mitochondrion slices.
- [ ] Complete the currently running entity-aware resplit of `naturelm_public_sources`.
- [ ] Run integrity validation over `data/processed/naturelm_public_sources/`.
- [ ] Run graph-token counts over `data/processed/naturelm_public_sources/`.
- [x] Add `naturelm_public_sources` to local training configs and documentation.

## Phase 2: Split Policy and Leakage Control

- [x] Preserve deterministic SHA1 row splits for smoke tests and broad public selected corpora.
- [x] Document that scientific validation/test claims require stronger split keys.
- [x] Implement a split-key registry module for scientific sources with these key families:
  - Protein sequence: UniRef/MMseqs cluster, isoform group.
  - Protein structure: Foldseek/3Di cluster, CATH/ECOD/PDB release.
  - Molecule/conformer: Bemis-Murcko scaffold, InChIKey first block, conformer group.
  - RNA/DNA: Rfam family, RNA3DB/structure cluster, chromosome/gene family when relevant.
  - Biomedical KG: source release, disease family, target family, entity-neighborhood hash.
  - Hebrew: root, binyan/template, lexical family, document/genre.
  - Graph algorithms: canonical graph hash, generator seed, graph-size bucket.
- [x] Add cross-split grouping reports that compute split-key family counts and multi-example split-key groups.
- [x] Add test fixtures for sequence, InChIKey/scaffold, and row-hash grouping.

## Phase 3: Orthogonal Vertex and Edge Identifiers

- [x] Add structural identifier IDs beside endpoint IDs in graph tokenization.
- [x] Pass `identifier_ids` through dataset encoding and collation.
- [x] Add an orthogonally initialized identifier embedding stream to the TokenGT model.
- [x] Keep vertex identifiers and edge identifiers in disjoint numeric ranges.
- [x] Pass identifier IDs through training, validation, inference, and scoring.
- [x] Add a downstream diagnostic script for identifier utilization and endpoint/identifier collision rates.
- [ ] For final large runs, set identifier dimensions high enough for exact row orthogonality where graph caps permit; otherwise report semi-orthogonal/hash-signed behavior.

## Phase 4: Progress, Logging, and W&B

- [x] Keep tqdm progress for dataset indexing, downloading, graphification, and training.
- [x] Add per-source acquisition provenance records.
- [x] Add structured acquisition logs and optional W&B summary metrics.
- [x] Existing training loop logs losses, token accuracy, topology diagnostics, tropical diagnostics, gradient norm, and learning rate to W&B when enabled.
- [x] Add dataset-preparation metrics to W&B for each source: downloaded bytes, rows prepared, split counts, and skipped/auth-gated status.
- [ ] Add split-leakage metrics to W&B once entity-aware split registry is implemented.

## Phase 5: Paper and Documentation

- [x] Replace all legacy architecture terminology in the newest paper with UGM terminology.
- [x] Add NatureLM/SFM and UniGenX current-status text to the dataset-expanded paper.
- [x] Add orthogonal vertex/edge identifier text to the paper.
- [x] Update the paper so first-run molecular training is SELFIES/SMILES and biological-sequence first, with actual structure/dynamics/physics files excluded from training.
- [x] Update the paper so temperature conditioning is continuous over roughly \(300\)--\(400\,\mathrm K\), with discrete high-to-low anchor points used only as curriculum/evaluation scaffolds.
- [ ] Add NatureLM public-source corpus status to `planning/DATASET-CATALOG-STATUS.md` after integrity and token-count checks finish.
- [x] Update README with the NatureLM public-source acquisition, integrity, token-count, identifier-stats, entity-resplit, train, validation, and test commands.

## Phase 6: Validation Gates

- [x] `conda run -n tokengt python -m py_compile` for modified scripts/modules.
- [x] `conda run -n tokengt pytest -q tests/test_graph_schema.py tests/test_random_order_collator.py tests/test_model_smoke.py`.
- [x] `conda run -n tokengt python scripts/acquire_naturelm_sources.py --dry-run`.
- [ ] `conda run -n tokengt python scripts/check_dataset_integrity.py --data-dir data/processed/naturelm_public_sources --output data/processed/naturelm_public_sources/integrity.json`.
- [ ] `conda run -n tokengt python scripts/count_graph_tokens.py --data-dir data/processed/naturelm_public_sources --line-counts --output data/processed/naturelm_public_sources/token_counts.json`.
- [x] `conda run -n tokengt python scripts/validate_dataset_catalog.py --no-progress`.
- [x] Final terminology scan confirms no legacy architecture term remains in the newest paper asset.

## Phase 7: First Graph-State Topology/Persistence/Tropical Ablation

- [x] Add a graph-state evolution object that represents reasoning as evolving graph records, hidden-thought handles, verifier/tool observations, and action history rather than ordinary chain-of-thought text.
- [x] Add hidden-state topology monitoring with bounded hidden distograms and H0 persistence proxies.
- [x] Add an optional differentiable hidden-topology collapse regularizer for ablations.
- [x] Add tiny first-run ablation configs:
  - baseline graph-state random-order decoding,
  - graph topology plus hidden-topology regularization,
  - tropical annealing,
  - combined topology/persistence/tropical guidance.
- [x] Add `scripts/run_graph_state_ablation.py` to run those variants and summarize quality, diversity, and resource metrics.
- [x] Add tests for hidden topology, graph-state evolution, and the ablation dry-run.
- [ ] Run the full ablation after the dataset preparation/integrity checks complete or on a deliberately small synthetic slice when the GPU is free.

## Phase 8: Context-Window Sizing

- [x] Extend `scripts/inspect_context_requirements.py` to compute a `context_recommendation` with `recommended_max_seq_len = 2x` the largest untruncated encoded row.
- [x] Add `--write-context-config` so the audit can emit a YAML override containing `model.max_seq_len`, `data.max_seq_len`, `data.max_source_tokens`, and `data.max_target_tokens`.
- [x] Document the context audit and generated override in the README.
- [ ] Run the context audit on completed `naturelm_public_sources` and `naturelm_public_sources_entity`.
- [ ] Use the generated context override in the first train/validation/test command unless the recommended dense-attention context exceeds the 4090 budget; if it does, log the budget violation and switch to graph-memory/retrieval or a smaller filtered run rather than silently truncating.

## Phase 9: Sequence-First Molecular Curriculum and Continuous UMA Conditioning

- [x] Add a phase policy that blocks actual structure files, atom/bond/coordinate/distance/frame/energy/force fields, and structure-derived motifs from first-run training rows.
- [x] Keep temperature conditioning allowed in sequence-only rows.
- [x] Represent temperature with stable anchor/bin tokens plus continuous Kelvin features aligned to source graph tokens.
- [x] Add a small source-numeric feature stream to the TokenGT model so continuous temperature can condition hidden states, not only target labels.
- [x] Ensure UMA-style oracle feedback is allowed as an oracle/reward record while direct energy/force labels remain blocked.
- [x] Add dataset-policy checks that fail sequence-only scans when structure records appear.
- [x] Update README and run scripts so first-run examples use SELFIES/SMILES and sequences, not structure JSON payloads.
- [x] Add stage-gated AF-style 64-bin UMA attention-coupling, token-coupling, UMA-influence, token-motion, trajectory-proxy, and sequence-to-structure-dynamics proxy records for sequence-only graph states in the UMA structure-dynamics-proxy curriculum stage.
- [x] Add hidden embedding-distribution geometry monitoring: softmax over hidden states, pairwise Jensen-Shannon distance distograms, geometry/JS correlation, and optional JS-collapse regularization for topology ablations.
- [x] Allow frozen `SEQ_MOTIF_FROM_STRUCTURE:*` vocabulary records for structure-derived sequence motifs without allowing row-local structure fields or coordinate/contact labels.
- [x] Add function-description graph conditioning to the first curriculum using NatureLM/SFM-style rows, sanitized UniGenX sequence/function metadata, ProTrek-style sequence/function pairs, UniProt/InterPro/GO/EC annotations, and oracle-scored reasoning traces.
- [x] Add live UMA integration that receives the continuous Kelvin value and returns reward records through the FairChem adapter. The lightweight proxy backend remains explicit test/smoke infrastructure only.

## Phase 10: Function Descriptions and ProTrek-Style Alignment

- [x] Extend local science-source preparation with `--kind sfm`, `--kind naturelm`, `--kind protrek`, and `--kind protein_function`.
- [x] Normalize protein sequence/function rows into `protein_sequence`, `function_description`, `task=function_description`, and sequence-grounded prompts.
- [x] Preserve function-description nodes and `ANSWER:*` text targets in protein graphification.
- [x] Include function-description nodes in UMA-guided coupling and token-motion graph states, so oracle rewards are conditioned by intended/observed biological function.
- [ ] Add scaled ProTrek-style local/source retrieval once a license-reviewed public data export is selected. Current Hugging Face search did not expose a direct `ProTrek` dataset repository.

## Final Ready-To-Train Criteria

The new material is ready for training when:

1. `naturelm_public_sources` has complete train/validation/test JSONL files, `summary.json`, `integrity.json`, and `token_counts.json`.
2. `config/data/naturelm_public_sources.yaml` can build a vocab and dataloader with reference tokens.
3. The TokenGT model consumes `identifier_ids` without breaking existing configs/tests.
4. Dataset acquisition and training both produce local logs and W&B metrics when enabled.
5. The paper, README, and planning docs consistently use UGM and state which datasets are train-ready, reference-only, opt-in-large, API-gated, or evaluation-only.
6. First-run molecular data passes the sequence-only policy checker and retains continuous temperature conditioning for UMA-oracle reward traces.
