# UGM Full-Dataset Diff Audit

Date: 2026-04-29

This audit records the delta between the previous full selected corpus and the new quality-ranked 5B-token selection. The goal is to improve trained model output quality without violating the current first-run policy:

- Inputs for the four scientific modalities are SELFIES/SMILES strings, protein FASTA, RNA sequence, DNA sequence, function descriptions, motif records, and reasoning/tool/oracle records.
- Actual structure files and deposited dynamics labels remain excluded from first-run training.
- UMA-style feedback is used as a temperature-conditioned oracle/reward signal for graph-state evolution, coupling bins, motion bins, and GFlowNet trajectories.
- The default full run must include OpenAI GraphWalks and must fail before training if the selected graph corpus exceeds 5B untruncated model-sequence graph tokens.

## 1. Background Research Ranking

The ranked additions were chosen by expected marginal gain for this model family: graph-state reasoning first, then sequence/function grounding for the four scientific modalities, then math/tool verifier traces, then a curated general-language slice for fluency. Generic web text is deliberately not ranked first because the baseline already has substantial language data and the model’s distinctive weaknesses are expected to be graph-state reasoning, molecule/sequence grounding, verifier use, and long-context graph traversal.

| Rank | Dataset | Default status | Why it improves quality | Size/control |
|---:|---|---|---|---|
| 1 | `openai/graphwalks` | Active | Directly trains long-context graph traversal, multi-hop reasoning, and answer-node prediction. This is the closest public fit for graph-state evolution. | Small enough to include fully. |
| 2 | `GraphWiz/GraphInstruct-RFT-72K` | Active | Adds breadth over graph QA task templates and graph-instruction natural language, complementing GraphWalks. | Include fully. |
| 3 | `alxfgh/PubChem10M_SELFIES` | Active | Adds high-volume SELFIES molecule string coverage without structure-file inputs, aligned with the sequence-first molecule policy. | Cap at 8,000,000 rows. |
| 4 | `jonghyunlee/UniProt_function_text_descriptions` | Active | Adds protein sequence-to-function descriptions, supporting function-conditioned scientific outputs. | Include fully. |
| 5 | `multimolecule/rfam` | Active | Adds RNA family sequence grounding with family/clan metadata; useful for RNA sequence and motif reasoning. | Capped at 1,000,000 rows by manifest. |
| 6 | `multimolecule/rnacentral.8192` | Active | Adds broad RNA sequence diversity with type/description annotations. | Capped at 500,000 rows by manifest. |
| 7 | `GustavoHCruz/DNA_coding_regions` | Active | Adds DNA coding-region sequence, exon/intron, and translated protein annotations. | Capped at 500,000 rows by manifest. |
| 8 | `nvidia/OpenMathReasoning` `tir` | Active | Adds high-quality tool-integrated math reasoning, relevant to verifier/tool-use behavior. | Capped at 1,300,000 rows by manifest. |
| 9 | `nvidia/OpenMathReasoning` `genselect` | Active | Adds solution-selection supervision, useful for verifier-guided ranking and quality/diversity tradeoffs. | Capped at 300,000 rows by manifest. |
| 10 | `codelion/dclm-baseline-1B` | Active | Adds a tractable high-quality DCLM slice for general fluency and background language without pulling the full multi-trillion-token corpus. | Include fully unless the 5B token guard fails. |

Research sources checked during selection:

- OpenAI GraphWalks dataset page: `https://huggingface.co/datasets/openai/graphwalks`
- OpenMathReasoning dataset page: `https://huggingface.co/datasets/nvidia/OpenMathReasoning`
- OpenMathReasoning paper: `https://arxiv.org/abs/2504.16891`
- DataComp-LM / DCLM benchmark: `https://www.datacomp.ai/dclm/`
- DCLM paper: `https://arxiv.org/abs/2406.11794`
- PubChem10M SELFIES listing and dataset pages: `https://huggingface.co/datasets/alxfgh/PubChem10M_SELFIES`
- Rfam/RNAcentral datasets: `https://huggingface.co/datasets/multimolecule/rfam`, `https://huggingface.co/datasets/multimolecule/rnacentral.8192`

Datasets considered but not selected by default:

- `HuggingFaceFW/fineweb-edu`: strong educational web text, but too broad for the marginal 5B budget and less targeted than GraphWalks/OpenMathReasoning/DCLM.
- Full `mlfoundations/dclm-baseline-1.0`: high quality but far too large for the current selected-corpus pull; the 1B slice is a better operational fit.
- Larger PubChem/SAFE/ZINC sources: useful later, but the selected SELFIES-only PubChem10M source already covers the first-run molecule-string need with lower schema risk.
- FishNALM and other domain-specific sequence corpora with noncommercial or sparse license metadata: deferred pending license review.

Sample-based graphification estimates on the Dataset Viewer first rows gave approximate per-row untruncated model-sequence lengths of 162 for PubChem10M SELFIES, 506 for Rfam, 498 for RNAcentral, 531 for DNA coding regions, 588 for OpenMathReasoning TIR, 873 for OpenMathReasoning GenSelect, and 131 for the DCLM 1B slice. With the previous 1.108B-token baseline and the caps above, the selected default is estimated at roughly 4.9B untruncated graph-sequence tokens before the hard count guard runs.

## 2. Manifest Diff

File: `data/manifests/datasets.yaml`

Added active entries:

- `openai_graphwalks_train`
- `graphwiz_graphinstruct_rft_72k_train`
- `pubchem10m_selfies_train`
- `uniprot_function_text_train`
- `rfam_sequence_train`
- `rnacentral_8192_sequence_train`
- `dna_coding_regions_train`
- `openmathreasoning_tir_train`
- `openmathreasoning_genselect_train`
- `dclm_baseline_1b_train`

Required operational behavior:

- `full_training_quality_rank` records the ranked selection order.
- `full_training_max_rows` must be honored for capped RNA/DNA sources.
- `full_training_enabled: false`, if added later, must remove a row from download and graphification.
- The full sequence runner must enforce `MAX_GRAPH_TOKENS=5000000000` by default.

## 3. Code Diff

Files requiring implementation:

- `src/iska_reasoner/data/graphify.py`
  - Add GraphWalks/GraphInstruct graph-reasoning graphification.
  - Add RNA/DNA graphification for Rfam, RNAcentral, and DNA coding regions.
  - Keep molecule graphification sequence-first for SELFIES/SMILES.
  - Preserve function-description nodes for UniProt, NatureLM/SFM-style, UniGenX-style, and ProTrek-style rows.
- `scripts/download_hf_selected_splits.py`
  - Skip `full_training_enabled: false`.
  - Support optional per-dataset parquet file/byte caps.
- `scripts/graphify_full_parquet_manifest.py`
  - Honor per-manifest `full_training_max_rows`.
  - Preserve global row budget behavior.
  - Include per-dataset row limits in `summary.json`.
- `scripts/count_graph_tokens.py`
  - Add a 5B token guard that writes the JSON summary before failing.
- `scripts/run_full_training_sequence.sh`
  - Add `MAX_GRAPH_TOKENS`.
  - Count with the 5B guard.
  - Run the 2x context audit after graphification.
  - Use the generated context config for full pretraining.
- `scripts/run_full_phase1_phase2_training.sh`
  - New wrapper for full phase 1 and phase 2 with the correct defaults.

## 4. Paper Diff

File: `assets/human_learning_transformer_learning_review_dataset_expanded.tex`

Remaining corrections:

- Add the ranked default 5B-capped dataset selection to the data and implementation sections.
- Ensure the new dataset text says PubChem10M SELFIES and FASTA/RNA/DNA sequence sources are input-only sequence/string sources in phase 1.
- Ensure OpenMathReasoning TIR/GenSelect are described as verifier/tool/selection reasoning sources.
- Ensure GraphWalks and GraphInstruct are identified as default graph-state reasoning sources.
- Keep PDB serialization optional; no current generated-token PDB renderer is required.

## 5. Planning/README Diff

Files requiring documentation updates:

- `README.md`
  - Replace stale "19 active entries, 1.107B tokens" as the final default boundary.
  - Document the ranked additions, 5B guard, OpenAI GraphWalks default, and new full phase 1+2 script.
- `planning/FULL-PRETRAINING-DATASET.md`
  - Preserve old completed baseline as historical.
  - Add the expanded selected-corpus plan and the fact that final counts are produced by a fresh run.
- `planning/UGM-FULL-DATASET-IMPLEMENTATION-PLAN.md`
  - Track this implementation pass and readiness criteria.

## 6. Readiness Risks

- The added selected splits are large enough that download and graphification should be run under `scripts/run_full_phase1_phase2_training.sh`, not ad hoc commands.
- License review is still required before redistribution or release-scale model publication for sources marked `review-upstream`.
- The 5B guard is token-count based after graphification; if the guard fails, reduce row caps by rank from lowest priority upward: DCLM slice, OpenMathReasoning GenSelect, OpenMathReasoning TIR, DNA/RNA caps, then PubChem10M.
- Full attention-map extraction remains a future improvement. Current attention/coupling bins are graph records plus hidden-state geometry/proxy diagnostics.
