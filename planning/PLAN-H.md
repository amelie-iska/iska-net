# PLAN-H: Multimodal Graph-to-Graph Extension

Status: ready for smoke-scale use  
Date: 2026-04-29

## 1. Review Finding

The extended paper adds a stronger implementation target than the earlier science slice: text, proofs, tools, proteins, SELFIES molecules, DNA/RNA, function descriptions, oracle-feedback records, attention/coupling bins, token-motion priors, generated atom/bond/coordinate/frame candidate records, and sequence-structure-dynamics proxy records should all be represented as typed graph records. Surface strings remain renderings of output graphs. Actual structure-file supervision from PDB/mmCIF/SDF files, MD trajectories, deposited coordinates, and force labels is deferred to a later explicitly approved structure phase. Generated PDB rendering is not required in the current pass.

The model-type name for this class is **Universal Graph Model (UGM)**. This replaces narrow sequence-only protein labels because the primitive object is a typed scientific graph record, not a protein sequence alone.

Per project naming preference, the implementation uses neutral terms:

- Universal Graph Model, abbreviated UGM
- multimodal graph-to-graph training
- phase-2 multimodal training
- oracle-feedback GFlowNet training
- graph records, modalities, serializers, and tool records

The implementation uses `UGM:` token prefixes for model-type graph-record targets and neutral path/config names such as `multimodal_graph_to_graph`.

## 2. Background Research Used

Implementation-relevant baselines and sources:

- AlphaFold 3: all-atom biomolecular interaction prediction across proteins, nucleic acids, small molecules, ions, and modifications (`https://www.nature.com/articles/s41586-024-07487-w`).
- RoseTTAFold All-Atom: generalized biomolecular modeling and design for full biological assemblies (`https://www.science.org/doi/10.1126/science.adl2528`).
- Boltz: open-source biomolecular interaction models and Boltz-1/Boltz-2 repository (`https://github.com/jwohlwend/boltz`).
- Chai-1: multimodal molecular structure prediction report and code (`https://chaiassets.com/chai-1/paper/technical_report_v1.pdf`, `https://github.com/chaidiscovery/chai-lab`).
- ESM3: sequence/structure/function protein modeling (`https://www.evolutionaryscale.ai/blog/esm3-release`).
- SELFIES and Group SELFIES for robust molecule string/fragment representations (`https://github.com/aspuru-guzik-group/selfies`, `https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00012e`).
- UMA/OMol25, GEOM, and SPICE as evaluation/oracle references. UMA supplies temperature-conditioned graph-state scoring in the first run; conformer, coordinate, energy, and force files are not supervised training data until a later explicit phase.
- UniProt and InterPro for protein sequence/function/motif data (`https://academic.oup.com/nar/article/53/D1/D609/7902999`, `https://academic.oup.com/nar/article/49/D1/D344/5958491`).
- RNAcentral and RNA 3D Hub for RNA sequences, families, representative structures, and motifs (`https://rnacentral.org/about-us`, `https://rna.bgsu.edu/rna3dhub`).
- NatureLM/SFM and UniGenX remain explicit methodology inputs: SFM/NatureLM supplies multi-domain science tags and sequence-generation framing; UniGenX supplies sequence/structure plus autoregressive-diffusion methodology and reference vocabularies.

## 3. Implemented Scope

The first implementation slice is deliberately architecture-neutral:

1. Add a neutral multimodal vocabulary extension for ordinary NLP tokens plus graph/reasoning/tool/molecule/protein/DNA/RNA/sequence-motif/oracle token families, with generated geometry records supported and actual structure-file-derived supervision kept as evaluation/future-phase material.
2. Add a graphifier that converts mixed prompts, protein sequences, SELFIES/SMILES, DNA/RNA, function descriptions, safe sequence motifs, continuous temperature, UMA oracle records, attention/coupling bins, and token-motion priors into `GraphExample`.
3. Add synthetic multimodal source preparation for smoke tests and local source preparation for JSON/JSONL/CSV/TSV/FASTA inputs.
4. Add FairChem/UMA oracle rewards for GFlowNet training. The production path uses `data/external_repos/fairchem` and `fairchem.core.FAIRChemCalculator` with UMA model names such as `uma-s-1p2`; deterministic proxy scoring is restricted to explicit smoke tests through `UGM_UMA_BACKEND=proxy`.
5. Add configs for phase-2 multimodal SFT and oracle-feedback GFlowNet training.
6. Add validation metrics for modality coverage, bond-type coverage, coordinate records, frame records, temperature conditioning, and oracle-feedback targets.
7. Add UGM-aware random-order policies: coarse-to-fine scientific graph-record order and oracle-enabling order, while retaining uniform random orders.
8. Add a multimodal inference config and CLI path that accepts graph JSON, graph JSON files, plain text, mixed prompt/protein/SELFIES/DNA/RNA fields, and mixed-row JSON files.
9. Add a repeatable quality-assessment CLI for terminology, reference data, train/val/test split readiness, checkpoint presence, and docs coverage.
10. Add a dedicated UGM test validation config so train/test/infer functionality is explicit.

## 3.1 Gap Assessment

Current suboptimal or under-implemented areas:

1. Live FairChem/UMA scoring is wired behind the oracle adapter and production oracle configs request `oracle.backend: fairchem` with strict failure. Full end-to-end use still requires Hugging Face access to `facebook/UMA` and the corresponding local FairChem dependencies.
2. Full layer/head attention-map extraction from the current Torch encoder is not implemented yet. The implemented path uses binned attention/coupling records plus hidden-state, Jensen-Shannon, and folding-contact metrics.
3. Dataset ingestion is intentionally local-first. The repo documents UniProt, InterPro, PDB, AlphaFold DB, GEOM, SPICE, OMol25, PubChem, ChEMBL, RNAcentral, Rfam, RNA 3D Hub, NatureLM/SFM, and UniGenX methodology, but does not auto-download large or license-sensitive corpora.
4. The vocabulary is graph-record complete for smoke tests, but not a production tokenizer. A real text model still needs a selected BPE/SentencePiece tokenizer merged with graph, tool, molecular, structure, and oracle tokens.
5. The UGM GFlowNet action space is still set-token construction. Delete, repair, refine, stop, and typed graph-edit operations exist conceptually but are not yet used by the trainer.
6. Generated PDB rendering is intentionally not required for this pass. Existing input-row PDB rendering remains an evaluation utility behind an explicit flag.
7. Inference is a smoke path over graph-record tokens. It can produce and verify token sets, but it does not yet parse generated tokens back into a complete typed output graph with constrained decoding.
8. Validation measures coverage, verifier metrics, and FairChem/UMA oracle rewards when the configured UMA weights are available. It does not yet run expensive external benchmarks such as CASP-style folds, ligand RMSD, conformer COV/MAT, ATLAS/mdCATH dynamics, Lean proof compilation, or code execution for this multimodal phase.
9. Long-context functionality remains graph-memory/retrieval methodology rather than a dense 256K training implementation. The current NatureLM/SFM-UniGenX entity split has a measured 2x context recommendation of `max_seq_len: 1552`.
10. Scaling remains bounded by a single-4090 envelope. Full-parameter 15B training and dense all-pairs long-context attention are explicitly out of scope.

## 3.2 Detailed Implementation Plan

Immediate implementation plan:

1. Keep the model type name as Universal Graph Model (UGM) and remove all old architecture-name terminology from code, configs, docs, and the paper asset.
2. Make UGM phase-2 data reproducible: build reference tokens, prepare synthetic multimodal rows, curate deterministic train/val/test splits, and register local/synthetic manifest entries.
3. Add UGM-specific random-order policies so phase-2 training sees coarse-to-fine, oracle-enabling, and random graph completion orders.
4. Add inference support for multimodal rows and a `config/inference/multimodal_tiny_inference.yaml` smoke config.
5. Add train/test validation support through `config/validate/multimodal_validation.yaml` and `config/validate/multimodal_test.yaml`.
6. Add QA automation through `scripts/quality_assess.py`, with checks for terminology, required files, train/val/test data, reference tokens, checkpoints, and docs coverage.
7. Run full unit tests, phase-2 training, phase-2 validation, phase-2 test validation, oracle-feedback GFlowNet training, oracle-feedback GFlowNet validation, UGM inference, readiness, and QA.
8. Update README, runbook, metrics, dataset, architecture, licensing, and background docs with the final commands and residual risks.

Next implementation plan after this slice:

1. Extend the existing FairChem/UMA oracle adapter with additional tool adapters for RDKit, OpenMM, PDB/mmCIF parsers, and other OMol-family MLIPs where those tools are licensed and installed.
2. Add graph-token constrained decoding and graph parser repair so generated UGM tokens become a typed `GraphExample` or output graph object.
3. Add typed GFlowNet edit actions: add/delete/refine/repair/stop for nodes, edges, coordinates, frames, and tool observations.
4. Add local source adapters for high-value reviewed corpora after license review.
5. Add evaluation harnesses for conformer coverage, parser validity, ligand contacts, function factuality, proof/code/tool checks, and dynamics consistency.

## 4. Training Methodology

Phase 1 remains the existing graph/science SFT path:

- SFM/NatureLM and UniGenX reference tokens are included through `data/processed/reference_tokens/naturelm_unigenx_tokens.txt`.
- Existing UniGenX-style examples are sanitized for the first run: SELFIES/SMILES, sequence/function metadata, and non-structure labels are allowed; coordinates, energy, force, and structure fields are excluded.

Phase 2 is multimodal graph-to-graph training:

- Use `data/processed/reference_tokens/multimodal_graph_tokens.txt` plus the NatureLM/SFM and UniGenX tokens.
- Build and include the motif vocabulary before phase-2 training. The implemented motif path downloads/parses PROSITE, InterPro, CATH, and Rfam public metadata, accepts local JSON/JSONL/CSV/TSV motif rows, and adds `SEQ_MOTIF:*` plus safe `SEQ_MOTIF_FROM_STRUCTURE:*` vocabulary tokens. Row-local coordinate/contact-derived structure motifs remain evaluation/future-phase only.
- Train on graphified mixed rows with random-order graph-record decoding.
- Targets include sequence records, sequence motif records, safe sequence-motif-from-structure vocabulary tokens, continuous temperature tokens/features, tool records, GoT/ToT/CoT reasoning records, and answer/function text records. The `ATTN_BIN:*`, `TOKEN_COUPLING:uma:*`, `UMA_INFLUENCE:uma:*`, `TOKEN_MOTION:uma:*`, `UMA_TRAJ_BIN:*`, and `SEQ_STRUCT_DYN_PROXY:*` families are stage-gated to the UMA structure-dynamics-proxy curriculum stage.
- Keep any optional geometric descriptors string-derived and off by default; do not add a separate equivariant structure head in the first run.

Phase 3 is oracle-feedback GFlowNet training:

- Candidate actions are output graph-record tokens.
- Rewards combine token recall, validators, and FairChem/UMA temperature-conditioned oracle feedback. The deterministic proxy backend is explicit test/smoke infrastructure only.
- The FairChem/UMA scorer keeps first-run training sequence-only by scoring generated candidate graph states externally; RDKit/string validators and domain verifiers can augment function/proof/code outputs.

## 5. 4090 Feasibility Boundary

The single-4090 path is still small-model/adapters-first:

- Tiny and small models can smoke-test graph-record objectives.
- Larger public bases should be adapted with LoRA/QLoRA when licensing and checkpoints are selected.
- Dense 256K full-attention training and full-parameter 15B Adam training remain out of scope.
- Long-context behavior should be retrieval/graph-memory based, not raw dense context by default.

## 6. Verification Targets

- `tests/test_multimodal_graphs.py` covers reference vocabulary, graphification, PDB rendering for explicit future-phase rows, numeric extraction policy, collator construction, 64-bin UMA candidate records, explicit proxy smoke mode, and oracle reward wiring.
- `tests/test_multimodal_graphs.py` also covers PROSITE, InterPro, CATH, Rfam, local motif parsing, and safe `SEQ_MOTIF_FROM_STRUCTURE:*` handling without allowing coordinate/contact-derived row supervision.
- Full test suite must pass.
- Smoke commands should generate neutral multimodal synthetic data, build public motif/reference tokens, curate train/val/test splits, train 20 phase-2 steps, train 20 GFlowNet steps, validate the phase-2 val/test checkpoints, and validate the GFlowNet checkpoint.
- `scripts/quality_assess.py` should report `ready_to_roll: true` after both checkpoints exist and docs/reference data are present.

## 7. Current Ready-To-Roll Status

Latest local smoke pass:

- UGM split: 23 train, 5 validation, 4 test examples.
- Motif vocabulary: full public metadata build completed locally with 74,789 motif records and 156,827 motif tokens from core defaults, PROSITE, InterPro, CATH, and Rfam.
- Phase-2 smoke training: completed 20 steps and wrote `outputs/multimodal_phase2_tiny/checkpoint_final.pt`.
- 4090 phase-2 config: `config/train/multimodal_phase2_4090.yaml`.
- 4090 phase-2 validation/test configs: `config/validate/multimodal_4090_validation.yaml` and `config/validate/multimodal_4090_test.yaml`.
- Oracle-feedback GFlowNet smoke: completed 20 steps and wrote `outputs/multimodal_oracle_gflownet_tiny/gflownet_final.pt`.
- 4090 oracle-feedback GFlowNet config: `config/train/multimodal_oracle_gflownet_4090.yaml`.
- Structure/dynamics extension: `scripts/prepare_structure_dynamics_sources.py` plus disabled future-phase configs for evaluation-only structure-file audits.
- File-based inference: `scripts/infer.py --multimodal-json-file ... --output ...` writes JSON output and can render provided input coordinates as PDB only when explicitly requested; generated-token PDB rendering is not required in this pass.
- Full tests: `40 passed`, 3 warnings.
- Readiness: CUDA RTX 4090, Lean/Elan, RDKit, topology packages, SFM/NatureLM references, UniGenX references, and UGM reference tokens are present.
- QA: `scripts/quality_assess.py` reports `ready_to_roll: true`.

This status means the scaffold is ready for smoke-scale training, validation, testing, and inference. It does not mean the tiny checkpoint is scientifically useful; it is a functionality checkpoint.
