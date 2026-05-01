# UGM Readiness Diff Audit

Date: 2026-04-29

This audit compares the current paper, codebase, and planning documents against the active sequence-first UGM requirement:

- Train first on SELFIES/SMILES, protein FASTA, RNA, DNA, function descriptions, reasoning traces, tool traces, and oracle-feedback graph records.
- Do not train on actual PDB/mmCIF/SDF files, MD trajectories, deposited coordinate labels, supervised RMSD, supervised force labels, or direct dynamics frames in the first run.
- Still treat generated structure-dynamics as an active output target: sampled graph states may contain atom, bond, coordinate, and frame records. PDB serialization is optional and not required for the current implementation pass.
- Use temperature-conditioned UMA/verifier feedback, GFlowNet trajectory balance, graph-state evolution, evolving attention/coupling bins, hidden-state geometry, Jensen-Shannon softmax geometry, persistent topology, and tropical diagnostics to guide and measure the sequence-only route.

## 1. Dataset Readiness Diff

Current train/val/test corpus:

- `data/processed/naturelm_public_sources_entity/train.jsonl`
- `data/processed/naturelm_public_sources_entity/val.jsonl`
- `data/processed/naturelm_public_sources_entity/test.jsonl`

Completed audits:

- Entity split: `data/processed/naturelm_public_sources_entity/summary.json`
  - train: 1,664,029
  - val: 16,112
  - test: 15,987
  - total: 1,696,128
  - split policy: entity
  - group families: `protein_seq` and `row_hash`
- Integrity audit: `data/processed/naturelm_public_sources_entity/integrity.json`
  - `ok: true`
- Sequence-only policy audit: `data/processed/naturelm_public_sources_entity/policy.json`
  - `ok: true`
  - scanned: 1,696,128
  - violations: 0
  - `sequence_only_molecules: true`
  - `forbid_actual_structure_files: true`
- Token-count audit: `data/processed/naturelm_public_sources_entity/token_counts.json`
  - source graph tokens: 905,584,603
  - model sequence tokens untruncated: 914,065,243
  - supervised prediction tokens: 3,392,256
- Orthogonal identifier audit: `data/processed/naturelm_public_sources_entity/identifier_stats.json`
  - max nodes: 258
  - max edges: 512
  - max identifier: 770
  - missing node identifiers: 0
  - missing edge identifiers: 0
  - node/edge identifier overlap examples: 0
- Context audit: `data/processed/naturelm_public_sources_entity/context_requirements.json`
  - largest untruncated model sequence: 776
  - largest source graph: 771
  - largest target: 2
  - recommended 2x context window: 1,552
  - generated config: `config/generated/naturelm_public_sources_context_2x.yaml`

Remaining dataset work:

- Add the higher-cardinality large PubChem CID/SMILES sources only behind the gated `--prepare-large` path after license, disk, and leakage review.
- Add Materials Project data only with a valid API key and explicit license review.
- Add ProTrek-style sequence/function rows when locally reviewed exports are present.
- Add live UMA oracle traces only as generated candidate feedback, not as direct structure-file labels.
- Add train-only motif mining summaries for sequence motifs and safe sequence motifs derived from structure-motif vocabularies; do not mine motifs from validation/test rows.

## 2. Paper Diff

File: `assets/human_learning_transformer_learning_review_dataset_expanded.tex`

The paper already contains the main UGM sequence-first rewrite, but several statements still implied that coordinate-like output belonged only to a later evaluation track. Those statements needed correction because the active requirement is: generated structure-dynamics candidate graphs are allowed now; structure-file supervision is not. PDB rendering is optional and not required in this pass.

Required corrections:

- Orientation section: replace the sentence that treated structure and trajectory renderers as future/evaluation-only artifacts. It should say generated coordinate/frame graph records are active candidate outputs, while actual structure-file datasets remain evaluation/future-phase sources and PDB serialization is optional.
- GFlowNet objective: replace the sentence that says candidate \(Y_G\) does not contain supervised coordinates, MD frames, or force labels. It should say \(Y_G\) may contain generated atom/bond/coordinate/frame records, but not supervised structure-file labels.
- Training objectives: replace the proxy-only set
  `{SEQ_STRUCT_DYN_PROXY, ATTN_BIN, TOKEN_COUPLING, TOKEN_MOTION, UMA_SCORE}`
  with a generated candidate output set that also includes `UMA_INFLUENCE`, `UMA_TRAJ_BIN`, generated `ATOM`, `BOND`, `COORD`, and `FRAME` records where the sampler creates them.
- Training objectives: replace the overly broad exclusion of coordinate/PDB-like targets with a narrower statement: the model is not supervised by copied coordinates, force labels, PDB files, or MD frames; generated coordinate/frame hypotheses are still target behavior.
- GFlowNet sampling: replace "output is a set of candidate records and explanations, not a supervised coordinate trajectory" with language that permits generated coordinate/frame records.
- Rendering corollary: leave PDB as an optional downstream serializer, not a required implementation target.
- Risks: replace the earlier blanket reporting caveat with a more precise caveat: first-run outputs can be reported as UMA-oracle-guided sequence-only structure-dynamics predictions, not as supervised PDB/MD-trained predictors or validated physical simulations.
- Conclusion: include generated structure-dynamics candidate graphs in the summary of output graphs.

## 3. MATH.md Diff

File: `MATH.md`

Current state:

- Correctly states that first-run restriction is about supervision source, not output capability.
- Includes UMA/GFlowNet equations for candidate graph reward.
- Includes evolving attention/contact fields, Euclidean embedding geometry, Jensen-Shannon softmax geometry, and generated coordinate/frame self-consistency equations.

Remaining work:

- Keep aligned with the paper after the paper patch.
- Add any future implementation-specific formulas if a live UMA adapter changes the reward shape.

## 4. Codebase Diff

Implemented and passing tests:

- Sequence-only policy enforcement exists in `src/iska_reasoner/data/phase_policy.py`.
- Stage-gated 64-bin UMA/attention/coupling/motion record emission exists in `src/iska_reasoner/data/multimodal.py`.
- Orthogonal identifiers are passed through dataset encoding, collator batches, model input, training, validation, and inference.
- Hidden topology metrics and JS geometry loss exist in `src/iska_reasoner/topology/hidden.py`.
- Folding contact utilities exist in `src/iska_reasoner/topology/folding.py`:
  - attention contact fields
  - Euclidean hidden-state contact fields
  - Jensen-Shannon softmax contact fields
  - fused folding-contact field
  - contact metrics
  - contact-coordinate self-consistency loss for generated coordinates
- W&B/logging wrappers exist in `src/iska_reasoner/utils/logging.py`.
- Acquisition and preparation scripts use `tqdm`, logs, and optional W&B.

Implementation status and remaining gaps:

- Implemented in this pass: training, in-training validation, and standalone validation can now log folding-contact metrics from hidden states when `hidden_topology.folding_contact_enabled` is true.
- `scripts/infer.py` can render input atoms/frames to PDB behind an explicit flag. Generated-token PDB rendering is intentionally not required in this pass.
- The current `RandomOrderTokenGT` does not expose true layer/head attention maps. Attention-bin targets are trained as graph records, and hidden-state geometry is available, but full attention-map diagnostics require a custom attention layer or hooks.
- The current production UMA reward path calls the FairChem adapter in `src/iska_reasoner/oracles/uma.py`; `src/iska_reasoner/tools/verifiers.py` retains an explicit deterministic proxy path only when `UGM_UMA_BACKEND=proxy` is set for tests or smoke runs. RDKit/OpenMM/PDB parser adapters remain future extensions.
- Structure-file phase configs are disabled correctly, but planning docs still need clearer distinction between generated coordinate/frame candidate graphs and structure-file training.

## 5. Planning Docs Diff

Files needing wording updates:

- `planning/PLAN-H.md`
  - Opening still says actual atom/coordinate/conformer/PDB training is deferred. It should distinguish generated candidate output from supervised structure-file training.
  - Gap list should say full attention-map extraction remains incomplete; live UMA is implemented but depends on local FairChem dependencies and gated `facebook/UMA` model access.
- `planning/ARCHITECTURE.md`
  - Current multimodal phase text says atom, coordinate, trajectory, and PDB records are evaluation/future-phase only. It should say structure-file-derived records are future-phase only, while generated coordinate/frame candidate records are active.
- `planning/DATASETS.md`
  - First multimodal phase text should clarify that generated candidate records may be emitted by samplers/oracle traces, but not sourced from actual structure files.
- `planning/STRUCTURE-DYNAMICS-TRAINING.md`
  - Needs a sentence that generated coordinate/frame candidate graphs are part of the active inference target and PDB serialization is optional.
- `planning/BACKGROUND-RESEARCH.md`
  - Needs a sequence-first update where it lists structure/dynamics resources and direct physics supervision.
- `planning/TRAINING-SEQUENCE.md`
  - The structure prediction gate should say evaluation-only structure files are barred from training, but generated candidate graph records can be emitted and oracle-scored.

## 6. Train/Val/Test/Infer Readiness Diff

Ready now:

- Entity-split NatureLM/SFM and UniGenX-derived public corpus has train/val/test files.
- Integrity, policy, token-count, identifier, and context audits passed.
- Recommended 2x context config exists.
- Unit tests passed before this second pass.
- W&B/offline logging config paths exist.

Implemented in this pass:

- Added folding-contact metrics to training, in-training validation, and standalone validation logging.
- Patched paper/planning contradictions about generated structure-dynamics candidate graphs versus structure-file training.
- Reran focused tests and the full test suite.
- Ran contradiction scans over paper/planning after patching.

## 7. Future Work Not Blocking Current Readiness

- Live UMA adapter and asynchronous oracle cache.
- Real attention-map extraction from model internals.
- Candidate coordinate graph decoder now emits identity-bearing autoregressive `COORD:f{frame}:a{atom}:{axis}:{bin}` records for bounded frame/atom slots.
- Generated PDB/mmCIF/SDF writers.
- Full chemical valence repair beyond current token/verifier checks.
- Large dataset acquisition gates for additional licensed or API-protected corpora.
