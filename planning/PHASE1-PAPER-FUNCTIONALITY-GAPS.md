# Phase 1 Paper Functionality Gaps

Created: 2026-04-29

This document tracks early Phase 1 training functionality from the updated paper that is not yet implemented in full. Phase 1 here means the graph-language, reasoning, code, proof, Hebrew/morphology, and science-reasoning pretraining stack before the second multimodal structure/dynamics phase. The current repository already has a working random-order graph-record training path, topology feature summaries, tropical diagnostics, verifier-aware GFlowNet training, and validation scripts. The items below are the remaining paper-level mechanisms that need additional implementation before the Phase 1 system fully matches the paper.

## Current Implemented Baseline

- Random-order graph-record language modeling with source graph records, target reveal records, position-query records, and graph-token vocabulary support.
- Lightweight graph topology summaries computed from graph structure and batched into training examples.
- A topology prediction head on the graph model and optional MSE topology auxiliary loss.
- Optional GUDHI-backed persistence summaries when dependencies are installed.
- Persistent-Laplacian-style graph spectral proxy over thresholded graph-distance filtrations.
- Tropical logit diagnostics: entropy, top-1 confidence, top-1 margin, and annealed temperature reporting.
- A standalone tropical attention utility, activation-cell signature helper, and maximum-spanning-arborescence parser helper.
- GFlowNet trajectory-balance training over target-token set construction, with topology context, learned backward policy support, subtrajectory loss support, verifier reward hooks, and validation metrics.
- Domain validation namespaces for code, Lean, chemistry, Hebrew morphology, and multimodal graph records.
- Full selected public Phase 1 graph corpus construction for public Hugging Face parquet splits, including SFM/NatureLM and UniGenX reference vocabulary.

## Priority Summary

| Priority | Area | Why It Matters |
|---|---|---|
| P0 | Embedding-space persistent homology and distograms | The paper analyzes hidden-state geometry, but the implementation mostly summarizes input graph topology. |
| P0 | Latent reasoning loops as actual recurrent dynamics | The paper's dynamic-systems thesis requires repeated hidden-state evolution, not only one-pass graph-token prediction. |
| P0 | Graph-valued reasoning trajectories | The current GFlowNet operates over target-token sets; the paper specifies typed graph state transitions. |
| P0 | Verifier-guided graph-of-thought inference | Current validation is mostly after-the-fact; paper requires online branching, verification, repair, and merging. |
| P1 | Differentiable topology losses | The topology head is useful, but it is not a true persistence-based regularizer. |
| P1 | Tropical attention and tropical transition logging | Tropical utilities exist but are not a selectable training backend or full layerwise diagnostic path. |
| P1 | Ergodic/stochastic loop diagnostics | The paper discusses Markov kernels and ergodicity, but no mixing diagnostics are implemented. |
| P1 | Process-supervised trajectory datasets | The corpus has many final-answer examples and fewer verified process traces than the paper assumes. |
| P2 | Full persistent Laplacians over simplicial filtrations | Current spectral proxy is useful, but not the mathematical persistent Laplacian described in the paper. |
| P2 | Theorem-to-code property tests | Random-order, invariance, and equivariance claims need systematic synthetic tests. |

## Detailed Gaps and Required Updates

### 1. Embedding-Space Persistent Homology and Distograms

**Status:** Under-implemented.

The paper defines embedding distograms over token, graph-token, morpheme, and latent-thought embeddings, then uses Vietoris-Rips filtrations and persistence summaries over those hidden-state point clouds. The current implementation computes topology features primarily from the input graph structure and optional graph-distance filtrations. It does not yet compute persistent homology from model hidden states at every selected layer or reasoning-loop step.

**Needed updates:**

- Add a hidden-state topology monitor that receives `hidden_states` from selected layers.
- Build pairwise distance matrices from hidden vectors with configurable metrics: cosine distance, Euclidean distance, attention-induced distance, and optional learned metric.
- Add distogram construction: deterministic bins from distances first, then optional uncertainty-aware distograms from sampled dropout, sampled reasoning trajectories, or auxiliary distance predictors.
- Add configurable Vietoris-Rips persistence summaries from hidden-state point clouds, with safeguards for `O(n^2)` distance memory.
- Log layerwise and loopwise summaries with names such as `hidden_topology/layer_03/h0_total_persistence`, `hidden_topology/layer_03/h1_count`, and `hidden_distogram/layer_03/bin_entropy`.
- Cache or subsample graph tokens for long contexts so topology monitoring is bounded and does not dominate training time.

**Acceptance tests:**

- Unit test hidden-state distance and distogram construction on small tensors.
- Unit test persistence summaries on synthetic point clouds with known connected components and one-cycle structure.
- Tiny training run with hidden topology logging enabled and no memory blow-up.
- Validation JSON contains both graph-topology and hidden-topology metric namespaces.

### 2. Differentiable Topology Losses

**Status:** Under-implemented.

The topology head currently regresses graph-level summaries. This is a real auxiliary task, but it is not a differentiable persistent-homology loss over hidden geometry. The paper proposes losses based on bottleneck/Wasserstein distance to target diagrams, collapse penalties, and topology preservation across layers.

**Needed updates:**

- Add a topology-loss interface with separate modes:
  - non-differentiable monitor only;
  - differentiable proxy loss over pairwise-distance summaries;
  - differentiable persistence backend when a supported library is available;
  - diagram-matching loss for target diagrams where available.
- Implement collapse penalties for premature component collapse and excessive hidden-state rank collapse.
- Add target construction for tasks with known structure: parse-tree constituents, Hebrew root families, synthetic graph motifs, and proof-dependency graphs.
- Add annealed loss weights so topology regularization starts after the model has learned basic reconstruction.

**Acceptance tests:**

- Gradient test proving the differentiable proxy topology loss backpropagates to hidden states.
- Tiny run with topology regularization enabled and finite loss values.
- Ablation config comparing no topology loss, topology-head loss, and differentiable proxy loss.

### 3. Full Persistent Laplacians Over Simplicial Filtrations

**Status:** Under-implemented.

The current implementation has a persistent-Laplacian-style proxy over graph-distance thresholds. The paper discusses Hodge Laplacians, boundary operators, persistent Laplacians, and spectra over simplicial complexes. That full construction is not yet present.

**Needed updates:**

- Build boundary matrices for `C_0`, `C_1`, and optionally `C_2` on small filtrations.
- Support clique-complex and Vietoris-Rips constructions from graph tokens and hidden-state distances.
- Compute `p`-Hodge Laplacian spectra for selected dimensions and scales.
- Add persistent spectral summaries: kernel dimension, lambda-2, spectral gap, trace, and stable eigenvalue counts across filtration scales.
- Add hard caps and sampling policies because exact simplicial computations can become expensive quickly.

**Acceptance tests:**

- Unit tests for boundary-of-boundary equals zero.
- Unit tests on a triangle, path, square cycle, and filled triangle.
- Validation metrics match expected Betti counts on synthetic graphs.

### 4. Topological Faithfulness and Intervention Tests

**Status:** Not implemented as a dedicated suite.

The paper warns about topological pareidolia and calls for causal tests: edge deletion, edge substitution, latent-thought lesioning, counterfactual morphology, and tropical temperature sweeps. The repository does not yet have a systematic intervention test harness.

**Needed updates:**

- Add intervention dataset wrappers:
  - delete syntactic/morphological/proof edges;
  - substitute plausible distractor edges;
  - remove selected thought nodes;
  - swap Hebrew root/template components;
  - perturb graph-token order while preserving graph identity.
- Measure output change, verifier-score change, topology-summary change, and tropical-signature change.
- Add a validation report that distinguishes correlation from causal sensitivity.

**Acceptance tests:**

- Synthetic graph task where deleting the critical edge predictably changes the answer.
- Hebrew root synthetic task where root swap changes semantic-family topology.
- Validation artifact with before/after metrics and examples.

### 5. Tropical Attention as a Selectable Training Backend

**Status:** Implemented as an experimental backend; broader backend variants remain future work.

The repository now has model config switches for pure MHTA (`model.attention_backend: tropical`) and hybrid Flash-eligible SDPA plus layer-sparse MHTA (`model.attention_backend: hybrid_flash_tropical`), tiny configs, train/validation logging, and unit tests. The paper describes a broader family of low-temperature attention, max-plus selection, tropical attention, and tropical training schedules as model-level experiments; the current implemented model-level path covers masked MHTA and the hybrid SDPA+MHTA encoder branch, with `hybrid_tropical_layers` or `hybrid_tropical_every` controlling how many layers pay the explicit max-plus attention cost.

**Implemented updates:**

- Add a model config switch for attention backend:
  - standard softmax attention;
  - tropical/MHTA experimental attention.
- Add a compatible transformer block with log-ReLU tropicalization, Hilbert scores, max-plus aggregation, causal masks, and padding masks.
- Keep output-logit tropical temperature schedules configurable per stage and validation run.
- Log sharp-selection metrics from the tropical backend.

**Still future work:**

- annealed softmax attention as a separate backend;
- hybrid top-k sparse attention;
- full per-head/per-layer artifacts beyond aggregate metrics.

**Acceptance tests:**

- Tiny train/validation runs with standard and tropical backend configs.
- Shape and mask tests for causal, padding, and graph-record attention masks.
- Regression test proving annealing changes entropy and top-1 confidence in the expected direction.

### 6. Tropical Dynamic-Programming Decoders and Parsers

**Status:** Utility exists, training integration missing.

The maximum-spanning-arborescence helper exists, but it is not wired into parser-style training objectives, parse supervision, or graph-of-thought reasoning choices. The paper proposes tropical dynamic programs for dependency parsing, proof-path selection, and exact graph decisions.

**Needed updates:**

- Add a parser objective for dependency or graph-edge selection from graph-token scores.
- Add synthetic and Hebrew morphosyntax tasks where graph edges have gold arborescences.
- Add a tropical parse decoder to validation and inference.
- Use selected parse edges as additional graph records for downstream training.

**Acceptance tests:**

- MST parser test on a known score matrix.
- Tiny supervised parse run over synthetic dependency graphs.
- Validation report with edge precision/recall and arborescence validity.

### 7. Tropical Transition Signatures Over Layers

**Status:** Under-implemented.

The paper asks for tropical cell-path analysis: which max-affine regions or high-confidence attention cells are traversed over layers and reasoning loop steps. The repository has an activation-cell signature helper but does not log full transition signatures through training and validation.

**Needed updates:**

- Capture selected hidden states, top-k attention supports, and MLP activation sign/proxy signatures.
- Compute transition edit distances across layers, examples, perturbations, and loop steps.
- Log stability metrics under input perturbations and temperature sweeps.

**Acceptance tests:**

- Unit tests for transition-signature distance.
- Validation artifact showing layerwise tropical signature summaries.
- Perturbation test where a small input change produces either stable or intentionally changed signatures.

### 8. Actual Latent Reasoning Loops

**Status:** Not implemented in the main model.

The paper's dynamic-systems section treats reasoning as `z_{t+1}=F_theta(z_t,x,u_t)`. The repository has graph tokens, thought token types, and GFlowNet target-set rollouts, but it does not yet have a Coconut-style or looped-transformer hidden-state recurrence in the main model.

**Needed updates:**

- Add latent thought slots to the model input and collator.
- Add a loop controller that reuses the transformer block or selected layers for `T` loop steps.
- Feed learned soft-thought embeddings or selected hidden states back into the next loop step.
- Allow loop budgets at train and test time, with optional curriculum over loop count.
- Decode from final loop state or aggregate states across loops.

**Acceptance tests:**

- Unit test loop unrolling with gradients through multiple loop steps.
- Tiny run with `loop_steps > 1`.
- Validation metrics grouped by loop step count.
- Inference command that increases loop steps at test time without changing weights.

### 9. Stability, Contractivity, and Lyapunov Diagnostics

**Status:** Not implemented.

The paper uses contraction, fixed-point, and energy/Lyapunov language to make looped latent reasoning measurable. The current implementation does not estimate loop stability, contraction, hidden-state energy, or convergence behavior.

**Needed updates:**

- Add hidden-state delta metrics: `||z_{t+1}-z_t||`, cosine drift, norm drift, and fixed-point residual.
- Add finite-difference Lipschitz estimates for loop maps on small batches.
- Add energy surrogates such as verifier loss, value estimate, topology collapse score, and answer entropy.
- Track monotonicity or violations across loop steps.

**Acceptance tests:**

- Synthetic contraction map test where residual decreases.
- Loop validation report with convergence and divergence counters.
- Configurable alert when loop dynamics explode or collapse.

### 10. Ergodic and Stochastic Loop Diagnostics

**Status:** Not implemented.

The paper discusses stochastic reasoning loops as Markov kernels and references ergodicity, total variation convergence, and Doeblin-style refresh conditions. No current training or validation path measures mixing behavior.

**Needed updates:**

- Add stochastic latent-loop sampling with configurable noise, random action proposals, and refresh probability.
- Hash or embed loop states to estimate state diversity, recurrence, and absorbing-state frequency.
- Add autocorrelation, effective sample size proxy, unique-state counts, and mode-collapse metrics.
- Add a refresh/no-refresh ablation to test whether stochastic loops explore or collapse.

**Acceptance tests:**

- Synthetic Markov-chain test with known stationary distribution.
- Validation report with unique states, autocorrelation proxy, and absorbing-state warnings.
- GFlowNet validation run comparing deterministic and stochastic rollout diversity.

### 11. Graph-Valued Reasoning Trajectories

**Status:** Under-implemented.

The current GFlowNet stage samples target-token sets. The paper specifies graph-valued states `S_t=(G_t,Z_t,C_t)` with typed actions such as add node, add edge, merge nodes, refine nodes, call tools, verify claims, and emit answer. That richer state/action system is not yet implemented.

**Needed updates:**

- Define a `ReasoningState` object with graph records, latent state handles, controller memory, verifier observations, and action history.
- Define typed actions:
  - `ADD_NODE`;
  - `ADD_EDGE`;
  - `MERGE_NODES`;
  - `REFINE_NODE`;
  - `CALL_TOOL`;
  - `ATTACH_OBSERVATION`;
  - `VERIFY_SUBCLAIM`;
  - `EMIT_ANSWER`.
- Add inverse actions for backward-policy training.
- Add trajectory serialization for training and replay.

**Acceptance tests:**

- Unit tests for action application and inverse-action validity.
- Tiny synthetic graph-of-thought dataset with known terminal graphs.
- GFlowNet rollout can produce a graph object, not only a flat target-token set.

### 12. GFlowNet Reward Completeness and Backward Policy Fidelity

**Status:** Partially implemented.

Trajectory balance, learned backward policy support, topology context, and verifier reward hooks exist. The reward is still a project-level proxy for many tasks, and the backward policy is token-set oriented rather than full graph-edit inverse modeling.

**Needed updates:**

- Add per-domain reward adapters for math answers, Lean proof checking, code tests, graph validity, Hebrew morphology, and science-reasoning constraints.
- Add exact inverse-action definitions for graph edits.
- Add replay-buffer support for high-reward and failed trajectories.
- Add reward normalization and reward-floor handling for long trajectories.
- Add off-policy trajectory-balance support where trajectories come from teacher traces or verifier repair traces.

**Acceptance tests:**

- Reward adapter tests for each domain.
- Backward-policy normalization test over valid predecessor states.
- GFlowNet validation shows nonzero valid terminal states, action coverage, and reward diversity.

### 13. Continuous or Hybrid GFlowNets

**Status:** Not implemented for Phase 1.

The paper references continuous and hybrid GFlowNets for latent thought trajectories. The current implementation is categorical over token/set actions.

**Needed updates:**

- Add a hybrid action representation with discrete graph edit plus optional continuous latent displacement.
- Add Gaussian or discretized-continuous backward kernels.
- Add density terms to trajectory-balance loss where continuous actions are used.
- Restrict first implementation to low-dimensional latent-thought probes before scaling.

**Acceptance tests:**

- Toy continuous target distribution test.
- Hybrid trajectory-balance loss finite and differentiable.
- Diversity metrics improve over categorical-only rollout on a synthetic latent target.

### 14. Verifier-Guided Graph-of-Thought Inference

**Status:** Under-implemented.

The system has validation and verifier metrics, but not a full inference-time graph-of-thought controller that branches, verifies, repairs, merges, and resamples partial reasoning graphs.

**Needed updates:**

- Add an inference controller with budgeted branching.
- Allow intermediate verifier calls on partial graphs.
- Add repair actions and merge actions.
- Rank or sample terminal graphs by verifier reward and cost.
- Emit a structured trace as graph records and optionally flatten it to a transcript.

**Acceptance tests:**

- Synthetic multi-hop task where graph-of-thought search beats greedy one-shot decoding.
- Trace artifact includes branch, verifier, repair, and merge records.
- Budget controls wall-clock time and max graph size.

### 15. Process-Supervised Reasoning-Trajectory Data

**Status:** Insufficient coverage.

The full public corpus has many final-answer examples, but the paper's graph-of-thought, verifier-guided repair, and GFlowNet trajectory goals require process traces. Current synthetic traces and proof/code datasets are useful but not enough for the full Phase 1 methodology.

**Needed updates:**

- Generate verified tool traces for math, code, Lean, science QA, and graph repair.
- Store traces as graph records with thought nodes, tool-call nodes, observation nodes, verifier nodes, and repair edges.
- Add source labels distinguishing teacher traces, self-generated verified traces, and human-authored traces.
- Add split-safe contamination checks for generated traces.

**Acceptance tests:**

- Curated process-trace dataset with train/val/test splits.
- Validation can measure intermediate-step verifier accuracy, not just final answer.
- Data quality report includes tool success, repair success, and duplicate rate.

### 16. Soft Thought Tokens and Hidden-State Feedback

**Status:** Not implemented beyond token vocabulary scaffolding.

The graph vocabulary supports thought-like records, but the model does not yet use learned soft thought vectors that bypass natural-language serialization.

**Needed updates:**

- Add special latent thought slots to the collator and model.
- Train projection modules that map final hidden states into next-step thought embeddings.
- Add losses that compare latent-loop outputs against final graph targets without requiring every thought to be rendered as text.
- Add optional thought-to-text probes for interpretability only.

**Acceptance tests:**

- Loop training works with soft thoughts and no decoded intermediate text.
- Inference can run hidden thought loops then produce a final graph answer.
- Probe output is optional and does not affect model state.

### 17. Agentic Memory, Retrieval, and Tool-State Graph Records

**Status:** Partially represented as tokens, under-implemented as a training loop.

Tool-call records and retrieval-like graph nodes can be represented, but there is not yet a full Phase 1 retrieval/memory controller that writes observations into the graph during training and inference.

**Needed updates:**

- Add retrieval node and memory-write node schemas for text, code, proof, and science snippets.
- Add a retriever interface that returns graph records rather than plain text.
- Train on traces where retrieval changes the graph state and downstream answer.
- Add memory-write policies for successful verified traces.

**Acceptance tests:**

- End-to-end retrieval-augmented inference over a small local corpus.
- Trace shows retrieval records connected to claims or proof/code steps.
- Retrieval ablation changes outputs on tasks that require external evidence.

### 18. Combined Topology, Tropical, Loop, and GFlowNet Curricula

**Status:** Not implemented as a coordinated curriculum.

Each component has some independent scaffolding, but there is no single Phase 1 curriculum that schedules graph reconstruction, hidden topology monitoring, tropical sharpness, latent loops, verifier rewards, and GFlowNet trajectory learning in a controlled sequence.

**Needed updates:**

- Add stage configs:
  - graph reconstruction warmup;
  - topology-head warmup;
  - hidden topology monitoring;
  - latent loop unroll;
  - verifier-guided traces;
  - GFlowNet fine-tuning;
  - tropical annealing ablation.
- Add loss-weight scheduling and per-stage early stopping criteria.
- Add ablation matrix configs so each paper claim can be tested independently.

**Acceptance tests:**

- `scripts/run_phase1_curriculum.sh` or equivalent can run a tiny complete curriculum.
- Metrics file records active objectives and loss weights per stage.
- Ablation configs train without manual code edits.

### 19. Metrics, Dashboards, and Failure Thresholds

**Status:** Basic metrics exist; paper-level monitoring is incomplete.

The repository records loss, perplexity, token accuracy, topology summaries, tropical diagnostics, verifier metrics, and GFlowNet metrics. It does not yet provide a single readiness dashboard for the Phase 1 theory-heavy mechanisms.

**Needed updates:**

- Add a unified Phase 1 metrics schema with namespaces:
  - `hidden_topology/*`;
  - `graph_topology/*`;
  - `tropical_signature/*`;
  - `loop_dynamics/*`;
  - `ergodic/*`;
  - `reasoning_trajectory/*`;
  - `gflownet/*`;
  - `verifier/*`.
- Add threshold checks for topology computation failures, loop divergence, verifier collapse, reward collapse, and action-space collapse.
- Add summary plots or JSONL logs suitable for later dashboarding.

**Acceptance tests:**

- Tiny run writes all enabled metric namespaces.
- Quality assessment script checks required Phase 1 metric files.
- Failure thresholds produce actionable warnings rather than silent pass/fail.

### 20. Dataset Coverage for Phase 1 Theory Mechanisms

**Status:** Under-covered.

The full selected corpus is large and useful for graph pretraining, but specific paper mechanisms need targeted data:

- topology-labeled graph tasks;
- parse and morphology graph tasks;
- verified graph-of-thought trajectories;
- stochastic trajectory data;
- Lean proof states with intermediate checker feedback;
- code traces with unit-test feedback;
- retrieval/tool-use traces with observations connected to claims.

**Needed updates:**

- Add synthetic controlled datasets for each mechanism.
- Add generated verified traces from existing public examples.
- Add contamination-safe splits and per-task metrics.
- Add manifest entries so these datasets are visible in the dataset catalog.

**Acceptance tests:**

- Each Phase 1 mechanism has at least one tiny and one scalable data config.
- Each config has train/val/test outputs and token-count reports.
- Validation can run on every task family without custom flags.

### 21. Graph Invariance and Random-Order Property Tests

**Status:** Partial.

Random-order decoding is implemented, but the paper's graph-to-graph framing needs stronger property tests for permutation invariance, equivariance, and order-sampled likelihood behavior.

**Needed updates:**

- Add synthetic graph isomorphism tests where node order changes but graph identity does not.
- Add target-record permutation tests for random-order likelihood.
- Add graph relabeling tests for edge and hyperedge records.
- Add regression tests for motif and multimodal vocabulary extension preserving original tokens.

**Acceptance tests:**

- Same graph under multiple record orderings yields equivalent target sets.
- Random-order sampling covers every target record over repeated runs.
- Graph relabeling preserves expected invariant outputs on synthetic tasks.

### 22. Performance and Budget Controls for Expensive Geometry

**Status:** Partially controlled, not complete.

Persistent topology, hidden-state distograms, loop dynamics, and GFlowNet branching can be expensive. The current training path has basic progress tracking, but not enough budget controls for full paper-level monitoring at scale.

**Needed updates:**

- Add max-token, max-node, max-distance-pairs, and max-persistence-points settings.
- Add async or periodic metric computation rather than every step.
- Add per-stage timing metrics for topology, tropical signatures, verifier calls, and GFlowNet rollouts.
- Add automatic fallback when optional topology packages are unavailable.

**Acceptance tests:**

- Long-context tiny run completes under a fixed time/memory budget.
- Disabling optional expensive metrics changes metrics coverage but not training correctness.
- Progress bars and logs identify slow stages clearly.

## Recommended Implementation Order

1. Implement hidden-state distograms and hidden topology monitoring, without making it differentiable.
2. Add latent loop unrolling and loop dynamics metrics.
3. Upgrade GFlowNet states from target-token sets to typed graph-edit trajectories.
4. Add verifier-guided graph-of-thought inference with branch, verify, repair, and merge actions.
5. Add process-supervised trace generation and datasets.
6. Extend the tropical attention backend from aggregate metrics to full tropical transition signature artifacts.
7. Add stochastic loop and ergodic diagnostics.
8. Add differentiable topology proxy losses.
9. Add full persistent Laplacian support behind strict size caps.
10. Add invariance, random-order, intervention, and curriculum property tests.

## Ready Definition for Phase 1 Paper Alignment

Phase 1 should be considered paper-aligned only when all of the following are true:

- Hidden-state topology metrics are computed and validated, not only graph-input topology.
- Latent reasoning loops can run at train and test time with loop dynamics metrics.
- GFlowNet trajectories operate over graph edit states, not only target-token sets.
- Verifier-guided graph-of-thought inference produces structured traces.
- Tropical attention or tropical selection is a selectable model/training mode.
- Stochastic loop diagnostics report diversity, recurrence, and collapse metrics.
- Each mechanism has at least one unit test, one tiny training config, and one validation report.
- Documentation and run scripts distinguish implemented, experimental, and future mechanisms.
