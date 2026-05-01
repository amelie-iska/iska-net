# Tropical Attention Integration Plan

This plan maps the optional integration of `Tropical-Attention` into UGM training. It is based on the cloned implementation at `data/external_repos/Tropical-Attention` and the NeurIPS 2025 paper `arXiv:2505.17190v2`, "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms" by Hashemi, Pasque, Teska, and Yoshida.

## Source Audit

The external repository provides two relevant kernels:

- `TropicalAttention.py`: a compact kernel with `TropicalLinear`, log-ReLU tropicalization, optional tropical projection, optional learned tropical normalization, Hilbert-projective scoring, and max-plus value aggregation.
- `models.py`: a larger experiment scaffold with vanilla/adaptive/tropical transformer blocks, Pluecker-space experiments, Sinkhorn-style tropical normalization, and the task-specific combinatorial benchmark models.

The paper's central attention operator is Multi-Head Tropical Attention (MHTA):

1. Map Euclidean token embeddings into a tropical projective representative using a valuation-like map, usually log-ReLU followed by projective normalization or a learnable shift.
2. Apply max-plus linear projections to obtain head-specific queries, keys, and values.
3. Score query-key compatibility by negative tropical Hilbert projective distance:
   `S_ij = -d_H(q_i, k_j)`, where `d_H(x,y)=max_r(x_r-y_r)-min_r(x_r-y_r)`.
4. Aggregate values by max-plus multiplication:
   `C_i = max_j(S_ij + v_j)`.
5. Devalue the tropical context back to a Euclidean tensor before the residual/MLP stack.

The paper's claims are strongest for combinatorial algorithmic reasoning and out-of-distribution stress tests. It explicitly notes that scaling to generative autoregressive language modeling is not established and that Hilbert-distance tropical operations can add runtime and memory overhead. UGM should therefore expose Tropical Attention as an experimental backend, not as the default.

## Integration Contract

UGM will keep the standard PyTorch transformer encoder as the default backend:

```yaml
model:
  attention_backend: standard
```

MHTA becomes opt-in:

```yaml
model:
  attention_backend: tropical
  tropical_projective_normalize: true
  tropical_use_projection: true
  tropical_use_norm_shift: true
  tropical_symmetric: true
  tropical_context_clamp: 20.0
  tropical_score_floor: -10000.0
```

The backend is model-level because MHTA replaces the attention sublayer inside every transformer block. The existing `tropical:` training section remains the logit-temperature diagnostic schedule; it is related but separate. That distinction is important:

- `model.attention_backend=tropical` changes the attention kernel.
- `tropical.enabled=true` controls tropical diagnostics and annealed logit summaries.

## Architecture Design

Add native UGM modules rather than importing the external Python file at runtime. The external checkout is a reference and reproducibility artifact; production training should not depend on unpinned import paths inside `data/`.

New classes:

- `HeadwiseTropicalLinear`: head-specific max-plus linear map with parameter shape `[num_heads, output_dim, input_dim]`.
- `MultiHeadTropicalAttention`: masked, batch-first MHTA implementation compatible with UGM causal masks and padding masks.
- `TropicalTransformerEncoderLayer`: norm-first residual block using MHTA plus the existing GELU feed-forward path.
- `TropicalTransformerEncoder`: small stack wrapper with `.layers` and `.norm` so gradient checkpointing remains compatible with the current training loop.

Mask behavior:

- `attention_mask` remains `True` for real tokens and `False` for padding.
- `causal_mask` remains `True` where attention is disallowed.
- Tropical scores are filled with a finite floor, not `-inf`, to avoid non-finite losses when all positions are masked by a bad batch.

Numerical safeguards:

- Use `log1p(relu(x))` rather than `log(relu(x))` so zeros stay finite.
- Initialize tropical projection weights near zero.
- Clamp tropical context before `expm1` by default.
- Keep all operations differentiable except diagnostics.
- Preserve AMP compatibility.

## Hyperparameter Guidance

Start with small and graph-state tasks:

```yaml
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  attention_backend: tropical
  tropical_context_clamp: 12.0
train:
  batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 0.0003
```

For 4090-scale runs:

- Reduce batch size relative to standard attention because MHTA materializes `[batch, heads, seq, seq, head_dim]` intermediate tensors.
- Keep `max_seq_len` modest for first ablations, typically `256` or `512`, before trying `1024`.
- Use gradient checkpointing for multi-layer tropical backends.
- Treat `tropical_context_clamp` as a stability hyperparameter; lower values are safer and may reduce expressivity.
- Compare under equal token budget and wall-clock budget against standard attention, because tropical attention changes both the inductive bias and the compute profile.

Recommended first comparisons:

- standard TokenGT baseline;
- standard backend plus existing tropical logit diagnostics;
- full MHTA backend;
- MHTA backend plus topology diagnostics;
- MHTA backend plus graph-of-thought/GFlowNet rewards.

## Metrics and Logging

Training and validation should log:

- `tropical_attention/enabled`;
- `tropical_attention/score_mean`;
- `tropical_attention/score_std`;
- `tropical_attention/distance_mean`;
- `tropical_attention/top1_margin`;
- `tropical_attention/selection_confidence`;
- `tropical_attention/unique_argmax_rate`;
- `tropical_attention/context_abs_mean`.

The metrics should appear in JSONL metrics, W&B, tqdm-derived train progress, and validation output. Existing `tropical/logit_*` metrics remain unchanged and measure output-logit sharpness rather than internal attention geometry.

## Validation Criteria

The integration is acceptable only if:

1. Standard backend tests continue to pass.
2. Tropical backend forward/backward runs on CPU.
3. Masked tropical attention respects padding and causal masks.
4. A tiny training run logs tropical-attention metrics.
5. Validation includes the same metrics.
6. Config files make the backend explicit and keep standard attention as the default.
7. README, architecture docs, and the paper describe the limits: MHTA is promising for graph/algorithmic reasoning but not yet proven as a universal replacement for softmax in large autoregressive language training.

## Risk Register

- **Memory blowup:** MHTA has explicit pairwise differences with shape `[B,H,S,S,Dh]`; long-context training can become infeasible.
- **Loss instability:** max-plus projections and `expm1` devaluation can create large values. Clamp and near-zero initialization are required.
- **Task mismatch:** the paper's strongest evidence is algorithmic reasoning; language and molecule graph training need ablations.
- **Ambiguity loss:** hard tropical selection can hurt tasks requiring calibrated uncertainty.
- **Metric confusion:** logit sharpness and attention sharpness are separate phenomena and should not be conflated.

## Implementation Checklist

- [x] Add native MHTA modules in `src/iska_reasoner/tropical/attention.py`.
- [x] Add config fields to `RandomOrderTokenGTConfig`.
- [x] Add tropical encoder path in `RandomOrderTokenGT`.
- [x] Log backend configuration in `stage_runner.py`.
- [x] Add attention metrics to training and validation.
- [x] Add tiny tropical backend config.
- [x] Add unit tests for MHTA masking, model forward/backward, and tiny training metrics.
- [x] Update README, architecture docs, metrics docs, and the research paper.
