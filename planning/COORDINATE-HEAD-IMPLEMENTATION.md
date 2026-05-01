# Optional UGM Coordinate Head

UGM remains a random-order autoregressive graph-to-graph model. Coordinate records are symbolic graph tokens first: a generated position is named by records such as `COORD:f0:a17:x:pos_near`, where the token identifies the frame, atom slot, axis, and coordinate bin. The optional coordinate head is a continuous refinement side-channel for an explicit structure phase, not a replacement for graph-token decoding.

## Contract

- Default sequence-only training keeps `model.coordinate_head_enabled: false` and `loss.coordinate_loss_weight: 0.0`.
- Coordinate supervision is masked per axis and is applied only at `<POS>` slots whose target token is a `COORD:f*:a*:axis:*` record.
- The graph-token cross-entropy still decides whether a coordinate record exists and what graph identity it has.
- The coordinate head predicts six values per token: mean `(x,y,z)` and log variance `(log sigma_x^2, log sigma_y^2, log sigma_z^2)`.
- The loss is a masked diagonal Gaussian negative log likelihood, scaled by `model.coordinate_target_scale`, and logged with RMSE, supervised-axis count, and predicted sigma.

## Enabling

Use the explicit override only with a structure-approved dataset:

```bash
conda run -n tokengt python scripts/train_stage.py \
  --config config/model/max_4090_tokengt.yaml \
  --config config/data/structure_dynamics_graphs.yaml \
  --config config/train/structure_dynamics_4090.yaml \
  --config config/train/overrides/coordinate_head.yaml
```

For 4090 runs, keep the initial batch size conservative. The coordinate head is small, but structure rows have long graph records, and the main memory cost is still transformer activation storage.

## Validation

The implementation is covered by tests that verify:

- coordinate targets are derived from coordinate nodes and aligned with `COORD:f*:a*:axis:*` target tokens;
- the collator emits `coordinate_targets` and `coordinate_mask` tensors;
- the model computes coordinate NLL/RMSE and backpropagates through the coordinate head;
- default sequence-only model paths continue to run without coordinate supervision.

The coordinate head should be evaluated with held-out structure rows only after the structure-file phase is explicitly enabled. For the first sequence-only run, UMA/contact feedback can still shape attention maps and embedding geometry, but direct coordinate labels stay disabled.
