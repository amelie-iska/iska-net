from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from iska_reasoner.tools import domain_metric_dict, verify_example_tokens
from iska_reasoner.topology import TOPOLOGY_FEATURE_NAMES, folding_contact_field, folding_contact_metrics, hidden_state_topology_metrics
from iska_reasoner.tropical import logit_diagnostics
from iska_reasoner.training.metrics import MetricAverager


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix: str = "val/",
    max_batches: int | None = None,
    hidden_topology_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    avg = MetricAverager()
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"{prefix}eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        tensor_batch: dict[str, Any] = {}
        for key, value in batch.items():
            tensor_batch[key] = value.to(device) if torch.is_tensor(value) else value
        out = model(**{k: tensor_batch[k] for k in [
            "input_ids",
            "kind_ids",
            "slot_ids",
            "endpoint_ids",
            "identifier_ids",
            "source_numeric_features",
            "attention_mask",
            "causal_mask",
            "labels",
        ]}, topology_targets=tensor_batch.get("topology_features"), numeric_targets=tensor_batch.get("numeric_targets"), numeric_mask=tensor_batch.get("numeric_mask"))
        metrics = {"loss": out["loss"].item(), "token_accuracy": out["token_accuracy"].item()}
        if "topology_loss" in out:
            metrics["topology_loss"] = out["topology_loss"].item()
        if "numeric_diffusion_loss" in out:
            metrics["numeric_diffusion_loss"] = out["numeric_diffusion_loss"].item()
        metrics.update(logit_diagnostics(out["logits"], tensor_batch["labels"]))
        topo = tensor_batch.get("topology_features")
        if topo is not None:
            for idx, name in enumerate(TOPOLOGY_FEATURE_NAMES):
                metrics[f"topology/{name}_mean"] = topo[:, idx].float().mean().item()
        if hidden_topology_cfg and hidden_topology_cfg.get("enabled", False):
            metrics.update(
                hidden_state_topology_metrics(
                    out["hidden_states"],
                    tensor_batch["attention_mask"],
                    max_points=int(hidden_topology_cfg.get("max_points", 64)),
                    bins=int(hidden_topology_cfg.get("bins", 8)),
                )
            )
        if hidden_topology_cfg and hidden_topology_cfg.get("folding_contact_enabled", False):
            contact = folding_contact_field(
                hidden_states=out["hidden_states"],
                token_mask=tensor_batch["attention_mask"],
            )
            metrics.update(folding_contact_metrics(contact, tensor_batch["attention_mask"]))
        pred = out["logits"].argmax(dim=-1)
        verifier_avg = MetricAverager()
        domain_avg = MetricAverager()
        for row, example in enumerate(batch.get("examples", [])):
            label_mask = tensor_batch["labels"][row].ne(-100)
            pred_tokens = [loader.collate_fn.vocab.decode(int(tok)) for tok in pred[row][label_mask].detach().cpu()] if hasattr(loader, "collate_fn") else []
            result = verify_example_tokens(example, pred_tokens)
            verifier_avg.update(result.metric_dict(prefix=""))
            domain_avg.update(domain_metric_dict(example, pred_tokens))
        metrics.update({f"verifier/{key}": value for key, value in verifier_avg.compute().items()})
        metrics.update(domain_avg.compute())
        avg.update(metrics)
    metrics = avg.compute(prefix=prefix)
    if f"{prefix}loss" in metrics:
        metrics[f"{prefix}perplexity"] = math.exp(min(20.0, metrics[f"{prefix}loss"]))
    model.train()
    return metrics
