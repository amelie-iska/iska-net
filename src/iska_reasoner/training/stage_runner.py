from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from iska_reasoner.data.dataset import GraphJsonlDataset, RandomOrderCollator
from iska_reasoner.data.phase_policy import actual_structure_file_source, graph_structure_violations, sanitize_graph_example_for_sequence_only
from iska_reasoner.data.vocab import GraphVocab, build_vocab, read_extra_tokens
from iska_reasoner.gflownet.trainer import train_gflownet_stage
from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGTConfig, RandomOrderTokenGT
from iska_reasoner.topology import (
    TOPOLOGY_FEATURE_NAMES,
    folding_contact_field,
    folding_contact_metrics,
    hidden_js_geometry_loss,
    hidden_state_topology_metrics,
    hidden_topology_collapse_loss,
)
from iska_reasoner.training.checkpointing import load_checkpoint, save_checkpoint
from iska_reasoner.training.metrics import MetricAverager
from iska_reasoner.tropical import TropicalSchedule, logit_diagnostics
from iska_reasoner.utils.io import ensure_dir
from iska_reasoner.utils.logging import WandbLogger, get_device, set_seed, setup_logging
from iska_reasoner.validation.evaluate import evaluate_model


def _tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _check_dataset_policy(dataset: GraphJsonlDataset, cfg: dict[str, Any], split: str, logger: Any) -> None:
    sequence_only = bool(cfg["data"].get("enforce_sequence_only_molecules", False))
    forbid_structure_files = bool(cfg["data"].get("forbid_actual_structure_files", True))
    if bool(cfg["data"].get("skip_policy_check", False)):
        logger.warning(
            "Skipping dataset policy check for %s split because data.skip_policy_check=true. "
            "Use this only after the same corpus has already passed the full sequence-only policy scan.",
            split,
        )
        return
    if not sequence_only and not forbid_structure_files:
        return
    max_rows = cfg["data"].get("policy_check_max_rows")
    if max_rows is None:
        max_rows = cfg.get("train", {}).get("policy_check_max_rows")
    if max_rows is None:
        max_rows = cfg.get("training", {}).get("policy_check_max_rows")
    max_rows = None if max_rows is None else int(max_rows)
    checked = 0
    violations: list[str] = []
    total = len(dataset) if max_rows is None else min(len(dataset), max_rows)
    for idx in tqdm(range(total), desc=f"policy/{split}", leave=False):
        ex = dataset[idx]
        checked += 1
        source_path = actual_structure_file_source(ex)
        if forbid_structure_files and source_path:
            violations.append(f"{ex.id}: actual structure file source {source_path}")
        if sequence_only:
            found = graph_structure_violations(ex)
            if found:
                violations.append(f"{ex.id}: sequence-only violation count={len(found)} first={found[:8]}")
        if len(violations) >= 20:
            break
    logger.info("Dataset policy checked %d %s rows: sequence_only=%s forbid_structure_files=%s", checked, split, sequence_only, forbid_structure_files)
    if violations:
        joined = "\n".join(violations)
        raise ValueError(f"Dataset policy violation in {split} split:\n{joined}")


def _resolve_total_steps(cfg: dict[str, Any], train_examples: int) -> int:
    train_cfg = cfg["train"]
    batch_size = max(1, int(train_cfg.get("batch_size", 4)))
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, math.ceil(train_examples / effective_batch))

    max_steps = train_cfg.get("max_steps", 100)
    max_steps_text = str(max_steps).strip().lower() if max_steps is not None else ""
    full_epoch_requested = max_steps_text in {"auto", "epoch", "full_epoch", "full-dataset", "full_dataset"}
    epochs = train_cfg.get("full_epochs", train_cfg.get("epochs"))
    if epochs is not None or full_epoch_requested:
        epoch_count = float(epochs if epochs is not None else 1.0)
        if epoch_count <= 0:
            raise ValueError(f"full_epochs must be positive, got {epoch_count}")
        return max(1, math.ceil(steps_per_epoch * epoch_count))
    return max(1, int(max_steps))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "full", "all", "0"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"Expected a non-negative integer or 'full', got {value!r}")
    return parsed or None


def _scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        try:
            return float(value.detach().float().cpu().item())
        except (RuntimeError, ValueError, TypeError):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def run_training_stage(cfg: dict[str, Any]) -> None:
    stage = cfg["stage"]["name"]
    if cfg.get("run", {}).get("train_disabled"):
        raise SystemExit(cfg["run"].get("disable_reason", f"Training stage {stage} is disabled by config."))
    if stage == "gflownet_got":
        train_gflownet_stage(cfg)
        return

    seed = int(cfg.get("seed", 17))
    set_seed(seed)
    output_dir = ensure_dir(cfg["run"]["output_dir"])
    logger = setup_logging(output_dir)
    logger.info("Starting stage %s", stage)

    sequence_only = bool(cfg["data"].get("enforce_sequence_only_molecules", False))
    transform = sanitize_graph_example_for_sequence_only if sequence_only else None
    dataset = GraphJsonlDataset(cfg["data"]["train_path"], transform=transform)
    _check_dataset_policy(dataset, cfg, "train", logger)
    if cfg["data"].get("val_path"):
        train_ds = dataset
        val_ds = GraphJsonlDataset(cfg["data"]["val_path"], transform=transform)
        _check_dataset_policy(val_ds, cfg, "val", logger)
    else:
        val_fraction = float(cfg["data"].get("val_fraction", 0.1))
        val_size = max(1, int(len(dataset) * val_fraction)) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size > 0:
            train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
        else:
            train_ds, val_ds = dataset, None

    vocab_path = Path(cfg["data"].get("vocab_path", output_dir / "vocab.jsonl"))
    if vocab_path.exists() and cfg["data"].get("reuse_vocab", True):
        vocab = GraphVocab.load(vocab_path)
    else:
        extra_tokens = read_extra_tokens(cfg["data"].get("extra_vocab_paths", []))
        vocab = build_vocab(
            (dataset[i] for i in range(len(dataset))),
            min_freq=int(cfg["data"].get("min_freq", 1)),
            max_size=cfg["data"].get("max_vocab_size"),
            extra_tokens=extra_tokens,
            total=len(dataset),
            progress_desc=f"vocab/{stage}",
        )
        vocab.save(vocab_path)
    logger.info("Vocab size: %d", len(vocab.token_to_id))

    collator = RandomOrderCollator(
        vocab=vocab,
        max_source_tokens=int(cfg["data"].get("max_source_tokens", 128)),
        max_target_tokens=int(cfg["data"].get("max_target_tokens", 64)),
        max_seq_len=int(cfg["model"].get("max_seq_len", 256)),
        max_numeric_targets=int(cfg["data"].get("max_numeric_targets", cfg["model"].get("numeric_dim", 0))),
        order_mode=cfg["data"].get("order_mode", "sample"),
        seed=seed,
    )
    requested_device = str(cfg["run"].get("device", "cuda"))
    train_workers = int(cfg["train"].get("num_workers", 0))
    eval_workers = int(cfg["train"].get("eval_num_workers", 0))
    pin_memory = bool(cfg["train"].get("pin_memory", requested_device.startswith("cuda")))
    persistent_workers = bool(cfg["train"].get("persistent_workers", True)) and train_workers > 0
    prefetch_factor = cfg["train"].get("prefetch_factor")
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg["train"].get("batch_size", 4)),
        "shuffle": True,
        "num_workers": train_workers,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None and train_workers > 0:
        train_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(
        train_ds,
        **train_loader_kwargs,
    )
    val_loader = None
    if val_ds is not None:
        val_loader_kwargs: dict[str, Any] = {
            "batch_size": int(cfg["train"].get("eval_batch_size", cfg["train"].get("batch_size", 4))),
            "shuffle": False,
            "num_workers": eval_workers,
            "collate_fn": collator,
            "pin_memory": pin_memory,
            "persistent_workers": bool(cfg["train"].get("persistent_workers", True)) and eval_workers > 0,
        }
        if prefetch_factor is not None and eval_workers > 0:
            val_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        val_loader = DataLoader(
            val_ds,
            **val_loader_kwargs,
        )

    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = len(vocab.token_to_id)
    model = RandomOrderTokenGT(RandomOrderTokenGTConfig(**model_cfg))
    logger.info(
        "Model attention backend: %s tropical_projection=%s tropical_norm_shift=%s tropical_projective_normalize=%s",
        model.cfg.attention_backend,
        model.cfg.tropical_use_projection,
        model.cfg.tropical_use_norm_shift,
        model.cfg.tropical_projective_normalize,
    )
    device = get_device(cfg["run"].get("device", "cuda"))
    model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["train"].get("learning_rate", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )
    total_steps = _resolve_total_steps(cfg, len(train_ds))
    effective_batch = int(cfg["train"].get("batch_size", 4)) * int(cfg["train"].get("gradient_accumulation_steps", 1))
    logger.info(
        "Training schedule: train_examples=%d effective_batch=%d total_steps=%d",
        len(train_ds),
        effective_batch,
        total_steps,
    )
    logger.info(
        "DataLoader settings: train_workers=%d eval_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        train_workers,
        eval_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    )
    scheduler = None
    scheduler_name = cfg["train"].get("scheduler", "none")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    elif scheduler_name == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=float(cfg["train"].get("end_lr_factor", 0.1)), total_iters=max(1, total_steps))
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run_id = os.environ.get("RUN_ID")
    wandb_run_name = cfg.get("wandb", {}).get("run_name") or (f"{run_id}-{stage}" if run_id else stage)
    wandb = WandbLogger({"enabled": cfg.get("wandb", {}).get("enabled", False), **cfg.get("wandb", {}), "config": cfg}, run_name=wandb_run_name)

    grad_accum = int(cfg["train"].get("gradient_accumulation_steps", 1))
    log_every = max(1, int(cfg["train"].get("log_every", 10)))
    eval_every = int(cfg["train"].get("eval_every", 50))
    eval_enabled = eval_every > 0
    eval_max_batches = _optional_positive_int(cfg["train"].get("eval_max_batches"))
    ckpt_every = int(cfg["train"].get("checkpoint_every", 100))
    ckpt_enabled = ckpt_every > 0
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    topo_weight = float(cfg.get("loss", {}).get("topology_weight", 0.0))
    numeric_weight = float(cfg.get("loss", {}).get("numeric_diffusion_weight", 0.0))
    hidden_topology_cfg = cfg.get("hidden_topology", {})
    hidden_topology_enabled = bool(hidden_topology_cfg.get("enabled", False))
    hidden_topology_every = max(1, int(hidden_topology_cfg.get("log_every", log_every)))
    hidden_topology_max_points = int(hidden_topology_cfg.get("max_points", 64))
    hidden_topology_bins = int(hidden_topology_cfg.get("bins", 8))
    folding_contact_enabled = bool(hidden_topology_cfg.get("folding_contact_enabled", False))
    hidden_collapse_weight = float(cfg.get("loss", {}).get("hidden_topology_collapse_weight", 0.0))
    hidden_collapse_margin = float(hidden_topology_cfg.get("collapse_margin", 0.5))
    hidden_js_weight = float(cfg.get("loss", {}).get("hidden_js_geometry_weight", 0.0))
    hidden_js_margin = float(hidden_topology_cfg.get("js_margin", 0.05))
    if stage == "topology_aux" and topo_weight == 0.0:
        topo_weight = 1.0
    tropical_cfg = cfg.get("tropical", {})
    tropical_schedule = TropicalSchedule(
        temperature=float(tropical_cfg.get("temperature", 1.0)),
        temperature_min=float(tropical_cfg.get("temperature_min", 0.25)),
        anneal_steps=int(tropical_cfg.get("anneal_steps", max(1, total_steps))),
    )

    metrics_file = Path(output_dir) / "metrics.jsonl"
    avg = MetricAverager()
    step = 0
    if cfg["train"].get("resume_from"):
        step = load_checkpoint(cfg["train"]["resume_from"], model, optimizer)
        logger.info("Resumed from %s at step %d", cfg["train"]["resume_from"], step)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, initial=min(step, total_steps), desc=f"train/{stage}")
    micro_step = 0
    while step < total_steps:
        for batch in train_loader:
            batch = _tensor_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                out = model(**{k: batch[k] for k in [
                    "input_ids",
                    "kind_ids",
                    "slot_ids",
                    "endpoint_ids",
                    "identifier_ids",
                    "source_numeric_features",
                    "attention_mask",
                    "causal_mask",
                    "labels",
                ]}, topology_targets=batch.get("topology_features"), numeric_targets=batch.get("numeric_targets"), numeric_mask=batch.get("numeric_mask"))
                full_loss = out["loss"]
                if topo_weight > 0 and "topology_loss" in out:
                    full_loss = full_loss + topo_weight * out["topology_loss"]
                if numeric_weight > 0 and "numeric_diffusion_loss" in out:
                    full_loss = full_loss + numeric_weight * out["numeric_diffusion_loss"]
                hidden_collapse = torch.tensor(0.0, device=device)
                hidden_js_loss = torch.tensor(0.0, device=device)
                if hidden_collapse_weight > 0:
                    hidden_collapse = hidden_topology_collapse_loss(
                        out["hidden_states"],
                        batch["attention_mask"],
                        margin=hidden_collapse_margin,
                        max_points=hidden_topology_max_points,
                    )
                    full_loss = full_loss + hidden_collapse_weight * hidden_collapse
                if hidden_js_weight > 0:
                    hidden_js_loss = hidden_js_geometry_loss(
                        out["hidden_states"],
                        batch["attention_mask"],
                        margin=hidden_js_margin,
                        max_points=hidden_topology_max_points,
                    )
                    full_loss = full_loss + hidden_js_weight * hidden_js_loss
                if not bool(torch.isfinite(full_loss.detach()).all().item()):
                    emergency_path = Path(output_dir) / f"checkpoint_nonfinite_step_{step}_micro_{micro_step}.pt"
                    save_checkpoint(emergency_path, model, optimizer, step, cfg)
                    example_ids = batch.get("example_ids", [])[:8]
                    diagnostics = {
                        "loss": _scalar_or_none(out.get("loss")),
                        "total_loss": _scalar_or_none(full_loss),
                        "topology_loss": _scalar_or_none(out.get("topology_loss")),
                        "numeric_diffusion_loss": _scalar_or_none(out.get("numeric_diffusion_loss")),
                        "hidden_topology/collapse_loss": _scalar_or_none(hidden_collapse),
                        "hidden_topology/js_geometry_loss": _scalar_or_none(hidden_js_loss),
                    }
                    raise FloatingPointError(
                        "Non-finite training loss detected "
                        f"stage={stage} step={step} micro_step={micro_step} "
                        f"example_ids={example_ids} checkpoint={emergency_path} diagnostics={diagnostics}"
                    )
                loss = full_loss / grad_accum
            scaler.scale(loss).backward()
            micro_step += 1
            step_temp = tropical_schedule.value(step)
            metrics = {
                "loss": out["loss"].item(),
                "token_accuracy": out["token_accuracy"].item(),
                "total_loss": full_loss.item(),
                "topology_loss": out.get("topology_loss", torch.tensor(0.0, device=device)).item(),
                "hidden_topology/collapse_loss": hidden_collapse.item(),
                "hidden_topology/js_geometry_loss": hidden_js_loss.item(),
                "numeric_diffusion_loss": out.get("numeric_diffusion_loss", torch.tensor(0.0, device=device)).item(),
                "tropical/temperature": step_temp,
            }
            for key, value in out.get("attention_metrics", {}).items():
                metrics[key] = value.item() if torch.is_tensor(value) else value
            metrics.update(logit_diagnostics(out["logits"], batch["labels"], temperature=step_temp))
            topo = batch.get("topology_features")
            if topo is not None:
                for idx, name in enumerate(TOPOLOGY_FEATURE_NAMES):
                    metrics[f"topology/{name}_mean"] = topo[:, idx].float().mean().item()
            if hidden_topology_enabled and (step % hidden_topology_every == 0):
                metrics.update(
                    hidden_state_topology_metrics(
                        out["hidden_states"],
                        batch["attention_mask"],
                        max_points=hidden_topology_max_points,
                        bins=hidden_topology_bins,
                    )
                )
            if folding_contact_enabled and (step % hidden_topology_every == 0):
                contact = folding_contact_field(
                    hidden_states=out["hidden_states"],
                    token_mask=batch["attention_mask"],
                )
                metrics.update(folding_contact_metrics(contact, batch["attention_mask"]))
            avg.update(metrics)

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
                if scheduler is not None and optimizer_stepped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                postfix = {"loss": f"{out['loss'].item():.4f}", "acc": f"{out['token_accuracy'].item():.3f}"}
                if "tropical_attention/top1_margin" in metrics:
                    postfix["trop_margin"] = f"{metrics['tropical_attention/top1_margin']:.3f}"
                pbar.set_postfix(**postfix)

                if step % log_every == 0:
                    metrics = avg.compute(prefix=f"{stage}/train/")
                    metrics[f"{stage}/train/grad_norm"] = float(grad_norm)
                    metrics[f"{stage}/train/lr"] = optimizer.param_groups[0]["lr"]
                    wandb.log(metrics, step)
                    with metrics_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
                    avg.reset()

                if eval_enabled and val_loader is not None and step % eval_every == 0:
                    metrics = evaluate_model(model, val_loader, device, prefix=f"{stage}/val/", max_batches=eval_max_batches, hidden_topology_cfg=hidden_topology_cfg)
                    wandb.log(metrics, step)
                    with metrics_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")

                if ckpt_enabled and step % ckpt_every == 0:
                    save_checkpoint(Path(output_dir) / f"checkpoint_step_{step}.pt", model, optimizer, step, cfg)
                if step >= total_steps:
                    break
    pbar.close()
    save_checkpoint(Path(output_dir) / "checkpoint_final.pt", model, optimizer, step, cfg)
    wandb.finish()
    logger.info("Finished stage %s at step %d", stage, step)
