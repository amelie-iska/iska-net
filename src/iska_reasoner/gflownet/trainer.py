from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from iska_reasoner.data.dataset import GraphJsonlDataset
from iska_reasoner.gflownet.trajectory import GraphSetPolicy, SubtrajectoryBalanceLoss, TrajectoryBalanceLoss, sample_trajectories
from iska_reasoner.tools import domain_metric_dict, verify_example_tokens
from iska_reasoner.topology import topology_feature_tensor
from iska_reasoner.training.metrics import MetricAverager
from iska_reasoner.utils.io import ensure_dir
from iska_reasoner.utils.logging import WandbLogger, get_device, set_seed, setup_logging


def _example_temperature_norm(example: Any) -> float | None:
    for node in getattr(example, "nodes", []) or []:
        if getattr(node, "type", "") != "temperature":
            continue
        features = getattr(node, "features", {}) or {}
        for key in ("kelvin", "kelvin_clamped", "temperature_k"):
            try:
                return max(0.0, min(1.0, (float(features[key]) - 300.0) / 100.0))
            except Exception:
                pass
        try:
            return max(0.0, min(1.0, (float(str(getattr(node, "value", "")).rstrip("Kk")) - 300.0) / 100.0))
        except Exception:
            return None
    metadata = getattr(example, "metadata", {}) or {}
    try:
        return max(0.0, min(1.0, (float(metadata.get("temperature")) - 300.0) / 100.0))
    except Exception:
        return None


def _temperature_diversity_bonuses(examples: list[Any], terminal_state: torch.Tensor, weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    if weight <= 0 or terminal_state.numel() == 0:
        return torch.zeros(terminal_state.size(0), dtype=torch.float32), {
            "temperature_diversity_bonus_mean": 0.0,
            "high_temperature_unique_terminal_states": 0.0,
            "high_temperature_terminal_hamming": 0.0,
        }
    state = terminal_state.float().cpu()
    temp_norms = [_example_temperature_norm(example) for example in examples]
    bonuses = torch.zeros(state.size(0), dtype=torch.float32)
    high_rows = [idx for idx, temp in enumerate(temp_norms) if temp is not None and temp >= 0.6]
    high_hamming = 0.0
    if high_rows:
        high_state = state[high_rows]
        unique = float(torch.unique(high_state, dim=0).size(0))
        if high_state.size(0) > 1:
            pairwise = torch.cdist(high_state, high_state, p=1) / max(1, high_state.size(1))
            non_diag = ~torch.eye(high_state.size(0), dtype=torch.bool)
            row_diversity = pairwise.masked_fill(~non_diag, 0.0).sum(dim=1) / max(1, high_state.size(0) - 1)
            high_hamming = float(row_diversity.mean().item())
        else:
            row_diversity = torch.ones(1, dtype=torch.float32)
            high_hamming = 1.0
        for local_idx, row in enumerate(high_rows):
            bonuses[row] = float(weight) * float(temp_norms[row] or 0.0) * row_diversity[local_idx]
    else:
        unique = 0.0
    return bonuses, {
        "temperature_diversity_bonus_mean": float(bonuses.mean().item()),
        "high_temperature_unique_terminal_states": unique,
        "high_temperature_terminal_hamming": high_hamming,
    }


def _candidate_vocab(dataset: GraphJsonlDataset, max_actions: int) -> tuple[list[str], torch.Tensor]:
    counts: dict[str, int] = {}
    for ex in tqdm((dataset[i] for i in range(len(dataset))), total=len(dataset), desc="gflownet/candidate_vocab"):
        for tok in ex.target_tokens:
            counts[tok] = counts.get(tok, 0) + 1
    candidates = [tok for tok, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_actions]]
    return candidates, torch.zeros(0)


def _collate_target_masks(candidates: list[str], use_context: bool = False):
    index = {tok: i for i, tok in enumerate(candidates)}

    def collate(examples):
        mask = torch.zeros(len(examples), len(candidates), dtype=torch.float32)
        for row, ex in enumerate(examples):
            for tok in ex.target_tokens:
                if tok in index:
                    mask[row, index[tok]] = 1.0
        batch = {"target_mask": mask, "example_ids": [ex.id for ex in examples], "examples": examples}
        if use_context:
            batch["context"] = topology_feature_tensor(examples)
        return batch

    return collate


def train_gflownet_stage(cfg: dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 17))
    set_seed(seed)
    oracle_cfg = cfg.get("oracle", {})
    if oracle_cfg:
        if oracle_cfg.get("backend"):
            os.environ["UGM_UMA_BACKEND"] = str(oracle_cfg["backend"])
        if oracle_cfg.get("strict") is not None:
            os.environ["UGM_UMA_STRICT"] = "1" if bool(oracle_cfg["strict"]) else "0"
        if oracle_cfg.get("model_name"):
            os.environ["UGM_UMA_MODEL"] = str(oracle_cfg["model_name"])
        if oracle_cfg.get("task_name"):
            os.environ["UGM_UMA_TASK"] = str(oracle_cfg["task_name"])
        if oracle_cfg.get("device"):
            os.environ["UGM_UMA_DEVICE"] = str(oracle_cfg["device"])
        if oracle_cfg.get("fairchem_repo"):
            os.environ["UGM_FAIRCHEM_REPO"] = str(oracle_cfg["fairchem_repo"])
    output_dir = ensure_dir(cfg["run"]["output_dir"])
    logger = setup_logging(output_dir)
    logger.info("Starting GFlowNet graph-of-thought stage")

    dataset = GraphJsonlDataset(cfg["data"]["train_path"])
    max_actions = int(cfg["gflownet"].get("max_actions", 64))
    candidates, _ = _candidate_vocab(dataset, max_actions)
    if not candidates:
        raise ValueError("No candidate target tokens for GFlowNet training")
    with (Path(output_dir) / "gflownet_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 8)),
        shuffle=True,
        collate_fn=_collate_target_masks(candidates, use_context=bool(cfg["gflownet"].get("use_context", False))),
    )
    device = get_device(cfg["run"].get("device", "cuda"))
    context_dim = 7 if bool(cfg["gflownet"].get("use_context", False)) else int(cfg["gflownet"].get("context_dim", 0))
    policy = GraphSetPolicy(len(candidates), hidden_dim=int(cfg["gflownet"].get("hidden_dim", 128)), context_dim=context_dim).to(device)
    backward_policy = None
    if bool(cfg["gflownet"].get("learn_backward_policy", False)):
        backward_policy = GraphSetPolicy(len(candidates), hidden_dim=int(cfg["gflownet"].get("hidden_dim", 128)), context_dim=context_dim).to(device)
    tb_loss = TrajectoryBalanceLoss(init_log_z=float(cfg["gflownet"].get("init_log_z", 0.0))).to(device)
    subtb_loss = SubtrajectoryBalanceLoss(init_log_z=float(cfg["gflownet"].get("init_log_z", 0.0))).to(device)
    subtrajectory_weight = float(cfg["gflownet"].get("subtrajectory_weight", 0.0))
    param_list = list(policy.parameters()) + list(tb_loss.parameters())
    if backward_policy is not None:
        param_list.extend(backward_policy.parameters())
    if subtrajectory_weight > 0:
        param_list.extend(subtb_loss.parameters())
    optimizer = torch.optim.AdamW(
        param_list,
        lr=float(cfg["train"].get("learning_rate", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    run_id = os.environ.get("RUN_ID")
    stage = cfg.get("stage", {}).get("name", "gflownet_got")
    wandb_run_name = cfg.get("wandb", {}).get("run_name") or (f"{run_id}-{stage}" if run_id else stage)
    wandb = WandbLogger({"enabled": cfg.get("wandb", {}).get("enabled", False), **cfg.get("wandb", {}), "config": cfg}, run_name=wandb_run_name)
    metrics_file = Path(output_dir) / "metrics.jsonl"
    total_steps = int(cfg["train"].get("max_steps", 100))
    max_traj_steps = int(cfg["gflownet"].get("max_traj_steps", 8))
    epsilon = float(cfg["gflownet"].get("epsilon", 0.05))
    temperature_diversity_reward_weight = float(cfg["gflownet"].get("temperature_diversity_reward_weight", 0.0))
    log_every = int(cfg["train"].get("log_every", 10))

    step = 0
    pbar = tqdm(total=total_steps, desc="train/gflownet_got")
    rollout_path = Path(output_dir) / "rollouts.jsonl"
    while step < total_steps:
        for batch in loader:
            target_mask = batch["target_mask"].to(device)
            context = batch.get("context")
            if context is not None:
                context = context.to(device)
            traj = sample_trajectories(policy, target_mask, max_steps=max_traj_steps, epsilon=epsilon, backward_policy=backward_policy, context=context)
            rewards = []
            exact = []
            recalls = []
            predicted_token_rows: list[list[str]] = []
            domain_avg = MetricAverager()
            terminal_cpu = traj.terminal_state.detach().cpu()
            for row, example in enumerate(batch["examples"]):
                predicted_tokens = [candidates[i] for i, flag in enumerate(terminal_cpu[row].tolist()) if flag >= 0.5]
                predicted_token_rows.append(predicted_tokens)
                result = verify_example_tokens(example, predicted_tokens)
                rewards.append(result.reward)
                exact.append(1.0 if result.exact_token_match else 0.0)
                recalls.append(result.token_recall)
                domain_avg.update(domain_metric_dict(example, predicted_tokens))
            diversity_bonuses, diversity_metrics = _temperature_diversity_bonuses(
                batch["examples"],
                terminal_cpu,
                temperature_diversity_reward_weight,
            )
            traj.rewards = (
                torch.tensor(rewards, dtype=torch.float32, device=device)
                + diversity_bonuses.to(device)
            ).clamp_min(1e-4)
            tb = tb_loss(traj)
            sub_tb = subtb_loss(traj) if subtrajectory_weight > 0 else torch.tensor(0.0, device=device)
            loss = tb + subtrajectory_weight * sub_tb
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(param_list, 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(tb=f"{loss.item():.4f}", reward=f"{traj.rewards.mean().item():.3f}")

            if step % log_every == 0:
                metrics = {
                    "gflownet/tb_loss": loss.item(),
                    "gflownet/trajectory_balance_loss": tb.item(),
                    "gflownet/subtrajectory_balance_loss": sub_tb.item(),
                    "gflownet/logZ": tb_loss.log_z.item(),
                    "gflownet/subtb_logZ": subtb_loss.log_z.item(),
                    "gflownet/reward_mean": traj.rewards.mean().item(),
                    "gflownet/reward_max": traj.rewards.max().item(),
                    "gflownet/trajectory_len": traj.lengths.float().mean().item(),
                    "gflownet/terminal_valid_rate": traj.terminal_valid.mean().item(),
                    "gflownet/verifier_exact_rate": sum(exact) / max(1, len(exact)),
                    "gflownet/token_recall": sum(recalls) / max(1, len(recalls)),
                    "gflownet/unique_terminal_states": float(torch.unique(traj.terminal_state, dim=0).size(0)),
                    "gflownet/action_entropy": traj.action_entropy.mean().item(),
                    "gflownet/action_coverage": float(traj.action_counts.gt(0).float().mean().item()),
                    "gflownet/context_enabled": 1.0 if context is not None else 0.0,
                    "gflownet/backward_policy_learned": 1.0 if backward_policy is not None else 0.0,
                    "gflownet/grad_norm": float(grad_norm),
                }
                metrics.update({f"gflownet/{key}": value for key, value in diversity_metrics.items()})
                metrics.update({f"gflownet/{key}": value for key, value in domain_avg.compute().items()})
                wandb.log(metrics, step)
                with metrics_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, **metrics}, sort_keys=True) + "\n")
                with rollout_path.open("a", encoding="utf-8") as f:
                    for example, toks, reward in zip(batch["examples"], predicted_token_rows, rewards):
                        f.write(json.dumps({"step": step, "example_id": example.id, "task": example.task, "tokens": toks, "reward": reward}, sort_keys=True) + "\n")
            if step >= total_steps:
                break

    pbar.close()
    torch.save(
        {
            "policy": policy.state_dict(),
            "backward_policy": backward_policy.state_dict() if backward_policy is not None else None,
            "tb_loss": tb_loss.state_dict(),
            "subtb_loss": subtb_loss.state_dict(),
            "candidates": candidates,
            "config": cfg,
            "context_dim": context_dim,
        },
        Path(output_dir) / "gflownet_final.pt",
    )
    wandb.finish()
    logger.info("Finished GFlowNet stage at step %d", step)
