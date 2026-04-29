#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from iska_reasoner.data.dataset import GraphJsonlDataset
from iska_reasoner.gflownet.trajectory import GraphSetPolicy, sample_trajectories
from iska_reasoner.tools import domain_metric_dict, verify_example_tokens
from iska_reasoner.topology import topology_feature_tensor
from iska_reasoner.training.metrics import MetricAverager
from iska_reasoner.utils.config import load_config


def _collate_target_masks(candidates: list[str], use_context: bool = False):
    index = {tok: i for i, tok in enumerate(candidates)}

    def collate(examples):
        mask = torch.zeros(len(examples), len(candidates), dtype=torch.float32)
        for row, ex in enumerate(examples):
            for tok in ex.target_tokens:
                if tok in index:
                    mask[row, index[tok]] = 1.0
        batch = {"target_mask": mask, "examples": examples}
        if use_context:
            batch["context"] = topology_feature_tensor(examples)
        return batch

    return collate


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a saved GFlowNet graph-of-thought policy.")
    parser.add_argument("--config", action="append", help="YAML config path used as defaults.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-traj-steps", type=int)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else {}
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
    gcfg = cfg.get("gflownet_validation", {})
    checkpoint = args.checkpoint or gcfg.get("checkpoint") or str(Path(cfg.get("run", {}).get("output_dir", "")) / "gflownet_final.pt")
    data_path = args.data or gcfg.get("data_path") or cfg.get("data", {}).get("val_path") or cfg.get("data", {}).get("train_path")
    if not checkpoint or not data_path:
        raise SystemExit("Provide --checkpoint/--data or config values for them")
    batch_size = int(args.batch_size or gcfg.get("batch_size", cfg.get("train", {}).get("eval_batch_size", 8)))
    max_traj_steps = int(args.max_traj_steps or gcfg.get("max_traj_steps", cfg.get("gflownet", {}).get("max_traj_steps", 8)))
    epsilon = float(args.epsilon if args.epsilon is not None else gcfg.get("epsilon", cfg.get("gflownet", {}).get("epsilon", 0.0)))
    device = torch.device(args.device or gcfg.get("device", cfg.get("run", {}).get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint, map_location="cpu")
    candidates = list(ckpt["candidates"])
    ckpt_cfg = ckpt.get("config", {})
    merged_gcfg = {**ckpt_cfg.get("gflownet", {}), **cfg.get("gflownet", {})}
    hidden_dim = int(merged_gcfg.get("hidden_dim", 128))
    context_dim = int(ckpt.get("context_dim", 0))
    use_context = context_dim > 0
    policy = GraphSetPolicy(len(candidates), hidden_dim=hidden_dim, context_dim=context_dim).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    dataset = GraphJsonlDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_target_masks(candidates, use_context=use_context))
    avg = MetricAverager()
    sample_rows = []
    for batch in tqdm(loader, desc="validate/gflownet"):
        target_mask = batch["target_mask"].to(device)
        context = batch.get("context")
        if context is not None:
            context = context.to(device)
        traj = sample_trajectories(policy, target_mask, max_steps=max_traj_steps, epsilon=epsilon, context=context)
        terminal_cpu = traj.terminal_state.detach().cpu()
        verifier_avg = MetricAverager()
        domain_avg = MetricAverager()
        rewards = []
        for row, example in enumerate(batch["examples"]):
            predicted_tokens = [candidates[i] for i, flag in enumerate(terminal_cpu[row].tolist()) if flag >= 0.5]
            result = verify_example_tokens(example, predicted_tokens)
            rewards.append(result.reward)
            verifier_avg.update(result.metric_dict(prefix=""))
            domain_avg.update(domain_metric_dict(example, predicted_tokens))
            if len(sample_rows) < 20:
                sample_rows.append({"example_id": example.id, "task": example.task, "tokens": predicted_tokens, "reward": result.reward})
        metrics = {
            "gflownet_val/reward_mean": float(sum(rewards) / max(1, len(rewards))),
            "gflownet_val/trajectory_len": traj.lengths.float().mean().item(),
            "gflownet_val/terminal_valid_rate": traj.terminal_valid.mean().item(),
            "gflownet_val/unique_terminal_states": float(torch.unique(traj.terminal_state, dim=0).size(0)),
            "gflownet_val/action_entropy": traj.action_entropy.mean().item(),
            "gflownet_val/action_coverage": float(traj.action_counts.gt(0).float().mean().item()),
            "gflownet_val/context_enabled": 1.0 if context is not None else 0.0,
        }
        metrics.update({f"gflownet_val/verifier/{key}": value for key, value in verifier_avg.compute().items()})
        metrics.update({f"gflownet_val/{key}": value for key, value in domain_avg.compute().items()})
        avg.update(metrics)
    out = avg.compute()
    out["gflownet_val/example_count"] = float(len(dataset))
    if args.output or gcfg.get("output"):
        path = Path(args.output or gcfg["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metrics": out, "samples": sample_rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
