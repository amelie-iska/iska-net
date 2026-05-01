#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from iska_reasoner.models.random_order_tokengt import RandomOrderTokenGT, RandomOrderTokenGTConfig
from iska_reasoner.utils.config import load_config


def _default_seq_len(cfg: dict[str, Any]) -> int:
    model_len = int(cfg.get("model", {}).get("max_seq_len", 256))
    data_cfg = cfg.get("data", {})
    if "max_source_tokens" in data_cfg and "max_target_tokens" in data_cfg:
        encoded_len = int(data_cfg["max_source_tokens"]) + 1 + 2 * int(data_cfg["max_target_tokens"])
        return min(model_len, encoded_len)
    return model_len


def _run_single(cfg: dict[str, Any], batch_size: int, seq_len: int, vocab_size: int, optimizer_step: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch-size probing")
    torch.manual_seed(17)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = vocab_size
    model = RandomOrderTokenGT(RandomOrderTokenGTConfig(**model_cfg)).cuda()
    model.train()

    train_cfg = cfg.get("train", {})
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)))

    max_nodes = int(model_cfg.get("max_nodes", 1024))
    max_slots = int(model_cfg.get("max_slots", 256))
    max_identifiers = int(model_cfg.get("max_identifiers", 2 * max_nodes))
    num_kinds = int(model_cfg.get("num_kinds", 8))
    topology_dim = int(model_cfg.get("topology_dim", 7))

    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), device="cuda")
    kind_ids = torch.randint(0, num_kinds, (batch_size, seq_len), device="cuda")
    slot_ids = torch.randint(0, max_slots + 1, (batch_size, seq_len), device="cuda")
    endpoint_ids = torch.randint(0, max_nodes + 1, (batch_size, seq_len, 2), device="cuda")
    identifier_ids = torch.randint(0, max_identifiers + 1, (batch_size, seq_len), device="cuda")
    source_numeric_features = torch.randn(batch_size, seq_len, 4, device="cuda")
    attention_mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.bool)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1)
    labels = torch.full((batch_size, seq_len), -100, device="cuda", dtype=torch.long)
    supervised = min(16, seq_len)
    labels[:, -supervised:] = torch.randint(4, vocab_size, (batch_size, supervised), device="cuda")
    topology_targets = torch.randn(batch_size, topology_dim, device="cuda")

    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
        out = model(
            input_ids=input_ids,
            kind_ids=kind_ids,
            slot_ids=slot_ids,
            endpoint_ids=endpoint_ids,
            identifier_ids=identifier_ids,
            source_numeric_features=source_numeric_features,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            labels=labels,
            topology_targets=topology_targets,
        )
        loss = out["loss"]
        if "topology_loss" in out:
            loss = loss + float(cfg.get("loss", {}).get("topology_weight", 0.0)) * out["topology_loss"]
    scaler.scale(loss).backward()
    if optimizer_step:
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    props = torch.cuda.get_device_properties(0)
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "optimizer_step": optimizer_step,
        "loss": float(loss.detach().cpu()),
        "seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "total_memory_mib": props.total_memory / 2**20,
    }


def _single_main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seq_len = args.seq_len or _default_seq_len(cfg)
    try:
        result = _run_single(cfg, args.single_batch, seq_len, args.vocab_size, args.optimizer_step)
        print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(json.dumps({"ok": False, "batch_size": args.single_batch, "error": f"CUDA OOM: {exc}"}, indent=2, sort_keys=True))
        return 2


def _batch_list(args: argparse.Namespace) -> list[int]:
    if args.batch_sizes:
        return [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    sizes: list[int] = []
    value = max(1, int(args.min_batch))
    while value <= int(args.max_batch):
        sizes.append(value)
        value *= 2
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the largest CUDA batch size for a configured UGM model.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--vocab-size", type=int, default=262144)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--batch-sizes", help="Comma-separated batch sizes to test, for example 1,2,4,8.")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--optimizer-step", action="store_true", help="Also run optimizer.step() to include AdamW state allocation.")
    parser.add_argument("--single-batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_batch is not None:
        raise SystemExit(_single_main(args))

    results = []
    best = None
    for batch_size in _batch_list(args):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sum((["--config", path] for path in args.config), []),
            "--vocab-size",
            str(args.vocab_size),
            "--single-batch",
            str(batch_size),
        ]
        if args.seq_len:
            command.extend(["--seq-len", str(args.seq_len)])
        if args.optimizer_step:
            command.append("--optimizer-step")
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        stdout = proc.stdout.strip()
        try:
            payload = json.loads(stdout[stdout.find("{") :]) if "{" in stdout else {"ok": False, "error": stdout}
        except json.JSONDecodeError:
            payload = {"ok": False, "error": stdout, "stderr": proc.stderr}
        payload["returncode"] = proc.returncode
        results.append(payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if payload.get("ok"):
            best = payload
        else:
            break
    print(json.dumps({"best": best, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
