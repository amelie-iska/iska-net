from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TropicalSchedule:
    temperature: float = 1.0
    temperature_min: float = 0.25
    anneal_steps: int = 10_000

    def value(self, step: int) -> float:
        if self.anneal_steps <= 0:
            return max(self.temperature_min, self.temperature)
        frac = min(1.0, max(0.0, step / self.anneal_steps))
        return float(self.temperature + frac * (self.temperature_min - self.temperature))


@torch.no_grad()
def logit_diagnostics(logits: torch.Tensor, labels: torch.Tensor | None = None, temperature: float = 1.0) -> dict[str, float]:
    if labels is not None:
        mask = labels.ne(-100)
        if mask.any():
            logits = logits[mask]
        else:
            return {
                "tropical/logit_entropy": 0.0,
                "tropical/top1_margin": 0.0,
                "tropical/top1_confidence": 0.0,
            }
    if logits.numel() == 0:
        return {
            "tropical/logit_entropy": 0.0,
            "tropical/top1_margin": 0.0,
            "tropical/top1_confidence": 0.0,
        }
    scaled = logits.float() / max(float(temperature), 1e-6)
    scaled = torch.nan_to_num(scaled, nan=0.0, posinf=1e4, neginf=-1e4)
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
    top2 = torch.topk(scaled, k=min(2, scaled.size(-1)), dim=-1).values
    margin = top2[:, 0] - top2[:, 1] if top2.size(-1) > 1 else top2[:, 0]
    confidence = probs.max(dim=-1).values.mean()
    return {
        "tropical/logit_entropy": float(entropy.item()),
        "tropical/top1_margin": float(margin.mean().item()),
        "tropical/top1_confidence": float(confidence.item()),
    }
