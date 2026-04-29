from __future__ import annotations

from collections import defaultdict
from typing import Any


class MetricAverager:
    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            self.sums[key] += val
            self.counts[key] += 1

    def compute(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}{key}": self.sums[key] / max(1, self.counts[key])
            for key in sorted(self.sums)
        }

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()

