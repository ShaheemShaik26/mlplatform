from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlplatform.config import SETTINGS
from mlplatform.registry import ModelRegistry


@dataclass(frozen=True)
class PromotionDecision:
    allowed: bool
    reason: str
    promoted_version: int | None = None


class PromotionPolicy:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def evaluate(
        self,
        model_name: str,
        candidate_metrics: dict[str, Any],
        candidate_latency_ms: float,
        minimum_validation_f1: float | None = None,
        min_improvement: float | None = None,
        max_latency_ms: float | None = None,
    ) -> PromotionDecision:
        min_improvement = SETTINGS.promotion_min_improvement if min_improvement is None else min_improvement
        max_latency_ms = SETTINGS.promotion_max_latency_ms if max_latency_ms is None else max_latency_ms
        minimum_validation_f1 = SETTINGS.min_validation_f1 if minimum_validation_f1 is None else minimum_validation_f1

        candidate_f1 = float(candidate_metrics.get("f1", 0.0))
        if candidate_f1 < minimum_validation_f1:
            return PromotionDecision(False, f"candidate f1 {candidate_f1:.4f} below minimum {minimum_validation_f1:.4f}")
        if candidate_latency_ms > max_latency_ms:
            return PromotionDecision(False, f"latency {candidate_latency_ms:.2f}ms exceeds max {max_latency_ms:.2f}ms")

        current = self.registry.get_production_version(model_name)
        if current is None:
            return PromotionDecision(True, "no production version exists; promotion allowed")

        current_metrics = current.metrics_json
        current_f1 = 0.0
        try:
            import json

            current_metrics_dict = json.loads(current_metrics)
            current_f1 = float(current_metrics_dict.get("f1", 0.0))
        except Exception:
            pass

        improvement = candidate_f1 - current_f1
        if improvement < min_improvement:
            return PromotionDecision(
                False,
                f"improvement {improvement:.4f} below threshold {min_improvement:.4f}",
            )
        return PromotionDecision(True, "promotion checks passed", promoted_version=current.version)
