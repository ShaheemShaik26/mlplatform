from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from mlplatform.db import session_scope
from mlplatform.models import InferenceEvent, ModelVersion


@dataclass
class MetricWindow:
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    request_count: int = 0
    drift_scores: deque[float] = field(default_factory=lambda: deque(maxlen=1000))


class ObservabilityService:
    def __init__(self) -> None:
        self._windows: dict[str, MetricWindow] = defaultdict(MetricWindow)

    def record_inference(
        self,
        model_name: str,
        model_version: int,
        latency_ms: float,
        features: list[float],
        baseline_mean: list[float] | None = None,
    ) -> float:
        window = self._windows[model_name]
        window.latencies.append(latency_ms)
        window.request_count += 1
        drift_score = self._drift_score(features, baseline_mean)
        window.drift_scores.append(drift_score)
        with session_scope() as session:
            event = InferenceEvent(
                request_id=str(uuid4()),
                model_name=model_name,
                model_version=model_version,
                latency_ms=latency_ms,
                drift_score=drift_score,
                feature_snapshot_json=json.dumps(features),
            )
            session.add(event)
        return drift_score

    def _drift_score(self, features: list[float], baseline_mean: list[float] | None) -> float:
        if not baseline_mean:
            return 0.0
        if not features:
            return 0.0
        size = min(len(features), len(baseline_mean))
        if size == 0:
            return 0.0
        diffs = [abs(features[index] - baseline_mean[index]) for index in range(size)]
        return sum(diffs) / size

    def get_metrics(self, model_name: str) -> dict[str, Any]:
        window = self._windows[model_name]
        return {
            "request_count": window.request_count,
            "mean_latency_ms": mean(window.latencies) if window.latencies else 0.0,
            "mean_drift_score": mean(window.drift_scores) if window.drift_scores else 0.0,
        }

    def fetch_drift_history(self, model_name: str) -> list[InferenceEvent]:
        with session_scope() as session:
            return list(
                session.execute(
                    select(InferenceEvent)
                    .where(InferenceEvent.model_name == model_name)
                    .order_by(InferenceEvent.created_at.desc())
                ).scalars()
            )
