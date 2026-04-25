from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from mlplatform.observability import ObservabilityService
from mlplatform.registry import ModelRegistry
from mlplatform.training.pipeline import TorchTrainingPipeline


class ServingService:
    def __init__(self, registry: ModelRegistry | None = None, observability: ObservabilityService | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self.observability = observability or ObservabilityService()
        self.pipeline = TorchTrainingPipeline()

    @lru_cache(maxsize=64)
    def _artifact_for(self, model_name: str, version: int) -> str:
        model_version = self.registry.get_version(model_name, version)
        if model_version is None:
            raise ValueError(f"unknown model version {model_name}:{version}")
        return model_version.artifact_uri

    def predict(self, model_name: str, features: list[float], version: int | None = None) -> dict[str, Any]:
        selected_version = version
        if selected_version is None:
            production = self.registry.get_production_version(model_name)
            if production is None:
                versions = self.registry.get_model_versions(model_name)
                if not versions:
                    raise ValueError(f"no versions available for model {model_name}")
                selected_version = versions[-1].version
            else:
                selected_version = production.version
        artifact_uri = self._artifact_for(model_name, selected_version)
        start = time.perf_counter()
        prediction, probability = self.pipeline.predict(artifact_uri, features)
        latency_ms = (time.perf_counter() - start) * 1000.0
        model_version = self.registry.get_version(model_name, selected_version)
        baseline_mean = []
        if model_version is not None:
            baseline_mean = self._extract_baseline_mean(artifact_uri)
        drift_score = self.observability.record_inference(model_name, selected_version, latency_ms, features, baseline_mean)
        return {
            "model_name": model_name,
            "model_version": selected_version,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": latency_ms,
            "drift_score": drift_score,
        }

    def _extract_baseline_mean(self, artifact_uri: str) -> list[float]:
        path = artifact_uri
        if path.endswith(".json"):
            import json
            from pathlib import Path

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return payload.get("baseline_mean", [])
        try:
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            return payload.get("baseline_mean", [])
        except Exception:
            return []
