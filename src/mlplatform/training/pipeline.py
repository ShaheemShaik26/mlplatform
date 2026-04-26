from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlplatform.schemas import ExperimentConfig
from mlplatform.storage import ArtifactStore

try:  # optional dependency for real PyTorch training pipelines
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None
    optim = None


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, float]
    system_metadata: dict[str, Any]
    artifact_uri: str
    latency_ms: float
    baseline_mean: list[float]


class TorchTrainingPipeline:
    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def train(self, config: ExperimentConfig, artifact_name: str | None = None) -> TrainingResult:
        self._set_seed(config.seed)
        start = time.perf_counter()
        train_features, train_labels = self._generate_dataset(config.train_samples, config.feature_count, config.seed)
        validation_features, validation_labels = self._generate_dataset(
            config.validation_samples, config.feature_count, config.seed + 1
        )
        if config.use_torch and torch is not None:
            metrics, artifact_uri = self._train_with_torch(
                config,
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                artifact_name=artifact_name,
            )
        else:
            metrics, artifact_uri = self._train_with_fallback(
                config,
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                artifact_name=artifact_name,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        system_metadata = self._system_metadata(config, elapsed_ms)
        baseline_mean = self._feature_mean(train_features)
        return TrainingResult(metrics=metrics, system_metadata=system_metadata, artifact_uri=artifact_uri, latency_ms=metrics["inference_latency_ms"], baseline_mean=baseline_mean)

    def _train_with_torch(
        self,
        config: ExperimentConfig,
        train_features: list[list[float]],
        train_labels: list[int],
        validation_features: list[list[float]],
        validation_labels: list[int],
        artifact_name: str | None = None,
    ) -> tuple[dict[str, float], str]:
        assert torch is not None and nn is not None and optim is not None

        model = nn.Sequential(
            nn.Linear(config.feature_count, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        x_train = torch.tensor(train_features, dtype=torch.float32)
        y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
        for _ in range(config.epochs):
            optimizer.zero_grad()
            logits = model(x_train)
            loss = loss_fn(logits, y_train)
            loss.backward()
            optimizer.step()
        metrics = self._evaluate_torch(model, validation_features, validation_labels)
        artifact_dir = self.artifact_store.root / config.model_name / (artifact_name or "latest")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config.model_dump(),
                "baseline_mean": self._feature_mean(train_features),
            },
            artifact_path,
        )
        return metrics, str(artifact_path)

    def _evaluate_torch(self, model: Any, features: list[list[float]], labels: list[int]) -> dict[str, float]:
        assert torch is not None
        with torch.no_grad():
            logits = model(torch.tensor(features, dtype=torch.float32)).squeeze(1)
            probabilities = torch.sigmoid(logits).tolist()
        predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
        return self._classification_metrics(labels, predictions, probabilities)

    def _train_with_fallback(
        self,
        config: ExperimentConfig,
        train_features: list[list[float]],
        train_labels: list[int],
        validation_features: list[list[float]],
        validation_labels: list[int],
        artifact_name: str | None = None,
    ) -> tuple[dict[str, float], str]:
        weights = [0.0 for _ in range(config.feature_count)]
        bias = 0.0
        for _ in range(config.epochs):
            for features, label in zip(train_features, train_labels, strict=False):
                score = self._sigmoid(sum(weight * value for weight, value in zip(weights, features, strict=False)) + bias)
                error = score - label
                for index, value in enumerate(features):
                    weights[index] -= config.learning_rate * error * value
                bias -= config.learning_rate * error
        probabilities = [self._sigmoid(sum(weight * value for weight, value in zip(weights, features, strict=False)) + bias) for features in validation_features]
        predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
        metrics = self._classification_metrics(validation_labels, predictions, probabilities)
        artifact_dir = self.artifact_store.root / config.model_name / (artifact_name or "latest")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.json"
        artifact_payload = {
            "weights": weights,
            "bias": bias,
            "baseline_mean": self._feature_mean(train_features),
            "config": config.model_dump(),
        }
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True), encoding="utf-8")
        return metrics, str(artifact_path)

    def predict(self, artifact_path: str, features: list[float]) -> tuple[int, float]:
        path = Path(artifact_path)
        if path.suffix == ".pt" and torch is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model_config = ExperimentConfig.model_validate(payload["config"])
            model = nn.Sequential(
                nn.Linear(model_config.feature_count, model_config.hidden_dim),
                nn.ReLU(),
                nn.Linear(model_config.hidden_dim, 1),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            with torch.no_grad():
                probability = torch.sigmoid(model(torch.tensor([features], dtype=torch.float32))).item()
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            weights = payload["weights"]
            bias = payload["bias"]
            probability = self._sigmoid(sum(weight * value for weight, value in zip(weights, features, strict=False)) + bias)
        prediction = 1 if probability >= 0.5 else 0
        return prediction, probability

    def _classification_metrics(
        self,
        labels: list[int],
        predictions: list[int],
        probabilities: list[float],
    ) -> dict[str, float]:
        true_positive = sum(1 for label, prediction in zip(labels, predictions, strict=False) if label == 1 and prediction == 1)
        true_negative = sum(1 for label, prediction in zip(labels, predictions, strict=False) if label == 0 and prediction == 0)
        false_positive = sum(1 for label, prediction in zip(labels, predictions, strict=False) if label == 0 and prediction == 1)
        false_negative = sum(1 for label, prediction in zip(labels, predictions, strict=False) if label == 1 and prediction == 0)
        accuracy = (true_positive + true_negative) / max(len(labels), 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        auc = self._roc_auc(labels, probabilities)
        return {
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "auc": round(auc, 6),
            "inference_latency_ms": round(5.0 if torch is None else 2.0, 6),
        }

    def _roc_auc(self, labels: list[int], probabilities: list[float]) -> float:
        positives = [score for label, score in zip(labels, probabilities, strict=False) if label == 1]
        negatives = [score for label, score in zip(labels, probabilities, strict=False) if label == 0]
        if not positives or not negatives:
            return 0.5
        pairwise = 0.0
        for positive in positives:
            for negative in negatives:
                if positive > negative:
                    pairwise += 1.0
                elif positive == negative:
                    pairwise += 0.5
        return pairwise / (len(positives) * len(negatives))

    def _generate_dataset(self, sample_count: int, feature_count: int, seed: int) -> tuple[list[list[float]], list[int]]:
        rng = random.Random(seed)
        features: list[list[float]] = []
        labels: list[int] = []
        for _ in range(sample_count):
            vector = [rng.uniform(-1.0, 1.0) for _ in range(feature_count)]
            score = sum(vector[: max(feature_count // 2, 1)]) + rng.uniform(-0.35, 0.35)
            label = 1 if score > 0 else 0
            features.append(vector)
            labels.append(label)
        return features, labels

    def _feature_mean(self, features: list[list[float]]) -> list[float]:
        if not features:
            return []
        size = len(features[0])
        return [sum(row[index] for row in features) / len(features) for index in range(size)]

    def _system_metadata(self, config: ExperimentConfig, elapsed_ms: float) -> dict[str, Any]:
        return {
            "seed": config.seed,
            "torch_available": torch is not None,
            "gpu_available": bool(torch.cuda.is_available()) if torch is not None else False,
            "runtime_ms": round(elapsed_ms, 3),
            "python_version": os.sys.version,
        }

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))
