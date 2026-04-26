from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    experiment_name: str
    model_name: str
    dataset_version: str = "synthetic-v1"
    seed: int = 42
    epochs: int = 20
    learning_rate: float = 0.01
    batch_size: int = 32
    train_samples: int = 256
    validation_samples: int = 128
    feature_count: int = 8
    hidden_dim: int = 16
    target_latency_ms: float = 100.0
    promotion_threshold: float = 0.01
    use_torch: bool = True


class TrainingJobSubmitRequest(BaseModel):
    config: ExperimentConfig
    priority: int = 0


class TrainingJobResponse(BaseModel):
    id: str
    status: str
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result_run_id: str | None = None
    error_message: str | None = None


class ExperimentRunResponse(BaseModel):
    id: str
    job_id: str | None
    experiment_name: str
    model_name: str
    status: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    dataset_version: str
    seed: int
    config: dict[str, Any]
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    system_metadata: dict[str, Any]
    artifact_uri: str | None = None
    error_message: str | None = None


class ModelVersionResponse(BaseModel):
    model_name: str
    version: int
    stage: str
    artifact_uri: str
    source_run_id: str
    created_at: datetime
    updated_at: datetime
    promoted_at: datetime | None
    metrics: dict[str, Any]
    latency_ms: float


class PromotionDecisionResponse(BaseModel):
    allowed: bool
    reason: str
    promoted_version: int | None = None


class PredictionRequest(BaseModel):
    features: list[float]
    version: int | None = None


class PredictionResponse(BaseModel):
    model_name: str
    model_version: int
    prediction: int
    probability: float
    latency_ms: float
    drift_score: float


class CompareExperimentsResponse(BaseModel):
    experiment_ids: list[str]
    runs: list[ExperimentRunResponse]
