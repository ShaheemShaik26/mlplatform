from __future__ import annotations

import time

from fastapi.testclient import TestClient

from mlplatform.api.app import create_app
from mlplatform.db import init_db
from mlplatform.promotion import PromotionPolicy
from mlplatform.registry import ModelRegistry
from mlplatform.schemas import ExperimentConfig
from mlplatform.serving.app import create_serving_app
from mlplatform.training.service import TrainingService


def test_training_service_registers_and_promotes_model() -> None:
    init_db()
    service = TrainingService()
    result = service.run_sync(
        ExperimentConfig(
            experiment_name="unit-test",
            model_name="classifier-a",
            seed=7,
            epochs=8,
            train_samples=96,
            validation_samples=64,
            feature_count=6,
            hidden_dim=8,
            use_torch=False,
        )
    )
    registry = ModelRegistry()
    production = registry.get_production_version("classifier-a")
    assert result["metrics"]["accuracy"] >= 0.6
    assert production is not None
    assert production.stage == "production"


def test_api_queue_and_inference_flow() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/training/jobs",
            json={
                "priority": 1,
                "config": {
                    "experiment_name": "api-flow",
                    "model_name": "classifier-b",
                    "seed": 11,
                    "epochs": 8,
                    "train_samples": 96,
                    "validation_samples": 64,
                    "feature_count": 6,
                    "hidden_dim": 8,
                    "use_torch": False,
                },
            },
        )
        assert response.status_code == 200
        job_id = response.json()["id"]

        for _ in range(50):
            job = client.get(f"/training/jobs/{job_id}")
            assert job.status_code == 200
            status = job.json()["status"]
            if status == "completed":
                break
            time.sleep(0.1)
        assert status == "completed"

        versions = client.get("/models/classifier-b")
        assert versions.status_code == 200
        payload = versions.json()
        assert payload

        inference = client.post("/infer/classifier-b", json={"features": [0.25] * 6})
        assert inference.status_code == 200
        body = inference.json()
        assert body["model_name"] == "classifier-b"
        assert 0.0 <= body["probability"] <= 1.0
        assert body["latency_ms"] >= 0.0

        observability = client.get("/observability/classifier-b")
        assert observability.status_code == 200
        assert observability.json()["request_count"] >= 1


def test_promotion_policy_blocks_latency_regression() -> None:
    policy = PromotionPolicy()
    decision = policy.evaluate(
        model_name="unknown-model",
        candidate_metrics={"f1": 0.9},
        candidate_latency_ms=10.0,
        min_improvement=0.01,
        max_latency_ms=5.0,
    )
    assert decision.allowed is False
    assert "latency" in decision.reason
