from __future__ import annotations

from fastapi.testclient import TestClient

from mlplatform.db import init_db
from mlplatform.schemas import ExperimentConfig
from mlplatform.training.service import TrainingService
from mlplatform.serving.app import create_serving_app


def test_dedicated_serving_app_predicts() -> None:
    init_db()
    service = TrainingService()
    service.run_sync(
        ExperimentConfig(
            experiment_name="serving",
            model_name="serving-model",
            seed=33,
            epochs=6,
            train_samples=64,
            validation_samples=32,
            feature_count=4,
            hidden_dim=6,
            use_torch=False,
        )
    )

    app = create_serving_app()
    with TestClient(app) as client:
        response = client.post("/infer/serving-model", json={"features": [0.1, 0.2, 0.3, 0.4]})
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_name"] == "serving-model"
        assert payload["model_version"] >= 1
