from __future__ import annotations

from mlplatform.db import init_db
from mlplatform.registry import ModelRegistry
from mlplatform.schemas import ExperimentConfig
from mlplatform.training.service import TrainingService


def test_rollback_restores_previous_version() -> None:
    init_db()
    service = TrainingService()
    service.run_sync(
        ExperimentConfig(
            experiment_name="rollback-1",
            model_name="classifier-c",
            seed=21,
            epochs=6,
            train_samples=64,
            validation_samples=48,
            feature_count=5,
            hidden_dim=6,
            use_torch=False,
        )
    )
    service.run_sync(
        ExperimentConfig(
            experiment_name="rollback-2",
            model_name="classifier-c",
            seed=22,
            epochs=6,
            train_samples=64,
            validation_samples=48,
            feature_count=5,
            hidden_dim=6,
            use_torch=False,
        )
    )
    registry = ModelRegistry()
    versions = registry.get_model_versions("classifier-c")
    assert len(versions) >= 2
    production = registry.get_production_version("classifier-c")
    assert production is not None
    rolled_back = registry.rollback("classifier-c", versions[0].version)
    assert rolled_back is not None
    assert registry.get_production_version("classifier-c").version == versions[0].version
