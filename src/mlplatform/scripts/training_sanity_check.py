from __future__ import annotations

from mlplatform.db import init_db
from mlplatform.schemas import ExperimentConfig
from mlplatform.training.service import TrainingService


def main() -> None:
    init_db()
    service = TrainingService()
    config = ExperimentConfig(
        experiment_name="sanity-check",
        model_name="sanity-model",
        train_samples=64,
        validation_samples=32,
        epochs=5,
        use_torch=False,
    )
    result = service.run_sync(config)
    print(result["metrics"])


if __name__ == "__main__":
    main()
