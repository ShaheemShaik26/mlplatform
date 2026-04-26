from __future__ import annotations

from typing import Any

from mlplatform.promotion import PromotionPolicy
from mlplatform.registry import ModelRegistry
from mlplatform.schemas import ExperimentConfig
from mlplatform.tracking import ExperimentTracker
from mlplatform.training.pipeline import TorchTrainingPipeline


class TrainingService:
    def __init__(
        self,
        tracker: ExperimentTracker | None = None,
        registry: ModelRegistry | None = None,
        pipeline: TorchTrainingPipeline | None = None,
    ) -> None:
        self.tracker = tracker or ExperimentTracker()
        self.registry = registry or ModelRegistry()
        self.pipeline = pipeline or TorchTrainingPipeline()
        self.promotion_policy = PromotionPolicy(self.registry)

    def submit(self, config: ExperimentConfig, priority: int = 0) -> str:
        return self.tracker.create_job(config_json=config.model_dump_json(), priority=priority)

    def run_job(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        config = ExperimentConfig.model_validate(payload["config"])
        self.tracker.update_job_status(job_id, "running")
        run_id = self.tracker.create_run(job_id, config)
        try:
            result = self.pipeline.train(config, artifact_name=run_id)
            metrics = dict(result.metrics)
            self.tracker.complete_run(run_id, metrics, result.system_metadata, result.artifact_uri)
            registered = self.registry.register_model_version(
                model_name=config.model_name,
                source_run_id=run_id,
                artifact_uri=result.artifact_uri,
                metrics=metrics,
                latency_ms=result.latency_ms,
            )
            decision = self.promotion_policy.evaluate(
                model_name=config.model_name,
                candidate_metrics=metrics,
                candidate_latency_ms=result.latency_ms,
                min_improvement=config.promotion_threshold,
                max_latency_ms=config.target_latency_ms,
            )
            if decision.allowed:
                self.registry.set_stage(config.model_name, registered.version, "production")
            self.tracker.update_job_status(job_id, "completed", result_run_id=run_id)
            return {
                "run_id": run_id,
                "model_version": registered.version,
                "metrics": metrics,
                "promotion": decision.__dict__,
            }
        except Exception as exc:  # pragma: no cover - defensive persistence path
            self.tracker.fail_run(run_id, str(exc))
            self.tracker.update_job_status(job_id, "failed", result_run_id=run_id, error_message=str(exc))
            raise

    def run_sync(self, config: ExperimentConfig) -> dict[str, Any]:
        job_id = self.submit(config)
        return self.run_job(job_id, {"config": config.model_dump()})
