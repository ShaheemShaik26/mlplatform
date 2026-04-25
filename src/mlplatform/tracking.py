from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from mlplatform.db import session_scope
from mlplatform.models import ExperimentRun, TrainingJob
from mlplatform.schemas import ExperimentConfig


class ExperimentTracker:
    def create_run(self, job_id: str | None, config: ExperimentConfig) -> str:
        with session_scope() as session:
            run = ExperimentRun(
                job_id=job_id,
                experiment_name=config.experiment_name,
                model_name=config.model_name,
                status="running",
                started_at=datetime.now(timezone.utc),
                config_json=config.model_dump_json(),
                hyperparameters_json=json.dumps(
                    {
                        "epochs": config.epochs,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "hidden_dim": config.hidden_dim,
                        "target_latency_ms": config.target_latency_ms,
                        "promotion_threshold": config.promotion_threshold,
                    },
                    sort_keys=True,
                ),
                dataset_version=config.dataset_version,
                seed=config.seed,
                metrics_json="{}",
                system_metadata_json="{}",
            )
            session.add(run)
            session.flush()
            return run.id

    def complete_run(self, run_id: str, metrics: dict[str, Any], system_metadata: dict[str, Any], artifact_uri: str) -> None:
        with session_scope() as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                return
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            run.metrics_json = json.dumps(metrics, sort_keys=True)
            run.system_metadata_json = json.dumps(system_metadata, sort_keys=True)
            run.artifact_uri = artifact_uri

    def fail_run(self, run_id: str, error_message: str) -> None:
        with session_scope() as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                return
            run.status = "failed"
            run.finished_at = datetime.now(timezone.utc)
            run.error_message = error_message

    def list_runs(self) -> list[ExperimentRun]:
        with session_scope() as session:
            return list(session.execute(select(ExperimentRun).order_by(ExperimentRun.created_at.desc())).scalars())

    def get_run(self, run_id: str) -> ExperimentRun | None:
        with session_scope() as session:
            return session.get(ExperimentRun, run_id)

    def update_job_status(self, job_id: str, status: str, result_run_id: str | None = None, error_message: str | None = None) -> None:
        with session_scope() as session:
            job = session.get(TrainingJob, job_id)
            if job is None:
                return
            job.status = status
            if status == "running":
                job.started_at = datetime.now(timezone.utc)
            if status in {"completed", "failed"}:
                job.finished_at = datetime.now(timezone.utc)
            job.result_run_id = result_run_id or job.result_run_id
            job.error_message = error_message

    def create_job(self, config_json: str, priority: int) -> str:
        with session_scope() as session:
            job = TrainingJob(config_json=config_json, priority=priority, status="queued")
            session.add(job)
            session.flush()
            return job.id

    def get_job(self, job_id: str) -> TrainingJob | None:
        with session_scope() as session:
            return session.get(TrainingJob, job_id)
