from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from mlplatform.promotion import PromotionPolicy
from mlplatform.schemas import (
    CompareExperimentsResponse,
    ExperimentRunResponse,
    ModelVersionResponse,
    PredictionRequest,
    PredictionResponse,
    PromotionDecisionResponse,
    TrainingJobResponse,
    TrainingJobSubmitRequest,
)
from mlplatform.tracking import ExperimentTracker
from mlplatform.registry import ModelRegistry

router = APIRouter()


def _tracker() -> ExperimentTracker:
    return ExperimentTracker()


def _registry() -> ModelRegistry:
    return ModelRegistry()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/training/jobs", response_model=TrainingJobResponse)
def submit_training_job(request: Request, payload: TrainingJobSubmitRequest) -> TrainingJobResponse:
    job_queue = request.app.state.job_queue
    training_service = request.app.state.training_service
    job_id = training_service.submit(payload.config, priority=payload.priority)
    job_queue.submit(job_id, {"config": payload.config.model_dump()}, priority=payload.priority)
    job = _tracker().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="failed to create training job")
    return TrainingJobResponse(
        id=job.id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result_run_id=job.result_run_id,
        error_message=job.error_message,
    )


@router.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(job_id: str) -> TrainingJobResponse:
    job = _tracker().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return TrainingJobResponse(
        id=job.id,
        status=job.status,
        submitted_at=job.submitted_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result_run_id=job.result_run_id,
        error_message=job.error_message,
    )


@router.get("/experiments", response_model=list[ExperimentRunResponse])
def list_experiments() -> list[ExperimentRunResponse]:
    runs = _tracker().list_runs()
    return [
        ExperimentRunResponse(
            id=run.id,
            job_id=run.job_id,
            experiment_name=run.experiment_name,
            model_name=run.model_name,
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            dataset_version=run.dataset_version,
            seed=run.seed,
            config=json.loads(run.config_json),
            hyperparameters=json.loads(run.hyperparameters_json),
            metrics=json.loads(run.metrics_json),
            system_metadata=json.loads(run.system_metadata_json),
            artifact_uri=run.artifact_uri,
            error_message=run.error_message,
        )
        for run in runs
    ]


@router.get("/experiments/compare", response_model=CompareExperimentsResponse)
def compare_experiments(experiment_ids: str) -> CompareExperimentsResponse:
    ids = [value.strip() for value in experiment_ids.split(",") if value.strip()]
    tracker = _tracker()
    runs = []
    for experiment_id in ids:
        run = tracker.get_run(experiment_id)
        if run is None:
            continue
        runs.append(
            ExperimentRunResponse(
                id=run.id,
                job_id=run.job_id,
                experiment_name=run.experiment_name,
                model_name=run.model_name,
                status=run.status,
                created_at=run.created_at,
                started_at=run.started_at,
                finished_at=run.finished_at,
                dataset_version=run.dataset_version,
                seed=run.seed,
                config=json.loads(run.config_json),
                hyperparameters=json.loads(run.hyperparameters_json),
                metrics=json.loads(run.metrics_json),
                system_metadata=json.loads(run.system_metadata_json),
                artifact_uri=run.artifact_uri,
                error_message=run.error_message,
            )
        )
    return CompareExperimentsResponse(experiment_ids=ids, runs=runs)


@router.get("/models/{model_name}", response_model=list[ModelVersionResponse])
def get_model_versions(model_name: str) -> list[ModelVersionResponse]:
    versions = _registry().get_model_versions(model_name)
    return [
        ModelVersionResponse(
            model_name=version.model_name,
            version=version.version,
            stage=version.stage,
            artifact_uri=version.artifact_uri,
            source_run_id=version.source_run_id,
            created_at=version.created_at,
            updated_at=version.updated_at,
            promoted_at=version.promoted_at,
            metrics=json.loads(version.metrics_json),
            latency_ms=version.latency_ms,
        )
        for version in versions
    ]


@router.post("/models/{model_name}/promote", response_model=PromotionDecisionResponse)
def promote_model(model_name: str, version: int) -> PromotionDecisionResponse:
    registry = _registry()
    candidate = registry.get_version(model_name, version)
    if candidate is None:
        raise HTTPException(status_code=404, detail="model version not found")
    decision = PromotionPolicy(registry).evaluate(
        model_name=model_name,
        candidate_metrics=json.loads(candidate.metrics_json),
        candidate_latency_ms=candidate.latency_ms,
    )
    if decision.allowed:
        registry.set_stage(model_name, version, "production")
    return PromotionDecisionResponse(**decision.__dict__)


@router.post("/models/{model_name}/rollback", response_model=PromotionDecisionResponse)
def rollback_model(model_name: str, version: int) -> PromotionDecisionResponse:
    registry = _registry()
    target = registry.rollback(model_name, version)
    if target is None:
        raise HTTPException(status_code=404, detail="model version not found")
    return PromotionDecisionResponse(allowed=True, reason="rollback completed", promoted_version=target.version)


@router.post("/infer/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, request: PredictionRequest, http_request: Request) -> PredictionResponse:
    service = http_request.app.state.serving_service
    try:
        result = service.predict(model_name, request.features, request.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.get("/observability/{model_name}")
def observability(model_name: str, request: Request) -> dict[str, Any]:
    service = request.app.state.serving_service.observability
    return service.get_metrics(model_name)
