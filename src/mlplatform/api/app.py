from __future__ import annotations

from fastapi import FastAPI

from mlplatform.db import init_db
from mlplatform.api.routes import router
from mlplatform.queue import AsyncJobQueue
from mlplatform.training.service import TrainingService
from mlplatform.serving.service import ServingService


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="MLPlatform", version="0.1.0")
    training_service = TrainingService()
    serving_service = ServingService()
    job_queue = AsyncJobQueue()

    def handle_job(job_id: str, payload: dict[str, object]) -> dict[str, object]:
        return training_service.run_job(job_id, payload)

    job_queue.start(handle_job)
    app.state.training_service = training_service
    app.state.serving_service = serving_service
    app.state.job_queue = job_queue
    app.include_router(router)

    @app.on_event("shutdown")
    def shutdown_queue() -> None:
        job_queue.stop()

    return app
