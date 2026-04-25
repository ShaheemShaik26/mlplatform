from __future__ import annotations

from fastapi import FastAPI

from mlplatform.db import init_db
from mlplatform.api.routes import router
from mlplatform.queue import AsyncJobQueue
from mlplatform.serving.service import ServingService
from mlplatform.training.service import TrainingService


def create_serving_app() -> FastAPI:
    init_db()
    app = FastAPI(title="MLPlatform Serving", version="0.1.0")
    app.state.serving_service = ServingService()
    app.state.training_service = TrainingService()
    app.state.job_queue = AsyncJobQueue()
    app.state.job_queue.start(lambda job_id, payload: app.state.training_service.run_job(job_id, payload))
    app.include_router(router)
    return app
