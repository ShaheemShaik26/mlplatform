from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select

from mlplatform.db import session_scope
from mlplatform.models import ModelVersion, ExperimentRun


class ModelRegistry:
    def register_model_version(
        self,
        model_name: str,
        source_run_id: str,
        artifact_uri: str,
        metrics: dict[str, Any],
        latency_ms: float,
    ) -> ModelVersion:
        with session_scope() as session:
            latest_version = session.execute(
                select(ModelVersion.version)
                .where(ModelVersion.model_name == model_name)
                .order_by(ModelVersion.version.desc())
            ).scalar_one_or_none()
            next_version = 1 if latest_version is None else latest_version + 1
            version = ModelVersion(
                model_name=model_name,
                version=next_version,
                stage="staging",
                artifact_uri=artifact_uri,
                source_run_id=source_run_id,
                metrics_json=json.dumps(metrics, sort_keys=True),
                latency_ms=latency_ms,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(version)
            session.flush()
            return version

    def get_model_versions(self, model_name: str) -> list[ModelVersion]:
        with session_scope() as session:
            result = session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_name == model_name)
                .order_by(ModelVersion.version.asc())
            )
            return list(result.scalars())

    def get_version(self, model_name: str, version: int) -> ModelVersion | None:
        with session_scope() as session:
            return session.execute(
                select(ModelVersion).where(
                    ModelVersion.model_name == model_name,
                    ModelVersion.version == version,
                )
            ).scalar_one_or_none()

    def get_production_version(self, model_name: str) -> ModelVersion | None:
        with session_scope() as session:
            return session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_name == model_name, ModelVersion.stage == "production")
                .order_by(ModelVersion.version.desc())
            ).scalar_one_or_none()

    def set_stage(self, model_name: str, version: int, stage: str) -> ModelVersion | None:
        with session_scope() as session:
            candidate = session.execute(
                select(ModelVersion).where(
                    ModelVersion.model_name == model_name,
                    ModelVersion.version == version,
                )
            ).scalar_one_or_none()
            if candidate is None:
                return None
            if stage == "production":
                productions = session.execute(
                    select(ModelVersion).where(
                        ModelVersion.model_name == model_name,
                        ModelVersion.stage == "production",
                    )
                ).scalars().all()
                for current in productions:
                    current.stage = "archived"
                    current.updated_at = datetime.now(timezone.utc)
            candidate.stage = stage
            candidate.updated_at = datetime.now(timezone.utc)
            if stage == "production":
                candidate.promoted_at = datetime.now(timezone.utc)
            return candidate

    def rollback(self, model_name: str, target_version: int) -> ModelVersion | None:
        return self.set_stage(model_name, target_version, "production")

    def latest_completed_run(self, model_name: str) -> ExperimentRun | None:
        with session_scope() as session:
            return session.execute(
                select(ExperimentRun)
                .where(ExperimentRun.model_name == model_name, ExperimentRun.status == "completed")
                .order_by(desc(ExperimentRun.finished_at))
            ).scalar_one_or_none()
