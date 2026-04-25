from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("MLPLATFORM_DATABASE_URL", "sqlite:///./mlplatform.db")
    artifact_root: Path = Path(os.getenv("MLPLATFORM_ARTIFACT_ROOT", "./artifacts")).resolve()
    seed: int = int(os.getenv("MLPLATFORM_DEFAULT_SEED", "42"))
    promotion_min_improvement: float = float(os.getenv("MLPLATFORM_PROMOTION_MIN_IMPROVEMENT", "0.01"))
    promotion_max_latency_ms: float = float(os.getenv("MLPLATFORM_PROMOTION_MAX_LATENCY_MS", "200.0"))
    min_validation_f1: float = float(os.getenv("MLPLATFORM_MIN_VALIDATION_F1", "0.70"))
    model_cache_ttl_seconds: int = int(os.getenv("MLPLATFORM_MODEL_CACHE_TTL_SECONDS", "300"))

    def ensure_directories(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
