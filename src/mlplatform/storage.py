from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlplatform.config import SETTINGS


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or SETTINGS.artifact_root
        self.root.mkdir(parents=True, exist_ok=True)

    def model_dir(self, model_name: str, version: int) -> Path:
        directory = self.root / model_name / f"v{version}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_json(self, relative_path: Path, payload: dict[str, Any]) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_text(self, relative_path: Path, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def resolve(self, uri: str) -> Path:
        return Path(uri)
