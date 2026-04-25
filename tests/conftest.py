from __future__ import annotations

import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / ".testdata"
TEST_DB = TEST_DATA / "mlplatform-test.db"
TEST_ARTIFACTS = TEST_DATA / "artifacts"


def pytest_configure() -> None:
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    if TEST_DB.exists():
        TEST_DB.unlink()
    if TEST_ARTIFACTS.exists():
        shutil.rmtree(TEST_ARTIFACTS)
    os.environ["MLPLATFORM_DATABASE_URL"] = "sqlite:///./.testdata/mlplatform-test.db"
    os.environ["MLPLATFORM_ARTIFACT_ROOT"] = "./.testdata/artifacts"
    os.environ["MLPLATFORM_DEFAULT_SEED"] = "42"
