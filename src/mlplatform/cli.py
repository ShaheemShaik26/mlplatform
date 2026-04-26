from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

from mlplatform.db import init_db
from mlplatform.schemas import ExperimentConfig
from mlplatform.training.service import TrainingService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlplatform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db")

    submit = subparsers.add_parser("submit-training")
    submit.add_argument("--config", required=True, help="Path to YAML experiment config")
    submit.add_argument("--priority", type=int, default=0)

    run = subparsers.add_parser("run-training")
    run.add_argument("--config", required=True, help="Path to YAML experiment config")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "init-db":
        init_db()
        print("database initialized")
        return 0

    if args.command == "submit-training":
        init_db()
        config = _load_config(Path(args.config))
        service = TrainingService()
        job_id = service.submit(config, priority=args.priority)
        print(json.dumps({"job_id": job_id, "status": "queued"}))
        return 0

    if args.command == "run-training":
        init_db()
        config = _load_config(Path(args.config))
        service = TrainingService()
        result = service.run_sync(config)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("mlplatform.api.app:create_app", factory=True, host=args.host, port=args.port, reload=False)
        return 0

    return 1


def _load_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(payload)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
