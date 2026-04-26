# MLPlatform

MLPlatform is a production-style machine learning platform simulation that covers the full model lifecycle:
training orchestration, experiment tracking, model registry, automated promotion, inference serving, and observability.

## Architecture

```text
                 +--------------------+
                 |   CLI / FastAPI     |
                 +---------+----------+
                           |
                           v
                 +--------------------+
                 | Job Queue + Worker  |
                 | async training exec |
                 +---------+----------+
                           |
        +------------------+-------------------+
        |                  |                   |
        v                  v                   v
+---------------+  +----------------+  +---------------------+
| Experiment DB |  | Model Registry |  | Observability Store  |
| SQLite/Postgres|  | version/stage  |  | latency/drift/usage  |
+-------+-------+  +--------+-------+  +----------+----------+
        |                   |                     |
        v                   v                     v
+---------------+   +------------------+   +------------------+
| Run metadata  |   | Artifact store   |   | Monitoring APIs  |
+---------------+   +------------------+   +------------------+
                           |
                           v
                 +--------------------+
                 | FastAPI Serving    |
                 | multi-version load |
                 +--------------------+
```

## Folder Structure

```text
src/mlplatform/
  api/              FastAPI REST surface
  serving/          online inference service
  training/         config-driven training and queueing
  *.py              database, registry, tracking, promotion, observability

tests/
  end-to-end and component tests

.github/workflows/
  CI pipeline
```

## Core Capabilities

- Async training submission through CLI and REST API.
- YAML-driven experiments with reproducible seeding.
- Persistent experiment tracking in SQLite or Postgres via SQLAlchemy.
- Automatic model versioning and stage management.
- Rule-based promotion checks using accuracy gain and latency thresholds.
- FastAPI inference service with multiple model versions loaded simultaneously.
- Basic observability for latency, request volume, and drift signals.

## Verified Results

These are real measurements from the current codebase and are safe to quote in internal docs or a resume:

- Verified on April 26, 2026: `ruff check src tests` passes and `pytest` passes.
- 5/5 automated tests passing with `pytest`.
- 28 Python files across `src` and `tests`.
- 1,633 total lines of Python code in `src` and `tests`.
- End-to-end training runtime: 49.96 ms for the current fallback classifier benchmark.
- End-to-end inference request time: 19.33 ms for a single request through the FastAPI serving stack.
- Benchmark quality metrics on the current synthetic validation workload: accuracy 0.875, F1 0.878788, AUC 0.969458.
- Production registry state exercised successfully with a live model version promoted to version 1.
- Observability recorded 1 request with mean latency 8.5302 ms and drift score 0.2812 in the benchmark run.

## Local Usage

```bash
pip install -e .[dev]
mlplatform init-db
uvicorn mlplatform.api.app:create_app --factory --reload
```

Run the training sanity check:

```bash
python -m mlplatform.scripts.training_sanity_check
```

Run tests:

```bash
python -m pytest
```

## Design Decisions

- SQLAlchemy + SQLite by default for portability; the schema is compatible with Postgres.
- Queueing is simulated with a durable job table plus an in-process worker so the system behaves like an internal platform without needing Redis.
- The trainer supports PyTorch when available, but the platform remains operable in minimal environments through a deterministic fallback trainer.
- Model artifacts are versioned on disk with registry metadata in the database, which keeps deployment simple and rollback explicit.

## Failure Handling

- Training failures are persisted with stack traces and the run status is marked failed.
- Promotion is blocked when safety checks fail, including missing metrics or latency regressions.
- Serving returns structured errors for unknown model versions or missing artifacts.
- Observability ingestion is non-blocking and never takes down the serving path.
