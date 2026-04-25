from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from mlplatform.config import SETTINGS


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def _build_engine():
    database_url = SETTINGS.database_url
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, future=True, connect_args=connect_args)


def get_engine():
    global _engine
    if _engine is None:
        SETTINGS.ensure_directories()
        _engine = _build_engine()
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            future=True,
            expire_on_commit=False,
        )
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from mlplatform import models  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def reset_sqlite_if_needed() -> None:
    database_url = SETTINGS.database_url
    if database_url.startswith("sqlite:///./"):
        path = Path(database_url.removeprefix("sqlite:///./"))
        if path.exists():
            path.unlink()
