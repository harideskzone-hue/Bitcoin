"""
backend/app/database.py
=======================
SQLAlchemy engine + session factory.

Primary:  PostgreSQL 16 (docker-compose.yml)
Fallback: SQLite  (USE_SQLITE_FALLBACK=true in .env)

The active backend is printed at import time so it is always visible in
startup logs and in the P0.8.1 dry-run report. A silent database switch
during the demo would be undetectable — this prevents that.

Usage
-----
Normal:    DATABASE_BACKEND=postgresql  (default)
Venue fallback: USE_SQLITE_FALLBACK=true → DATABASE_BACKEND=sqlite ⚠

All ORM models are backend-agnostic (no PostgreSQL-only type extensions used).
"""

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── Backend selection ─────────────────────────────────────────────────────────

_SQLITE_FALLBACK = os.getenv("USE_SQLITE_FALLBACK", "false").lower() in ("true", "1", "yes")
_SQLITE_PATH     = Path("data/sih26146.sqlite3")


def _build_url() -> str:
    if _SQLITE_FALLBACK:
        _SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{_SQLITE_PATH.resolve()}"

    # Import here to avoid the settings validation cost when using SQLite
    from backend.app.config import get_settings
    s = get_settings()
    return (
        f"postgresql+psycopg2://{s.db_user}:{s.db_password}"
        f"@{s.db_host}:{s.db_port}/{s.db_name}"
    )


_URL = _build_url()

# ── Startup visibility (required by approved plan — never silent) ──────────────

if _SQLITE_FALLBACK:
    _backend_label = f"sqlite  (fallback) → {_SQLITE_PATH}"
    _warning       = "⚠ SQLite fallback enabled — not suitable for production"
    print(f"DATABASE_BACKEND=sqlite\n{_warning}", file=sys.stderr)
else:
    _backend_label = "postgresql"
    print("DATABASE_BACKEND=postgresql", file=sys.stderr)


# ── Engine ────────────────────────────────────────────────────────────────────

_engine_kwargs: dict = {"pool_pre_ping": True}
if _SQLITE_FALLBACK:
    # SQLite requires check_same_thread=False for FastAPI's threaded request handling
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    del _engine_kwargs["pool_pre_ping"]   # not supported by SQLite dialect

engine       = create_engine(_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Enable WAL mode on SQLite for concurrent read performance
if _SQLITE_FALLBACK:
    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA foreign_keys=ON")


class Base(DeclarativeBase):
    """All ORM models inherit from this."""
    pass


def get_db():
    """
    FastAPI dependency: yields a database session and closes it after the
    request finishes, even if an exception is raised.

    Usage in a route:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def database_backend() -> str:
    """Return a human-readable string describing the active database backend.
    Used by GET /api/v1/health to expose the backend in the startup report."""
    return _backend_label
