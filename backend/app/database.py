"""
backend/app/database.py
=======================
SQLAlchemy engine + session factory for PostgreSQL 16.

Uses environment variables for credentials so no secrets are ever in code.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import get_settings


def _build_url() -> str:
    s = get_settings()
    return (
        f"postgresql+psycopg2://{s.db_user}:{s.db_password}"
        f"@{s.db_host}:{s.db_port}/{s.db_name}"
    )


engine = create_engine(_build_url(), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """All ORM models inherit from this."""
    pass


def get_db():
    """
    FastAPI dependency: yields a database session and guarantees it is
    closed after the request finishes, even if an exception is raised.

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
