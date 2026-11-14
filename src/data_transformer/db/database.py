"""
Database connection and session management.
Supports PostgreSQL in production and SQLite for local development/testing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Load database URL from environment or default to SQLite
DEFAULT_DB_PATH = Path("./datatransformer.db").resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """Create all registered database tables and ensure schema consistency."""
    Base.metadata.create_all(bind=engine)
    # Ensure column migrations for SQLite / existing databases
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            # Check if file_id exists in raw_records
            result = conn.execute(text("PRAGMA table_info(raw_records)"))
            cols = [row[1] for row in result.fetchall()]
            if cols and "file_id" not in cols:
                conn.execute(text("ALTER TABLE raw_records ADD COLUMN file_id VARCHAR(36)"))
                conn.commit()
    except Exception:
        pass
