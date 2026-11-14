"""
Database seeder: populates initial users and configuration.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from .models import MatchingConfig, User, UserRole


def _now() -> datetime:
    return datetime.now(timezone.utc)


def seed_database(db: Session, hash_password_func) -> None:
    """Creates default recruiter user and active matching configuration if not present."""
    if not db.query(User).filter(User.username == "recruiter").first():
        recruiter = User(
            username=os.getenv("RECRUITER_USERNAME", "recruiter"),
            email=os.getenv("RECRUITER_EMAIL", "recruiter@datatransformer.io"),
            hashed_password=hash_password_func(os.getenv("RECRUITER_PASSWORD", "recruiter123")),
            role=UserRole.recruiter,
            is_active=True,
            created_at=_now(),
        )
        db.add(recruiter)

    # Initial matching configuration version 1
    if not db.query(MatchingConfig).filter(MatchingConfig.version == 1).first():
        initial_config = MatchingConfig(
            version=1,
            thresholds_json={
                "high_confidence_threshold": 0.85,
                "uncertain_threshold": 0.50,
            },
            signal_weights_json={
                "exact_email": 1.00,
                "exact_phone": 1.00,
                "exact_linkedin": 0.95,
                "exact_github": 0.95,
                "fuzzy_name_high": 0.70,
                "fuzzy_name_medium": 0.45,
                "shared_employer": 0.25,
                "shared_location": 0.10,
            },
            is_active=True,
            notes="Default baseline matching thresholds and signal weights",
            created_at=_now(),
        )
        db.add(initial_config)

    db.commit()
