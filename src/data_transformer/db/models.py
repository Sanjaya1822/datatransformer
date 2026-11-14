"""
SQLAlchemy ORM models for the Candidate Reconciliation & Intelligence Platform.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Enumerations ───────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    recruiter = "recruiter"


class MatchClassification(str, enum.Enum):
    high_confidence = "high_confidence"
    uncertain = "uncertain"
    non_match = "non_match"
    pending = "pending"


class FeedbackLabel(str, enum.Enum):
    match = "match"
    non_match = "non_match"
    unsure = "unsure"


class RunStatus(str, enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"


# ─── User Model ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), nullable=False, default=UserRole.recruiter)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)
    last_login = Column(DateTime, nullable=True)

    audit_entries = relationship("AuditLog", back_populates="user", lazy="dynamic")
    feedback = relationship("MatchFeedback", back_populates="reviewer", lazy="dynamic")


# ─── Pipeline Run Model ─────────────────────────────────────────────────────

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(String(64), unique=True, nullable=False, index=True)
    config_json = Column(JSON, nullable=True)
    summary_json = Column(JSON, nullable=True)
    quality_report_json = Column(JSON, nullable=True)
    status = Column(SAEnum(RunStatus), default=RunStatus.running, nullable=False)
    files_processed = Column(Integer, default=0)
    candidates_detected = Column(Integer, default=0)
    duplicates_merged = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    duration_seconds = Column(Float, nullable=True)
    started_at = Column(DateTime, default=_now, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    triggered_by = Column(String(36), ForeignKey("users.id"), nullable=True)

    __table_args__ = (Index("ix_pipeline_runs_started_at", "started_at"),)


# ─── Candidate Profile & Versions ───────────────────────────────────────────

class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id = Column(String(36), primary_key=True, default=_uuid)
    candidate_id = Column(String(64), unique=True, nullable=False, index=True)
    canonical_json = Column(JSON, nullable=False)
    projected_json = Column(JSON, nullable=True)
    overall_confidence = Column(Float, default=0.0)
    sources_merged = Column(JSON, nullable=True)
    current_version = Column(Integer, default=1)
    is_deleted = Column(Boolean, default=False)
    pipeline_run_id = Column(String(36), ForeignKey("pipeline_runs.id"), nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    updated_at = Column(DateTime, default=_now, onupdate=_now, nullable=False)

    versions = relationship("ProfileVersion", back_populates="profile", lazy="dynamic", cascade="all, delete-orphan", order_by="ProfileVersion.version")
    match_pairs_a = relationship("MatchPair", foreign_keys="MatchPair.profile_a_id", back_populates="profile_a", lazy="dynamic")
    match_pairs_b = relationship("MatchPair", foreign_keys="MatchPair.profile_b_id", back_populates="profile_b", lazy="dynamic")

    __table_args__ = (
        Index("ix_candidate_profiles_confidence", "overall_confidence"),
        Index("ix_candidate_profiles_created_at", "created_at"),
    )


class ProfileVersion(Base):
    __tablename__ = "profile_versions"

    id = Column(String(36), primary_key=True, default=_uuid)
    profile_id = Column(String(36), ForeignKey("candidate_profiles.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    snapshot_json = Column(JSON, nullable=False)
    field_diffs_json = Column(JSON, nullable=True)
    changed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    change_reason = Column(Text, nullable=True)
    changed_at = Column(DateTime, default=_now, nullable=False)

    profile = relationship("CandidateProfile", back_populates="versions")

    __table_args__ = (
        Index("ix_profile_versions_profile_version", "profile_id", "version", unique=True),
    )


# ─── Uploaded Files Management ──────────────────────────────────────────────

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(String(36), primary_key=True, default=_uuid)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(32), default="unknown")
    file_size = Column(Integer, default=0)
    status = Column(String(32), default="processed")  # processed, pending, failed
    pipeline_run_id = Column(String(36), ForeignKey("pipeline_runs.id"), nullable=True)
    is_deleted = Column(Boolean, default=False)
    uploaded_at = Column(DateTime, default=_now, nullable=False)

    __table_args__ = (
        Index("ix_uploaded_files_is_deleted", "is_deleted"),
        Index("ix_uploaded_files_uploaded_at", "uploaded_at"),
    )


# ─── Raw Source Records ─────────────────────────────────────────────────────

class RawRecordDB(Base):
    __tablename__ = "raw_records"

    id = Column(String(36), primary_key=True, default=_uuid)
    profile_id = Column(String(36), ForeignKey("candidate_profiles.id"), nullable=True, index=True)
    pipeline_run_id = Column(String(36), ForeignKey("pipeline_runs.id"), nullable=True)
    file_id = Column(String(36), ForeignKey("uploaded_files.id"), nullable=True, index=True)
    source_type = Column(String(32), nullable=False)
    source_path = Column(String(512), nullable=True)
    raw_fields_json = Column(JSON, nullable=False)
    extraction_stats_json = Column(JSON, nullable=True)
    ingested_at = Column(DateTime, default=_now, nullable=False)


# ─── Matching Pairs & Active Learning Feedback ─────────────────────────────

class MatchPair(Base):
    __tablename__ = "match_pairs"

    id = Column(String(36), primary_key=True, default=_uuid)
    profile_a_id = Column(String(36), ForeignKey("candidate_profiles.id"), nullable=False)
    profile_b_id = Column(String(36), ForeignKey("candidate_profiles.id"), nullable=False)
    score = Column(Float, nullable=False)
    signals_json = Column(JSON, nullable=True)
    classification = Column(SAEnum(MatchClassification), default=MatchClassification.pending)
    was_merged = Column(Boolean, default=False)
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    profile_a = relationship("CandidateProfile", foreign_keys=[profile_a_id], back_populates="match_pairs_a")
    profile_b = relationship("CandidateProfile", foreign_keys=[profile_b_id], back_populates="match_pairs_b")
    feedback_entries = relationship("MatchFeedback", back_populates="match_pair", lazy="dynamic", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_match_pairs_score", "score"),
        Index("ix_match_pairs_classification", "classification"),
    )


class MatchFeedback(Base):
    __tablename__ = "match_feedback"

    id = Column(String(36), primary_key=True, default=_uuid)
    pair_id = Column(String(36), ForeignKey("match_pairs.id"), nullable=False, index=True)
    label = Column(SAEnum(FeedbackLabel), nullable=False)
    reviewer_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_now, nullable=False)

    match_pair = relationship("MatchPair", back_populates="feedback_entries")
    reviewer = relationship("User", back_populates="feedback")


class MatchingConfig(Base):
    __tablename__ = "matching_configs"

    id = Column(String(36), primary_key=True, default=_uuid)
    version = Column(Integer, nullable=False, unique=True)
    thresholds_json = Column(JSON, nullable=False)
    signal_weights_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=False)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)
    notes = Column(Text, nullable=True)


# ─── Conflict Resolution Decisions ──────────────────────────────────────────

class ConflictDecision(Base):
    __tablename__ = "conflict_decisions"

    id = Column(String(36), primary_key=True, default=_uuid)
    profile_id = Column(String(36), ForeignKey("candidate_profiles.id"), nullable=False, index=True)
    field_name = Column(String(128), nullable=False)
    selected_value_json = Column(JSON, nullable=True)
    rejected_values_json = Column(JSON, nullable=True)
    resolution_method = Column(String(64), nullable=False)
    explanation = Column(Text, nullable=True)
    is_manual_override = Column(Boolean, default=False)
    reviewer_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    decided_at = Column(DateTime, default=_now, nullable=False)



# ─── Job Descriptions & Matches ─────────────────────────────────────────────

class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(String(36), primary_key=True, default=_uuid)
    title = Column(String(256), nullable=False)
    content = Column(Text, nullable=False)
    extracted_requirements_json = Column(JSON, nullable=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    match_results = relationship("JDMatchResult", back_populates="job_description", lazy="dynamic", cascade="all, delete-orphan")


class JDMatchResult(Base):
    __tablename__ = "jd_match_results"

    id = Column(String(36), primary_key=True, default=_uuid)
    jd_id = Column(String(36), ForeignKey("job_descriptions.id"), nullable=False, index=True)
    profile_id = Column(String(36), ForeignKey("candidate_profiles.id"), nullable=False, index=True)
    overall_score = Column(Float, nullable=False)
    explanation_json = Column(JSON, nullable=True)
    matched_requirements_json = Column(JSON, nullable=True)
    missing_requirements_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    job_description = relationship("JobDescription", back_populates="match_results")


# ─── Tamper-Evident Audit Log ───────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=True)
    resource_id = Column(String(128), nullable=True)
    old_value_json = Column(JSON, nullable=True)
    new_value_json = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=_now, nullable=False)

    user = relationship("User", back_populates="audit_entries")

    __table_args__ = (
        Index("ix_audit_log_timestamp", "timestamp"),
        Index("ix_audit_log_user_action", "user_id", "action"),
    )
