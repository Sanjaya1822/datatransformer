"""
Tests for the DB Repositories (in-memory SQLite).

Covers:
- CandidateRepository: create, get_by_id/candidate_id, list, soft_delete, versioning
- PipelineRunRepository: create, complete, fail
- AuditRepository: log and list
- MatchPairRepository: create, review queue, feedback
- JobDescriptionRepository: create, list, get, save_match_result

All tests use an isolated in-memory SQLite session — no shared state.
"""
from __future__ import annotations

import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from data_transformer.db.database import Base
from data_transformer.db.models import (
    CandidateProfile,
    PipelineRun,
    MatchPair,
    AuditLog,
    JobDescription,
    RunStatus,
    MatchClassification,
)
from data_transformer.db.repositories import (
    CandidateRepository,
    PipelineRunRepository,
    MatchPairRepository,
    AuditRepository,
    JobDescriptionRepository,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session() -> Session:
    """In-memory SQLite session, tables recreated per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _uid() -> str:
    return str(uuid.uuid4())


def _canonical(name: str = "Alice Smith") -> dict:
    return {
        "full_name": {"value": name},
        "emails": [{"value": "alice@example.com"}],
        "skills": [{"name": "Python"}],
    }


# ---------------------------------------------------------------------------
# CandidateRepository
# ---------------------------------------------------------------------------

class TestCandidateRepository:
    def test_create_and_get_by_id(self, db_session):
        repo = CandidateRepository(db_session)
        cid = _uid()
        profile = repo.create(candidate_id=cid, canonical=_canonical(), confidence=0.9)
        assert profile.id is not None
        fetched = repo.get_by_id(profile.id)
        assert fetched is not None
        assert fetched.candidate_id == cid

    def test_get_by_nonexistent_id_returns_none(self, db_session):
        repo = CandidateRepository(db_session)
        result = repo.get_by_id(99999)
        assert result is None

    def test_get_by_candidate_id(self, db_session):
        repo = CandidateRepository(db_session)
        cid = _uid()
        repo.create(candidate_id=cid, canonical=_canonical())
        found = repo.get_by_candidate_id(cid)
        assert found is not None
        assert found.candidate_id == cid

    def test_list_returns_created_profiles(self, db_session):
        repo = CandidateRepository(db_session)
        for i in range(3):
            repo.create(candidate_id=_uid(), canonical=_canonical(f"Candidate {i}"))
        items, total = repo.list()
        assert total >= 3
        assert len(items) >= 3

    def test_soft_delete_hides_profile(self, db_session):
        repo = CandidateRepository(db_session)
        profile = repo.create(candidate_id=_uid(), canonical=_canonical())
        repo.soft_delete(profile)
        # get_by_id filters is_deleted=False
        fetched = repo.get_by_id(profile.id)
        assert fetched is None

    def test_update_increments_version(self, db_session):
        repo = CandidateRepository(db_session)
        profile = repo.create(candidate_id=_uid(), canonical=_canonical())
        assert profile.current_version == 1
        updated = repo.update(profile, canonical=_canonical("Alice Updated"), reason="Test update")
        assert updated.current_version == 2

    def test_get_versions_returns_history(self, db_session):
        repo = CandidateRepository(db_session)
        profile = repo.create(candidate_id=_uid(), canonical=_canonical())
        repo.update(profile, canonical=_canonical("Alice V2"), reason="V2")
        versions = repo.get_versions(profile.id)
        assert len(versions) == 2

    def test_count_active_excludes_deleted(self, db_session):
        repo = CandidateRepository(db_session)
        p1 = repo.create(candidate_id=_uid(), canonical=_canonical())
        p2 = repo.create(candidate_id=_uid(), canonical=_canonical())
        repo.soft_delete(p1)
        count = repo.count_active()
        assert count == 1


# ---------------------------------------------------------------------------
# PipelineRunRepository
# ---------------------------------------------------------------------------

class TestPipelineRunRepository:
    def test_create_run(self, db_session):
        repo = PipelineRunRepository(db_session)
        run = repo.create(run_id=_uid())
        assert run.id is not None
        assert run.status == RunStatus.running

    def test_get_by_run_id(self, db_session):
        repo = PipelineRunRepository(db_session)
        rid = _uid()
        repo.create(run_id=rid)
        found = repo.get_by_run_id(rid)
        assert found is not None
        assert found.run_id == rid

    def test_complete_run(self, db_session):
        repo = PipelineRunRepository(db_session)
        run = repo.create(run_id=_uid())
        completed = repo.complete(
            run,
            summary={"ok": True},
            report={},
            files=5,
            candidates=10,
            duplicates=2,
            failed=0,
            duration=1.5,
        )
        assert completed.status == RunStatus.completed
        assert completed.files_processed == 5
        assert completed.candidates_detected == 10

    def test_fail_run(self, db_session):
        repo = PipelineRunRepository(db_session)
        run = repo.create(run_id=_uid())
        failed = repo.fail(run, error_message="Something went wrong")
        assert failed.status == RunStatus.failed

    def test_list_runs(self, db_session):
        repo = PipelineRunRepository(db_session)
        repo.create(run_id=_uid())
        repo.create(run_id=_uid())
        runs = repo.list()
        assert len(runs) >= 2


# ---------------------------------------------------------------------------
# MatchPairRepository
# ---------------------------------------------------------------------------

class TestMatchPairRepository:
    def _create_candidates(self, db_session):
        c_repo = CandidateRepository(db_session)
        a = c_repo.create(candidate_id=_uid(), canonical=_canonical("Alice"))
        b = c_repo.create(candidate_id=_uid(), canonical=_canonical("Bob"))
        return a, b

    def test_create_match_pair(self, db_session):
        repo = MatchPairRepository(db_session)
        a, b = self._create_candidates(db_session)
        pair = repo.create(
            profile_a_id=a.id,
            profile_b_id=b.id,
            score=0.87,
            signals=[{"signal": "fuzzy_name", "value": 0.90}],
            classification=MatchClassification.high_confidence,
        )
        assert pair.id is not None
        assert pair.score == pytest.approx(0.87)

    def test_review_queue_returns_uncertain(self, db_session):
        repo = MatchPairRepository(db_session)
        a, b = self._create_candidates(db_session)
        repo.create(
            profile_a_id=a.id,
            profile_b_id=b.id,
            score=0.65,
            signals=[],
            classification=MatchClassification.uncertain,
        )
        queue = repo.get_review_queue()
        assert len(queue) >= 1
        assert all(p.classification == MatchClassification.uncertain for p in queue)

    def test_count_uncertain(self, db_session):
        repo = MatchPairRepository(db_session)
        a, b = self._create_candidates(db_session)
        repo.create(
            profile_a_id=a.id, profile_b_id=b.id,
            score=0.60, signals=[],
            classification=MatchClassification.uncertain,
        )
        assert repo.count_uncertain() >= 1

    def test_add_feedback(self, db_session):
        from data_transformer.db.models import FeedbackLabel
        repo = MatchPairRepository(db_session)
        a, b = self._create_candidates(db_session)
        pair = repo.create(
            profile_a_id=a.id, profile_b_id=b.id,
            score=0.70, signals=[],
            classification=MatchClassification.uncertain,
        )
        fb = repo.add_feedback(
            pair_id=pair.id,
            label=FeedbackLabel.match,
            reviewer_id="reviewer-1",
            notes="Confirmed same person",
        )
        assert fb.id is not None
        all_fb = repo.get_all_feedback()
        assert len(all_fb) >= 1


# ---------------------------------------------------------------------------
# AuditRepository
# ---------------------------------------------------------------------------

class TestAuditRepository:
    def test_log_creates_entry(self, db_session):
        repo = AuditRepository(db_session)
        entry = repo.log(
            action="profile_update",
            resource_type="candidate",
            resource_id="cand-001",
        )
        assert entry.id is not None
        assert entry.action == "profile_update"

    def test_list_returns_entries(self, db_session):
        repo = AuditRepository(db_session)
        for i in range(3):
            repo.log(action=f"action_{i}", resource_type="test", resource_id=str(i))
        entries = repo.list(limit=10)
        assert len(entries) >= 3

    def test_list_filtered_by_resource_type(self, db_session):
        repo = AuditRepository(db_session)
        repo.log(action="match_review", resource_type="match_pair", resource_id="pair-1")
        repo.log(action="profile_create", resource_type="candidate", resource_id="cand-1")
        match_entries = repo.list(resource_type="match_pair")
        assert all(e.resource_type == "match_pair" for e in match_entries)


# ---------------------------------------------------------------------------
# JobDescriptionRepository
# ---------------------------------------------------------------------------

class TestJobDescriptionRepository:
    def test_create_and_get_jd(self, db_session):
        repo = JobDescriptionRepository(db_session)
        jd = repo.create(
            title="Senior Python Engineer",
            content="5+ years Python, PostgreSQL, Docker required.",
            requirements=[{"category": "required_skill", "text": "Python"}],
        )
        assert jd.id is not None
        fetched = repo.get(jd.id)
        assert fetched is not None
        assert fetched.title == "Senior Python Engineer"

    def test_list_jds(self, db_session):
        repo = JobDescriptionRepository(db_session)
        repo.create(title="JD1", content="Python", requirements=[])
        repo.create(title="JD2", content="Java", requirements=[])
        jds = repo.list()
        assert len(jds) >= 2

    def test_save_match_result(self, db_session):
        c_repo = CandidateRepository(db_session)
        profile = c_repo.create(candidate_id=_uid(), canonical=_canonical())
        jd_repo = JobDescriptionRepository(db_session)
        jd = jd_repo.create(title="Test JD", content="Python required", requirements=[])
        result = jd_repo.save_match_result(
            jd_id=jd.id,
            profile_id=profile.id,
            score=85.0,
            explanation={"summary": "Strong match"},
            matched=[{"requirement": "Python", "status": "verified"}],
            missing=[],
        )
        assert result.id is not None
        assert result.overall_score == pytest.approx(85.0)

    def test_get_results_for_jd(self, db_session):
        c_repo = CandidateRepository(db_session)
        profile = c_repo.create(candidate_id=_uid(), canonical=_canonical())
        jd_repo = JobDescriptionRepository(db_session)
        jd = jd_repo.create(title="Test JD", content="Python", requirements=[])
        jd_repo.save_match_result(
            jd_id=jd.id, profile_id=profile.id, score=75.0,
            explanation={}, matched=[], missing=[],
        )
        results = jd_repo.get_results_for_jd(jd.id)
        assert len(results) >= 1
