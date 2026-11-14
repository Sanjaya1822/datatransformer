"""
End-to-End Tests for Candidate Identity Resolution, De-duplication, and Accurate Data Merging.
Validates:
- Exactly one canonical profile generated for each real candidate from multi-source files.
- Accurate Source-to-Candidate mapping (no unrelated files in Merged Sources).
- Reprocessing files does not create duplicate candidate profiles.
- Candidate Comparison and JD Matching operate on the canonical profiles.
"""
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data_transformer.db.models import Base, CandidateProfile, RawRecordDB, UploadedFile, User, UserRole
from data_transformer.pipeline.runner import PipelineRunner
from data_transformer.schema.canonical import Source
from data_transformer.db.repositories import CandidateRepository, FileRepository
from data_transformer.jd_matching.matcher import JobDescriptionMatcher, JobDescriptionParser


@pytest.fixture
def clean_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_identity_resolution_actual_uploaded_dataset(clean_db):
    upload_dir = Path("data/uploads")
    if not upload_dir.exists():
        pytest.skip("data/uploads directory not found")

    files = [
        "DEVA_JOEL_RAJ_J_FlowCV_Resume_2026-03-23.pdf",
        "Preethi_AI_RESUME.pdf",
        "Sanjaya M - 23108049.pdf",
        "Sanjaya_ATS_Report.pdf",
        "SANJAYA_M_23108049 (7).pdf",
        "Varsha_Resume (1).pdf"
    ]

    existing_files = [f for f in files if (upload_dir / f).exists()]
    if len(existing_files) < 6:
        pytest.skip(f"Expected 6 sample files, found {len(existing_files)}")

    sources = [Source(type="resume", path=str(upload_dir / f)) for f in existing_files]

    config = {
        "clustering": {
            "auto_merge_threshold": 90.0,
            "manual_review_threshold": 70.0
        }
    }
    schema = {
        "fields": [
            {"path": "candidate_id"},
            {"path": "full_name"},
            {"path": "emails"},
            {"path": "phones"},
            {"path": "skills"},
            {"path": "experience"},
        ]
    }

    runner = PipelineRunner(config, schema)
    result = runner.run(sources)

    # 1. Verify candidate count from the 6 files: 4 real candidates
    assert len(result.profiles) == 4, f"Expected 4 canonical profiles, got {len(result.profiles)}"

    cand_repo = CandidateRepository(clean_db)

    # Persist profiles to database
    for p in result.profiles:
        cid = p.get("candidate_id")
        conf = float(p.get("overall_confidence", 0.85))
        sources_merged = p.get("sources_merged", [])
        existing = cand_repo.find_matching_candidate(p)
        if existing:
            cand_repo.update(profile=existing, canonical=p, projected=p)
        else:
            cand_repo.create(
                candidate_id=cid,
                canonical=p,
                projected=p,
                confidence=conf,
                sources=sources_merged,
            )

    db_profiles = clean_db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
    assert len(db_profiles) == 4

    # 2. Check each candidate's attributes and merged sources
    profiles_by_name = {}
    for prof in db_profiles:
        canon = prof.canonical_json or {}
        name = canon.get("full_name")
        profiles_by_name[name] = prof

    # Verify Deva Joel Raj J
    deva = profiles_by_name.get("Deva Joel Raj J")
    assert deva is not None
    assert deva.sources_merged == ["DEVA_JOEL_RAJ_J_FlowCV_Resume_2026-03-23.pdf"]
    deva_emails = [e.get("value") if isinstance(e, dict) else str(e) for e in (deva.canonical_json.get("emails") or [])]
    assert "devajoelraj@gmail.com" in deva_emails

    # Verify Preethi S
    preethi = profiles_by_name.get("Preethi S")
    assert preethi is not None
    assert preethi.sources_merged == ["Preethi_AI_RESUME.pdf"]
    preethi_emails = [e.get("value") if isinstance(e, dict) else str(e) for e in (preethi.canonical_json.get("emails") or [])]
    assert "prepreethi2611@gmail.com" in preethi_emails

    # Verify Varsha S
    varsha = profiles_by_name.get("Varsha S")
    assert varsha is not None
    assert varsha.sources_merged == ["Varsha_Resume (1).pdf"]
    varsha_emails = [e.get("value") if isinstance(e, dict) else str(e) for e in (varsha.canonical_json.get("emails") or [])]
    assert "varsharaja04.s@gmail.com" in varsha_emails

    # Verify Sanjaya M
    sanjaya = profiles_by_name.get("Sanjaya M")
    assert sanjaya is not None
    # Sanjaya should have merged all 3 source files
    expected_sanjaya_sources = sorted([
        "SANJAYA_M_23108049 (7).pdf",
        "Sanjaya M - 23108049.pdf",
        "Sanjaya_ATS_Report.pdf",
    ])
    assert sorted(sanjaya.sources_merged) == expected_sanjaya_sources
    sanjaya_emails = [e.get("value") if isinstance(e, dict) else str(e) for e in (sanjaya.canonical_json.get("emails") or [])]
    assert "sanjayakumar3105@gmail.com" in sanjaya_emails or "san9345420@gmail.com" in sanjaya_emails

    # 3. Verify Reprocessing the same files does NOT create duplicate candidates
    result2 = runner.run(sources)
    for p in result2.profiles:
        existing = cand_repo.find_matching_candidate(p)
        assert existing is not None, "Candidate should already exist in DB"
        cand_repo.update(profile=existing, canonical=p, projected=p)

    db_profiles_after = clean_db.query(CandidateProfile).filter(CandidateProfile.is_deleted == False).all()
    assert len(db_profiles_after) == 4, f"Reprocessing should not duplicate profiles, got {len(db_profiles_after)}"

    # 4. Verify Candidate Comparison & JD Matching with canonical candidates
    jd_text = """
    Job Title: Senior Java Backend Developer
    Requirements:
    - 2+ years experience in Java and Spring Boot
    - Strong database skills with PostgreSQL or MySQL
    - Strong knowledge of DSA and OOP concepts
    - Experience with REST APIs and Git
    """
    reqs = JobDescriptionParser.parse_requirements(jd_text)
    matcher = JobDescriptionMatcher()

    matches = []
    for p in db_profiles_after:
        score_res = matcher.match_candidate(p.canonical_json, reqs)
        matches.append((score_res.candidate_name, score_res.overall_score, score_res.strengths, score_res.gaps))

    assert len(matches) == 4
    # All 4 candidates are evaluated transparently based on skills/experience
    for name, score, strengths, gaps in matches:
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
