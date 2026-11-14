"""
End-to-End Test Suite for Critical Fixes and Features.

Tests:
1. File loading, retrieval, reprocess, and deletion (HTTP 200, meaningful error handling).
2. Candidate deduplication:
   - Same identity records merge properly into a single canonical profile.
   - Distinct candidates with similar names and different contact info are NOT merged.
3. User role simplification:
   - Recruiter role works everywhere, admin/reviewer roles removed.
4. Job Description semantic matcher & candidate comparison:
   - Parsing text and file uploads.
   - Evidence-based matching, strengths, gaps, and experience calculations.
   - Explanatory ranking and relative comparative advantages.
   - Head-to-head candidate comparison matrix and winner explanation.
5. Verification of module removal (Knowledge graph endpoints not mounted).
"""
import io
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data_transformer.api.app import app
from data_transformer.auth.service import create_access_token, hash_password
from data_transformer.db.database import Base, get_db
from data_transformer.db.models import CandidateProfile, UploadedFile, User, UserRole
from data_transformer.db.repositories import CandidateRepository, FileRepository
from data_transformer.db.seeder import seed_database
from data_transformer.pipeline.runner import PipelineRunner
from data_transformer.schema.canonical import RawRecord, Source


@pytest.fixture
def client_with_clean_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    seed_database(db, hash_password_func=hash_password)

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, db, TestingSessionLocal
    app.dependency_overrides.clear()
    db.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def recruiter_token(client_with_clean_db):
    client, db, _ = client_with_clean_db
    resp = client.post("/auth/login", json={"username": "recruiter", "password": "recruiter123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ── 1. File Loading, Retrieval, Reprocess & Deletion ──────────────────────────

def test_files_listing_and_retrieval(client_with_clean_db, recruiter_token):
    client, _, TestingSessionLocal = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    # Seed a file
    with TestingSessionLocal() as session:
        repo = FileRepository(session)
        f = repo.create(
            filename="sample_resume.pdf",
            file_path="data/uploads/sample_resume.pdf",
            file_type="resume",
            file_size=1024,
            status="processed",
        )
        file_id = f.id

    # List files
    resp = client.get("/v1/files/", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert data["total"] >= 1
    assert any(item["filename"] == "sample_resume.pdf" for item in data["items"])

    # Retrieve single file
    resp_single = client.get(f"/v1/files/{file_id}", headers=headers)
    assert resp_single.status_code == 200
    assert resp_single.json()["filename"] == "sample_resume.pdf"


def test_file_deletion_by_recruiter(client_with_clean_db, recruiter_token):
    client, _, TestingSessionLocal = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    with TestingSessionLocal() as session:
        repo = FileRepository(session)
        f = repo.create(
            filename="delete_me.pdf",
            file_path="data/uploads/delete_me.pdf",
            file_type="resume",
            file_size=512,
        )
        file_id = f.id

    del_resp = client.delete(f"/v1/files/{file_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # 404 on already deleted file
    del_again = client.delete(f"/v1/files/{file_id}", headers=headers)
    assert del_again.status_code == 404
    assert "not found" in del_again.json()["detail"].lower()


def test_file_reprocessing(client_with_clean_db, recruiter_token):
    client, _, TestingSessionLocal = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write("Priya Sharma\npriya@example.com\n+91-9876543210\nPython, Django, AWS\n2020-2023 Software Engineer at Infosys\n")
        tmp_path = tmp.name

    try:
        with TestingSessionLocal() as session:
            repo = FileRepository(session)
            f = repo.create(
                filename="priya_resume.txt",
                file_path=tmp_path,
                file_type="resume",
                file_size=200,
            )
            file_id = f.id

        reproc_resp = client.post(f"/v1/files/{file_id}/reprocess", headers=headers)
        assert reproc_resp.status_code == 200
        res_data = reproc_resp.json()
        assert res_data["success"] is True
        assert res_data["candidates_extracted"] >= 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── 2. Candidate Deduplication & Merging ─────────────────────────────────────

def test_deduplication_merges_same_candidate():
    """Verify that records with identical emails are clustered into ONE candidate profile."""
    config = {"clustering": {"auto_merge_threshold": 90.0, "manual_review_threshold": 70.0}}
    schema = {"fields": [{"path": "candidate_id"}, {"path": "full_name"}, {"path": "emails"}, {"path": "skills"}]}
    runner = PipelineRunner(config, schema)

    source1 = Source(type="resume", content="Alice Smith\nalice@techcorp.io\nSkills\nPython, FastAPI\n")
    source2 = Source(type="ats", content=json.dumps({"name": "Alice Smith", "email": "alice@techcorp.io", "skills": ["Docker", "AWS"]}))

    result = runner.run([source1, source2])
    # Must produce exactly 1 merged candidate profile
    assert len(result.profiles) == 1
    p = result.profiles[0]
    # Skills from both sources must be merged
    skill_names = [s.get("name") if isinstance(s, dict) else s for s in p.get("skills", [])]
    assert any("python" in str(s).lower() for s in skill_names)


def test_deduplication_does_not_merge_different_candidates_with_same_name():
    """Verify that two people named 'John Smith' with different emails DO NOT merge."""
    config = {"clustering": {"auto_merge_threshold": 90.0, "manual_review_threshold": 70.0}}
    schema = {"fields": [{"path": "candidate_id"}, {"path": "full_name"}, {"path": "emails"}, {"path": "skills"}]}
    runner = PipelineRunner(config, schema)

    source1 = Source(type="resume", content="John Smith\njohn.smith.ny@gmail.com\nPython Developer in New York\n")
    source2 = Source(type="resume", content="John Smith\njohn.smith.sf@yahoo.com\nJava Developer in San Francisco\n")

    result = runner.run([source1, source2])
    # Must NOT be merged — two separate profiles
    assert len(result.profiles) == 2


# ── 3. Role Simplification ───────────────────────────────────────────────────

def test_single_recruiter_role_access(client_with_clean_db, recruiter_token):
    client, _, _ = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    # /auth/me returns recruiter
    me_resp = client.get("/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["role"] == "recruiter"

    # Only UserRole.recruiter is defined in UserRole enum
    assert [r.value for r in UserRole] == ["recruiter"]


# ── 4 & 5. Job Description Semantic Matcher & Comparison ─────────────────────

def test_jd_creation_and_matching(client_with_clean_db, recruiter_token):
    client, _, TestingSessionLocal = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    # Seed candidates in DB
    with TestingSessionLocal() as session:
        cand_repo = CandidateRepository(session)
        # Candidate 1: Senior with Python, FastAPI, Docker, 5 years exp
        cand_repo.create(
            candidate_id="cand-senior-01",
            canonical={
                "candidate_id": "cand-senior-01",
                "full_name": {"value": "Rohan Patel"},
                "skills": [{"name": "Python"}, {"name": "FastAPI"}, {"name": "Docker"}, {"name": "PostgreSQL"}],
                "years_experience": 5.0,
                "education": [{"degree": "Bachelor of Technology in Computer Science"}],
            },
        )
        # Candidate 2: Junior with Python only, 1 year exp
        cand_repo.create(
            candidate_id="cand-junior-02",
            canonical={
                "candidate_id": "cand-junior-02",
                "full_name": {"value": "Amit Kumar"},
                "skills": [{"name": "Python"}],
                "years_experience": 1.0,
                "education": [{"degree": "Bachelor of Science"}],
            },
        )

    # 1. Create JD
    jd_resp = client.post(
        "/v1/jobs/",
        json={
            "title": "Senior Backend Engineer",
            "content": "Looking for a Senior Backend Engineer with 4+ years of professional experience. Must have strong Python, FastAPI, and Docker skills. Bachelor's degree required.",
        },
        headers=headers,
    )
    assert jd_resp.status_code == 201
    jd_data = jd_resp.json()
    jd_id = jd_data["id"]

    # 2. Match candidates
    match_resp = client.post(
        f"/v1/jobs/{jd_id}/match",
        json={"min_score": 0.0},
        headers=headers,
    )
    assert match_resp.status_code == 200
    match_data = match_resp.json()
    rankings = match_data["rankings"]

    assert len(rankings) == 2
    # Rohan (Senior) must rank #1 over Amit (Junior)
    assert rankings[0]["candidate_id"] == "cand-senior-01"
    assert rankings[0]["overall_score"] > rankings[1]["overall_score"]
    assert "why_better_match" in rankings[0]
    assert len(rankings[0]["strengths"]) > len(rankings[1]["strengths"])

    # 3. Head-to-Head Comparison
    comp_resp = client.post(
        f"/v1/jobs/{jd_id}/compare",
        json={"candidate_id_a": "cand-senior-01", "candidate_id_b": "cand-junior-02"},
        headers=headers,
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["stronger_candidate_id"] == "cand-senior-01"
    assert "Rohan Patel" in comp_data["comparison_explanation"]
    assert len(comp_data["matrix"]) > 0


def test_jd_file_upload(client_with_clean_db, recruiter_token):
    client, _, _ = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    jd_content = b"DevOps Engineer\nWe need an experienced Kubernetes, Docker, and AWS specialist with 3+ years of experience.\n"
    files = {"file": ("job_description.txt", io.BytesIO(jd_content), "text/plain")}
    data = {"title": "DevOps Engineer"}

    upload_resp = client.post("/v1/jobs/upload", files=files, data=data, headers=headers)
    assert upload_resp.status_code == 201
    res = upload_resp.json()
    assert res["title"] == "DevOps Engineer"
    assert len(res["extracted_requirements"]) >= 2


# ── 6. Unnecessary Modules Removal ───────────────────────────────────────────

def test_knowledge_graph_endpoints_removed(client_with_clean_db, recruiter_token):
    client, _, _ = client_with_clean_db
    headers = {"Authorization": f"Bearer {recruiter_token}"}

    # /v1/graph/overview must return 404
    resp = client.get("/v1/graph/overview", headers=headers)
    assert resp.status_code == 404
