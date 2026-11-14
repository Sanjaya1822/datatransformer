"""
Tests for Candidate Comparison, Reconciliation, and Job Description Matching.

Covers:
1. Candidate identity matching:
   - Strong identity signals (email, normalized phone)
   - Supporting signals (name, skills, company)
   - Missing values handled without treating as mismatches or conflicts
   - Preventing false merges based only on similar names
   - Excluding UUIDs from identity/similarity evidence
2. Accurate field-level comparison:
   - Displays Candidate A & B values, status, and difference explanation
   - The 6 required statuses: Exact Match, Partial Match, Conflict,
     Missing in Candidate A, Missing in Candidate B, Insufficient Evidence
   - Missing values NOT marked as conflicts
3. Meaningful weighted similarity score:
   - Transparent weights, score breakdown
   - Identity evidence considered alongside similarity
4. Accurate profile merging:
   - Merging valid attributes into canonical profile
   - Preserving source records and tracking attribute sources
5. Job description matching & ranking:
   - Structured criteria extraction (required skills, preferred skills, experience, education, certifications, responsibilities)
   - Evaluation of candidates against requirements
   - Matched skills, missing required skills, relevant experience, education match, strengths, gaps, score explanation
   - Head-to-head JD candidate comparison
   - Resilience against empty/null candidate fields (preventing 500 crashes)
"""
import pytest
from fastapi.testclient import TestClient

from data_transformer.api.app import app
from data_transformer.matching.candidate_identity import (
    compare_candidates_detailed,
    evaluate_candidate_identity,
    merge_canonical_profiles,
    extract_candidate_attributes,
)
from data_transformer.jd_matching.matcher import (
    JobDescriptionMatcher,
    JobDescriptionParser,
    JDRequirement,
)


# ─── 1. Identity Matching Tests ──────────────────────────────────────────────

def test_same_candidate_email_phone_match():
    cand_a = {
        "candidate_id": "uuid-1111",
        "full_name": "Sanjaya M",
        "emails": [{"value": "sanjayakumar3105@gmail.com", "is_primary": True}],
        "phones": [{"value": "+19345420856", "is_primary": True}],
        "skills": [{"name": "Java"}, {"name": "Python"}],
    }
    cand_b = {
        "candidate_id": "uuid-2222",  # Different internal UUID
        "full_name": "SANJAYA M",
        "primary_email": "sanjayakumar3105@gmail.com",
        "phone": "+1 (934) 542-0856",
        "skills": ["Java", "Docker"],
    }

    ident = evaluate_candidate_identity(cand_a, cand_b)
    assert ident["is_same_candidate"] is True
    assert ident["verdict"] == "Same Candidate"

    comp = compare_candidates_detailed(cand_a, cand_b)
    assert comp["is_same_candidate"] is True
    assert comp["match_score"] >= 0.85
    assert comp["classification"] == "high_confidence"

    # Verify status on fields
    diffs = comp["field_diffs"]
    assert diffs["full_name"]["status"] == "Exact Match"
    assert diffs["email"]["status"] == "Exact Match"
    assert diffs["phone"]["status"] == "Exact Match"
    assert diffs["skills"]["status"] == "Partial Match"
    assert diffs["location"]["status"] == "Insufficient Evidence"  # Missing in both, NOT conflict!


def test_different_candidates_distinct_emails_and_phones():
    cand_a = {
        "candidate_id": "uuid-a",
        "full_name": "Deva Joel Raj J",
        "emails": [{"value": "devajoelraj@gmail.com"}],
        "phones": [{"value": "+919677607931"}],
        "skills": ["Java", "CSS"],
    }
    cand_b = {
        "candidate_id": "uuid-b",
        "full_name": "Varsha S",
        "emails": [{"value": "varsharaja04.s@gmail.com"}],
        "phones": [{"value": "+19344873369"}],
        "skills": ["Python", "SQL"],
    }

    ident = evaluate_candidate_identity(cand_a, cand_b)
    assert ident["is_same_candidate"] is False
    assert ident["verdict"] == "Different Candidates"

    comp = compare_candidates_detailed(cand_a, cand_b)
    assert comp["is_same_candidate"] is False
    diffs = comp["field_diffs"]
    assert diffs["full_name"]["status"] == "Conflict"
    assert diffs["email"]["status"] == "Conflict"
    assert diffs["phone"]["status"] == "Conflict"


def test_no_merge_on_name_alone_without_corroboration():
    # Two candidates with similar name but no matching contact info or corroborating career history
    cand_a = {
        "full_name": "John Smith",
        "emails": [{"value": "jsmith.boston@example.com"}],
        "skills": ["Accounting"],
    }
    cand_b = {
        "full_name": "John Smith",
        "emails": [{"value": "john.smith.chicago@test.com"}],
        "skills": ["Mechanical Engineering"],
    }

    ident = evaluate_candidate_identity(cand_a, cand_b)
    assert ident["is_same_candidate"] is False


def test_missing_values_not_marked_as_conflicts():
    cand_a = {
        "full_name": "Candidate A",
        "location": {"city": "San Francisco", "country": "USA"},
        "skills": ["Python", "AWS"],
    }
    cand_b = {
        "full_name": "Candidate B",
        # Location is completely missing
        # Skills is completely missing
    }

    comp = compare_candidates_detailed(cand_a, cand_b)
    diffs = comp["field_diffs"]

    # Location missing in B
    assert diffs["location"]["status"] == "Missing in Candidate B"
    assert diffs["location"]["status"] != "Conflict"

    # Skills missing in B
    assert diffs["skills"]["status"] == "Missing in Candidate B"
    assert diffs["skills"]["status"] != "Conflict"

    # Phone missing in both
    assert diffs["phone"]["status"] == "Insufficient Evidence"
    assert diffs["phone"]["status"] != "Conflict"


# ─── 2. Profile Merging Tests ────────────────────────────────────────────────

def test_accurate_profile_merging():
    existing = {
        "candidate_id": "cand-123",
        "full_name": "Sanjaya M",
        "emails": [{"value": "sanjayakumar3105@gmail.com", "is_primary": True}],
        "phones": [{"value": "+19345420856", "is_primary": True}],
        "skills": [{"name": "Java", "sources": ["resume1.pdf"]}],
        "sources_merged": ["resume1.pdf"],
    }
    incoming = {
        "candidate_id": "new-uuid",
        "full_name": "Sanjaya M",
        "emails": [{"value": "san9345420@gmail.com", "is_primary": False}],
        "skills": [{"name": "SQL", "sources": ["resume2.pdf"]}, {"name": "Java", "sources": ["resume2.pdf"]}],
        "experience": [{"company": "Internpe", "title": "Developer"}],
        "sources_merged": ["resume2.pdf"],
    }

    merged = merge_canonical_profiles(existing, incoming)

    # 1. Sources merged contains both files
    assert "resume1.pdf" in merged["sources_merged"]
    assert "resume2.pdf" in merged["sources_merged"]

    # 2. Both emails preserved
    email_vals = [e["value"] for e in merged["emails"]]
    assert "sanjayakumar3105@gmail.com" in email_vals
    assert "san9345420@gmail.com" in email_vals

    # 3. Skills merged without duplicates and sources tracked
    skill_names = [s["name"] for s in merged["skills"]]
    assert "Java" in skill_names
    assert "Sql" in skill_names or "SQL" in skill_names
    java_skill = next(s for s in merged["skills"] if s["name"].lower() == "java")
    assert "resume1.pdf" in java_skill["sources"] or "existing" in java_skill["sources"]

    # 4. Experience preserved
    assert len(merged["experience"]) == 1
    assert merged["experience"][0]["company"] == "Internpe"


# ─── 3. Job Description Matching & Ranking Tests ─────────────────────────────

def test_jd_parser_structured_requirements():
    jd_text = """
    Job Title: Senior Cloud Engineer
    Responsibilities:
    - Architect and deploy resilient microservices on Kubernetes
    - Optimize high-throughput event processing pipelines
    Requirements:
    - 4+ years of professional experience in distributed systems
    - Bachelor's degree in Computer Science or related discipline
    - Required skills: Python, Docker, Kubernetes, AWS, SQL
    Preferred skills:
    - React, Go, Kafka
    Certifications:
    - AWS Certified Solutions Architect
    """

    reqs = JobDescriptionParser.parse_requirements(jd_text)
    categories = {r.category for r in reqs}

    assert "experience" in categories
    assert "education" in categories
    assert "required_skill" in categories
    assert "preferred_skill" in categories
    assert "certification" in categories

    exp_req = next(r for r in reqs if r.category == "experience")
    assert "4+" in exp_req.text

    req_skills = [r.text.lower() for r in reqs if r.category == "required_skill"]
    assert "python" in req_skills
    assert "kubernetes" in req_skills


def test_jd_matcher_null_and_empty_candidate_safety():
    # Must never raise TypeError or 500 error when candidate attributes are None or empty
    matcher = JobDescriptionMatcher()
    reqs = [
        JDRequirement(category="required_skill", text="Python", weight=0.2, is_required=True),
        JDRequirement(category="experience", text="3+ years", weight=0.2, is_required=True),
        JDRequirement(category="education", text="Bachelor degree", weight=0.2, is_required=True),
    ]

    empty_candidate = {
        "candidate_id": "test-empty",
        "full_name": None,
        "skills": None,
        "experience": None,
        "education": None,
        "certifications": None,
    }

    # Should run smoothly without exceptions
    score = matcher.match_candidate(empty_candidate, reqs)
    assert score.overall_score == 0.0
    assert score.fit_level == "Low Alignment"
    assert "Python" in score.missing_required_skills
    assert score.relevant_experience == "No recorded professional experience"


def test_jd_matcher_candidate_ranking_transparency():
    matcher = JobDescriptionMatcher()
    reqs = [
        JDRequirement(category="required_skill", text="Python", weight=0.3, is_required=True),
        JDRequirement(category="required_skill", text="FastAPI", weight=0.3, is_required=True),
        JDRequirement(category="experience", text="2+ years", weight=0.4, is_required=True),
    ]

    candidate = {
        "candidate_id": "c1",
        "full_name": "Test Engineer",
        "skills": [{"name": "Python"}, {"name": "FastAPI"}],
        "experience": [
            {
                "company": "Tech Corp",
                "title": "Software Engineer",
                "start": "2023-01",
                "end": "2025-06",
                "summary": "Built backend APIs using Python and FastAPI."
            }
        ],
        "education": [{"degree": "Bachelor of Science", "institution": "State University"}],
    }

    score = matcher.match_candidate(candidate, reqs)
    assert score.overall_score >= 80.0
    assert score.fit_level == "Strong Fit"
    assert "Python" in score.matched_skills
    assert "FastAPI" in score.matched_skills
    assert len(score.missing_required_skills) == 0
    assert score.experience_years >= 2.0


# ─── 4. End-to-End API Integration Tests ────────────────────────────────────

def test_api_candidate_compare_endpoint():
    client = TestClient(app)
    # Auth
    login_res = client.post("/v1/auth/login", json={"username": "recruiter", "password": "recruiter123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Retrieve candidates
    cands_res = client.get("/v1/candidates/", headers=headers)
    assert cands_res.status_code == 200
    items = cands_res.json().get("items", [])
    if len(items) >= 2:
        id_a = items[0]["candidate_id"]
        id_b = items[1]["candidate_id"]

        comp_res = client.get(f"/v1/matching/compare/{id_a}/{id_b}", headers=headers)
        assert comp_res.status_code == 200
        data = comp_res.json()

        assert "similarity_percentage" in data
        assert "identity_verdict" in data
        assert "score_breakdown" in data
        assert "field_diffs" in data

        # Check required fields in field_diffs
        expected_fields = ["full_name", "email", "phone", "location", "skills", "experience", "education", "certifications", "source_records"]
        for f in expected_fields:
            assert f in data["field_diffs"]
            field_obj = data["field_diffs"][f]
            assert "status" in field_obj
            assert "explanation" in field_obj
            assert field_obj["status"] in [
                "Exact Match", "Partial Match", "Conflict",
                "Missing in Candidate A", "Missing in Candidate B", "Insufficient Evidence"
            ]


def test_api_jd_create_and_match():
    client = TestClient(app)
    login_res = client.post("/v1/auth/login", json={"username": "recruiter", "password": "recruiter123"})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create JD
    create_res = client.post("/v1/jobs/", json={
        "title": "Backend Python Developer",
        "content": "Requirements: Python, SQL, REST APIs. 1+ years experience. Bachelor degree."
    }, headers=headers)
    assert create_res.status_code == 201
    jd_id = create_res.json()["id"]

    # 2. Match candidates
    match_res = client.post(f"/v1/jobs/{jd_id}/match", json={"min_score": 0.0}, headers=headers)
    assert match_res.status_code == 200
    data = match_res.json()

    assert "rankings" in data
    assert "job_title" in data
    for r in data["rankings"]:
        assert "overall_score" in r
        assert "fit_level" in r
        assert "matched_skills" in r
        assert "missing_required_skills" in r
        assert "relevant_experience" in r
        assert "education_match" in r
        assert "strengths" in r
        assert "gaps" in r
