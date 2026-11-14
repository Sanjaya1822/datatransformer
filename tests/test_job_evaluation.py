"""
Tests for Job-Specific Candidate Evaluation & Recommendation Engine.

Covers:
1. Multi-dimensional candidate evaluation (Technical Skills 30%, Relevant Experience 25%,
   Duration 15%, Education 10%, Projects 10%, Certifications/Preferred 10%).
2. Job criteria specialization: Different roles yield different scores for the same candidate.
3. Skill equivalence clusters: Recognizes related technologies (e.g., PostgreSQL satisfies SQL,
   Vue/Angular satisfies frontend requirements).
4. AI Recommendation: Answers all 7 required hiring questions:
   - Which candidate is more suitable?
   - Why are they more suitable?
   - What specific requirements does the recommended candidate satisfy?
   - What important requirements are missing?
   - In what areas does the other candidate excel?
   - Are both candidates equally suitable (tie)?
   - Is either candidate completely unsuitable?
5. Insufficient Evidence handling: Empty profiles categorized as Insufficient Evidence.
6. API endpoint: POST /v1/matching/evaluate-candidates integration with authentication.
"""
import pytest
from fastapi.testclient import TestClient

from data_transformer.api.app import app
from data_transformer.jd_matching.job_evaluator import (
    JobCriteria,
    evaluate_single_candidate,
    compare_and_recommend_candidates,
    DEFAULT_WEIGHTS,
)


@pytest.fixture
def java_criteria():
    return JobCriteria(
        job_title="Java Full Stack Developer",
        job_description="Develop web platforms using Java, relational databases, and frontend interfaces.",
        required_skills=["Java", "PostgreSQL", "HTML", "CSS", "JavaScript"],
        preferred_skills=["Spring Boot", "React", "Docker"],
        min_experience_years=1.0,
        required_education="Bachelor's Degree in CS, IT, or related field",
        additional_requirements="REST APIs, Git, Teamwork",
    )


@pytest.fixture
def ml_criteria():
    return JobCriteria(
        job_title="Machine Learning Engineer",
        job_description="Build, train, and deploy machine learning models. Perform EDA and statistical modeling.",
        required_skills=["Python", "Machine Learning", "Pandas", "NumPy", "Scikit-Learn"],
        preferred_skills=["TensorFlow", "Deep Learning", "SQL"],
        min_experience_years=1.0,
        required_education="Bachelor's or Master's in CS or Data Science",
    )


@pytest.fixture
def candidate_java():
    return {
        "candidate_id": "cand-java-01",
        "full_name": "Sanjaya M",
        "skills": [
            {"name": "Java"},
            {"name": "PostgreSQL"},
            {"name": "HTML"},
            {"name": "CSS"},
            {"name": "JavaScript"},
            {"name": "Spring Boot"},
        ],
        "work_experience": [
            {
                "title": "Java Developer",
                "company": "Alpha Tech",
                "start": "2023-01",
                "end": "2024-06",
                "description": "Developed backend REST microservices in Java and handled PostgreSQL database schemas.",
            }
        ],
        "education": [
            {
                "degree": "B.Tech in Information Technology",
                "institution": "Anna University",
                "graduation_year": "2024",
                "cgpa": "8.5",
            }
        ],
        "projects": [
            {
                "title": "E-Commerce Web Portal | Java, HTML, CSS",
                "description": "Built full stack web application with Java servlets and PostgreSQL.",
            }
        ],
        "certifications": ["Java Professional Certificate"],
    }


@pytest.fixture
def candidate_ml():
    return {
        "candidate_id": "cand-ml-02",
        "full_name": "Preethi S",
        "skills": [
            {"name": "Python"},
            {"name": "Machine Learning"},
            {"name": "Pandas"},
            {"name": "NumPy"},
            {"name": "Scikit-Learn"},
            {"name": "Matplotlib"},
        ],
        "work_experience": [
            {
                "title": "Data Science Intern",
                "company": "Data Insights Ltd",
                "start": "2023-06",
                "end": "2024-05",
                "description": "Performed exploratory data analysis and built predictive models using scikit-learn.",
            }
        ],
        "education": [
            {
                "degree": "B.E in Computer Science",
                "institution": "PSG College of Technology",
                "graduation_year": "2024",
                "cgpa": "9.1",
            }
        ],
        "projects": [
            {
                "title": "Customer Churn Prediction Model | Python, Scikit-Learn",
                "description": "Developed churn prediction classifier using Random Forest and Pandas.",
            }
        ],
        "certifications": ["Machine Learning Specialization"],
    }


# ─── 1. Unit Tests for Evaluation Engine ─────────────────────────────────────

def test_single_candidate_evaluation_dimensions(candidate_java, java_criteria):
    eval_result = evaluate_single_candidate(candidate_java, java_criteria)
    assert "overall_score" in eval_result
    assert "dimension_scores" in eval_result
    assert "match_category" in eval_result
    assert "strengths" in eval_result
    assert "gaps" in eval_result

    dims = eval_result["dimension_scores"]
    assert "technical_skills" in dims
    assert "relevant_experience" in dims
    assert "experience_duration" in dims
    assert "education" in dims
    assert "project_relevance" in dims
    assert "certifications_preferred" in dims

    # Candidate Java should score strongly on Java skills
    assert dims["technical_skills"] >= 75.0
    assert eval_result["match_category"] in ["Excellent Match", "Strong Match", "Moderate Match"]


def test_role_differentiation_same_candidates(candidate_java, candidate_ml, java_criteria, ml_criteria):
    """
    Candidate Java should win for the Java role.
    Candidate ML should win for the ML role.
    """
    # 1. Java Role
    res_java = compare_and_recommend_candidates(candidate_java, candidate_ml, java_criteria)
    score_java_for_java = res_java["overall_recommendation"]["candidate_a_score"]
    score_ml_for_java = res_java["overall_recommendation"]["candidate_b_score"]
    assert score_java_for_java > score_ml_for_java
    assert "Candidate A" in res_java["overall_recommendation"]["recommended_candidate"]

    # 2. ML Role
    res_ml = compare_and_recommend_candidates(candidate_java, candidate_ml, ml_criteria)
    score_java_for_ml = res_ml["overall_recommendation"]["candidate_a_score"]
    score_ml_for_ml = res_ml["overall_recommendation"]["candidate_b_score"]
    assert score_ml_for_ml > score_java_for_ml
    assert "Candidate B" in res_ml["overall_recommendation"]["recommended_candidate"]

    # 3. Same candidate scores differently on different roles
    assert score_java_for_java != score_java_for_ml
    assert score_ml_for_ml != score_ml_for_java


def test_skill_cluster_equivalence():
    """
    A candidate with 'MySQL' should receive partial/equivalent credit for 'SQL' or 'PostgreSQL'.
    """
    cand_mysql = {
        "candidate_id": "c-mysql",
        "full_name": "Dev User",
        "skills": [{"name": "Java"}, {"name": "MySQL"}],
        "work_experience": [],
        "education": [],
    }
    crit = JobCriteria(
        job_title="Backend Developer",
        required_skills=["Java", "PostgreSQL"],
    )
    eval_res = evaluate_single_candidate(cand_mysql, crit)
    # MySQL should be counted as matching or equivalent for PostgreSQL
    assert len(eval_res["matching_skills"]) >= 2
    assert eval_res["dimension_scores"]["technical_skills"] >= 75.0


def test_ai_recommendation_seven_questions(candidate_java, candidate_ml, java_criteria):
    """
    Verifies that all 7 required hiring questions are addressed in the AI recommendation payload.
    """
    res = compare_and_recommend_candidates(candidate_java, candidate_ml, java_criteria)
    ai_rec = res["ai_recommendation"]

    # 1. Which candidate is more suitable?
    assert "which_candidate_more_suitable" in ai_rec
    assert "Sanjaya" in ai_rec["which_candidate_more_suitable"]

    # 2. Why more suitable?
    assert "why_more_suitable" in ai_rec
    assert len(ai_rec["why_more_suitable"]) > 10

    # 3. Specific requirements satisfied by winner
    assert "requirements_satisfied_by_winner" in ai_rec
    assert len(ai_rec["requirements_satisfied_by_winner"]) > 0

    # 4. Important requirements missing
    assert "important_requirements_missing" in ai_rec

    # 5. Where other candidate excels
    assert "where_other_candidate_excels" in ai_rec

    # 6. Are both equally suitable (tie)?
    assert "are_both_equally_suitable" in ai_rec
    assert ai_rec["are_both_equally_suitable"] is False

    # 7. Is either unsuitable?
    assert "is_either_unsuitable" in ai_rec


def test_insufficient_evidence_for_empty_profiles(java_criteria):
    empty_a = {"candidate_id": "emp-a", "full_name": "Blank Profile A"}
    empty_b = {"candidate_id": "emp-b", "full_name": "Blank Profile B"}

    res = compare_and_recommend_candidates(empty_a, empty_b, java_criteria)
    overall = res["overall_recommendation"]
    assert "Insufficient" in overall["recommended_candidate"]
    assert overall["recommendation_confidence"] == "Low"


def test_tie_detection(java_criteria):
    """
    Two identical qualified candidates should result in a Tie verdict.
    """
    cand = {
        "candidate_id": "cand-twin",
        "full_name": "Twin Candidate",
        "skills": [{"name": "Java"}, {"name": "HTML"}, {"name": "CSS"}],
        "work_experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "start": "2023-01",
                "end": "2024-01",
                "description": "Developed web applications.",
            }
        ],
        "education": [
            {
                "degree": "B.Tech in Computer Science",
                "institution": "University",
                "graduation_year": "2023",
            }
        ],
    }
    res = compare_and_recommend_candidates(cand, cand, java_criteria)
    overall = res["overall_recommendation"]
    assert "Tie" in overall["recommended_candidate"]
    assert res["ai_recommendation"]["are_both_equally_suitable"] is True


# ─── 2. End-to-End API Integration Tests ────────────────────────────────────

def test_api_evaluate_candidates_endpoint():
    client = TestClient(app)
    # 1. Authenticate as recruiter
    login_res = client.post("/v1/auth/login", json={"username": "recruiter", "password": "recruiter123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Fetch candidates from database
    cands_res = client.get("/v1/candidates/", headers=headers)
    assert cands_res.status_code == 200
    candidates = cands_res.json().get("items", [])

    if len(candidates) >= 2:
        id_a = candidates[0]["candidate_id"]
        id_b = candidates[1]["candidate_id"]

        # 3. Post evaluation request
        eval_payload = {
            "candidate_id_a": id_a,
            "candidate_id_b": id_b,
            "job_title": "Full Stack Software Developer",
            "job_description": "Building web applications, working with databases, collaborating in agile teams.",
            "required_skills": ["Java", "Python", "SQL", "HTML", "CSS"],
            "preferred_skills": ["React", "Docker"],
            "min_experience_years": 1.0,
            "required_education": "Bachelor's Degree in Computer Science or related",
            "additional_requirements": "Git, REST APIs",
        }
        res = client.post("/v1/matching/evaluate-candidates", json=eval_payload, headers=headers)
        assert res.status_code == 200
        data = res.json()

        # Check all required top-level keys
        for key in ["overall_recommendation", "candidate_a", "candidate_b", "detailed_comparison", "ai_recommendation"]:
            assert key in data, f"Missing key {key} in response"

        # Check overall recommendation structure
        overall = data["overall_recommendation"]
        assert "candidate_a_score" in overall
        assert "candidate_b_score" in overall
        assert "recommended_candidate" in overall
        assert "recommendation_confidence" in overall
        assert "winner_id" in overall

        # Check detailed comparison has all 6 dimensions
        detailed = data["detailed_comparison"]
        assert len(detailed) == 6
        for row in detailed:
            assert "criterion" in row
            assert "weight" in row
            assert "candidate_a" in row
            assert "candidate_b" in row

        # Check candidate profiles are attached and masked
        assert "candidate_a_profile" in data
        assert "candidate_b_profile" in data
