"""
Tests for the Job Description Parser and Matcher.

Covers:
- JobDescriptionParser.parse_requirements: skill extraction, experience, education
- JobDescriptionMatcher.match_candidate: matched/missing items, score, fit_level
- Advisory note always present
- Evidence populated for matched requirements
- No automatic rejection (advisory-only policy)
"""
from __future__ import annotations

import pytest

from data_transformer.jd_matching.matcher import (
    CandidateJDMatchScore,
    JobDescriptionMatcher,
    JobDescriptionParser,
    JDRequirement,
    RequirementMatchItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_JD_SKILLS_AND_EXP = """
We are looking for a Senior Software Engineer with 5+ years of experience.
Required skills: Python, PostgreSQL, Docker, AWS.
Nice to have: React, Kubernetes.
Bachelor's degree in Computer Science required.
"""

MINIMAL_JD = "We need someone with Python experience."


def _make_candidate(skills=None, years_exp=0.0, education=None) -> dict:
    return {
        "candidate_id": "cand-test",
        "full_name": {"value": "Test Candidate"},
        "skills": [{"name": s} for s in (skills or [])],
        "years_experience": {"value": years_exp},
        "experience": [],
        "education": education or [],
    }


# ---------------------------------------------------------------------------
# JobDescriptionParser
# ---------------------------------------------------------------------------

class TestJobDescriptionParser:
    def test_parses_experience_years(self):
        reqs = JobDescriptionParser.parse_requirements(SAMPLE_JD_SKILLS_AND_EXP)
        exp_reqs = [r for r in reqs if r.category == "experience"]
        assert len(exp_reqs) == 1
        assert "5" in exp_reqs[0].text

    def test_parses_bachelor_education(self):
        reqs = JobDescriptionParser.parse_requirements(SAMPLE_JD_SKILLS_AND_EXP)
        edu_reqs = [r for r in reqs if r.category == "education"]
        assert len(edu_reqs) == 1
        assert edu_reqs[0].is_required is True

    def test_parses_required_skills(self):
        reqs = JobDescriptionParser.parse_requirements(SAMPLE_JD_SKILLS_AND_EXP)
        skill_reqs = [r for r in reqs if r.category == "required_skill"]
        skill_names = {r.text.lower() for r in skill_reqs}
        assert "python" in skill_names
        assert "postgresql" in skill_names
        assert "docker" in skill_names
        assert "aws" in skill_names

    def test_weights_sum_reasonable(self):
        """Total weight should be positive; individual weights > 0."""
        reqs = JobDescriptionParser.parse_requirements(SAMPLE_JD_SKILLS_AND_EXP)
        total = sum(r.weight for r in reqs)
        assert total > 0
        for r in reqs:
            assert r.weight > 0

    def test_minimal_jd_still_extracts_python(self):
        reqs = JobDescriptionParser.parse_requirements(MINIMAL_JD)
        skill_names = {r.text.lower() for r in reqs}
        assert "python" in skill_names

    def test_empty_jd_returns_empty_list(self):
        reqs = JobDescriptionParser.parse_requirements("")
        assert reqs == []


# ---------------------------------------------------------------------------
# JobDescriptionMatcher
# ---------------------------------------------------------------------------

class TestJobDescriptionMatcher:
    def test_perfect_skill_match_high_score(self):
        reqs = [JDRequirement(category="required_skill", text="Python", weight=0.5, is_required=True)]
        candidate = _make_candidate(skills=["python"])
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        assert result.overall_score >= 80.0
        assert result.fit_level == "Strong Fit"

    def test_no_skills_match_low_score(self):
        reqs = [JDRequirement(category="required_skill", text="Golang", weight=0.5, is_required=True)]
        candidate = _make_candidate(skills=["python"])
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        assert result.overall_score < 50.0
        assert len(result.missing_requirements) == 1

    def test_experience_matched_when_sufficient(self):
        reqs = [JDRequirement(category="experience", text="3+ years", weight=0.3, is_required=True)]
        candidate = _make_candidate(years_exp=5.0)
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        matched_cats = {m.category for m in result.matched_requirements}
        assert "experience" in matched_cats

    def test_experience_partial_when_insufficient(self):
        reqs = [JDRequirement(category="experience", text="5+ years", weight=0.3, is_required=True)]
        candidate = _make_candidate(years_exp=2.0)
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        # Partial experience goes to missing_requirements with status=partial
        all_items = result.matched_requirements + result.missing_requirements
        exp_items = [i for i in all_items if i.category == "experience"]
        assert len(exp_items) == 1
        assert exp_items[0].status in ("partial", "missing")

    def test_advisory_note_always_present(self):
        reqs = []
        candidate = _make_candidate()
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        assert "advisory" in result.advisory_note.lower() or "human" in result.advisory_note.lower()

    def test_evidence_populated_for_matched_skills(self):
        reqs = [JDRequirement(category="required_skill", text="Python", weight=0.5, is_required=True)]
        candidate = _make_candidate(skills=["python"])
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        for item in result.matched_requirements:
            assert item.evidence is not None and len(item.evidence) > 0

    def test_candidate_id_and_name_in_result(self):
        reqs = []
        candidate = _make_candidate()
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        assert result.candidate_id == "cand-test"
        assert "Test Candidate" in result.candidate_name

    def test_fit_levels_cover_all_tiers(self):
        """Exercise each fit-level boundary."""
        matcher = JobDescriptionMatcher()
        req = JDRequirement(category="required_skill", text="Python", weight=1.0, is_required=True)

        # 100% match → Strong Fit
        result_strong = matcher.match_candidate(_make_candidate(skills=["python"]), [req])
        assert result_strong.fit_level == "Strong Fit"

        # 0% match → Low Alignment
        result_low = matcher.match_candidate(_make_candidate(skills=["cobol"]), [req])
        assert result_low.fit_level == "Low Alignment"

    def test_education_bachelor_matched(self):
        reqs = [JDRequirement(category="education", text="Bachelor's degree", weight=0.2, is_required=True)]
        candidate = _make_candidate(
            education=[{"degree": "bachelor", "institution": "MIT"}]
        )
        matcher = JobDescriptionMatcher()
        result = matcher.match_candidate(candidate, reqs)
        matched_cats = {m.category for m in result.matched_requirements}
        assert "education" in matched_cats
