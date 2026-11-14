"""
Tests for Explainable Candidate Matching Scorer.
Verifies signal explanations, classification thresholds, and false-merge safeguards.
"""
import pytest
from data_transformer.deduplication.explainable_scorer import ExplainableScorer
from data_transformer.schema.canonical import RawRecord


@pytest.fixture
def scorer():
    return ExplainableScorer(high_confidence_threshold=0.85, uncertain_threshold=0.50)


def test_exact_email_match_signal(scorer):
    rec_a = RawRecord(source="ats", fields={"full_name": "Alice Smith", "emails": ["alice@example.com"]})
    rec_b = RawRecord(source="resume", fields={"full_name": "A. Smith", "emails": ["alice@example.com"]})

    res = scorer.score(rec_a, rec_b)
    assert res.score == 1.0
    assert res.classification == "high_confidence"
    assert not res.requires_review
    assert any(s.signal_type == "exact_email" and s.matched for s in res.signals)


def test_conflicting_email_safeguard(scorer):
    # Same name but different emails: should NOT match, score = 0.0
    rec_a = RawRecord(source="ats", fields={"full_name": "Alice Smith", "emails": ["alice.smith1@gmail.com"]})
    rec_b = RawRecord(source="resume", fields={"full_name": "Alice Smith", "emails": ["alice.smith2@yahoo.com"]})

    res = scorer.score(rec_a, rec_b)
    assert res.conflicting_hard_identifiers
    assert res.score == 0.0
    assert res.classification == "non_match"


def test_fuzzy_name_and_company_uncertain(scorer):
    # Similar name, shared company, no email -> uncertain match, requires review
    rec_a = RawRecord(
        source="resume",
        fields={
            "full_name": "Michael Johnson",
            "experience": [{"company": "Acme Corp", "title": "Developer"}]
        }
    )
    rec_b = RawRecord(
        source="linkedin",
        fields={
            "full_name": "Mike Johnson",
            "experience": [{"company": "Acme Corp Inc", "title": "Senior Developer"}]
        }
    )

    res = scorer.score(rec_a, rec_b)
    assert res.score >= 0.50
    assert res.classification in ("uncertain", "high_confidence")
    assert any(s.signal_type == "shared_employer" and s.matched for s in res.signals)


def test_completely_different_candidates(scorer):
    rec_a = RawRecord(source="ats", fields={"full_name": "Carlos Gomez", "emails": ["carlos@tech.com"]})
    rec_b = RawRecord(source="resume", fields={"full_name": "Sarah Connor", "emails": ["sarah@sky.net"]})

    res = scorer.score(rec_a, rec_b)
    assert res.score == 0.0
    assert res.classification == "non_match"
    assert not res.requires_review
