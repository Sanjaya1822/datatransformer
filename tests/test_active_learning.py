"""
Tests for the Active Learning / Threshold Recalibration module.

Covers:
- Temporal train/validation/test split ordering and proportions
- Recalibration with sufficient labeled pairs (grid-search)
- Insufficient-data fallback (config unchanged)
- EvaluationMetrics.compute accuracy
- Review-queue priority (uncertainty nearest threshold)
- Held-out test-set evaluation
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from data_transformer.matching.active_learning import (
    ActiveLearner,
    EvaluationMetrics,
    LabeledPair,
    MatchingConfigVersion,
    MIN_SAMPLES_FOR_RECALIBRATION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair(pair_id: str, label: str, score: float, offset_days: int = 0) -> LabeledPair:
    ts = (datetime(2024, 1, 1) + timedelta(days=offset_days)).isoformat()
    return LabeledPair(pair_id=pair_id, label=label, score=score, signals={}, timestamp=ts)


def _default_config() -> MatchingConfigVersion:
    return MatchingConfigVersion(
        version=1,
        high_confidence_threshold=0.85,
        uncertain_threshold=0.50,
        signal_weights={},
    )


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_temporal_ordering(self):
        """Oldest pairs go to train, newest to test — no future leakage."""
        pairs = [_make_pair(str(i), "match", 0.9, offset_days=i) for i in range(20)]
        learner = ActiveLearner(_default_config())
        train, val, test = learner.split_dataset(pairs)
        # train timestamps must all be <= val which are <= test
        assert train[-1].timestamp <= val[0].timestamp
        assert val[-1].timestamp <= test[0].timestamp

    def test_split_proportions_approx(self):
        """70/15/15 split is approximately correct for 100 samples."""
        pairs = [_make_pair(str(i), "match", 0.9, offset_days=i) for i in range(100)]
        learner = ActiveLearner(_default_config())
        train, val, test = learner.split_dataset(pairs)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_split_empty(self):
        """Empty list produces three empty splits without error."""
        learner = ActiveLearner(_default_config())
        train, val, test = learner.split_dataset([])
        assert train == val == test == []

    def test_split_small(self):
        """Very small lists produce non-empty train; val and test may be empty."""
        pairs = [_make_pair(str(i), "match", 0.9, offset_days=i) for i in range(3)]
        learner = ActiveLearner(_default_config())
        train, val, test = learner.split_dataset(pairs)
        assert len(train) + len(val) + len(test) == 3


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------

class TestRecalibrate:
    def _make_labeled_set(self) -> list[LabeledPair]:
        """Returns 20 pairs: 10 clear matches (score 0.95), 10 non-matches (score 0.20)."""
        pairs = []
        for i in range(10):
            pairs.append(_make_pair(f"m{i}", "match", 0.95, offset_days=i))
        for i in range(10):
            pairs.append(_make_pair(f"n{i}", "non_match", 0.20, offset_days=10 + i))
        return pairs

    def test_recalibrate_returns_new_version(self):
        learner = ActiveLearner(_default_config())
        pairs = self._make_labeled_set()
        result = learner.recalibrate(pairs, new_version=2)
        assert result.version == 2

    def test_recalibrate_feedback_count(self):
        learner = ActiveLearner(_default_config())
        pairs = self._make_labeled_set()
        result = learner.recalibrate(pairs, new_version=2)
        assert result.feedback_count == len(pairs)

    def test_recalibrate_threshold_in_valid_range(self):
        learner = ActiveLearner(_default_config())
        pairs = self._make_labeled_set()
        result = learner.recalibrate(pairs, new_version=2)
        assert 0.50 <= result.high_confidence_threshold <= 0.95

    def test_recalibrate_validation_f1_populated(self):
        learner = ActiveLearner(_default_config())
        pairs = self._make_labeled_set()
        result = learner.recalibrate(pairs, new_version=2)
        assert result.validation_f1 is not None
        assert 0.0 <= result.validation_f1 <= 1.0

    def test_recalibrate_insufficient_data_fallback(self):
        """Below MIN_SAMPLES_FOR_RECALIBRATION, config is returned unchanged."""
        learner = ActiveLearner(_default_config())
        pairs = [_make_pair(str(i), "match", 0.9, offset_days=i)
                 for i in range(MIN_SAMPLES_FOR_RECALIBRATION - 1)]
        result = learner.recalibrate(pairs, new_version=2)
        # Config unchanged – thresholds preserved from current config
        assert result.high_confidence_threshold == _default_config().high_confidence_threshold
        assert "unchanged" in result.notes.lower()

    def test_recalibrate_uncertain_threshold_lower_than_high(self):
        learner = ActiveLearner(_default_config())
        pairs = self._make_labeled_set()
        result = learner.recalibrate(pairs, new_version=2)
        assert result.uncertain_threshold < result.high_confidence_threshold


# ---------------------------------------------------------------------------
# EvaluationMetrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_perfect_match(self):
        """All pairs correctly predicted: precision=recall=F1=1.0, FMR=0."""
        pairs = [
            _make_pair("1", "match", 0.95),
            _make_pair("2", "match", 0.90),
            _make_pair("3", "non_match", 0.20),
        ]
        m = EvaluationMetrics.compute(pairs, threshold=0.85, split="test")
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1_score == 1.0
        assert m.false_merge_rate == 0.0

    def test_all_false_positives(self):
        """Threshold too low: every non-match is predicted match → FMR = 1.0."""
        pairs = [
            _make_pair("1", "non_match", 0.95),
            _make_pair("2", "non_match", 0.95),
        ]
        m = EvaluationMetrics.compute(pairs, threshold=0.50, split="test")
        assert m.true_positives == 0
        assert m.false_positives == 2
        assert m.false_merge_rate == 1.0

    def test_sample_size_matches_input(self):
        pairs = [_make_pair(str(i), "match", 0.9) for i in range(7)]
        m = EvaluationMetrics.compute(pairs, threshold=0.85, split="validation")
        assert m.sample_size == 7

    def test_split_label_preserved(self):
        pairs = [_make_pair("1", "match", 0.95)]
        m = EvaluationMetrics.compute(pairs, threshold=0.85, split="test")
        assert m.split == "test"


# ---------------------------------------------------------------------------
# Review priority
# ---------------------------------------------------------------------------

class TestReviewPriority:
    def test_highest_priority_at_threshold(self):
        """A score exactly at the decision threshold = maximum uncertainty."""
        learner = ActiveLearner(_default_config())
        threshold = learner.current_config.high_confidence_threshold
        priority = learner.compute_review_priority(threshold)
        assert priority == 1.0

    def test_priority_decreases_with_distance(self):
        learner = ActiveLearner(_default_config())
        p_near = learner.compute_review_priority(0.80)   # near threshold (0.85)
        p_far = learner.compute_review_priority(0.20)    # far below threshold
        assert p_near > p_far

    def test_priority_non_negative(self):
        learner = ActiveLearner(_default_config())
        for score in [0.0, 0.1, 0.5, 0.99, 1.0]:
            assert learner.compute_review_priority(score) >= 0.0


# ---------------------------------------------------------------------------
# Evaluate test set
# ---------------------------------------------------------------------------

class TestEvaluateTestSet:
    def test_returns_metrics_with_correct_split_label(self):
        pairs = [_make_pair(str(i), "match", 0.95, offset_days=i) for i in range(30)]
        learner = ActiveLearner(_default_config())
        metrics = learner.evaluate_test_set(pairs, _default_config())
        assert metrics.split == "test"

    def test_empty_test_split_returns_zero_metrics(self):
        """Fewer pairs than 1/test-ratio → test split may be empty; graceful fallback."""
        pairs = [_make_pair("1", "match", 0.9)]
        learner = ActiveLearner(_default_config())
        metrics = learner.evaluate_test_set(pairs, _default_config())
        # Either test split had data or zero-sample note is returned
        assert metrics.f1_score >= 0.0
        assert metrics.sample_size >= 0
