"""
Active Learning Module for Entity Resolution & Matching Model Calibration.

Provides:
  - Temporal dataset partitioning: 70% train (oldest first to prevent leakage),
    15% validation (threshold tuning), 15% test (held-out final evaluation).
  - Threshold recalibration: searches threshold space to maximize validation F1.
  - Evaluation metrics: true positives, false positives, true negatives,
    false negatives, precision, recall, F1 score, false-merge rate.
  - Prioritized review queue based on uncertainty and information gain
    (samples closest to decision boundary evaluated first).
  - Versioned matching configurations with rollback support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


MIN_SAMPLES_FOR_RECALIBRATION = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


@dataclass
class LabeledPair:
    """A reviewer-labeled pair with signal features."""
    pair_id: str
    label: str          # "match" | "non_match"
    score: float        # overall heuristic score [0.0 - 1.0]
    signals: Dict[str, float]
    timestamp: str      # ISO timestamp for temporal ordering


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for entity resolution."""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_merge_rate: float
    threshold_used: float
    sample_size: int
    split: str          # "train" | "validation" | "test"
    notes: str = ""

    @classmethod
    def compute(
        cls,
        pairs: List[LabeledPair],
        threshold: float,
        split: str = "validation",
    ) -> "EvaluationMetrics":
        tp = fp = tn = fn = 0
        for p in pairs:
            predicted_match = p.score >= threshold
            actual_match = p.label == "match"
            if predicted_match and actual_match:
                tp += 1
            elif predicted_match and not actual_match:
                fp += 1
            elif not predicted_match and actual_match:
                fn += 1
            else:
                tn += 1

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall) / max(1e-9, precision + recall)
        fmr = fp / max(1, tp + fp)

        return cls(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_merge_rate=round(fmr, 4),
            threshold_used=threshold,
            sample_size=len(pairs),
            split=split,
        )


@dataclass
class MatchingConfigVersion:
    version: int
    high_confidence_threshold: float
    uncertain_threshold: float
    signal_weights: Dict[str, float]
    training_f1: Optional[float] = None
    validation_f1: Optional[float] = None
    feedback_count: int = 0
    notes: str = ""


class ActiveLearner:
    """
    Active Learning Manager.
    Recalibrates matching thresholds from feedback and prioritizes the review queue.
    """

    def __init__(self, current_config: Optional[MatchingConfigVersion] = None):
        self.current_config = current_config or MatchingConfigVersion(
            version=1,
            high_confidence_threshold=0.85,
            uncertain_threshold=0.50,
            signal_weights={},
        )

    def split_dataset(
        self, pairs: List[LabeledPair]
    ) -> Tuple[List[LabeledPair], List[LabeledPair], List[LabeledPair]]:
        """
        Partitions pairs temporally (oldest to newest) into train/validation/test
        to avoid future-to-past data leakage.
        """
        sorted_pairs = sorted(pairs, key=lambda p: p.timestamp)
        n = len(sorted_pairs)
        train_end = math.floor(n * TRAIN_RATIO)
        val_end = train_end + math.floor(n * VAL_RATIO)
        return sorted_pairs[:train_end], sorted_pairs[train_end:val_end], sorted_pairs[val_end:]

    def recalibrate(
        self, labeled_pairs: List[LabeledPair], new_version: int
    ) -> MatchingConfigVersion:
        """
        Recalibrates matching thresholds based on labeled feedback.
        Uses grid-search on the validation split to find the threshold maximizing F1.
        """
        if len(labeled_pairs) < MIN_SAMPLES_FOR_RECALIBRATION:
            return MatchingConfigVersion(
                version=new_version,
                high_confidence_threshold=self.current_config.high_confidence_threshold,
                uncertain_threshold=self.current_config.uncertain_threshold,
                signal_weights=self.current_config.signal_weights,
                feedback_count=len(labeled_pairs),
                notes=f"Sample size ({len(labeled_pairs)}) below minimum ({MIN_SAMPLES_FOR_RECALIBRATION}). Config unchanged.",
            )

        train, val, test = self.split_dataset(labeled_pairs)
        if not val:
            val = train  # fallback for small sets

        best_threshold = self.current_config.high_confidence_threshold
        best_val_f1 = -1.0
        candidate_thresholds = [round(0.55 + i * 0.05, 2) for i in range(8)]  # 0.55 - 0.90

        for thresh in candidate_thresholds:
            metrics = EvaluationMetrics.compute(val, threshold=thresh, split="validation")
            if metrics.f1_score > best_val_f1:
                best_val_f1 = metrics.f1_score
                best_threshold = thresh

        train_metrics = EvaluationMetrics.compute(train, threshold=best_threshold, split="train")

        return MatchingConfigVersion(
            version=new_version,
            high_confidence_threshold=best_threshold,
            uncertain_threshold=max(0.30, round(best_threshold - 0.30, 2)),
            signal_weights=self.current_config.signal_weights,
            training_f1=train_metrics.f1_score,
            validation_f1=best_val_f1,
            feedback_count=len(labeled_pairs),
            notes=f"Recalibrated on {len(train)} train & {len(val)} val pairs. Optimal Val F1={best_val_f1:.3f} at threshold={best_threshold}",
        )

    def evaluate_test_set(
        self, labeled_pairs: List[LabeledPair], config: MatchingConfigVersion
    ) -> EvaluationMetrics:
        """
        Evaluates the matching configuration strictly on the held-out test split.
        """
        _, _, test = self.split_dataset(labeled_pairs)
        if not test:
            return EvaluationMetrics(
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                false_merge_rate=0.0,
                threshold_used=config.high_confidence_threshold,
                sample_size=0,
                split="test",
                notes="Test split contains zero samples.",
            )
        return EvaluationMetrics.compute(
            test, threshold=config.high_confidence_threshold, split="test"
        )

    def compute_review_priority(self, score: float) -> float:
        """
        Information-gain priority for active learning review queue.
        Pairs located closest to the decision threshold offer the highest learning value.
        Returns a score in [0.0, 1.0], where 1.0 = highest review priority.
        """
        threshold = self.current_config.high_confidence_threshold
        distance = abs(score - threshold)
        priority = max(0.0, 1.0 - (distance * 2.0))
        return round(priority, 4)
