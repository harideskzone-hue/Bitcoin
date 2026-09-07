"""
ml/tests/test_constants.py
===========================
Verify that backend/constants.py is correct.

These tests confirm:
  1. Every integer 0–100 maps to exactly one label.
  2. The threshold boundaries are exact (24→LOW, 25→MEDIUM, etc.)
  3. score_to_label raises on out-of-range input.
  4. The SCORE_DISCLAIMER string is present and correct.
"""

import os
import sys

import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.constants import RISK_LABEL_THRESHOLDS, SCORE_DISCLAIMER, score_to_label


class TestScoreToLabel:
    """Verify every boundary of the four risk-label bands."""

    def test_low_lower_bound(self):
        assert score_to_label(0) == "LOW"

    def test_low_upper_bound(self):
        assert score_to_label(24) == "LOW"

    def test_medium_lower_bound(self):
        assert score_to_label(25) == "MEDIUM"

    def test_medium_upper_bound(self):
        assert score_to_label(49) == "MEDIUM"

    def test_high_lower_bound(self):
        assert score_to_label(50) == "HIGH"

    def test_high_upper_bound(self):
        assert score_to_label(74) == "HIGH"

    def test_critical_lower_bound(self):
        assert score_to_label(75) == "CRITICAL"

    def test_critical_upper_bound(self):
        assert score_to_label(100) == "CRITICAL"

    def test_exhaustive_coverage(self):
        """Every integer 0–100 must map to exactly one label with no gaps."""
        labels = {score_to_label(s) for s in range(101)}
        assert labels == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_all_scores_map(self):
        """No score in 0–100 should raise."""
        for s in range(101):
            label = score_to_label(s)
            assert label in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}, \
                f"Unexpected label '{label}' for score {s}"

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="risk_score must be in"):
            score_to_label(-1)

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="risk_score must be in"):
            score_to_label(101)


class TestScoreDisclaimer:
    """The disclaimer string must contain the key phrases from the plan."""

    def test_disclaimer_not_probability(self):
        assert "Not a calibrated probability" in SCORE_DISCLAIMER

    def test_disclaimer_prioritization(self):
        assert "prioritization" in SCORE_DISCLAIMER.lower()

    def test_disclaimer_human_review(self):
        assert "human review" in SCORE_DISCLAIMER.lower()


class TestThresholdCompleteness:
    """The threshold table itself must cover [0, 100] with no gaps or overlaps."""

    def test_covers_full_range(self):
        covered = set()
        for lo, hi in RISK_LABEL_THRESHOLDS.values():
            for s in range(lo, hi + 1):
                covered.add(s)
        assert covered == set(range(101)), "Threshold bands do not cover 0–100 exhaustively"

    def test_no_overlaps(self):
        all_scores = []
        for lo, hi in RISK_LABEL_THRESHOLDS.values():
            all_scores.extend(range(lo, hi + 1))
        assert len(all_scores) == len(set(all_scores)), "Threshold bands overlap"
