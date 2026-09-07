"""
ml/tests/test_deterministic.py
================================
P0.6.1 acceptance tests — one unit test per deterministic rule.

Each test verifies:
  - Rule fires (returns a non-None string) when its condition is met
  - Rule does NOT fire when its condition is not met
  - explain_address() returns at most 3 reasons
  - explain_address() returns no duplicates
"""

import pytest

from ml.explain.deterministic import (
    BURST_THRESHOLD,
    HOPS_THRESHOLD,
    RAPID_TX_THRESHOLD,
    REUSE_THRESHOLD,
    GraphContext,
    explain_address,
    rule_address_reuse,
    rule_change_address,
    rule_cluster_flagged_member,
    rule_hops_to_flagged,
    rule_rapid_consecutive_tx,
    rule_tx_burst,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_ctx():
    return GraphContext()


@pytest.fixture
def ctx_with_hops():
    return GraphContext(
        hops_map={"abc123": 1, "far_away": 5, "unreachable": None},
        flagged_clusters={"CLUSTER_1"},
        cluster_sizes={"CLUSTER_1": 4, "CLUSTER_2": 1},
    )


def _row(**kwargs) -> dict:
    """Build a minimal address feature row."""
    defaults = {
        "address_hash":           "abc123",
        "tx_count":               1,
        "tx_burst_score":         0.0,
        "address_reuse_count":    0,
        "avg_time_between_tx_hrs": 10.0,
        "candidate_change_address": False,
        "cluster_id":             None,
        "weak_label":             "unknown",
    }
    defaults.update(kwargs)
    return defaults


# ── Rule 1: hops_to_flagged ────────────────────────────────────────────────────

class TestRuleHopsToFlagged:
    def test_fires_within_threshold(self, ctx_with_hops):
        row = _row(address_hash="abc123")       # hops=1
        assert rule_hops_to_flagged(row, ctx_with_hops) is not None

    def test_fires_at_exact_threshold(self):
        ctx = GraphContext(hops_map={"x": HOPS_THRESHOLD})
        assert rule_hops_to_flagged(_row(address_hash="x"), ctx) is not None

    def test_does_not_fire_beyond_threshold(self, ctx_with_hops):
        row = _row(address_hash="far_away")     # hops=5
        assert rule_hops_to_flagged(row, ctx_with_hops) is None

    def test_does_not_fire_when_unreachable(self, ctx_with_hops):
        row = _row(address_hash="unreachable")  # hops=None
        assert rule_hops_to_flagged(row, ctx_with_hops) is None

    def test_does_not_fire_when_not_in_map(self, empty_ctx):
        assert rule_hops_to_flagged(_row(address_hash="unknown_addr"), empty_ctx) is None

    def test_reason_string_contains_hop_count(self, ctx_with_hops):
        reason = rule_hops_to_flagged(_row(address_hash="abc123"), ctx_with_hops)
        assert "1" in reason
        assert "hop" in reason.lower()


# ── Rule 2: tx_burst ──────────────────────────────────────────────────────────

class TestRuleTxBurst:
    def test_fires_above_threshold(self, empty_ctx):
        row = _row(tx_burst_score=BURST_THRESHOLD + 0.1)
        assert rule_tx_burst(row, empty_ctx) is not None

    def test_does_not_fire_at_threshold(self, empty_ctx):
        row = _row(tx_burst_score=BURST_THRESHOLD)
        assert rule_tx_burst(row, empty_ctx) is None

    def test_does_not_fire_below_threshold(self, empty_ctx):
        row = _row(tx_burst_score=0.5)
        assert rule_tx_burst(row, empty_ctx) is None

    def test_reason_string(self, empty_ctx):
        reason = rule_tx_burst(_row(tx_burst_score=BURST_THRESHOLD + 1), empty_ctx)
        assert "burst" in reason.lower() or "volume" in reason.lower()


# ── Rule 3: address_reuse ─────────────────────────────────────────────────────

class TestRuleAddressReuse:
    def test_fires_above_threshold(self, empty_ctx):
        row = _row(address_reuse_count=REUSE_THRESHOLD + 1)
        assert rule_address_reuse(row, empty_ctx) is not None

    def test_does_not_fire_at_threshold(self, empty_ctx):
        row = _row(address_reuse_count=REUSE_THRESHOLD)
        assert rule_address_reuse(row, empty_ctx) is None

    def test_does_not_fire_at_zero(self, empty_ctx):
        assert rule_address_reuse(_row(address_reuse_count=0), empty_ctx) is None


# ── Rule 4: cluster_flagged_member ────────────────────────────────────────────

class TestRuleClusterFlaggedMember:
    def test_fires_when_in_flagged_cluster_of_size_2(self):
        ctx = GraphContext(
            flagged_clusters={"CLUSTER_1"},
            cluster_sizes={"CLUSTER_1": 2},
        )
        row = _row(cluster_id="CLUSTER_1")
        assert rule_cluster_flagged_member(row, ctx) is not None

    def test_does_not_fire_for_singleton_cluster(self):
        ctx = GraphContext(
            flagged_clusters={"CLUSTER_1"},
            cluster_sizes={"CLUSTER_1": 1},
        )
        row = _row(cluster_id="CLUSTER_1")
        assert rule_cluster_flagged_member(row, ctx) is None

    def test_does_not_fire_for_unflagged_cluster(self):
        ctx = GraphContext(
            flagged_clusters={"CLUSTER_1"},
            cluster_sizes={"CLUSTER_2": 3},
        )
        row = _row(cluster_id="CLUSTER_2")
        assert rule_cluster_flagged_member(row, ctx) is None

    def test_does_not_fire_without_cluster(self, empty_ctx):
        assert rule_cluster_flagged_member(_row(cluster_id=None), empty_ctx) is None


# ── Rule 5: change_address ────────────────────────────────────────────────────

class TestRuleChangeAddress:
    def test_fires_when_candidate(self, empty_ctx):
        row = _row(candidate_change_address=True)
        assert rule_change_address(row, empty_ctx) is not None

    def test_does_not_fire_when_not_candidate(self, empty_ctx):
        row = _row(candidate_change_address=False)
        assert rule_change_address(row, empty_ctx) is None

    def test_reason_string(self, empty_ctx):
        reason = rule_change_address(_row(candidate_change_address=True), empty_ctx)
        assert "change" in reason.lower()


# ── Rule 6: rapid_consecutive_tx ─────────────────────────────────────────────

class TestRuleRapidConsecutiveTx:
    def test_fires_when_rapid_and_multiple_tx(self, empty_ctx):
        row = _row(tx_count=5, avg_time_between_tx_hrs=RAPID_TX_THRESHOLD - 0.01)
        assert rule_rapid_consecutive_tx(row, empty_ctx) is not None

    def test_does_not_fire_when_slow(self, empty_ctx):
        row = _row(tx_count=5, avg_time_between_tx_hrs=RAPID_TX_THRESHOLD + 1.0)
        assert rule_rapid_consecutive_tx(row, empty_ctx) is None

    def test_does_not_fire_with_only_one_tx(self, empty_ctx):
        # avg_time_between_tx_hrs is meaningless with 1 transaction
        row = _row(tx_count=1, avg_time_between_tx_hrs=0.0)
        assert rule_rapid_consecutive_tx(row, empty_ctx) is None


# ── explain_address() integration tests ───────────────────────────────────────

class TestExplainAddress:
    def test_returns_at_most_3_reasons(self):
        """API contract: max 3 reasons."""
        ctx = GraphContext(
            hops_map={"a": 1},
            flagged_clusters={"C1"},
            cluster_sizes={"C1": 3},
        )
        row = _row(
            address_hash="a",
            tx_burst_score=BURST_THRESHOLD + 1,
            address_reuse_count=REUSE_THRESHOLD + 1,
            avg_time_between_tx_hrs=0.0,
            tx_count=5,
            candidate_change_address=True,
            cluster_id="C1",
        )
        reasons = explain_address(row, ctx, max_reasons=3)
        assert len(reasons) <= 3

    def test_no_duplicate_reasons(self):
        ctx = GraphContext(hops_map={"a": 1}, flagged_clusters=set(), cluster_sizes={})
        row = _row(address_hash="a", tx_burst_score=BURST_THRESHOLD + 1)
        reasons = explain_address(row, ctx)
        assert len(reasons) == len(set(reasons))

    def test_empty_when_no_rules_fire(self, empty_ctx):
        row = _row()  # all defaults below thresholds
        reasons = explain_address(row, empty_ctx)
        assert reasons == []

    def test_returns_list(self, empty_ctx):
        reasons = explain_address(_row(), empty_ctx)
        assert isinstance(reasons, list)
