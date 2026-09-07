"""
ml/tests/test_features.py
==========================
Unit tests for all 14 Bitcoin address features defined in data/FEATURE_SPEC.md.

Uses the mini-graph fixture specified in FEATURE_SPEC.md — Acceptance Test Fixtures:

  addr_A, addr_B, addr_C
  tx_1  Mon 02:00 UTC — addr_A inputs 1.0 BTC, outputs 0.9→addr_B, 0.1→addr_A (reuse)
  tx_2  Mon 03:00 UTC — addr_B inputs 0.9 BTC, outputs 0.85→addr_C
  tx_3  Tue 14:00 UTC — addr_A inputs 0.1 BTC, outputs 0.09→addr_C

  snapshot_now = ts(tx_3)

All expected values are hand-computed from the formulas in FEATURE_SPEC.md.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.build_graph import (
    SATOSHI,
    _compute_address_features,
    _compute_projection_features,
    _hash_string,
)

# ── Mini-graph fixture ─────────────────────────────────────────────────────────

def _ts(day: str, hour: int) -> datetime:
    """Return a UTC-aware datetime for the fixture."""
    # tx_1/tx_2 = Monday 2026-09-01, tx_3 = Tuesday 2026-09-02
    dates = {"Mon": "2026-09-01", "Tue": "2026-09-02"}
    return pd.Timestamp(f"{dates[day]} {hour:02d}:00:00", tz="UTC")


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    """
    Build the mini-graph DataFrame matching FEATURE_SPEC.md fixture exactly.

    Each row = one (input_address, tx, output_address) triple.
    In the real pipeline, this comes from a BigQuery join.
    """
    rows = [
        # tx_1: addr_A inputs 1.0 BTC; outputs 0.9→addr_B, 0.1→addr_A
        {
            "tx_hash": "tx_1", "block_timestamp": _ts("Mon", 2),
            "input_address": "addr_A", "input_value_sat": int(1.0 * SATOSHI),
            "output_address": "addr_B", "output_value_sat": int(0.9 * SATOSHI),
            "output_type": "pubkeyhash", "output_index": 0,
            "fee_satoshi": 0, "input_count": 1, "output_count": 2,
            "total_input_sat": int(1.0 * SATOSHI), "total_output_sat": int(1.0 * SATOSHI),
            "is_coinbase": False,
        },
        {
            "tx_hash": "tx_1", "block_timestamp": _ts("Mon", 2),
            "input_address": "addr_A", "input_value_sat": int(1.0 * SATOSHI),
            "output_address": "addr_A", "output_value_sat": int(0.1 * SATOSHI),  # ← reuse
            "output_type": "pubkeyhash", "output_index": 1,
            "fee_satoshi": 0, "input_count": 1, "output_count": 2,
            "total_input_sat": int(1.0 * SATOSHI), "total_output_sat": int(1.0 * SATOSHI),
            "is_coinbase": False,
        },
        # tx_2: addr_B inputs 0.9 BTC; output 0.85→addr_C
        {
            "tx_hash": "tx_2", "block_timestamp": _ts("Mon", 3),
            "input_address": "addr_B", "input_value_sat": int(0.9 * SATOSHI),
            "output_address": "addr_C", "output_value_sat": int(0.85 * SATOSHI),
            "output_type": "pubkeyhash", "output_index": 0,
            "fee_satoshi": 0, "input_count": 1, "output_count": 1,
            "total_input_sat": int(0.9 * SATOSHI), "total_output_sat": int(0.85 * SATOSHI),
            "is_coinbase": False,
        },
        # tx_3: addr_A inputs 0.1 BTC; output 0.09→addr_C
        {
            "tx_hash": "tx_3", "block_timestamp": _ts("Tue", 14),
            "input_address": "addr_A", "input_value_sat": int(0.1 * SATOSHI),
            "output_address": "addr_C", "output_value_sat": int(0.09 * SATOSHI),
            "output_type": "pubkeyhash", "output_index": 0,
            "fee_satoshi": 0, "input_count": 1, "output_count": 1,
            "total_input_sat": int(0.1 * SATOSHI), "total_output_sat": int(0.09 * SATOSHI),
            "is_coinbase": False,
        },
    ]
    df = pd.DataFrame(rows)
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)
    return df


@pytest.fixture
def snapshot_now(fixture_df) -> pd.Timestamp:
    return fixture_df["block_timestamp"].max()  # = tx_3 timestamp = Tue 14:00 UTC


@pytest.fixture
def features(fixture_df, snapshot_now) -> pd.DataFrame:
    """Run the feature engineering pipeline on the mini fixture."""
    return _compute_address_features(fixture_df, snapshot_now)


@pytest.fixture
def features_with_proj(fixture_df, snapshot_now) -> pd.DataFrame:
    """Features including projection-based F1 and F7."""
    from scripts.build_graph import build_bipartite_and_projection
    feat = _compute_address_features(fixture_df, snapshot_now)
    _, proj = build_bipartite_and_projection(fixture_df, feat)
    return _compute_projection_features(feat, proj)


def _get(feat: pd.DataFrame, addr: str, col: str) -> float:
    """Retrieve a feature value for a raw (unhashed) address."""
    return feat.loc[_hash_string(addr), col]


# ── F4: tx_count ──────────────────────────────────────────────────────────────

class TestTxCount:
    """addr_A appears in tx_1 + tx_3 = 2; addr_B in tx_1 + tx_2 = 2; addr_C in tx_2 + tx_3 = 2"""

    def test_addr_a(self, features):
        assert _get(features, "addr_A", "tx_count") == 2

    def test_addr_b(self, features):
        assert _get(features, "addr_B", "tx_count") == 2

    def test_addr_c(self, features):
        assert _get(features, "addr_C", "tx_count") == 2


# ── F2: total_sent_btc ────────────────────────────────────────────────────────

class TestTotalSentBtc:
    """
    F2: sum of ALL output values for txs where address is an input.
    addr_A: tx_1 outputs = 0.9 + 0.1 = 1.0 BTC, tx_3 outputs = 0.09 BTC → total = 1.09 BTC
    addr_B: tx_2 outputs = 0.85 BTC → total = 0.85 BTC
    addr_C: never an input → 0.0 BTC
    """

    def test_addr_a(self, features):
        # tx_1: addr_A input, outputs sum = 1.0; tx_3: addr_A input, output = 0.09
        assert abs(_get(features, "addr_A", "total_sent_btc") - 1.09) < 1e-6

    def test_addr_b(self, features):
        assert abs(_get(features, "addr_B", "total_sent_btc") - 0.85) < 1e-6

    def test_addr_c(self, features):
        assert _get(features, "addr_C", "total_sent_btc") == 0.0


# ── F3: total_received_btc ────────────────────────────────────────────────────

class TestTotalReceivedBtc:
    """
    addr_A: receives 0.1 BTC change in tx_1 → 0.1 BTC
    addr_B: receives 0.9 BTC in tx_1 → 0.9 BTC
    addr_C: receives 0.85 in tx_2 + 0.09 in tx_3 → 0.94 BTC
    """

    def test_addr_a(self, features):
        assert abs(_get(features, "addr_A", "total_received_btc") - 0.1) < 1e-6

    def test_addr_b(self, features):
        assert abs(_get(features, "addr_B", "total_received_btc") - 0.9) < 1e-6

    def test_addr_c(self, features):
        assert abs(_get(features, "addr_C", "total_received_btc") - 0.94) < 1e-6


# ── F8: input_count ───────────────────────────────────────────────────────────

class TestInputCount:
    """Number of txs where address appears as an input."""

    def test_addr_a(self, features):
        assert _get(features, "addr_A", "input_count") == 2  # tx_1 and tx_3

    def test_addr_b(self, features):
        assert _get(features, "addr_B", "input_count") == 1  # tx_2 only

    def test_addr_c(self, features):
        assert _get(features, "addr_C", "input_count") == 0  # never an input


# ── F9: output_count ──────────────────────────────────────────────────────────

class TestOutputCount:
    """Number of txs where address appears as an output."""

    def test_addr_a(self, features):
        assert _get(features, "addr_A", "output_count") == 1  # tx_1 change output

    def test_addr_b(self, features):
        assert _get(features, "addr_B", "output_count") == 1  # tx_1

    def test_addr_c(self, features):
        assert _get(features, "addr_C", "output_count") == 2  # tx_2 and tx_3


# ── F11: address_reuse_count ─────────────────────────────────────────────────

class TestAddressReuseCount:
    """
    reuse(a) = |T_in(a) ∩ T_out(a)| — txs where addr is BOTH input AND output.
    addr_A: tx_1 (input AND output via change) → reuse = 1
    addr_B: tx_1 (output only), tx_2 (input only) → reuse = 0
    addr_C: never an input → reuse = 0
    """

    def test_addr_a_reuse_is_one(self, features):
        assert _get(features, "addr_A", "address_reuse_count") == 1

    def test_addr_b_no_reuse(self, features):
        assert _get(features, "addr_B", "address_reuse_count") == 0

    def test_addr_c_no_reuse(self, features):
        assert _get(features, "addr_C", "address_reuse_count") == 0


# ── F14: avg_time_between_tx_hrs ─────────────────────────────────────────────

class TestAvgTimeBetweenTx:
    """
    addr_A: tx_1 Mon 02:00, tx_3 Tue 14:00 → Δ = 36h → avg_ibi = 36.0h
    addr_B: tx_1 Mon 02:00, tx_2 Mon 03:00 → Δ = 1h  → avg_ibi = 1.0h
    addr_C: tx_2 Mon 03:00, tx_3 Tue 14:00 → Δ = 35h → avg_ibi = 35.0h
    """

    def test_addr_a(self, features):
        assert abs(_get(features, "addr_A", "avg_time_between_tx_hrs") - 36.0) < 0.01

    def test_addr_b(self, features):
        assert abs(_get(features, "addr_B", "avg_time_between_tx_hrs") - 1.0) < 0.01

    def test_addr_c(self, features):
        assert abs(_get(features, "addr_C", "avg_time_between_tx_hrs") - 35.0) < 0.01


# ── F6: time_since_last_tx_hrs ───────────────────────────────────────────────

class TestTimeSinceLastTx:
    """
    snapshot_now = Tue 14:00 UTC
    addr_A last tx = tx_3 Tue 14:00 → 0.0h
    addr_B last tx = tx_2 Mon 03:00 → 35h
    addr_C last tx = tx_3 Tue 14:00 → 0.0h
    """

    def test_addr_a(self, features, snapshot_now):
        assert _get(features, "addr_A", "time_since_last_tx_hrs") < 0.01

    def test_addr_b(self, features):
        # tx_2 is Mon 03:00, tx_3 (snapshot_now) is Tue 14:00 → 35h
        assert abs(_get(features, "addr_B", "time_since_last_tx_hrs") - 35.0) < 0.01

    def test_addr_c(self, features, snapshot_now):
        assert _get(features, "addr_C", "time_since_last_tx_hrs") < 0.01


# ── F10: is_script_hash ───────────────────────────────────────────────────────

class TestIsScriptHash:
    """All outputs in the fixture use 'pubkeyhash' → is_script_hash = 0 for all."""

    def test_no_script_hash_in_fixture(self, features):
        for addr in ["addr_A", "addr_B", "addr_C"]:
            assert _get(features, addr, "is_script_hash") == 0


# ── F5: avg_tx_value_btc ─────────────────────────────────────────────────────

class TestAvgTxValueBtc:
    """
    avg_val(a) = (sent + recv) / tx_count
    addr_A: (1.09 + 0.1) / 2 = 0.595
    addr_B: (0.85 + 0.9) / 2 = 0.875
    addr_C: (0.0  + 0.94) / 2 = 0.47
    """

    def test_addr_a(self, features):
        assert abs(_get(features, "addr_A", "avg_tx_value_btc") - 0.595) < 1e-4

    def test_addr_b(self, features):
        assert abs(_get(features, "addr_B", "avg_tx_value_btc") - 0.875) < 1e-4

    def test_addr_c(self, features):
        assert abs(_get(features, "addr_C", "avg_tx_value_btc") - 0.47) < 1e-4


# ── F1 + F7: address_degree and clustering_coefficient ────────────────────────

class TestProjectionFeatures:
    """
    Projection edges (a1≠a2, from shared tx):
      tx_1: addr_A→addr_B, addr_A→addr_A (SELF LOOP — excluded)
      tx_2: addr_B→addr_C
      tx_3: addr_A→addr_C

    After self-loop removal, projection graph:
      addr_A — addr_B (from tx_1)
      addr_A — addr_C (from tx_3)
      addr_B — addr_C (from tx_2)

    address_degree: addr_A=2, addr_B=2, addr_C=2
    clustering_coefficient:
      All three form a complete triangle → C=1.0 for all
    """

    def test_degree_addr_a(self, features_with_proj):
        assert _get(features_with_proj, "addr_A", "address_degree") == 2

    def test_degree_addr_b(self, features_with_proj):
        assert _get(features_with_proj, "addr_B", "address_degree") == 2

    def test_degree_addr_c(self, features_with_proj):
        assert _get(features_with_proj, "addr_C", "address_degree") == 2

    def test_clustering_all_form_triangle(self, features_with_proj):
        # All three addresses are connected to each other → triangle → C=1.0
        for addr in ["addr_A", "addr_B", "addr_C"]:
            cc = _get(features_with_proj, addr, "clustering_coefficient")
            assert abs(cc - 1.0) < 1e-6, f"{addr}: clustering={cc}, expected 1.0"

    def test_no_self_loop_inflates_degree(self, features_with_proj):
        # addr_A has a self-loop candidate from tx_1 (appears as both input and output)
        # The self-loop must be excluded, so addr_A's degree = 2 (not 3)
        assert _get(features_with_proj, "addr_A", "address_degree") == 2


# ── General validity checks ────────────────────────────────────────────────────

class TestFeatureValidity:
    """Sanity checks that apply across all addresses."""

    def test_no_nan_in_any_feature(self, features_with_proj):
        from scripts.build_graph import ADDRESS_FEATURE_COLS
        nan_mask = features_with_proj[ADDRESS_FEATURE_COLS].isna()
        assert not nan_mask.any().any(), \
            f"NaN detected:\n{features_with_proj[ADDRESS_FEATURE_COLS][nan_mask.any(axis=1)]}"

    def test_all_monetary_features_non_negative(self, features_with_proj):
        for col in ["total_sent_btc", "total_received_btc", "avg_tx_value_btc"]:
            assert (features_with_proj[col] >= 0).all(), f"Negative values in {col}"

    def test_is_script_hash_binary(self, features_with_proj):
        vals = features_with_proj["is_script_hash"].unique()
        assert set(vals).issubset({0, 1})

    def test_clustering_in_unit_interval(self, features_with_proj):
        cc = features_with_proj["clustering_coefficient"]
        assert (cc >= 0).all() and (cc <= 1).all()

    def test_address_degree_non_negative(self, features_with_proj):
        assert (features_with_proj["address_degree"] >= 0).all()
