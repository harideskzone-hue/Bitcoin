"""
ml/tests/test_p2p_integration.py
==================================
HP4.1 — P2P fusion integration tests

Verifies all four acceptance criteria from the frozen spec:

  1. 20 records loaded from data/p2p_signals_simulated.json
  2. All 20 addresses have p2p_signals in API response (via annotations)
  3. signal_source == "SIMULATED" for every record
  4. No IPv4-like strings in any signal field
"""

import json
import re
from pathlib import Path

import pytest

SIGNALS_PATH     = Path("data/p2p_signals_simulated.json")
ANNOTATIONS_PATH = Path("data/p2p_node_annotations.json")
IP_PATTERN       = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def signals() -> list[dict]:
    """Load the flat signal records. Skip all tests if file doesn't exist."""
    if not SIGNALS_PATH.exists():
        pytest.skip(f"P2P signals file not found: {SIGNALS_PATH} — run generate_p2p_signals.py")
    return json.loads(SIGNALS_PATH.read_text())


@pytest.fixture(scope="module")
def annotations() -> dict[str, list[dict]]:
    """Load address → [signal] mapping."""
    if not ANNOTATIONS_PATH.exists():
        pytest.skip(f"P2P annotations file not found: {ANNOTATIONS_PATH}")
    return json.loads(ANNOTATIONS_PATH.read_text())


# ── HP4 Acceptance Test 1: 20 records ─────────────────────────────────────────

class TestSignalCount:
    def test_exactly_20_signals(self, signals):
        """P0.7.1 / HP4.1: File must have exactly 20 signal records."""
        assert len(signals) == 20, (
            f"Expected 20 signals, got {len(signals)}. "
            "Re-run generate_p2p_signals.py."
        )

    def test_all_records_have_required_fields(self, signals):
        required = {"signal_source", "ip_cluster_id", "timestamp",
                    "address_hash", "tx_hash", "peer_count", "disclaimer"}
        for i, sig in enumerate(signals):
            missing = required - set(sig.keys())
            assert not missing, f"Signal[{i}] missing fields: {missing}"


# ── HP4 Acceptance Test 2: 20 addresses annotated ─────────────────────────────

class TestAnnotationCoverage:
    def test_annotations_cover_20_addresses(self, annotations, signals):
        """HP4.2: All signal addresses must appear in annotations."""
        signal_addresses = {s["address_hash"] for s in signals}
        annotated        = set(annotations.keys())
        uncovered = signal_addresses - annotated
        assert not uncovered, (
            f"{len(uncovered)} addresses have signals but no annotation: {list(uncovered)[:5]}"
        )

    def test_annotation_entries_have_required_fields(self, annotations):
        required = {"signal_source", "ip_cluster_id", "timestamp", "tx_hash", "disclaimer"}
        for addr, sigs in annotations.items():
            for i, sig in enumerate(sigs):
                missing = required - set(sig.keys())
                assert not missing, f"Annotation for {addr}[{i}] missing fields: {missing}"

    def test_total_annotation_count_matches_signals(self, annotations, signals):
        """Total annotated signal entries must equal the number of flat signals."""
        total_annotated = sum(len(v) for v in annotations.values())
        assert total_annotated == len(signals), (
            f"Total annotated entries ({total_annotated}) != signal count ({len(signals)})"
        )


# ── HP4 Acceptance Test 3: signal_source == "SIMULATED" always ────────────────

class TestSimulatedLabel:
    def test_all_flat_signals_are_simulated(self, signals):
        """HP4.3: signal_source must be 'SIMULATED' for every record."""
        non_simulated = [
            (i, s["signal_source"])
            for i, s in enumerate(signals)
            if s.get("signal_source") != "SIMULATED"
        ]
        assert not non_simulated, (
            f"Found non-SIMULATED signal_source values: {non_simulated}"
        )

    def test_all_annotated_signals_are_simulated(self, annotations):
        """HP4.3: signal_source must be 'SIMULATED' in the annotations map too."""
        non_simulated = [
            (addr, i, sig.get("signal_source"))
            for addr, sigs in annotations.items()
            for i, sig in enumerate(sigs)
            if sig.get("signal_source") != "SIMULATED"
        ]
        assert not non_simulated, (
            f"Found non-SIMULATED entries in annotations: {non_simulated[:3]}"
        )

    def test_disclaimer_present_on_all_signals(self, signals):
        """Disclaimer field must be non-empty on every record."""
        missing_disclaimer = [
            i for i, s in enumerate(signals)
            if not s.get("disclaimer")
        ]
        assert not missing_disclaimer, (
            f"Missing disclaimer on signals at indices: {missing_disclaimer}"
        )

    def test_disclaimer_contains_simulated_wording(self, signals):
        """Disclaimer must include 'SIMULATED' so it's unambiguous."""
        bad = [
            (i, s["disclaimer"]) for i, s in enumerate(signals)
            if "SIMULATED" not in s.get("disclaimer", "").upper()
        ]
        assert not bad, f"Disclaimer missing 'SIMULATED' wording: {bad[:3]}"


# ── HP4 Acceptance Test 4: no real IP strings ─────────────────────────────────

class TestNoRealIPStrings:
    def test_no_ip_in_flat_signals_file(self, signals):
        """P0.7.3 / HP4.4: No IPv4-pattern strings anywhere in the signals JSON."""
        payload = json.dumps(signals)
        matches = IP_PATTERN.findall(payload)
        assert not matches, (
            f"Found IP-like strings in p2p_signals_simulated.json: {matches}"
        )

    def test_no_ip_in_annotations_file(self, annotations):
        """HP4.4: No IPv4-pattern strings anywhere in the annotations JSON."""
        payload = json.dumps(annotations)
        matches = IP_PATTERN.findall(payload)
        assert not matches, (
            f"Found IP-like strings in p2p_node_annotations.json: {matches}"
        )

    def test_ip_cluster_ids_use_correct_format(self, signals):
        """All ip_cluster_id values must match IP_CLUSTER_XX pattern."""
        cluster_pattern = re.compile(r'^IP_CLUSTER_\d{2}$')
        bad = [
            (i, s["ip_cluster_id"]) for i, s in enumerate(signals)
            if not cluster_pattern.match(s.get("ip_cluster_id", ""))
        ]
        assert not bad, f"Invalid ip_cluster_id format: {bad}"

    def test_no_ip_in_tx_hash(self, signals):
        """tx_hash must not look like a real IP-containing identifier."""
        for i, sig in enumerate(signals):
            tx = sig.get("tx_hash", "")
            assert not IP_PATTERN.search(tx), (
                f"Signal[{i}] tx_hash contains IP-like string: {tx}"
            )


# ── Simulated mode integration: API schema shape ───────────────────────────────

class TestAPISchemaShape:
    def test_p2p_signal_fields_match_api_schema(self, annotations):
        """
        The annotation entries must contain all fields required by the P2PSignal
        Pydantic schema: signal_source, ip_cluster_id, timestamp, tx_hash, disclaimer.
        """
        required_api_fields = {"signal_source", "ip_cluster_id", "timestamp",
                               "tx_hash", "disclaimer"}
        for addr, sigs in list(annotations.items())[:5]:  # spot-check first 5
            for sig in sigs:
                missing = required_api_fields - set(sig.keys())
                assert not missing, (
                    f"API schema mismatch for {addr}: missing {missing}"
                )

    def test_peer_count_is_positive_integer(self, signals):
        """peer_count must be a positive integer (not a string or float)."""
        for i, sig in enumerate(signals):
            pc = sig.get("peer_count")
            assert isinstance(pc, int) and pc > 0, (
                f"Signal[{i}] peer_count invalid: {pc!r}"
            )
