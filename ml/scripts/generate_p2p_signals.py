"""
ml/scripts/generate_p2p_signals.py
=====================================
P0.7 — Synthetic P2P Simulation Data

Generates 20 synthetic P2P network signals and joins them to snapshot
address nodes. Every field is clearly marked as simulated.

P0.7.1 — 20 entries, all with signal_source = "SIMULATED"
P0.7.2 — Address nodes annotated with p2p_signals list
P0.7.3 — No real IP strings anywhere in the output file

Design constraints (from frozen spec):
  - ip_cluster_id uses "IP_CLUSTER_XX" format, never real IP addresses
  - disclaimer field on every record
  - signal_source is always "SIMULATED"
  - peer_count is a synthetic integer [4, 64]
  - timestamps are within the snapshot window + small jitter

Output:
  data/p2p_signals_simulated.json  — 20 flat signal records
  data/p2p_node_annotations.json   — {address_hash: [signals]} mapping for API
"""

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SNAPSHOT_PATH    = Path("data/btc_snapshot_10k.parquet")
FEATURES_PATH    = Path("data/address_features_10k.parquet")
RISK_SCORES_PATH = Path("data/btc_risk_scores.parquet")
OUT_SIGNALS      = Path("data/p2p_signals_simulated.json")
OUT_ANNOTATIONS  = Path("data/p2p_node_annotations.json")

SEED          = 42
N_SIGNALS     = 20           # frozen: exactly 20
N_IP_CLUSTERS = 12           # IP_CLUSTER_01 … IP_CLUSTER_12
PEER_COUNT_RANGE = (4, 64)

DISCLAIMER = "⚠ SIMULATED DATA — NOT REAL NETWORK TELEMETRY. Simulated/replayed demonstration data only."

# ── Snapshot time window (used to bound synthetic timestamps) ─────────────────
# Fallback if snapshot not available: use a plausible window
DEFAULT_WINDOW_START = datetime(2026, 9, 3, 13, 10, 0, tzinfo=timezone.utc)
DEFAULT_WINDOW_END   = datetime(2026, 9, 3, 13, 41, 0, tzinfo=timezone.utc)


def _load_address_hashes() -> list[str]:
    """Return ordered list of address_hash values from the features parquet."""
    if FEATURES_PATH.exists():
        df = pd.read_parquet(FEATURES_PATH)
        return list(df["address_hash"].values)
    raise FileNotFoundError(
        f"Address features not found: {FEATURES_PATH}\n"
        "Run: python scripts/build_graph.py --offline"
    )


def _load_window() -> tuple[datetime, datetime]:
    """Return (start, end) of snapshot transaction window."""
    if SNAPSHOT_PATH.exists():
        snap = pd.read_parquet(SNAPSHOT_PATH)
        if "block_timestamp" in snap.columns:
            ts = pd.to_datetime(snap["block_timestamp"], utc=True)
            return ts.min().to_pydatetime(), ts.max().to_pydatetime()
    return DEFAULT_WINDOW_START, DEFAULT_WINDOW_END


def _synthetic_tx_hash(seed_str: str) -> str:
    """Deterministic but opaque transaction hash (not a real Bitcoin tx)."""
    raw = hashlib.sha256(f"SIMULATED_TX_{seed_str}".encode()).hexdigest()
    return f"SIM_{raw[:32]}"


def generate_signals(
    address_hashes: list[str],
    window_start:   datetime,
    window_end:     datetime,
    n:              int = N_SIGNALS,
    rng_seed:       int = SEED,
) -> list[dict]:
    """
    Generate n synthetic P2P signal records.

    Selection: prefer high-risk addresses (by risk score) if available,
    otherwise sample uniformly from the address list.
    """
    rng = random.Random(rng_seed)

    # Weight selection toward high-risk addresses if scores available
    weights = None
    if RISK_SCORES_PATH.exists():
        scores_df = pd.read_parquet(RISK_SCORES_PATH)
        score_map = scores_df.set_index("address_hash")["risk_score"].to_dict()
        # Assign score=1 to unlisted addresses (uniform baseline weight)
        raw_weights = [float(score_map.get(ah, 1.0)) + 1.0 for ah in address_hashes]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

    # Sample n addresses (with replacement allowed, matching real P2P behaviour)
    selected = rng.choices(address_hashes, weights=weights, k=n)
    window_seconds = int((window_end - window_start).total_seconds())

    signals = []
    for i, addr_hash in enumerate(selected):
        ip_cluster_num = rng.randint(1, N_IP_CLUSTERS)
        ip_cluster_id  = f"IP_CLUSTER_{ip_cluster_num:02d}"

        # Timestamp within snapshot window ± small jitter (up to 10 min outside)
        jitter_s  = rng.randint(-600, 600)
        offset_s  = rng.randint(0, max(1, window_seconds))
        ts        = window_start + timedelta(seconds=offset_s + jitter_s)
        ts        = ts.replace(microsecond=0)
        timestamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        tx_seed  = f"{i}_{addr_hash}_{ip_cluster_id}"
        tx_hash  = _synthetic_tx_hash(tx_seed)
        peer_count = rng.randint(*PEER_COUNT_RANGE)

        signals.append({
            "signal_source": "SIMULATED",
            "ip_cluster_id": ip_cluster_id,
            "timestamp":     timestamp,
            "address_hash":  addr_hash,
            "tx_hash":       tx_hash,
            "peer_count":    peer_count,
            "disclaimer":    DISCLAIMER,
        })

    return signals


def build_node_annotations(signals: list[dict]) -> dict[str, list[dict]]:
    """
    Build {address_hash: [signal, ...]} mapping for the API's p2p_signals field.
    Each annotation strips address_hash (redundant in the nested structure).
    """
    annotations: dict[str, list[dict]] = {}
    for sig in signals:
        addr = sig["address_hash"]
        entry = {k: v for k, v in sig.items() if k != "address_hash"}
        annotations.setdefault(addr, []).append(entry)
    return annotations


def verify_no_ip_strings(signals: list[dict]) -> None:
    """
    P0.7.3: Assert no IPv4-looking strings appear anywhere in the signal data.
    Pattern: digits.digits.digits.digits
    Raises ValueError if any are found.
    """
    import re
    ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    payload = json.dumps(signals)
    matches = ip_pattern.findall(payload)
    if matches:
        raise ValueError(
            f"P0.7.3 FAILED: IP-like strings found in signal data: {matches}\n"
            "Remove all real IP address strings from the output."
        )
    print("P0.7.3 ✓  No real IP strings found in simulated data")


def main():
    print("P0.7 — Synthetic P2P signal generation")
    print("=" * 50)

    # ── Load addresses ─────────────────────────────────────────────────────────
    address_hashes = _load_address_hashes()
    window_start, window_end = _load_window()
    print(f"  Addresses available : {len(address_hashes):,}")
    print(f"  Window              : {window_start.isoformat()} → {window_end.isoformat()}")

    # ── P0.7.1 — Generate 20 signals ──────────────────────────────────────────
    signals = generate_signals(address_hashes, window_start, window_end,
                               n=N_SIGNALS, rng_seed=SEED)
    assert len(signals) == N_SIGNALS, f"Expected {N_SIGNALS} signals, got {len(signals)}"
    assert all(s["signal_source"] == "SIMULATED" for s in signals), \
        "All signals must have signal_source = 'SIMULATED'"
    print(f"\nP0.7.1 ✓  Generated {len(signals)} signals, all signal_source='SIMULATED'")

    # ── P0.7.3 — Verify no IP strings (run before writing) ────────────────────
    verify_no_ip_strings(signals)

    # ── Write signals file ─────────────────────────────────────────────────────
    OUT_SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SIGNALS, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"  Saved → {OUT_SIGNALS}")

    # ── P0.7.2 — Build node annotations ────────────────────────────────────────
    annotations = build_node_annotations(signals)
    annotated_addresses = len(annotations)
    assert annotated_addresses > 0, "No addresses annotated — something went wrong"
    print(f"\nP0.7.2 ✓  p2p_signals populated for {annotated_addresses} address nodes")

    with open(OUT_ANNOTATIONS, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"  Saved → {OUT_ANNOTATIONS}")

    # ── Summary ────────────────────────────────────────────────────────────────
    cluster_ids = sorted({s["ip_cluster_id"] for s in signals})
    peer_counts  = [s["peer_count"] for s in signals]
    print(f"\nSummary:")
    print(f"  IP clusters used  : {cluster_ids}")
    print(f"  Peer count range  : {min(peer_counts)} – {max(peer_counts)}")
    print(f"  Unique addresses  : {annotated_addresses}")
    print(f"\nP0.7 COMPLETE ✓")


if __name__ == "__main__":
    main()
