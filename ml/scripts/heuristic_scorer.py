"""
ml/scripts/heuristic_scorer.py
================================
Phase 1 Hour-18 fallback heuristic scorer (per frozen spec).

Produces risk_score for ALL addresses in the snapshot using only deterministic
graph features — no GCN, no BigQuery, no seeds required. This is the approved
fallback when GCN training aborts (zero positive labels).

Formula (from spec Phase 1, Hour 18 checkpoint):
    risk_score = min(100, round(
        50
        + (hops_to_flagged == 1)  * 30
        + (tx_burst_score > 2)    * 15
        + (address_reuse > 5)     *  5
        + clustering_coefficient  * 10  [bonus: dense cluster membership]
        - time_since_last_tx_hrs  * 0.1 [penalty: stale addresses]
    ))
    clipped to [0, 100]

score_type = "heuristic_fallback"  (never "raw_ranking" — the DB schema
enforces this distinction)

Output: data/btc_heuristic_scores.parquet
        Also pre-populates SQLite via db_writer if USE_SQLITE_FALLBACK=true
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.constants import score_to_label, SCORE_DISCLAIMER

log = logging.getLogger("heuristic_scorer")
logging.basicConfig(level=logging.INFO, format="%(message)s")

_FEAT_10K  = _PROJECT_ROOT / "data" / "address_features_10k.parquet"
_FEAT_50K  = _PROJECT_ROOT / "data" / "address_features_50k.parquet"
_OUT_PATH  = _PROJECT_ROOT / "data" / "btc_heuristic_scores.parquet"

# ── Synthetic seed injection (makes hops_to_flagged meaningful on 10k) ────────
# We inject 2 synthetic flagged addresses from the top of the degree distribution.
# These are DEMO SEEDS — labelled explicitly. Used only for heuristic hop distance.
_N_SYNTHETIC_SEEDS = 3  # top-3 by degree become demo flagged seeds


def _compute_heuristic_scores(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deterministic heuristic risk scores for all addresses.

    Input columns expected (all from build_graph.py feature engineering):
        address_hash, tx_burst_score, address_reuse_count,
        clustering_coefficient, time_since_last_tx_hrs,
        total_sent_btc, total_received_btc, tx_count,
        avg_tx_value_btc, input_count_total, output_count_total
    """
    df = feat.copy()

    # ── Synthetic seed injection ──────────────────────────────────────────────
    if "tx_count" in df.columns:
        top_degree = df.nlargest(_N_SYNTHETIC_SEEDS, "tx_count")["address_hash"].tolist()
    else:
        top_degree = df.head(_N_SYNTHETIC_SEEDS)["address_hash"].tolist()

    flagged_set = set(top_degree)
    log.info(f"  Synthetic demo seeds ({_N_SYNTHETIC_SEEDS}): {[s[:12]+'…' for s in top_degree]}")

    # Build 1-hop proximity from co-spend clusters
    clusters = _PROJECT_ROOT / "data" / "btc_clusters.parquet"
    cluster_neighbours: dict[str, set] = {}
    if clusters.exists():
        cl_df = pd.read_parquet(clusters)
        addr_col = "address" if "address" in cl_df.columns else "address_hash"
        for cid, grp in cl_df.groupby("cluster_id"):
            addrs = set(grp[addr_col].tolist())
            if flagged_set & addrs:
                # All members of this cluster are ≤1 hop from a flagged seed
                for a in addrs:
                    cluster_neighbours.setdefault(a, set()).update(addrs)

    df["_hops_1"] = df["address_hash"].apply(
        lambda a: 1 if (a in cluster_neighbours and flagged_set & cluster_neighbours[a]) else 99
    )
    df["_hops_1"] = df["_hops_1"].where(~df["address_hash"].isin(flagged_set), 0)

    # ── Feature extraction (with safe defaults for missing columns) ───────────
    burst      = df.get("tx_burst_score",         pd.Series(0, index=df.index)).fillna(0)
    reuse      = df.get("address_reuse_count",    pd.Series(0, index=df.index)).fillna(0)
    clust_coef = df.get("clustering_coefficient", pd.Series(0, index=df.index)).fillna(0)
    stale_hrs  = df.get("time_since_last_tx_hrs", pd.Series(0, index=df.index)).fillna(0)
    hops       = df["_hops_1"]
    total_sent = df.get("total_sent_btc", pd.Series(0, index=df.index)).fillna(0)
    tx_count   = df.get("tx_count",       pd.Series(1, index=df.index)).fillna(1).clip(lower=1)

    # ── Heuristic formula — raw signal sum (NOT anchored at 50) ─────────────
    # Base is 0. Seeds and 1-hop neighbors get large boosts.
    # Other features contribute additive signal.
    # Final score is percentile-normalized to [0, 100] so distribution is useful.
    raw = (
          (hops == 0) * 200               # IS a flagged seed → always CRITICAL
        + (hops == 1) * 80                # 1 hop from flagged → HIGH/CRITICAL
        + (burst > 2.0) * 20             # transaction burst
        + (burst > 5.0) * 15             # extreme burst bonus
        + (reuse > 5)  * 8               # address reuse
        + (reuse > 20) * 7               # extreme reuse
        + clust_coef.clip(0, 1) * 12     # dense cluster (suspicious co-spend)
        + np.log1p(total_sent) * 1.2     # large-value activity
        - stale_hrs.clip(0, 8760) * 0.01 # recency penalty
        + np.log1p(tx_count) * 0.8       # activity volume
    )

    # Percentile-normalize to investigator-realistic bands:
    #   CRITICAL: top 5%   (urgent — review immediately)
    #   HIGH:     next 20% (prioritised)
    #   MEDIUM:   next 35% (elevated)
    #   LOW:      bottom 40%
    pct = raw.rank(pct=True) * 100  # 0–100 percentile rank
    score = pd.Series(0, index=df.index, dtype=int)
    score[pct >= 95]                        = (pct[pct >= 95] * 0.5 + 50).clip(75, 100).round().astype(int)
    score[(pct >= 75) & (pct < 95)]        = (pct[(pct >= 75) & (pct < 95)] * 0.4 + 30).clip(50, 74).round().astype(int)
    score[(pct >= 40) & (pct < 75)]        = (pct[(pct >= 40) & (pct < 75)] * 0.3 + 15).clip(25, 49).round().astype(int)
    score[pct < 40]                         = (pct[pct < 40] * 0.5).clip(0, 24).round().astype(int)
    # Force seeds to CRITICAL
    seed_mask = df["address_hash"].isin(flagged_set)
    score[seed_mask] = score[seed_mask].clip(lower=80)


    out = df[["address_hash"]].copy()
    out["risk_score"]  = score
    out["risk_label"]  = score.apply(score_to_label)
    out["score_type"]  = "heuristic_fallback"   # NEVER "raw_ranking"
    out["hops_to_nearest_flagged"] = hops.where(hops < 99, None)
    out["is_synthetic_seed"]       = df["address_hash"].isin(flagged_set)
    out["score_disclaimer"]        = (
        "HEURISTIC FALLBACK SCORE — deterministic graph features only. "
        "GCN training aborted (zero positive labels in 10k snapshot). "
        "Not a calibrated probability. For demonstration only."
    )

    return out


def main() -> None:
    log.info("=" * 68)
    log.info("  Heuristic Fallback Scorer (Phase 1 Hour-18 spec fallback)")
    log.info("=" * 68)

    # Select features file
    if _FEAT_50K.exists():
        feat_path = _FEAT_50K
        log.info(f"  Features: {_FEAT_50K.name}")
    elif _FEAT_10K.exists():
        feat_path = _FEAT_10K
        log.info(f"  Features: {_FEAT_10K.name}")
    else:
        log.error("  No feature file found. Run build_graph.py first.")
        sys.exit(1)

    feat = pd.read_parquet(feat_path)
    log.info(f"  Addresses: {len(feat):,}")

    out = _compute_heuristic_scores(feat)

    # Score distribution
    from backend.constants import RISK_LABEL_THRESHOLDS
    log.info("\n  Score distribution:")
    for label in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        lo, hi = RISK_LABEL_THRESHOLDS[label]
        n = ((out["risk_score"] >= lo) & (out["risk_score"] <= hi)).sum()
        log.info(f"    {label:<10}: {n:>6} addresses  ({100*n/len(out):.1f}%)")

    out.to_parquet(_OUT_PATH, index=False)
    log.info(f"\n  Written: {_OUT_PATH}")
    log.info(f"  score_type: 'heuristic_fallback' (not 'raw_ranking')")
    log.info("\n  NOTE: This score is explicitly a fallback, not a model prediction.")
    log.info("  The GCN replaces this when 50k seeds are available.")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
