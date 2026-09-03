"""
backend/app/routers/wallets.py
================================
GET /api/v1/wallets/{address_hash}  —  Full risk profile for one address

Returns the complete risk profile including score, reasons, graph-derived
features, cluster membership, and simulated P2P signals (always null until P0.7).
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.schemas import WalletDetailResponse
from backend.constants import SCORE_DISCLAIMER, score_to_label

router = APIRouter(prefix="/api/v1", tags=["wallets"])

_RISK_SCORES_PATH = Path("data/btc_risk_scores.parquet")
_FEATURES_PATH    = Path("data/address_features_10k.parquet")
_CLUSTERS_PATH    = Path("data/btc_clusters.parquet")
_REASONS_PATH     = Path("data/btc_reasons.parquet")
_CHANGE_ADDR_PATH = Path("data/btc_change_candidates.parquet")  # written by P0.4.3


def _load_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"{label} not yet available at {path}. Run the relevant pipeline step.",
        )
    return pd.read_parquet(path)


@router.get(
    "/wallets/{address_hash}",
    response_model=WalletDetailResponse,
    summary="Full risk profile for a single address",
    description=(
        "Returns model-derived risk score (0–100), risk label, up to 3 explanation "
        "reasons, graph-derived feature values, co-spend cluster membership, and "
        "P2P network signals (always null in demo mode). "
        "score_disclaimer is always present and must be displayed in the UI."
    ),
)
def get_wallet(address_hash: str) -> WalletDetailResponse:
    # ── Risk score ─────────────────────────────────────────────────────────────
    if not _RISK_SCORES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk scores not yet available. "
                "Run: python ml/scripts/train_bitcoin_gcn.py (requires 50k snapshot)"
            ),
        )
    scores_df = pd.read_parquet(_RISK_SCORES_PATH)
    row = scores_df[scores_df["address_hash"] == address_hash]

    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Address '{address_hash}' not found in the current snapshot.",
        )

    risk_score = int(round(float(row.iloc[0]["risk_score"])))
    risk_score = max(0, min(100, risk_score))
    risk_label = score_to_label(risk_score)

    # ── Reasons (Layer 1, pending P0.6) ────────────────────────────────────────
    reasons: list[str]
    if _REASONS_PATH.exists():
        r_df    = pd.read_parquet(_REASONS_PATH)
        r_rows  = r_df[r_df["address_hash"] == address_hash].sort_values("rank")
        reasons = r_rows["reason"].tolist()[:3]
    else:
        reasons = ["Reason tokens pending explainability run (P0.6)"]

    # ── Address features (F-series from FEATURE_SPEC.md) ──────────────────────
    tx_burst_score    = 0.0
    timing_anomaly    = 0.0
    if _FEATURES_PATH.exists():
        feat_df = pd.read_parquet(_FEATURES_PATH)
        f_row   = feat_df[feat_df["address_hash"] == address_hash]
        if not f_row.empty:
            tx_burst_score = float(f_row.iloc[0].get("tx_burst_score", 0.0))
            timing_anomaly = float(f_row.iloc[0].get("weekday_vs_weekend_ratio", 0.0))

    # ── Cluster membership ─────────────────────────────────────────────────────
    cluster_id: str | None         = None
    cluster_confidence: str | None = None
    if _CLUSTERS_PATH.exists():
        cl_df = pd.read_parquet(_CLUSTERS_PATH)
        cl_row = cl_df[cl_df["address"] == address_hash]
        if not cl_row.empty:
            cluster_id = str(cl_row.iloc[0]["cluster_id"])
            # Confidence heuristic: cluster_size → LOW/MEDIUM/HIGH
            # (final confidence logic in P0.4.2 DB write step)
            cluster_confidence = "LOW"  # placeholder until DB write

    # ── Change-address candidate ────────────────────────────────────────────────
    candidate_change = False
    if _CHANGE_ADDR_PATH.exists():
        ca_df = pd.read_parquet(_CHANGE_ADDR_PATH)
        candidate_change = address_hash in ca_df["address_hash"].values

    # ── Hops to nearest flagged ─────────────────────────────────────────────────
    # Computed by P0.6 Layer 1; placeholder None until explainability runs.
    hops_to_flagged: int | None = None

    return WalletDetailResponse(
        address=address_hash,
        risk_score=risk_score,
        risk_label=risk_label,
        reasons=reasons,
        score_disclaimer=SCORE_DISCLAIMER,
        hops_to_nearest_flagged=hops_to_flagged,
        tx_burst_score=tx_burst_score,
        timing_anomaly_score=timing_anomaly,
        cluster_id=cluster_id,
        cluster_confidence=cluster_confidence,
        candidate_change_address=candidate_change,
        p2p_signals=None,   # Always null until P0.7 is implemented
    )
