"""
backend/app/routers/alerts.py
==============================
GET /api/v1/alerts  —  High-risk address alert feed

Returns addresses with risk_score >= min_risk, sorted by descending score.
In offline mode, reads from the btc_risk_scores parquet produced by P0.4.6.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import AlertListResponse, WalletAlert
from backend.constants import SCORE_DISCLAIMER, score_to_label

router = APIRouter(prefix="/api/v1", tags=["alerts"])

_RISK_SCORES_PATH = Path("data/btc_risk_scores.parquet")
_REASONS_PATH     = Path("data/btc_reasons.parquet")  # written by P0.6


def _load_scores() -> pd.DataFrame:
    """Load risk scores from offline parquet. Raises 503 when not yet generated."""
    if not _RISK_SCORES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk scores not yet available. "
                "The Bitcoin GCN model requires the 50k snapshot with seed-derived labels. "
                "Run: python ml/scripts/train_bitcoin_gcn.py"
            ),
        )
    df = pd.read_parquet(_RISK_SCORES_PATH)
    df["risk_score"] = df["risk_score"].round().astype(int).clip(0, 100)
    df["risk_label"] = df["risk_score"].apply(score_to_label)
    return df


def _get_reasons(address_hash: str) -> list[str]:
    """Load Layer-1 deterministic reasons for one address (P0.6). Empty if not yet run."""
    if not _REASONS_PATH.exists():
        return ["Reason tokens pending explainability run (P0.6)"]
    df = pd.read_parquet(_REASONS_PATH)
    rows = df[df["address_hash"] == address_hash].sort_values("rank")
    return rows["reason"].tolist()[:3] or ["No reason tokens available"]


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="High-risk address alert feed",
    description=(
        "Returns addresses with risk_score >= min_risk, sorted descending. "
        "Score is a model-derived ranking (0–100), NOT a calibrated probability. "
        "score_disclaimer is always present in the response and must be displayed in the UI."
    ),
)
def get_alerts(
    min_risk: Annotated[int, Query(ge=0, le=100, description="Minimum risk score (inclusive)")] = 70,
    limit:    Annotated[int, Query(ge=1, le=100, description="Max results returned")] = 20,
) -> AlertListResponse:
    df     = _load_scores()
    subset = df[df["risk_score"] >= min_risk].sort_values("risk_score", ascending=False)
    total  = len(subset)
    subset = subset.head(limit)

    alerts = []
    for _, row in subset.iterrows():
        reasons = _get_reasons(row["address_hash"])
        alerts.append(
            WalletAlert(
                address=row["address_hash"],
                risk_score=int(row["risk_score"]),
                risk_label=row["risk_label"],
                top_reason=reasons[0] if reasons else "—",
                reasons=reasons,
                cluster_id=None,   # populated after P0.4.2 DB write
                cluster_confidence=None,
            )
        )

    return AlertListResponse(
        alerts=alerts,
        total=total,
        score_disclaimer=SCORE_DISCLAIMER,
    )
