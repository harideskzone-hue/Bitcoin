"""
backend/app/routers/alerts.py
==============================
GET /api/v1/alerts  —  High-risk address alert feed

Data source priority (DB-first, parquet fallback):
  1. node_scores table in DB (pre-populated by db_writer + heuristic_scorer)
  2. data/btc_risk_scores.parquet (GCN output, 50k snapshot)
  3. data/btc_heuristic_scores.parquet (Phase 1 Hour-18 fallback, always available)

The score_type field distinguishes GCN scores ("raw_ranking") from heuristic
scores ("heuristic_fallback"). The UI must display the score_type.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import NodeReason, NodeScore
from backend.app.schemas import AlertListResponse, WalletAlert
from backend.constants import SCORE_DISCLAIMER, score_to_label

router = APIRouter(prefix="/api/v1", tags=["alerts"])

_GCN_SCORES_PATH        = Path("data/btc_risk_scores.parquet")
_HEURISTIC_SCORES_PATH  = Path("data/btc_heuristic_scores.parquet")
_REASONS_PATH           = Path("data/btc_reasons.parquet")


# ── Score source helpers ──────────────────────────────────────────────────────

def _scores_from_db(db: Session, min_risk: int, limit: int) -> tuple[list, int]:
    """Query node_scores from DB. Returns (rows, total_above_threshold)."""
    total = db.query(NodeScore).filter(NodeScore.risk_score >= min_risk).count()
    rows = (
        db.query(NodeScore)
        .filter(NodeScore.risk_score >= min_risk)
        .order_by(NodeScore.risk_score.desc())
        .limit(limit)
        .all()
    )
    return rows, total


def _scores_from_parquet(min_risk: int, limit: int) -> tuple[pd.DataFrame, int, str]:
    """Load from parquet files. Returns (df, total, score_type_label)."""
    if _GCN_SCORES_PATH.exists():
        df = pd.read_parquet(_GCN_SCORES_PATH)
        score_type = "GCN (raw_ranking)"
    elif _HEURISTIC_SCORES_PATH.exists():
        df = pd.read_parquet(_HEURISTIC_SCORES_PATH)
        score_type = "heuristic_fallback"
    else:
        return pd.DataFrame(), 0, "none"

    df["risk_score"] = df["risk_score"].round().astype(int).clip(0, 100)
    df["risk_label"] = df["risk_score"].apply(score_to_label)
    subset = df[df["risk_score"] >= min_risk].sort_values("risk_score", ascending=False)
    return subset.head(limit), len(subset), score_type


def _reasons_for(address_hash: str, db: Session) -> list[str]:
    """Get top-3 reason strings for one address (DB → parquet fallback)."""
    rows = (
        db.query(NodeReason)
        .filter(NodeReason.address_hash == address_hash)
        .order_by(NodeReason.rank)
        .limit(3)
        .all()
    )
    if rows:
        return [r.reason for r in rows]
    # Parquet fallback
    if _REASONS_PATH.exists():
        df = pd.read_parquet(_REASONS_PATH)
        hits = df[df["address_hash"] == address_hash].sort_values("rank")
        if not hits.empty:
            return hits["reason"].tolist()[:3]
    return []


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="High-risk address alert feed",
    description=(
        "Returns addresses with risk_score >= min_risk, sorted descending. "
        "Score is a model-derived ranking or heuristic fallback (0–100), NOT a calibrated probability. "
        "score_disclaimer is always present in the response and must be displayed in the UI. "
        "score_type distinguishes 'raw_ranking' (GCN) from 'heuristic_fallback'."
    ),
)
def get_alerts(
    min_risk: Annotated[int, Query(ge=0, le=100, description="Minimum risk score (default 70 per spec)")] = 70,
    limit:    Annotated[int, Query(ge=1, le=100, description="Max results returned")] = 20,
    db: Session = Depends(get_db),
) -> AlertListResponse:

    # ── 1. Try DB first ───────────────────────────────────────────────────────
    try:
        db_rows, total = _scores_from_db(db, min_risk, limit)
    except Exception:
        db_rows, total = [], 0

    if db_rows:
        alerts = []
        for row in db_rows:
            reasons = _reasons_for(row.address_hash, db)
            alerts.append(WalletAlert(
                address=row.address_hash,
                risk_score=int(row.risk_score),
                risk_label=score_to_label(int(row.risk_score)),
                top_reason=reasons[0] if reasons else "Flagged by heuristic risk ranking",
                reasons=reasons or ["Flagged by heuristic risk ranking"],
                cluster_id=None,
                cluster_confidence=None,
            ))
        return AlertListResponse(
            alerts=alerts,
            total=total,
            score_disclaimer=SCORE_DISCLAIMER,
        )

    # ── 2. Parquet fallback ───────────────────────────────────────────────────
    subset, total, _ = _scores_from_parquet(min_risk, limit)
    if isinstance(subset, pd.DataFrame) and len(subset) > 0:
        alerts = []
        for _, row in subset.iterrows():
            reasons = _reasons_for(row["address_hash"], db)
            alerts.append(WalletAlert(
                address=row["address_hash"],
                risk_score=int(row["risk_score"]),
                risk_label=row["risk_label"],
                top_reason=reasons[0] if reasons else "Flagged by heuristic risk ranking",
                reasons=reasons or ["Flagged by heuristic risk ranking"],
                cluster_id=None,
                cluster_confidence=None,
            ))
        return AlertListResponse(
            alerts=alerts,
            total=total,
            score_disclaimer=SCORE_DISCLAIMER,
        )

    # ── 3. Empty (nothing scored yet) ────────────────────────────────────────
    return AlertListResponse(
        alerts=[],
        total=0,
        score_disclaimer=(
            SCORE_DISCLAIMER +
            " [No scores available — run: python ml/scripts/heuristic_scorer.py]"
        ),
    )
