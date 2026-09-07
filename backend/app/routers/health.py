"""
backend/app/routers/health.py
==============================
GET /api/v1/health  —  Docker health-check endpoint
GET /api/v1/model/info  —  Active pipeline metadata
"""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.schemas import HealthResponse, ModelInfoResponse
from backend.constants import PIPELINE_BITCOIN, SCORE_DISCLAIMER

router = APIRouter(prefix="/api/v1", tags=["system"])

# Path to the Bitcoin GCN checkpoint — used to derive last_updated timestamp
_GCN_CKPT = Path("models/gcn_bitcoin_v1.pt")


def _model_last_updated() -> datetime:
    """Return mtime of the GCN checkpoint, or epoch if not yet trained."""
    if _GCN_CKPT.exists():
        return datetime.fromtimestamp(_GCN_CKPT.stat().st_mtime, tz=UTC)
    return datetime(2000, 1, 1, tzinfo=UTC)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description=(
        "Returns 200 when the API process is alive. "
        "Database connectivity is checked separately via the db service health check."
    ),
)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model_version=PIPELINE_BITCOIN,
        last_updated=_model_last_updated(),
        offline_mode=settings.offline_mode,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Active model pipeline metadata",
    description=(
        "Returns metadata distinguishing Pipeline A (Elliptic, methodology validation) "
        "from Pipeline B (Bitcoin GCN, risk ranking). "
        "The score_disclaimer field must be displayed alongside any risk score in the UI."
    ),
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=PIPELINE_BITCOIN,
        training_dataset=(
            "Weak supervision on Bitcoin snapshot "
            "(co-spend seeds: OFAC/sanctions list — demo data only)"
        ),
        validation_dataset="Elliptic (methodology validation only)",
        inference_dataset=(
            "Bitcoin snapshot — see /api/v1/health last_updated for snapshot date."
        ),
        model_status="demo",
        score_disclaimer=SCORE_DISCLAIMER,
        last_updated=_model_last_updated(),
    )
