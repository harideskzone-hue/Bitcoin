"""
backend/app/schemas.py
=======================
Pydantic v2 request/response schemas for all API endpoints.

These are the canonical type definitions for the frozen API contract
(P0.5 / SIH26146_Final_Backlog_v5.1_FROZEN.md § P0.5).

Every schema that carries a risk score also carries score_disclaimer.
The disclaimer text is sourced from backend/constants.py — never hard-coded inline.

Endpoint map:
  GET /api/v1/health                  → HealthResponse
  GET /api/v1/model/info              → ModelInfoResponse
  GET /api/v1/alerts                  → AlertListResponse  (query: AlertQuery)
  GET /api/v1/wallets/{address_hash}  → WalletDetailResponse
  GET /api/v1/graph                   → GraphResponse       (query: GraphQuery)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.constants import SCORE_DISCLAIMER


# ── P0.4.7 — Disclaimer injected at schema level ──────────────────────────────
# Any schema that returns risk_score must inherit ScoreCarrier or explicitly
# add score_disclaimer. This makes it structurally impossible to omit.

class ScoreCarrier(BaseModel):
    """Mixin: adds score_disclaimer to every schema that returns a risk score."""
    score_disclaimer: str = Field(
        default=SCORE_DISCLAIMER,
        description=(
            "Hard-coded disclaimer that must accompany every risk score. "
            "NOT a calibrated probability. For human review and prioritization only."
        ),
        examples=[SCORE_DISCLAIMER],
    )


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """
    GET /api/v1/health

    Docker health-check endpoint. Returns 200 when the API process is alive.
    """
    status:        str      = Field(..., examples=["ok"])
    model_version: str      = Field(..., examples=["gcn_btc_v1"])
    last_updated:  datetime = Field(..., description="Timestamp of last model artifact write")
    offline_mode:  bool     = Field(
        ...,
        description=(
            "True when running against local parquet snapshots instead of BigQuery. "
            "Always True during the hackathon demo."
        ),
    )


# ── Model info ─────────────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    """
    GET /api/v1/model/info

    Returns metadata about the active inference pipeline.
    Distinguishes Pipeline A (Elliptic, methodology validation) from
    Pipeline B (Bitcoin GCN, risk ranking).
    """
    model_config = {"protected_namespaces": ()}   # suppress model_ namespace warning

    model_version:       str = Field(..., examples=["gcn_btc_v1"])
    training_dataset:    str = Field(
        ...,
        examples=["Weak supervision on Bitcoin snapshot (co-spend seeds)"],
    )
    validation_dataset:  str = Field(
        ...,
        examples=["Elliptic benchmark (methodology validation only)"],
    )
    inference_dataset:   str = Field(
        ...,
        examples=["Bitcoin snapshot 50k txns, 2026-08-01 to 2026-09-01"],
    )
    model_status: Literal["demo", "production", "training", "unavailable"] = Field(
        ...,
        description=(
            "'demo' means the model is running in demonstration mode "
            "with weak-supervision labels. Not suitable for enforcement decisions."
        ),
    )
    score_disclaimer: str = Field(default=SCORE_DISCLAIMER)
    last_updated:     datetime


# ── Shared sub-schemas ─────────────────────────────────────────────────────────

class P2PSignal(BaseModel):
    """
    Simulated P2P network signal (P0.7).
    signal_source is always 'SIMULATED' — this is a demonstration artefact.
    """
    signal_source: Literal["SIMULATED"] = "SIMULATED"
    ip_cluster_id: str     = Field(..., examples=["IP_CLUSTER_07"])
    timestamp:     datetime
    tx_hash:       str     = Field(..., examples=["a0b05f87c91a36014d07..."])
    disclaimer:    Literal["Simulated/replayed demonstration data only."] = (
        "Simulated/replayed demonstration data only."
    )


class WalletAlert(BaseModel):
    """
    Single high-risk address record returned in /alerts.
    Used inside AlertListResponse.
    """
    address:            str = Field(..., description="SHA-256[:16] address hash")
    risk_score:         int = Field(..., ge=0, le=100)
    risk_label:         Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    top_reason:         str = Field(
        ...,
        description="Primary deterministic reason from Layer 1 explainability",
        examples=["Within 2 hops of a flagged address"],
    )
    reasons:            list[str] = Field(
        ...,
        max_length=3,
        description="Up to 3 explanation tokens (Layer 1 deterministic, Layer 2 optional)",
    )
    cluster_id:         Optional[str]  = Field(None, examples=["CLUSTER_042"])
    cluster_confidence: Optional[Literal["HIGH", "MEDIUM", "LOW"]] = None


# ── Alerts ─────────────────────────────────────────────────────────────────────

class AlertQuery(BaseModel):
    """Query parameters for GET /api/v1/alerts."""
    min_risk: int = Field(
        default=70,
        ge=0,
        le=100,
        description="Minimum risk_score threshold (inclusive). Default 70.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results. Default 20, max 100.",
    )


class AlertListResponse(ScoreCarrier):
    """
    GET /api/v1/alerts

    Returns addresses with risk_score ≥ min_risk, sorted descending.
    """
    alerts: list[WalletAlert]
    total:  int = Field(..., description="Total matching addresses (before limit)")


# ── Wallet detail ──────────────────────────────────────────────────────────────

class WalletDetailResponse(ScoreCarrier):
    """
    GET /api/v1/wallets/{address_hash}

    Full risk profile for a single address.
    """
    address:              str
    risk_score:           int  = Field(..., ge=0, le=100)
    risk_label:           Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    reasons:              list[str] = Field(..., max_length=3)
    score_type:           str = Field("heuristic_fallback", description="'raw_ranking' (GCN) or 'heuristic_fallback'")

    # Graph-derived features (from P0.2.2 / FEATURE_SPEC.md)
    hops_to_nearest_flagged:  Optional[int]   = Field(
        None,
        description="Shortest-path distance to nearest known-flagged cluster. "
                    "null when no flagged cluster is reachable within the snapshot.",
    )
    tx_burst_score:           float = Field(
        ...,
        description="Feature F11 from FEATURE_SPEC.md — ratio of tx count to active hours.",
    )
    timing_anomaly_score:     float = Field(
        ...,
        description="Feature F12 — weekday vs weekend transaction ratio.",
    )

    # Co-spend cluster
    cluster_id:           Optional[str]                          = None
    cluster_confidence:   Optional[Literal["HIGH", "MEDIUM", "LOW"]] = None
    candidate_change_address: bool = Field(
        ...,
        description=(
            "True if this address was identified as a likely change-address "
            "by the P0.4.3 heuristic (single-input, two-output, fresh address, "
            "smaller output value). This is a heuristic inference, not a certainty."
        ),
    )

    # P2P simulation data (P0.7)
    p2p_signals: Optional[list[P2PSignal]] = Field(
        None,
        description="Simulated P2P signals. Always null until P0.7 is implemented.",
    )

    # Temporal risk profile (Track 4 — hour-of-day analysis)
    temporal_profile: Optional[dict] = Field(
        None,
        description=(
            "Hour-of-day transaction activity profile. Keys: "
            "avg_time_between_tx_hrs, tx_burst_score, weekday_vs_weekend_ratio, "
            "avg_tx_value_btc, total_sent_btc, tx_count, is_offhours_active."
        ),
    )


# ── Graph ──────────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id:         str   = Field(..., description="address_hash or tx_hash")
    risk_score: Optional[int]  = Field(None, ge=0, le=100,
        description="null for transaction nodes")
    risk_label: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None
    node_type:  Literal["address", "transaction"]


class GraphEdge(BaseModel):
    source:     str
    target:     str
    edge_type:  Literal["INPUT_TO", "OUTPUT_TO"]
    value_btc:  Optional[float] = None
    timestamp:  Optional[datetime] = None


class GraphQuery(BaseModel):
    """Query parameters for GET /api/v1/graph."""
    address: str = Field(..., description="address_hash of the focal node")
    depth:   Literal[1, 2] = Field(
        default=1,
        description="Hop depth. 1 = direct neighbours, 2 = 2-hop neighbourhood.",
    )


class GraphResponse(BaseModel):
    """
    GET /api/v1/graph

    Returns the ego-graph (up to `depth` hops) around the queried address.
    Both address and transaction nodes are included.
    """
    nodes:      list[GraphNode]
    edges:      list[GraphEdge]
    disclaimer: str = Field(default=SCORE_DISCLAIMER)
