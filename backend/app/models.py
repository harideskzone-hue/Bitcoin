"""
backend/app/models.py
=====================
SQLAlchemy ORM models for all four tables required by P0.1.5.

Tables:
  address_clusters  — co-spend cluster membership and confidence
  node_scores       — model-derived risk rankings (0–100) per address
  node_reasons      — deterministic explanation tokens per address
  audit_log         — immutable record of every API query (for HITL + compliance)
"""

import datetime
from sqlalchemy import (
    Boolean, DateTime, Enum, Integer,
    String, Text, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database import Base

# ── Shared enum types ─────────────────────────────────────────────────────────

ClusterConfidence = Enum("HIGH", "MEDIUM", "LOW", name="cluster_confidence")
RiskLabel         = Enum("LOW", "MEDIUM", "HIGH", "CRITICAL", name="risk_label")
ScoreType         = Enum("raw_ranking", name="score_type")


# ── Table 1: address_clusters ─────────────────────────────────────────────────

class AddressCluster(Base):
    """
    Co-spend cluster membership for each address.

    Every address that participates in a co-spend transaction gets a cluster_id.
    Confidence reflects the strength of the co-spend evidence (not probability
    of illicit activity).

    Created by: P0.4.2 (co-spend clustering, Union-Find)
    """
    __tablename__ = "address_clusters"

    id:              Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    address_hash:    Mapped[str]      = mapped_column(String(128), nullable=False, index=True)
    cluster_id:      Mapped[str]      = mapped_column(String(64),  nullable=False, index=True)
    confidence:      Mapped[str]      = mapped_column(ClusterConfidence, nullable=False)
    # True if this cluster contains ≥1 known-bad seed address
    contains_seed:   Mapped[bool]     = mapped_column(Boolean, default=False, nullable=False)
    snapshot_tag:    Mapped[str]      = mapped_column(String(32), nullable=False)  # e.g. "50k_v1"
    created_at:      Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ── Table 2: node_scores ──────────────────────────────────────────────────────

class NodeScore(Base):
    """
    Model-derived risk ranking for each address node.

    risk_score is a 0–100 INTEGER derived from the raw GCN sigmoid output:
        risk_score = round(sigmoid(logit) * 100)

    It is NOT a calibrated probability. It is a ranking/prioritization score.
    score_type = "raw_ranking" makes this explicit at the data-layer.

    Created by: P0.4.6 (risk ranking computation)
    """
    __tablename__ = "node_scores"

    id:           Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    address_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    risk_score:   Mapped[int] = mapped_column(Integer,  nullable=False)   # 0–100
    risk_label:   Mapped[str] = mapped_column(RiskLabel, nullable=False)  # LOW/MEDIUM/HIGH/CRITICAL
    score_type:   Mapped[str] = mapped_column(ScoreType, nullable=False, default="raw_ranking")
    pipeline:     Mapped[str] = mapped_column(String(64), nullable=False) # "bitcoin_gcn_v1"
    snapshot_tag: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at:   Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ── Table 3: node_reasons ─────────────────────────────────────────────────────

class NodeReason(Base):
    """
    Deterministic explanation tokens for each address score.

    Each row is one reason token for one address. A single address can have
    up to 3 reason rows (the API returns max 3 reasons, per the API contract).

    Layer 1 (deterministic) reasons are always computed.
    Layer 2 (GNNExplainer) reasons are optional and only for top-5 addresses.

    Created by: P0.6 (two-layer explainability)
    """
    __tablename__ = "node_reasons"

    id:            Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    address_hash:  Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    reason:        Mapped[str] = mapped_column(Text, nullable=False)
    rank:          Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = primary, 2, 3
    layer:         Mapped[str] = mapped_column(String(16), nullable=False)  # "deterministic" | "gnnexplainer"
    snapshot_tag:  Mapped[str] = mapped_column(String(32), nullable=False)
    created_at:    Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ── Table 4: audit_log ────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Immutable audit record of every API request that returns risk scores.

    Required by the plan for HITL workflow (P2.3) and compliance (Track B).
    Every query that returns score data must produce exactly one audit_log row.

    This table is append-only — no UPDATE or DELETE operations are permitted.
    """
    __tablename__ = "audit_log"

    id:            Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    queried_hash:  Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    endpoint:      Mapped[str] = mapped_column(String(128), nullable=False)
    # Investigator identity — placeholder for Phase 2 MFA integration
    investigator:  Mapped[str | None] = mapped_column(String(128), nullable=True)
    response_code: Mapped[int]        = mapped_column(Integer, nullable=False)
    queried_at:    Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
