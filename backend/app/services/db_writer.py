"""
backend/app/services/db_writer.py
===================================
HP2.1 — Write GCN risk scores to node_scores table.

Design rules:
  - score_type is ALWAYS "raw_ranking" — hardcoded, never overridable by caller.
  - Clears the snapshot's existing rows before inserting new ones (idempotent).
  - Prints DATABASE_BACKEND to stderr on every call so the demo log is unambiguous.
  - Works identically on PostgreSQL and SQLite (USE_SQLITE_FALLBACK=true).
  - On any DB error, logs the error and returns False — caller decides whether to abort.

Usage
-----
  from backend.app.services.db_writer import write_node_scores
  ok = write_node_scores(scores_df, snapshot_tag="50k_v1", pipeline="bitcoin_gcn_v1")
  if not ok:
      sys.exit(1)  # caller handles fail-closed
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd

from backend.app.database import SessionLocal, database_backend
from backend.app.models import NodeScore
from backend.constants import score_to_label

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger("db_writer")

# ── score_type sentinel — never let callers override this ─────────────────────
_SCORE_TYPE = "raw_ranking"


def write_node_scores(
    scores_df:    pd.DataFrame,
    snapshot_tag: str,
    pipeline:     str = "bitcoin_gcn_v1",
) -> bool:
    """
    Persist GCN risk scores to the node_scores table.

    Args:
        scores_df:    DataFrame with columns: address_hash (str), risk_score (0–100 int).
        snapshot_tag: e.g. "50k_v1" or "10k_v1"
        pipeline:     e.g. "bitcoin_gcn_v1"

    Returns:
        True  — all rows committed successfully.
        False — database error; caller should treat this as a pipeline failure.

    Raises:
        ValueError — if scores_df is missing required columns or contains invalid scores.
    """
    # ── Input validation ───────────────────────────────────────────────────────
    required = {"address_hash", "risk_score"}
    missing  = required - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df missing required columns: {missing}")

    scores_df = scores_df.copy()
    scores_df["risk_score"] = scores_df["risk_score"].round().astype(int).clip(0, 100)
    scores_df["risk_label"] = scores_df["risk_score"].apply(score_to_label)

    n = len(scores_df)
    print(f"DATABASE_BACKEND={database_backend()}", file=sys.stderr)
    log.info(f"Writing {n:,} risk scores (snapshot={snapshot_tag}, pipeline={pipeline}) …")

    db: Session = SessionLocal()
    try:
        # Idempotent: delete existing rows for this snapshot before inserting
        deleted = (
            db.query(NodeScore)
            .filter(NodeScore.snapshot_tag == snapshot_tag)
            .delete(synchronize_session=False)
        )
        if deleted:
            log.info(f"  Cleared {deleted:,} existing rows for snapshot {snapshot_tag!r}")

        # Bulk insert
        rows = [
            NodeScore(
                address_hash=row["address_hash"],
                risk_score=  int(row["risk_score"]),
                risk_label=  row["risk_label"],
                score_type=  _SCORE_TYPE,   # always "raw_ranking"
                pipeline=    pipeline,
                snapshot_tag=snapshot_tag,
            )
            for _, row in scores_df.iterrows()
        ]
        db.add_all(rows)
        db.commit()

        log.info(f"  ✅ {n:,} rows committed to node_scores")
        return True

    except Exception as e:
        db.rollback()
        log.error(f"  ❌ DB write failed: {e}")
        return False

    finally:
        db.close()
