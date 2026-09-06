"""
backend/app/services/reasons_writer.py
=========================================
HP3.1 — Write Layer 1/2 explanation reasons to node_reasons table.

Design rules:
  - Clears existing rows for snapshot_tag before inserting (idempotent).
  - layer must be one of "deterministic" | "gnnexplainer" — validated at write time.
  - rank must be 1, 2, or 3 — validated at write time (matches DB CHECK constraint).
  - Works identically on PostgreSQL and SQLite.
  - On DB error: logs and returns False. Caller decides whether to abort.

Usage
-----
  from backend.app.services.reasons_writer import write_reasons
  ok = write_reasons(reasons_df, snapshot_tag="50k_v1")
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pandas as pd

from backend.app.database import SessionLocal, database_backend
from backend.app.models import NodeReason

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger("reasons_writer")

_VALID_LAYERS = frozenset({"deterministic", "gnnexplainer"})
_VALID_RANKS  = frozenset({1, 2, 3})


def write_reasons(
    reasons_df:   pd.DataFrame,
    snapshot_tag: str,
) -> bool:
    """
    Persist explanation reason tokens to the node_reasons table.

    Args:
        reasons_df:   DataFrame with columns:
                        address_hash (str)
                        reason       (str)
                        rank         (int, 1–3)
                        layer        (str, "deterministic" | "gnnexplainer")
                      Additional columns (e.g. snapshot_tag) are ignored.
        snapshot_tag: e.g. "50k_v1"

    Returns:
        True  — all rows committed.
        False — DB error; caller should treat as pipeline failure.

    Raises:
        ValueError — invalid layer values or out-of-range rank values.
    """
    # ── Input validation ───────────────────────────────────────────────────────
    required = {"address_hash", "reason", "rank", "layer"}
    missing  = required - set(reasons_df.columns)
    if missing:
        raise ValueError(f"reasons_df missing required columns: {missing}")

    bad_layers = set(reasons_df["layer"].unique()) - _VALID_LAYERS
    if bad_layers:
        raise ValueError(f"Invalid layer values: {bad_layers}. Must be {_VALID_LAYERS}")

    bad_ranks = set(reasons_df["rank"].unique()) - _VALID_RANKS
    if bad_ranks:
        raise ValueError(f"Invalid rank values: {bad_ranks}. Must be 1, 2, or 3.")

    n = len(reasons_df)
    print(f"DATABASE_BACKEND={database_backend()}", file=sys.stderr)
    log.info(f"Writing {n:,} reason tokens (snapshot={snapshot_tag}) …")

    db: Session = SessionLocal()
    try:
        deleted = (
            db.query(NodeReason)
            .filter(NodeReason.snapshot_tag == snapshot_tag)
            .delete(synchronize_session=False)
        )
        if deleted:
            log.info(f"  Cleared {deleted:,} existing reason rows for snapshot {snapshot_tag!r}")

        rows = [
            NodeReason(
                address_hash=row["address_hash"],
                reason=      str(row["reason"]),
                rank=         int(row["rank"]),
                layer=        str(row["layer"]),
                snapshot_tag=snapshot_tag,
            )
            for _, row in reasons_df.iterrows()
        ]
        db.add_all(rows)
        db.commit()

        log.info(f"  ✅ {n:,} reason rows committed to node_reasons")
        return True

    except Exception as e:
        db.rollback()
        log.error(f"  ❌ DB write failed: {e}")
        return False

    finally:
        db.close()
