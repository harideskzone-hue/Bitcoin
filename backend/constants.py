"""
backend/constants.py
====================
Single source of truth for shared constants.

Both the API response builder AND the frontend must derive risk labels from
these thresholds — never hard-code them inline anywhere else.

See: SIH26146_Final_Backlog_v5.1_FROZEN.md — API contract, RISK_LABEL_THRESHOLDS
"""

# ── Risk-label thresholds ─────────────────────────────────────────────────────
# These are PRESENTATION categories for prioritization consistency.
# They are NOT probability thresholds, do not represent estimated illicit-activity
# prevalence, and do not constitute an operational or enforcement decision rule.
# The sigmoid score distribution is not necessarily uniform, so score bands do
# not map directly to address-count percentiles.
# Final operational thresholds must be selected using investigator-reviewed
# agency data in Phase 2.

RISK_LABEL_THRESHOLDS: dict[str, tuple[int, int]] = {
    "LOW":      (0,  24),   # informational; monitor
    "MEDIUM":   (25, 49),   # elevated; review when capacity allows
    "HIGH":     (50, 74),   # prioritised; review within 24h
    "CRITICAL": (75, 100),  # urgent; review immediately
}

RISK_LABEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def score_to_label(risk_score: int) -> str:
    """
    Convert a 0–100 model-derived risk ranking to a presentation label.

    Args:
        risk_score: Integer in [0, 100]. Produced by Pipeline B (Bitcoin GCN).

    Returns:
        One of "LOW", "MEDIUM", "HIGH", "CRITICAL".

    Raises:
        ValueError: if risk_score is outside [0, 100].

    Note:
        This is a RANKING label, not a probability label.
        A score of 75 does NOT mean a 75% probability of illicit activity.
    """
    if not (0 <= risk_score <= 100):
        raise ValueError(f"risk_score must be in [0, 100], got {risk_score}")
    for label, (lo, hi) in RISK_LABEL_THRESHOLDS.items():
        if lo <= risk_score <= hi:
            return label
    # Should never reach here given the ranges cover 0–100 exhaustively
    raise ValueError(f"No label found for risk_score={risk_score}")  # pragma: no cover


# ── Score disclaimer ──────────────────────────────────────────────────────────
# This string must appear in every API response that contains a risk_score.
# See P0.4.7 acceptance test.
SCORE_DISCLAIMER = (
    "Model-derived risk ranking. For prioritization and human review only. "
    "Not a calibrated probability."
)

# ── Pipeline identifiers ──────────────────────────────────────────────────────
PIPELINE_ELLIPTIC = "elliptic_gcn_v1"   # Supervised, methodology validation
PIPELINE_BITCOIN  = "bitcoin_gcn_v1"    # Weakly supervised, risk ranking only

# ── Aliases (for backward compatibility and import ergonomics) ────────────────
THRESHOLDS = RISK_LABEL_THRESHOLDS  # spec-required name in correction #28

# ── Risk label colours ────────────────────────────────────────────────────────
# These match the CSS design tokens in frontend/src/constants.ts.
# Both must be updated together if colours change.
RISK_LABEL_COLOURS: dict[str, str] = {
    "LOW":      "#22c55e",
    "MEDIUM":   "#eab308",
    "HIGH":     "#f97316",
    "CRITICAL": "#ef4444",
}
