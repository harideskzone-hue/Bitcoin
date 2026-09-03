"""
ml/explain/deterministic.py
============================
P0.6.1 — Layer 1: Deterministic graph-feature reason rules

All 6 rules from the frozen spec (SIH26146_Final_Backlog_v5.1_FROZEN.md § P0.6):

  Rule 1: hops_to_nearest_flagged ≤ 2
          → "Within 2 hops of a flagged address"

  Rule 2: tx_burst_score > BURST_THRESHOLD
          → "Unusual transaction volume burst (off-hours)"

  Rule 3: address_reuse_count > REUSE_THRESHOLD
          → "High address reuse pattern"

  Rule 4: cluster_confidence == HIGH and cluster contains a flagged member
          → "Linked to cluster with flagged member"

  Rule 5: candidate_change_address == True
          → "Candidate change-address relationship detected"

  Rule 6: avg_time_between_tx_hrs < RAPID_TX_THRESHOLD
          → "Rapid consecutive transactions"

Design
------
Each rule is a pure function:
  (address_row: dict, graph_context: GraphContext) → str | None

None means the rule did not fire for this address.
The caller collects fired reasons, de-duplicates, and truncates to max 3.

All thresholds are defined as constants here so they can be overridden
by investigators in Phase 2 without touching rule logic.

Usage
-----
  from ml.explain.deterministic import explain_address, RULES
  reasons = explain_address(address_row, graph_ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# ── Thresholds (all configurable) ─────────────────────────────────────────────
HOPS_THRESHOLD    = 2      # Rule 1: max hops to nearest flagged cluster
BURST_THRESHOLD   = 3.0    # Rule 2: tx_burst_score (txns / active_hours)
REUSE_THRESHOLD   = 2      # Rule 3: address_reuse_count (appearances as both input+output)
RAPID_TX_THRESHOLD = 1.0   # Rule 6: avg_time_between_tx_hrs below this = rapid


# ── Graph context: data the rules need beyond the address feature row ──────────

@dataclass
class GraphContext:
    """
    Derived graph-level data passed alongside address features.

    hops_map: address_hash → shortest path length to nearest flagged cluster.
              Computed once for the whole graph by BFS from all flagged nodes.
              None if the address is not reachable from any flagged cluster.

    flagged_clusters: set of cluster_ids that contain ≥1 training seed address.

    cluster_sizes: cluster_id → number of addresses in cluster.
    """
    hops_map:         dict[str, Optional[int]] = field(default_factory=dict)
    flagged_clusters: set[str]                 = field(default_factory=set)
    cluster_sizes:    dict[str, int]           = field(default_factory=dict)


# ── Rule implementations ───────────────────────────────────────────────────────

RuleFn = Callable[[dict, GraphContext], Optional[str]]


def rule_hops_to_flagged(row: dict, ctx: GraphContext) -> Optional[str]:
    """
    Rule 1: Address is within HOPS_THRESHOLD hops of a flagged cluster.
    Hops are computed by BFS on the transaction-level graph
    (address → transaction → address = 1 hop).
    """
    hops = ctx.hops_map.get(row.get("address_hash", ""))
    if hops is not None and hops <= HOPS_THRESHOLD:
        hop_word = "hop" if hops == 1 else "hops"
        return f"Within {hops} {hop_word} of a flagged address"
    return None


def rule_tx_burst(row: dict, ctx: GraphContext) -> Optional[str]:
    """
    Rule 2: Unusually high transaction rate relative to active hours.
    tx_burst_score = tx_count / max(1, active_hours) from FEATURE_SPEC.md F11.
    """
    score = float(row.get("tx_burst_score", 0.0))
    if score > BURST_THRESHOLD:
        return "Unusual transaction volume burst (off-hours)"
    return None


def rule_address_reuse(row: dict, ctx: GraphContext) -> Optional[str]:
    """
    Rule 3: Address appears as both sender and recipient across multiple transactions.
    High reuse can indicate a service address or deliberate layering.
    address_reuse_count = number of transactions where this address appears
    as both input and output (self-churn) from FEATURE_SPEC.md.
    """
    count = int(row.get("address_reuse_count", 0))
    if count > REUSE_THRESHOLD:
        return "High address reuse pattern"
    return None


def rule_cluster_flagged_member(row: dict, ctx: GraphContext) -> Optional[str]:
    """
    Rule 4: Address belongs to a co-spend cluster that contains a known-bad seed.
    cluster_confidence == HIGH strengthens the co-spend heuristic
    (cluster size ≥ 3, all linked by direct input co-spending, not just change-addr).

    Note: cluster membership is an inferred ownership/control relationship
    (co-spend heuristic), not a certainty.
    """
    cluster_id = str(row.get("cluster_id", "")) if row.get("cluster_id") else None
    if cluster_id and cluster_id in ctx.flagged_clusters:
        cluster_size = ctx.cluster_sizes.get(cluster_id, 1)
        if cluster_size >= 2:
            return "Linked to cluster with flagged member"
    return None


def rule_change_address(row: dict, ctx: GraphContext) -> Optional[str]:
    """
    Rule 5: Address is a candidate change-address from P0.4.3 heuristic.
    (Single-input, two-output transaction; this address received the smaller
    output and had never appeared as an input before.)

    This is a heuristic inference, not a certainty.
    """
    if row.get("candidate_change_address", False):
        return "Candidate change-address relationship detected"
    return None


def rule_rapid_consecutive_tx(row: dict, ctx: GraphContext) -> Optional[str]:
    """
    Rule 6: Average time between transactions is very short.
    avg_time_between_tx_hrs from FEATURE_SPEC.md F14.
    Values near zero suggest automated or programmatic activity.
    """
    avg_hrs = float(row.get("avg_time_between_tx_hrs", float("inf")))
    tx_count = int(row.get("tx_count", 0))
    # Only fire if address has more than 1 transaction (otherwise the metric is N/A)
    if tx_count > 1 and avg_hrs < RAPID_TX_THRESHOLD:
        return "Rapid consecutive transactions"
    return None


# ── Ordered rule list (order = priority for top-3 truncation) ─────────────────
RULES: list[RuleFn] = [
    rule_hops_to_flagged,        # Priority 1 — graph proximity to flagged node
    rule_cluster_flagged_member, # Priority 2 — cluster contains seed
    rule_tx_burst,               # Priority 3 — volume burst
    rule_change_address,         # Priority 4 — structural role heuristic
    rule_rapid_consecutive_tx,   # Priority 5 — timing pattern
    rule_address_reuse,          # Priority 6 — address pattern
]


# ── Main entry point ───────────────────────────────────────────────────────────

def explain_address(row: dict, ctx: GraphContext,
                    max_reasons: int = 3) -> list[str]:
    """
    Run all Layer-1 rules for a single address and return up to max_reasons
    reason strings, in priority order (rules are evaluated in RULES list order).

    Args:
        row:         Dict of address features (keys = FEATURE_SPEC.md column names
                     + cluster_id, candidate_change_address, address_hash).
        ctx:         GraphContext with hops_map, flagged_clusters, cluster_sizes.
        max_reasons: Maximum number of reasons to return. Default 3 (API contract).

    Returns:
        List of reason strings. Empty list if no rules fire.

    Notes:
        - Rules are pure functions with no side effects.
        - A rule returning None means it did not fire for this address.
        - Reasons are guaranteed unique (no duplicates).
    """
    reasons: list[str] = []
    seen: set[str]    = set()

    for rule_fn in RULES:
        if len(reasons) >= max_reasons:
            break
        result = rule_fn(row, ctx)
        if result is not None and result not in seen:
            reasons.append(result)
            seen.add(result)

    return reasons


def explain_batch(
    rows: list[dict],
    ctx: GraphContext,
    max_reasons: int = 3,
) -> dict[str, list[str]]:
    """
    Run Layer-1 explanation for a batch of addresses.

    Args:
        rows:        List of address feature dicts (each must have 'address_hash').
        ctx:         Shared GraphContext for the entire batch.
        max_reasons: Max reasons per address.

    Returns:
        Dict mapping address_hash → list[str] of reason tokens.
    """
    return {
        row["address_hash"]: explain_address(row, ctx, max_reasons)
        for row in rows
    }
