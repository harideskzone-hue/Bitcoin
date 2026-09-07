"""
backend/app/routers/graph.py
==============================
GET /api/v1/graph  —  Ego-graph around a focal address

Returns all address and transaction nodes within `depth` hops, and all
edges (INPUT_TO, OUTPUT_TO) in the induced subgraph. Used by the frontend
graph visualisation (P1.3).
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas import GraphEdge, GraphNode, GraphResponse
from backend.constants import SCORE_DISCLAIMER, score_to_label

router = APIRouter(prefix="/api/v1", tags=["graph"])

_GRAPH_PATH       = Path("data/graph_10k.pt")
_RISK_SCORES_PATH = Path("data/btc_risk_scores.parquet")
_FEATURES_PATH    = Path("data/address_features_10k.parquet")

_MAX_NODES_RETURNED = 200   # cap to keep response size bounded


def _load_graph():
    """Load the HeteroData PyG graph from offline parquet. Cache-friendly (module-level)."""
    if not _GRAPH_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Graph not found at {_GRAPH_PATH}. "
                "Run: python scripts/build_graph.py --offline"
            ),
        )
    return torch.load(_GRAPH_PATH, weights_only=False)


@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="Ego-graph around a focal address",
    description=(
        "Returns all nodes and edges within `depth` hops of the queried address. "
        "Node types: 'address' (has risk_score) or 'transaction' (no risk_score). "
        "Edge types: 'INPUT_TO' (address → transaction), 'OUTPUT_TO' (transaction → address). "
        "Response is capped at 200 nodes for frontend rendering performance."
    ),
)
def get_graph(
    address: Annotated[str, Query(description="address_hash of the focal node")],
    depth:   Annotated[int, Query(ge=1, le=2, description="Hop depth: 1 or 2")] = 1,
) -> GraphResponse:
    pyg  = _load_graph()

    # ── Resolve address → node index ──────────────────────────────────────────
    addr_node_ids: list[str] = pyg["address"].node_ids
    tx_node_ids:   list[str] = pyg["transaction"].node_ids

    addr_idx_map = {nid: i for i, nid in enumerate(addr_node_ids)}
    focal_idx    = addr_idx_map.get(address)

    if focal_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Address '{address}' not found in the current graph snapshot.",
        )

    # ── Extract edges from HeteroData ─────────────────────────────────────────
    input_edges  = pyg["address", "INPUT_TO", "transaction"].edge_index   # (2, E1)
    output_edges = pyg["transaction", "OUTPUT_TO", "address"].edge_index  # (2, E2)

    # ── BFS: collect nodes within `depth` hops ────────────────────────────────
    # Start from focal address node
    visited_addr = {focal_idx}
    visited_tx: set[int] = set()
    frontier_addr = {focal_idx}

    for _hop in range(depth):
        new_tx = set()
        # Expand: address → transaction (INPUT_TO)
        for a_idx, t_idx in input_edges.t().tolist():
            if a_idx in frontier_addr:
                new_tx.add(t_idx)

        frontier_tx = new_tx - visited_tx
        visited_tx |= frontier_tx

        # Expand: transaction → address (OUTPUT_TO)
        new_addr = set()
        for t_idx, a_idx in output_edges.t().tolist():
            if t_idx in frontier_tx:
                new_addr.add(a_idx)

        frontier_addr = new_addr - visited_addr
        visited_addr |= frontier_addr

        if not frontier_addr and not frontier_tx:
            break

    # ── Load risk scores for colour-coding ─────────────────────────────────────
    score_map: dict[str, int] = {}
    if _RISK_SCORES_PATH.exists():
        s_df = pd.read_parquet(_RISK_SCORES_PATH)
        for _, row in s_df.iterrows():
            score_map[row["address_hash"]] = int(round(float(row["risk_score"])))

    # ── Assemble nodes ─────────────────────────────────────────────────────────
    nodes: list[GraphNode] = []

    addr_indices = sorted(visited_addr)[:_MAX_NODES_RETURNED]
    for idx in addr_indices:
        nid   = addr_node_ids[idx]
        score = score_map.get(nid)
        nodes.append(GraphNode(
            id=nid,
            risk_score=score,
            risk_label=score_to_label(score) if score is not None else None,
            node_type="address",
        ))

    # Add transaction nodes (no risk score)
    remaining_slots = _MAX_NODES_RETURNED - len(nodes)
    tx_indices = sorted(visited_tx)[:remaining_slots]
    for idx in tx_indices:
        nodes.append(GraphNode(
            id=tx_node_ids[idx],
            risk_score=None,
            risk_label=None,
            node_type="transaction",
        ))

    # ── Assemble edges ─────────────────────────────────────────────────────────
    node_addr_set = set(addr_indices)
    node_tx_set   = set(tx_indices)
    edges: list[GraphEdge] = []

    for a_idx, t_idx in input_edges.t().tolist():
        if a_idx in node_addr_set and t_idx in node_tx_set:
            edges.append(GraphEdge(
                source=addr_node_ids[a_idx],
                target=tx_node_ids[t_idx],
                edge_type="INPUT_TO",
                value_btc=None,
                timestamp=None,
            ))

    for t_idx, a_idx in output_edges.t().tolist():
        if t_idx in node_tx_set and a_idx in node_addr_set:
            edges.append(GraphEdge(
                source=tx_node_ids[t_idx],
                target=addr_node_ids[a_idx],
                edge_type="OUTPUT_TO",
                value_btc=None,
                timestamp=None,
            ))

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        disclaimer=SCORE_DISCLAIMER,
    )
