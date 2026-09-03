"""
ml/explain/gnn_explainer.py
=============================
P0.6.2 — Layer 2: GNNExplainer for top-5 high-risk addresses

Design
------
Layer 2 is OPTIONAL and runs only on the top-5 addresses by risk score.
If GNNExplainer fails or times out, Layer 1 reasons are used alone and
the demo continues without interruption.

GNNExplainer (Ying et al. 2019) learns per-instance masks over:
  - Node features: which of the 14 features contributed most to this prediction
  - Edges: which graph edges contributed most

We extract:
  - Top-3 node feature importances → human-readable feature name + direction
  - Top-3 edge importances → (source_hash, target_hash, weight)

Outputs
-------
Returns a dict per address:
  {
    "top_features": [{"name": str, "importance": float, "direction": "↑"|"↓"}, ...],
    "top_edges":    [{"source": str, "target": str, "weight": float}, ...],
  }

Fail-safe
---------
Any exception during GNNExplainer execution is caught and logged.
The caller always gets a result dict — with empty lists if Layer 2 failed.

Usage
-----
  from ml.explain.gnn_explainer import explain_top_k
  layer2 = explain_top_k(model, data, top_k_indices, feat_names, node_ids)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data

log = logging.getLogger("gnn_explainer")

# ── Constants ──────────────────────────────────────────────────────────────────
TOP_K          = 5     # number of high-risk addresses to explain
TOP_FEATURES   = 3     # number of top features to report per address
TOP_EDGES      = 3     # number of top edges to report per address
EPOCHS         = 200   # GNNExplainer optimisation epochs per explanation


# ── Feature name → human-readable description ─────────────────────────────────
# Maps the 14 FEATURE_SPEC.md feature columns to presentation-friendly strings.
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "tx_count":                   "transaction count",
    "total_sent_btc":             "total BTC sent",
    "total_received_btc":         "total BTC received",
    "input_count":                "number of inputs",
    "output_count":               "number of outputs",
    "is_script_hash":             "script-hash address type (P2SH)",
    "address_reuse_count":        "address reuse frequency",
    "time_since_last_tx_hrs":     "hours since last transaction",
    "tx_burst_score":             "transaction burst score",
    "weekday_vs_weekend_ratio":   "weekday-to-weekend activity ratio",
    "avg_time_between_tx_hrs":    "average hours between transactions",
    "avg_tx_value_btc":           "average transaction value (BTC)",
    "address_degree":             "graph degree (transaction connections)",
    "clustering_coefficient":     "local clustering coefficient",
}


def _importance_direction(value: float, feature_name: str) -> str:
    """
    Heuristic direction label.
    For most features, higher value = higher risk → "↑" means risk-increasing.
    For time_since_last_tx_hrs and avg_time_between_tx_hrs, direction is inverted.
    """
    inverted = {"time_since_last_tx_hrs", "avg_time_between_tx_hrs"}
    if feature_name in inverted:
        return "↓" if value > 0 else "↑"
    return "↑" if value > 0 else "↓"


def explain_top_k(
    model: torch.nn.Module,
    data:  Data,
    top_k_indices: list[int],
    feat_names:    list[str],
    node_ids:      list[str],
) -> dict[str, dict]:
    """
    P0.6.2: Run GNNExplainer on the top-k high-risk address nodes.

    Args:
        model:          Trained BitcoinGCN (or EllipticGCN for testing).
        data:           PyG Data object with x, edge_index, y.
        top_k_indices:  Node indices (in data.x) of the top-K addresses.
        feat_names:     Ordered list of feature column names (len = data.x.shape[1]).
        node_ids:       List of address_hash strings (len = data.num_nodes),
                        indexed same as data.x row order.

    Returns:
        Dict: address_hash → {
            "top_features": [{"name", "display_name", "importance", "direction"}, ...],
            "top_edges":    [{"source", "target", "weight"}, ...],
            "layer":        "gnnexplainer",
        }
        If explanation fails for a node, returns {"error": str, "layer": "gnnexplainer"}.
    """
    model.eval()
    results: dict[str, dict] = {}

    try:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=EPOCHS),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={
                "mode":       "multiclass_classification",
                "task_level": "node",
                "return_type": "raw",   # raw logits
            },
        )
    except Exception as e:
        log.error(f"GNNExplainer initialisation failed: {e}")
        for idx in top_k_indices:
            addr = node_ids[idx] if idx < len(node_ids) else str(idx)
            results[addr] = {"error": str(e), "layer": "gnnexplainer"}
        return results

    for node_idx in top_k_indices:
        addr = node_ids[node_idx] if node_idx < len(node_ids) else str(node_idx)
        try:
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                index=node_idx,
            )

            # ── Node feature importances ────────────────────────────────────────
            feat_imp: list[dict] = []
            if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
                # node_mask shape: (1, num_features) — importance per feature
                mask = explanation.node_mask[node_idx].detach().cpu()
                top_feat_idx = mask.abs().topk(
                    min(TOP_FEATURES, len(feat_names))
                ).indices.tolist()

                for fi in top_feat_idx:
                    fname    = feat_names[fi] if fi < len(feat_names) else f"f{fi}"
                    val      = float(mask[fi])
                    feat_imp.append({
                        "name":         fname,
                        "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                        "importance":   round(abs(val), 4),
                        "direction":    _importance_direction(val, fname),
                    })

            # ── Edge importances ────────────────────────────────────────────────
            top_edges: list[dict] = []
            if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None:
                edge_mask = explanation.edge_mask.detach().cpu()
                top_edge_idx = edge_mask.topk(
                    min(TOP_EDGES, edge_mask.shape[0])
                ).indices.tolist()

                for ei in top_edge_idx:
                    src_node = int(data.edge_index[0, ei])
                    dst_node = int(data.edge_index[1, ei])
                    src_id   = node_ids[src_node] if src_node < len(node_ids) else str(src_node)
                    dst_id   = node_ids[dst_node] if dst_node < len(node_ids) else str(dst_node)
                    top_edges.append({
                        "source": src_id,
                        "target": dst_id,
                        "weight": round(float(edge_mask[ei]), 4),
                    })

            results[addr] = {
                "top_features": feat_imp,
                "top_edges":    top_edges,
                "layer":        "gnnexplainer",
            }
            log.info(f"  GNNExplainer [{addr[:16]}…]: "
                     f"{len(feat_imp)} features, {len(top_edges)} edges")

        except Exception as e:
            log.warning(f"  GNNExplainer failed for {addr[:16]}…: {e}")
            results[addr] = {"error": str(e), "layer": "gnnexplainer"}

    return results
