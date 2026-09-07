"""
ml/explain/explainer.py
========================
P0.6.3 — Two-layer explainability pipeline

Wires Layer 1 (deterministic) and Layer 2 (GNNExplainer) together and
writes the final reason tokens to data/btc_reasons.parquet.

Execution order:
  1. Compute GraphContext (BFS hops_map, flagged_clusters, cluster_sizes)
  2. Layer 1: explain_batch() for ALL addresses (<100ms per address)
  3. Identify top-5 addresses by risk_score
  4. Layer 2: explain_top_k() for top-5 only (2–5s per address, optional)
  5. Merge Layer 2 tokens into Layer 1 reasons (Layer 2 appended if available)
  6. Write btc_reasons.parquet

Output schema
-------------
  address_hash : str
  reason       : str   — one reason token per row
  rank         : int   — 1 = primary reason, 2, 3
  layer        : str   — "deterministic" | "gnnexplainer"
  snapshot_tag : str

Usage
-----
  python -m ml.explain.explainer
  # or
  python ml/explain/explainer.py
"""

import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import torch

from ml.explain.deterministic import GraphContext, explain_batch
from ml.explain.gnn_explainer import explain_top_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("explainer")

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH    = Path("data/address_features_10k.parquet")
RISK_SCORES_PATH = Path("data/btc_risk_scores.parquet")
CLUSTERS_PATH    = Path("data/btc_clusters.parquet")
WEAK_LABELS_PATH = Path("data/btc_weak_labels.parquet")
GRAPH_PATH       = Path("data/graph_10k.pt")
GCN_CKPT         = Path("models/gcn_bitcoin_v1.pt")
OUT_PATH         = Path("data/btc_reasons.parquet")
CHANGE_ADDR_PATH = Path("data/btc_change_candidates.parquet")

SNAPSHOT_TAG = "10k_v1"
TOP_K        = 5   # addresses to run Layer 2 on


# ── Graph context builder ──────────────────────────────────────────────────────

def build_graph_context(
    pyg_data,
    labels_df:   pd.DataFrame,
    clusters_df: pd.DataFrame,
    node_ids:    list[str],
) -> GraphContext:
    """
    Compute the GraphContext needed by Layer 1 rules.

    hops_map: BFS from all flagged nodes outward through the address projection.
    flagged_clusters: clusters containing at least one high_risk label.
    cluster_sizes: cluster_id → member count.
    """
    log.info("Building GraphContext …")

    # ── Flagged clusters ────────────────────────────────────────────────────────
    flagged_mask = labels_df["weak_label"] == "high_risk"
    flagged_addrs = set(labels_df[flagged_mask]["address"].values)
    flagged_clusters: set[str] = set()
    if not clusters_df.empty:
        fc_rows = clusters_df[clusters_df["address"].isin(flagged_addrs)]
        flagged_clusters = set(fc_rows["cluster_id"].astype(str).values)
    log.info(f"  Flagged clusters: {len(flagged_clusters)}")

    # ── Cluster sizes ───────────────────────────────────────────────────────────
    cluster_sizes: dict[str, int] = {}
    if not clusters_df.empty:
        for cid, grp in clusters_df.groupby("cluster_id"):
            cluster_sizes[str(cid)] = len(grp)

    # ── BFS hops from flagged nodes ────────────────────────────────────────────
    # Build adjacency from edge_index (address projection, already reconstructed
    # as homogeneous in train_bitcoin_gcn.py — but here we use the bipartite graph
    # directly for BFS hop counting on the full transaction graph).
    addr_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    flagged_indices = {
        addr_to_idx[h] for h in flagged_addrs
        if h in addr_to_idx
    }
    log.info(f"  Flagged address nodes in graph: {len(flagged_indices)}")

    # Build adj list from address bipartite edges
    adj: dict[int, set[int]] = defaultdict(set)
    try:
        addr_input  = pyg_data["address", "INPUT_TO", "transaction"].edge_index
        tx_output   = pyg_data["transaction", "OUTPUT_TO", "address"].edge_index
        # addr→addr via shared transaction
        tx_to_in  = defaultdict(list)
        tx_to_out = defaultdict(list)
        for a, t in addr_input.t().tolist():
            tx_to_in[t].append(a)
        for t, a in tx_output.t().tolist():
            tx_to_out[t].append(a)
        for t in set(tx_to_in) & set(tx_to_out):
            for a_in in tx_to_in[t]:
                for a_out in tx_to_out[t]:
                    adj[a_in].add(a_out)
                    adj[a_out].add(a_in)
    except Exception as e:
        log.warning(f"  Edge extraction for BFS failed: {e} — hops_map will be empty")

    # Multi-source BFS from all flagged nodes
    hops: dict[int, int] = {idx: 0 for idx in flagged_indices}
    queue: deque[int]    = deque(flagged_indices)
    while queue:
        node = queue.popleft()
        for nbr in adj.get(node, set()):
            if nbr not in hops:
                hops[nbr] = hops[node] + 1
                queue.append(nbr)

    hops_map: dict[str, int | None] = {}
    for nid, idx in addr_to_idx.items():
        hops_map[nid] = hops.get(idx)

    n_reachable = sum(1 for v in hops_map.values() if v is not None)
    log.info(f"  BFS complete: {n_reachable:,} / {len(node_ids):,} nodes reachable from flagged")

    return GraphContext(
        hops_map=hops_map,
        flagged_clusters=flagged_clusters,
        cluster_sizes=cluster_sizes,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Fail-closed checks ─────────────────────────────────────────────────────
    for p, label in [
        (FEATURES_PATH,    "Address features"),
        (RISK_SCORES_PATH, "Risk scores"),
        (WEAK_LABELS_PATH, "Weak labels"),
        (GRAPH_PATH,       "PyG graph"),
    ]:
        if not p.exists():
            log.error(f"{label} not found: {p}")
            log.error("Run the upstream pipeline steps first.")
            sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading data …")
    feat_df    = pd.read_parquet(FEATURES_PATH).sort_values("address_hash").reset_index(drop=True)
    scores_df  = pd.read_parquet(RISK_SCORES_PATH)
    labels_df  = pd.read_parquet(WEAK_LABELS_PATH)
    clusters_df = pd.read_parquet(CLUSTERS_PATH) if CLUSTERS_PATH.exists() else pd.DataFrame()
    pyg_data   = torch.load(GRAPH_PATH, weights_only=False)

    node_ids: list[str] = pyg_data["address"].node_ids
    feat_cols = [c for c in feat_df.columns if c != "address_hash"]

    log.info(f"  Addresses: {len(feat_df):,}")
    log.info(f"  Risk scores: {len(scores_df):,}")

    # ── Merge features + scores + cluster info ─────────────────────────────────
    merged = feat_df.merge(
        scores_df[["address_hash", "risk_score"]],
        on="address_hash", how="left",
    )
    if not clusters_df.empty:
        # Hash the cluster 'address' column to match address_hash
        import hashlib
        def _hash(s): return hashlib.sha256(s.encode()).hexdigest()[:16]
        clusters_df["address_hash"] = clusters_df["address"].map(_hash)
        merged = merged.merge(
            clusters_df[["address_hash", "cluster_id"]].drop_duplicates("address_hash"),
            on="address_hash", how="left",
        )
    else:
        merged["cluster_id"] = None

    # Change-address candidates
    if CHANGE_ADDR_PATH.exists():
        ca_df = pd.read_parquet(CHANGE_ADDR_PATH)
        ca_hashes = set(ca_df["address_hash"].values)
    else:
        ca_hashes = set()
    merged["candidate_change_address"] = merged["address_hash"].isin(ca_hashes)

    # ── GraphContext ───────────────────────────────────────────────────────────
    ctx = build_graph_context(pyg_data, labels_df, clusters_df, node_ids)

    # ── Layer 1: deterministic rules for ALL addresses ─────────────────────────
    log.info("\nLayer 1 — deterministic rules (all addresses) …")
    rows_as_dicts = merged.to_dict(orient="records")
    layer1_results = explain_batch(rows_as_dicts, ctx, max_reasons=3)

    n_with_reasons = sum(1 for v in layer1_results.values() if v)
    log.info(f"  Addresses with ≥1 reason: {n_with_reasons:,} / {len(layer1_results):,}")

    # ── Layer 2: GNNExplainer for top-5 addresses ──────────────────────────────
    layer2_results: dict[str, dict] = {}
    if GCN_CKPT.exists():
        log.info(f"\nLayer 2 — GNNExplainer (top {TOP_K} by risk_score) …")
        try:
            from ml.scripts.train_bitcoin_gcn import BitcoinGCN

            ckpt = torch.load(GCN_CKPT, weights_only=False)
            cfg  = ckpt["config"]
            gcn  = BitcoinGCN(cfg["in_channels"], cfg["hidden"], cfg["out_channels"],
                              cfg["dropout"])
            gcn.load_state_dict(ckpt["model_state_dict"])

            # Reconstruct homogeneous edge_index (same as train_bitcoin_gcn.py)
            addr_input  = pyg_data["address", "INPUT_TO", "transaction"].edge_index
            tx_output   = pyg_data["transaction", "OUTPUT_TO", "address"].edge_index
            from collections import defaultdict as dd
            ti, to = dd(list), dd(list)
            for a, t in addr_input.t().tolist():
                ti[t].append(a)
            for t, a in tx_output.t().tolist():
                to[t].append(a)
            src, dst = [], []
            for t in set(ti) & set(to):
                for ai in ti[t]:
                    for ao in to[t]:
                        src += [ai, ao]
                        dst += [ao, ai]
            ei = torch.unique(torch.tensor([src, dst], dtype=torch.long), dim=1)

            from torch_geometric.data import Data
            g = Data(x=torch.tensor(merged[feat_cols].values, dtype=torch.float),
                     edge_index=ei)

            # Top-k node indices by risk_score
            top_scores = merged.nlargest(TOP_K, "risk_score")
            top_indices = top_scores.index.tolist()

            layer2_results = explain_top_k(
                model=gcn, data=g,
                top_k_indices=top_indices,
                feat_names=feat_cols,
                node_ids=list(merged["address_hash"].values),
            )
        except Exception as e:
            log.warning(f"Layer 2 skipped: {e}")
            log.info("  Demo continues with Layer 1 only (by design).")
    else:
        log.info(f"\nLayer 2 skipped: GCN checkpoint not found ({GCN_CKPT})")
        log.info("  This is expected on the 10k snapshot. Layer 1 reasons used only.")

    # ── Merge layers and write output ──────────────────────────────────────────
    log.info("\nMerging layers and writing btc_reasons.parquet …")
    out_rows = []

    for _, row in merged.iterrows():
        ahash = row["address_hash"]
        l1_reasons = layer1_results.get(ahash, [])
        l2_data    = layer2_results.get(ahash, {})
        l2_features = l2_data.get("top_features", [])

        # Layer 1 reasons (deterministic)
        for rank, reason in enumerate(l1_reasons, start=1):
            out_rows.append({
                "address_hash": ahash,
                "reason":       reason,
                "rank":         rank,
                "layer":        "deterministic",
                "snapshot_tag": SNAPSHOT_TAG,
            })

        # Layer 2 top-feature reasons (appended after L1, only for top-5)
        l2_rank_start = len(l1_reasons) + 1
        for fi, feat in enumerate(l2_features[:max(0, 3 - len(l1_reasons))]):
            direction_word = "elevated" if feat["direction"] == "↑" else "reduced"
            reason = (
                f"GNNExplainer: {direction_word} {feat['display_name']} "
                f"(importance={feat['importance']:.3f})"
            )
            out_rows.append({
                "address_hash": ahash,
                "reason":       reason,
                "rank":         l2_rank_start + fi,
                "layer":        "gnnexplainer",
                "snapshot_tag": SNAPSHOT_TAG,
            })

    reasons_df = pd.DataFrame(out_rows)
    reasons_df.to_parquet(OUT_PATH, index=False)

    total_with_reasons = reasons_df["address_hash"].nunique()
    log.info(f"  Addresses with reasons: {total_with_reasons:,}")
    log.info(f"  Total reason rows:      {len(reasons_df):,}")
    log.info(f"  Layer distribution:\n{reasons_df['layer'].value_counts().to_string()}")
    log.info(f"\nbtc_reasons.parquet saved → {OUT_PATH}")
    log.info("\nP0.6 COMPLETE ✓")


if __name__ == "__main__":
    main()
