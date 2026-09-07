"""
scripts/build_graph.py
======================
Builds the Bitcoin bipartite transaction graph from either:
  (a) A live BigQuery pull  (default, requires GOOGLE_APPLICATION_CREDENTIALS)
  (b) A local parquet snapshot  (--offline flag, no internet needed)

Produces:
  data/btc_snapshot_{n}k.parquet  — raw transaction rows (flat table)
  data/graph_{n}k.pt              — PyTorch Geometric HeteroData object

Usage:
  # Live pull (requires BigQuery credentials):
  python scripts/build_graph.py --n-tx 50000 --output data/

  # Offline mode (hackathon venue, no internet):
  python scripts/build_graph.py --offline --n-tx 50000 --output data/

See: data/SCHEMA.md for BigQuery table structure
See: data/FEATURE_SPEC.md for exact feature formulas
"""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Add project root to path so we can import ml/
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_graph")


# ── Constants ──────────────────────────────────────────────────────────────────

SATOSHI = 1e8  # 1 BTC = 100,000,000 satoshi

# The 14 address features in the exact order they enter the GCN tensor.
# Changing this order breaks saved model checkpoints — do not reorder.
ADDRESS_FEATURE_COLS = [
    "address_degree",           # F1
    "total_sent_btc",           # F2
    "total_received_btc",       # F3
    "tx_count",                 # F4
    "avg_tx_value_btc",         # F5
    "time_since_last_tx_hrs",   # F6
    "clustering_coefficient",   # F7
    "input_count",              # F8
    "output_count",             # F9
    "is_script_hash",           # F10
    "address_reuse_count",      # F11
    "tx_burst_score",           # F12
    "weekday_vs_weekend_ratio", # F13
    "avg_time_between_tx_hrs",  # F14
]

# Transaction node features (9 features)
TX_FEATURE_COLS = [
    "fee_btc",          # fee_satoshi / 1e8
    "input_count_tx",   # number of inputs in this transaction
    "output_count_tx",  # number of outputs in this transaction
    "total_input_btc",  # total input value in BTC
    "total_output_btc", # total output value in BTC
    "hour_of_day",      # 0–23 UTC
    "day_of_week",      # 0=Mon … 6=Sun
    "is_coinbase",      # bool, 0 or 1
]


# ── BigQuery pull ──────────────────────────────────────────────────────────────

BIGQUERY_SQL = """
WITH sampled_txs AS (
    SELECT `hash`, block_timestamp, fee, input_count, output_count,
           input_value, output_value, is_coinbase
    FROM `bigquery-public-data.crypto_bitcoin.transactions`
    WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
      AND is_coinbase = FALSE
    LIMIT {n_tx}
)
SELECT
    t.`hash`                                   AS tx_hash,
    t.block_timestamp,
    COALESCE(t.fee, 0)                         AS fee_satoshi,
    t.input_count,
    t.output_count,
    COALESCE(t.input_value,  0)                AS total_input_sat,
    COALESCE(t.output_value, 0)                AS total_output_sat,
    t.is_coinbase,
    inp_addr                                   AS input_address,
    COALESCE(i.value, 0)                       AS input_value_sat,
    out_addr                                   AS output_address,
    COALESCE(o.value, 0)                       AS output_value_sat,
    o.type                                     AS output_type,
    o.index                                    AS output_index
FROM
    sampled_txs t
JOIN
    `bigquery-public-data.crypto_bitcoin.inputs`  i
    ON i.transaction_hash = t.`hash`,
    UNNEST(i.addresses) AS inp_addr
JOIN
    `bigquery-public-data.crypto_bitcoin.outputs` o
    ON o.transaction_hash = t.`hash`,
    UNNEST(o.addresses) AS out_addr
WHERE
    o.type != 'nulldata'
"""


def pull_from_bigquery(n_tx: int) -> pd.DataFrame:
    """Pull n_tx rows from the BigQuery public Bitcoin dataset."""
    try:
        from google.cloud import bigquery
    except ImportError:
        log.error("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        sys.exit(1)

    import os

    from dotenv import load_dotenv
    load_dotenv()
    project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        log.error("GCP_PROJECT not set in .env. Add: GCP_PROJECT=your-project-id")
        sys.exit(1)

    log.info(f"Pulling {n_tx:,} rows from BigQuery (project={project}) …")
    client = bigquery.Client(project=project)
    query  = BIGQUERY_SQL.format(n_tx=n_tx)
    df     = client.query(query).to_dataframe()
    log.info(f"  → {len(df):,} rows received")
    return df


def load_from_parquet(parquet_path: Path) -> pd.DataFrame:
    """Load a previously saved parquet snapshot (offline mode)."""
    log.info(f"Loading offline snapshot: {parquet_path}")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    elapsed = time.time() - t0
    log.info(f"  → {len(df):,} rows loaded in {elapsed:.2f}s")
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def _hash_string(s: str) -> str:
    """SHA-256 hash a string to 16 hex chars. Used to anonymise addresses."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _compute_address_features(df: pd.DataFrame, snapshot_now: pd.Timestamp) -> pd.DataFrame:
    """
    Compute all 14 address node features from the raw transaction DataFrame.

    Each row in `df` is one (input_address, tx, output_address) triple.
    We aggregate at the address level.

    Returns a DataFrame indexed by address_hash with 14 feature columns.
    See data/FEATURE_SPEC.md for exact formulas.
    """
    log.info("Computing address features …")

    # ── Step 1: Identify T_in(a) and T_out(a) ─────────────────────────────────
    # T_in(a)  = set of tx_hashes where address appears as an INPUT
    # T_out(a) = set of tx_hashes where address appears as an OUTPUT

    # One row per (input_address, tx_hash) — deduplicated
    input_df = df[["input_address", "tx_hash", "block_timestamp", "output_value_sat",
                   "output_type"]].drop_duplicates(subset=["input_address", "tx_hash"])
    input_df = input_df.rename(columns={"input_address": "address"})

    # One row per (output_address, tx_hash) — deduplicated
    output_df = df[["output_address", "tx_hash", "block_timestamp", "output_value_sat",
                    "output_type"]].drop_duplicates(subset=["output_address", "tx_hash"])
    output_df = output_df.rename(columns={"output_address": "address"})

    # ── F2: total_sent_btc ─────────────────────────────────────────────────────
    # Sum of ALL output values for transactions where address is an input.
    # (See FEATURE_SPEC.md F2 — note on multi-input attribution)
    sent = (
        df.drop_duplicates(subset=["input_address", "tx_hash", "output_address"])
        .groupby("input_address")["output_value_sat"]
        .sum()
        .rename("total_sent_btc")
        / SATOSHI
    )

    # ── F3: total_received_btc ─────────────────────────────────────────────────
    recv = (
        output_df.groupby("address")["output_value_sat"]
        .sum()
        .rename("total_received_btc")
        / SATOSHI
    )

    # ── F4: tx_count ──────────────────────────────────────────────────────────
    # Count distinct tx_hashes for all transactions the address participates in
    all_addrs = pd.concat([
        input_df[["address", "tx_hash"]],
        output_df[["address", "tx_hash"]],
    ]).drop_duplicates()

    tx_count = all_addrs.groupby("address")["tx_hash"].nunique().rename("tx_count")

    # ── F8: input_count ────────────────────────────────────────────────────────
    input_count = input_df.groupby("address")["tx_hash"].nunique().rename("input_count")

    # ── F9: output_count ───────────────────────────────────────────────────────
    output_count = output_df.groupby("address")["tx_hash"].nunique().rename("output_count")

    # ── F10: is_script_hash ────────────────────────────────────────────────────
    # 1 if address ever received a P2SH or P2WSH output type
    script_hash_types = {"scripthash", "witness_v0_scripthash"}
    is_sh = (
        output_df.assign(is_sh=output_df["output_type"].isin(script_hash_types).astype(int))
        .groupby("address")["is_sh"]
        .max()
        .rename("is_script_hash")
    )

    # ── F11: address_reuse_count ───────────────────────────────────────────────
    # |T_in(a) ∩ T_out(a)| — txs where address is BOTH input and output
    t_in_set  = input_df.groupby("address")["tx_hash"].apply(set)
    t_out_set = output_df.groupby("address")["tx_hash"].apply(set)
    common    = t_in_set.index.intersection(t_out_set.index)
    reuse     = pd.Series(
        {addr: len(t_in_set[addr] & t_out_set[addr]) for addr in common},
        name="address_reuse_count",
        dtype=int,
    )

    # ── F6: time_since_last_tx_hrs ─────────────────────────────────────────────
    last_ts = all_addrs.merge(
        df[["tx_hash", "block_timestamp"]].drop_duplicates(),
        on="tx_hash",
    ).groupby("address")["block_timestamp"].max()
    time_since = ((snapshot_now - last_ts).dt.total_seconds() / 3600).rename("time_since_last_tx_hrs")

    # ── F12: tx_burst_score ────────────────────────────────────────────────────
    tx_with_ts = all_addrs.merge(
        df[["tx_hash", "block_timestamp"]].drop_duplicates(),
        on="tx_hash",
    )
    tx_with_ts["hour"] = pd.to_datetime(tx_with_ts["block_timestamp"], utc=True).dt.hour

    def _burst(grp: pd.Series) -> float:
        counts = grp.value_counts().reindex(range(24), fill_value=0)
        mu, sigma = counts.mean(), counts.std(ddof=0)
        if sigma == 0:
            return 0.0
        return float((counts.max() - mu) / sigma)

    burst = tx_with_ts.groupby("address")["hour"].apply(_burst).rename("tx_burst_score")

    # ── F13: weekday_vs_weekend_ratio ─────────────────────────────────────────
    tx_with_ts["dow"] = pd.to_datetime(tx_with_ts["block_timestamp"], utc=True).dt.dayofweek
    tx_with_ts["is_weekend"] = tx_with_ts["dow"].isin([5, 6])

    def _wk_ratio(grp: pd.DataFrame) -> float:
        wkday = (~grp["is_weekend"]).sum()
        wkend = grp["is_weekend"].sum()
        if wkend == 0:
            return float(wkday) if wkday > 0 else 1.0
        return float(wkday) / float(wkend)

    wk_ratio = (
        tx_with_ts.groupby("address")
        .apply(_wk_ratio, include_groups=False)
        .rename("weekday_vs_weekend_ratio")
    )

    # ── F14: avg_time_between_tx_hrs ──────────────────────────────────────────
    def _avg_ibi(grp: pd.Series) -> float:
        sorted_ts = grp.sort_values()
        if len(sorted_ts) < 2:
            return float("nan")
        deltas = sorted_ts.diff().dropna().dt.total_seconds() / 3600
        return float(deltas.mean())

    ts_by_addr = (
        all_addrs
        .merge(df[["tx_hash", "block_timestamp"]].drop_duplicates(), on="tx_hash")
        .drop_duplicates(subset=["address", "tx_hash"])
    )
    avg_ibi = (
        ts_by_addr.groupby("address")["block_timestamp"]
        .apply(_avg_ibi)
        .rename("avg_time_between_tx_hrs")
    )

    # ── Assemble base feature table ────────────────────────────────────────────
    feat = (
        tx_count
        .to_frame()
        .join(sent,         how="left")
        .join(recv,         how="left")
        .join(input_count,  how="left")
        .join(output_count, how="left")
        .join(is_sh,        how="left")
        .join(reuse,        how="left")
        .join(time_since,   how="left")
        .join(burst,        how="left")
        .join(wk_ratio,     how="left")
        .join(avg_ibi,      how="left")
        .fillna({
            "total_sent_btc":           0.0,
            "total_received_btc":       0.0,
            "input_count":              0,
            "output_count":             0,
            "is_script_hash":           0,
            "address_reuse_count":      0,
            "tx_burst_score":           0.0,
            "weekday_vs_weekend_ratio": 1.0,
        })
    )

    # ── F5: avg_tx_value_btc ──────────────────────────────────────────────────
    feat["avg_tx_value_btc"] = np.where(
        feat["tx_count"] > 0,
        (feat["total_sent_btc"] + feat["total_received_btc"]) / feat["tx_count"],
        0.0,
    )

    # Impute NaN temporal features with global median (F6 and F14)
    for col in ["time_since_last_tx_hrs", "avg_time_between_tx_hrs"]:
        median_val = feat[col].median()
        feat[col]  = feat[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Hash address strings for privacy
    feat.index = feat.index.map(_hash_string)
    feat.index.name = "address_hash"

    log.info(f"  → {len(feat):,} unique addresses, {len(ADDRESS_FEATURE_COLS)} features each")
    return feat


def _compute_projection_features(feat: pd.DataFrame, G_proj: nx.Graph) -> pd.DataFrame:
    """
    Compute address_degree (F1) and clustering_coefficient (F7) from the
    address-address projection graph.

    Self-loops are removed before any structural metric is computed.
    See FEATURE_SPEC.md — Graph Definitions / Self-loop rule.
    """
    log.info("Computing projection graph features (degree, clustering) …")

    # Remove self-loops BEFORE any metric — see FEATURE_SPEC.md
    G_proj.remove_edges_from(nx.selfloop_edges(G_proj))

    degree = dict(G_proj.degree())
    clust  = nx.clustering(G_proj)

    feat = feat.copy()
    feat["address_degree"]        = feat.index.map(degree).fillna(0).astype(int)
    feat["clustering_coefficient"]= feat.index.map(clust).fillna(0.0)
    return feat


# ── Graph construction ─────────────────────────────────────────────────────────

def build_bipartite_and_projection(
    df: pd.DataFrame,
    feat: pd.DataFrame,
) -> tuple[nx.DiGraph, nx.Graph]:
    """
    Build two graphs from the transaction DataFrame:

    1. bipartite_G  — directed bipartite: Address → Transaction → Address
    2. projection_G — undirected address-address projection (for structural features)

    Returns (bipartite_G, projection_G).
    """
    log.info("Building bipartite graph …")

    bipartite_G = nx.DiGraph()

    # Add unique address nodes and transaction nodes
    unique_inputs  = df["input_address"].dropna().unique()
    unique_outputs = df["output_address"].dropna().unique()
    unique_addrs   = set(unique_inputs) | set(unique_outputs)
    unique_txs     = df["tx_hash"].unique()

    for addr in unique_addrs:
        bipartite_G.add_node(_hash_string(addr), node_type="address")
    for tx in unique_txs:
        bipartite_G.add_node(tx, node_type="transaction")

    # Add INPUT_TO edges: address → transaction
    for _, row in df[["input_address", "tx_hash", "input_value_sat"]].drop_duplicates().iterrows():
        if pd.notna(row["input_address"]):
            bipartite_G.add_edge(
                _hash_string(row["input_address"]),
                row["tx_hash"],
                edge_type="INPUT_TO",
                value_btc=row["input_value_sat"] / SATOSHI,
            )

    # Add OUTPUT_TO edges: transaction → address
    for _, row in df[["tx_hash", "output_address", "output_value_sat"]].drop_duplicates().iterrows():
        if pd.notna(row["output_address"]):
            bipartite_G.add_edge(
                row["tx_hash"],
                _hash_string(row["output_address"]),
                edge_type="OUTPUT_TO",
                value_btc=row["output_value_sat"] / SATOSHI,
            )

    log.info(
        f"  Bipartite: {bipartite_G.number_of_nodes():,} nodes, "
        f"{bipartite_G.number_of_edges():,} edges"
    )

    # Build address-address projection
    # Edge (a1, a2) iff ∃ tx: a1 –INPUT_TO→ tx –OUTPUT_TO→ a2  AND  a1 ≠ a2
    log.info("Building address-address projection …")
    projection_G = nx.Graph()
    for tx in unique_txs:
        in_edges  = [(u, d) for u, v, d in bipartite_G.in_edges(tx, data=True)
                     if d.get("edge_type") == "INPUT_TO"]
        out_edges = [(v, d) for u, v, d in bipartite_G.out_edges(tx, data=True)
                     if d.get("edge_type") == "OUTPUT_TO"]
        for in_addr, _ in in_edges:
            for out_addr, out_d in out_edges:
                if in_addr != out_addr:  # exclude self-loops per FEATURE_SPEC.md
                    w = out_d.get("value_btc", 0.0)
                    if projection_G.has_edge(in_addr, out_addr):
                        projection_G[in_addr][out_addr]["weight"] += w
                    else:
                        projection_G.add_edge(in_addr, out_addr, weight=w)

    # Remove any self-loops that slipped through (defensive)
    projection_G.remove_edges_from(nx.selfloop_edges(projection_G))

    log.info(
        f"  Projection: {projection_G.number_of_nodes():,} nodes, "
        f"{projection_G.number_of_edges():,} edges"
    )

    return bipartite_G, projection_G


def build_pyg_heterodata(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    bipartite_G: nx.DiGraph,
) -> object:
    """
    Convert the bipartite graph + address features into a PyTorch Geometric
    HeteroData object.

    Node types: 'address', 'transaction'
    Edge types: ('address', 'INPUT_TO', 'transaction'),
                ('transaction', 'OUTPUT_TO', 'address')
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        log.warning(
            "torch or torch-geometric not installed. Skipping .pt output. "
            "Install with: pip install torch torch-geometric"
        )
        return None

    log.info("Building PyG HeteroData object …")
    data = HeteroData()

    # ── Address node features ──────────────────────────────────────────────────
    addr_nodes = sorted([
        n for n, d in bipartite_G.nodes(data=True) if d.get("node_type") == "address"
    ])
    addr_to_idx = {a: i for i, a in enumerate(addr_nodes)}

    addr_feat_matrix = feat.reindex(addr_nodes)[ADDRESS_FEATURE_COLS].values.astype(np.float32)
    data["address"].x        = torch.tensor(addr_feat_matrix, dtype=torch.float)
    data["address"].num_nodes= len(addr_nodes)
    data["address"].node_ids = addr_nodes  # keep mapping for lookup

    # ── Transaction node features ─────────────────────────────────────────────
    tx_meta = (
        df[["tx_hash", "fee_satoshi", "input_count", "output_count",
            "total_input_sat", "total_output_sat", "block_timestamp", "is_coinbase"]]
        .drop_duplicates(subset=["tx_hash"])
        .set_index("tx_hash")
    )
    tx_meta["fee_btc"]         = tx_meta["fee_satoshi"] / SATOSHI
    tx_meta["total_input_btc"] = tx_meta["total_input_sat"] / SATOSHI
    tx_meta["total_output_btc"]= tx_meta["total_output_sat"] / SATOSHI
    tx_meta["hour_of_day"]     = pd.to_datetime(tx_meta["block_timestamp"], utc=True).dt.hour
    tx_meta["day_of_week"]     = pd.to_datetime(tx_meta["block_timestamp"], utc=True).dt.dayofweek
    tx_meta["is_coinbase"]     = tx_meta["is_coinbase"].astype(float)

    tx_nodes = sorted([
        n for n, d in bipartite_G.nodes(data=True) if d.get("node_type") == "transaction"
    ])
    tx_to_idx = {t: i for i, t in enumerate(tx_nodes)}

    tx_feat_matrix = (
        tx_meta
        .rename(columns={"input_count": "input_count_tx", "output_count": "output_count_tx"})
        .reindex(tx_nodes)[TX_FEATURE_COLS]
        .values.astype(np.float32)
    )
    data["transaction"].x        = torch.tensor(tx_feat_matrix, dtype=torch.float)
    data["transaction"].num_nodes= len(tx_nodes)
    data["transaction"].node_ids = tx_nodes

    # ── Edge indices ──────────────────────────────────────────────────────────
    input_to_src, input_to_dst, input_to_w = [], [], []
    output_to_src, output_to_dst, output_to_w = [], [], []

    for u, v, d in bipartite_G.edges(data=True):
        if d.get("edge_type") == "INPUT_TO":
            if u in addr_to_idx and v in tx_to_idx:
                input_to_src.append(addr_to_idx[u])
                input_to_dst.append(tx_to_idx[v])
                input_to_w.append(d.get("value_btc", 0.0))
        elif d.get("edge_type") == "OUTPUT_TO":
            if u in tx_to_idx and v in addr_to_idx:
                output_to_src.append(tx_to_idx[u])
                output_to_dst.append(addr_to_idx[v])
                output_to_w.append(d.get("value_btc", 0.0))

    data["address", "INPUT_TO", "transaction"].edge_index = torch.tensor(
        [input_to_src, input_to_dst], dtype=torch.long
    )
    data["address", "INPUT_TO", "transaction"].edge_attr = torch.tensor(
        input_to_w, dtype=torch.float
    )
    data["transaction", "OUTPUT_TO", "address"].edge_index = torch.tensor(
        [output_to_src, output_to_dst], dtype=torch.long
    )
    data["transaction", "OUTPUT_TO", "address"].edge_attr = torch.tensor(
        output_to_w, dtype=torch.float
    )

    log.info(
        f"  PyG HeteroData: "
        f"{data['address'].num_nodes} address nodes, "
        f"{data['transaction'].num_nodes} tx nodes, "
        f"{len(input_to_src)} INPUT_TO edges, "
        f"{len(output_to_src)} OUTPUT_TO edges"
    )
    return data


# ── Graph validation ───────────────────────────────────────────────────────────

def validate_graph(feat: pd.DataFrame, projection_G: nx.Graph, tag: str) -> bool:
    """
    P0.2.6: Validate the graph and feature matrix.
    Returns True if all checks pass; False (with logged errors) otherwise.
    """
    log.info(f"Validating graph [{tag}] …")
    ok = True

    # Check for NaN in feature matrix
    nan_counts = feat[ADDRESS_FEATURE_COLS].isna().sum()
    if nan_counts.any():
        log.error(f"NaN features detected:\n{nan_counts[nan_counts > 0]}")
        ok = False
    else:
        log.info("  ✓ No NaN features")

    # Check node and edge counts
    n_nodes = projection_G.number_of_nodes()
    n_edges = projection_G.number_of_edges()
    log.info(f"  ✓ Projection graph: {n_nodes:,} nodes, {n_edges:,} edges")

    # Connected components
    components = list(nx.connected_components(projection_G))
    n_components = len(components)
    largest = max(len(c) for c in components) if components else 0
    isolated = sum(1 for c in components if len(c) == 1)
    log.info(
        f"  ✓ Connected components: {n_components:,} total, "
        f"largest={largest:,}, isolated={isolated:,}"
    )

    # Feature matrix shape
    log.info(f"  ✓ Feature matrix shape: {feat[ADDRESS_FEATURE_COLS].shape}")

    # risk_score range sanity (scores added later — just validate features here)
    for col in ["total_sent_btc", "total_received_btc", "avg_tx_value_btc"]:
        if (feat[col] < 0).any():
            log.error(f"  ✗ Negative values in {col}")
            ok = False

    if ok:
        log.info(f"  ✓ Validation PASSED for [{tag}]")
    else:
        log.error(f"  ✗ Validation FAILED for [{tag}]")
    return ok


# ── Main entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Bitcoin bipartite graph")
    parser.add_argument("--n-tx",   type=int,  default=50_000,
                        help="Number of transaction rows to fetch/use")
    parser.add_argument("--output", type=str,  default="data/",
                        help="Output directory")
    parser.add_argument("--offline", action="store_true",
                        help="Load from local parquet snapshot instead of BigQuery")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_k = args.n_tx // 1000
    tag = f"{n_k}k"
    parquet_path = out_dir / f"btc_snapshot_{tag}.parquet"

    t_start = time.time()

    # ── Step 1: Get raw data ──────────────────────────────────────────────────
    if args.offline:
        if not parquet_path.exists():
            log.error(
                f"Offline snapshot not found: {parquet_path}\n"
                f"Run without --offline first to create it."
            )
            sys.exit(1)
        df = load_from_parquet(parquet_path)
    else:
        df = pull_from_bigquery(args.n_tx)
        log.info(f"Saving snapshot → {parquet_path}")
        df.to_parquet(parquet_path, index=False, compression="snappy")
        log.info(f"  Saved ({parquet_path.stat().st_size / 1e6:.1f} MB)")

    # Ensure timestamp column is datetime
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)

    # BigQuery returns NUMERIC columns as decimal.Decimal — cast to float64
    # so all / SATOSHI divisions work correctly in feature engineering.
    numeric_cols = [
        "fee_satoshi", "total_input_sat", "total_output_sat",
        "input_value_sat", "output_value_sat",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    snapshot_now = df["block_timestamp"].max()
    log.info(f"Snapshot time range: {df['block_timestamp'].min()} → {snapshot_now}")

    # ── Step 2: Compute address features (F2–F14, excluding F1 and F7) ───────
    feat = _compute_address_features(df, snapshot_now)

    # ── Step 3: Build bipartite graph and address projection ──────────────────
    bipartite_G, projection_G = build_bipartite_and_projection(df, feat)

    # ── Step 4: Compute projection-based features (F1, F7) ───────────────────
    feat = _compute_projection_features(feat, projection_G)

    # ── Step 5: Validate ──────────────────────────────────────────────────────
    ok = validate_graph(feat, projection_G, tag)
    if not ok:
        log.error("Graph validation failed. Fix the issues above before proceeding.")
        sys.exit(1)

    # ── Step 6: Save feature table ────────────────────────────────────────────
    feat_path = out_dir / f"address_features_{tag}.parquet"
    feat.reset_index().to_parquet(feat_path, index=False, compression="snappy")
    log.info(f"Address features saved → {feat_path}")

    # ── Step 7: Build and save PyG HeteroData ─────────────────────────────────
    pyg_data = build_pyg_heterodata(df, feat, bipartite_G)
    if pyg_data is not None:
        import torch  # noqa: PLC0415 — lazy import; torch not required for feature-only runs
        pt_path = out_dir / f"graph_{tag}.pt"
        torch.save(pyg_data, pt_path)
        log.info(f"PyG graph saved → {pt_path}")

    elapsed = time.time() - t_start
    log.info(f"Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
