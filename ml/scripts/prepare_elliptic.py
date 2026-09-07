"""
ml/scripts/prepare_elliptic.py
================================
P0.3.1 + P0.3.2: Load, inspect, and create a fixed temporal train/val/test
split for the Elliptic Bitcoin dataset.

Elliptic dataset structure:
  elliptic_txs_features.csv  — (203769, 167): col0=txId, col1=time_step, cols2-166=165 node features
                                (94 local + 71 aggregate features per the Elliptic paper)
  elliptic_txs_classes.csv   — txId, class  (1=illicit, 2=licit, unknown)
  elliptic_txs_edgelist.csv  — txId1, txId2 (directed edges)

Temporal split rule (P0.3.2):
  - Time steps 1–34  → train   (~70%)
  - Time steps 35–41 → val     (~14%)
  - Time steps 42–49 → test    (~16%)
  Unknown-label nodes participate in message passing but are excluded
  from loss and evaluation metrics.

Outputs:
  data/elliptic_split.json   — {train: [txId, ...], val: [...], test: [...]}
  data/elliptic_graph.pt     — PyG HeteroData (or homogeneous Data object)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("elliptic")

ELLIPTIC_DIR = Path("data/elliptic/elliptic_bitcoin_dataset")
OUT_DIR = Path("data")

# Temporal split boundaries (inclusive, 1-indexed time steps)
TRAIN_MAX = 34
VAL_MAX   = 41
# steps 42-49 → test

LABEL_MAP = {"1": 1, "2": 0}   # illicit=1, licit=0, unknown→NaN


def load_elliptic(elliptic_dir: Path | None = None):
    """Load all three Elliptic CSVs and merge into a single node DataFrame."""
    log.info("Loading Elliptic dataset …")
    edir = elliptic_dir or ELLIPTIC_DIR

    # features: no header — col0=txId, col1=time_step, cols2-95=94 features
    features = pd.read_csv(
        edir / "elliptic_txs_features.csv",
        header=None,
        dtype={0: int, 1: int},
    )
    feat_cols = [f"f{i}" for i in range(features.shape[1] - 2)]
    features.columns = ["txId", "time_step"] + feat_cols
    log.info(f"  Features: {features.shape}  ({len(feat_cols)} feature columns)")

    classes = pd.read_csv(edir / "elliptic_txs_classes.csv")
    log.info(f"  Classes:  {classes.shape}")

    edges = pd.read_csv(edir / "elliptic_txs_edgelist.csv")
    log.info(f"  Edges:    {edges.shape}")

    # Merge features + labels
    nodes = features.merge(classes, on="txId", how="left")
    nodes["label"] = nodes["class"].map({"1": 1, "2": 0})  # unknown → NaN

    # P0.3.1: Log class distribution
    log.info("\nClass distribution:")
    vc = nodes["class"].value_counts(dropna=False)
    total = len(nodes)
    for cls, cnt in vc.items():
        log.info(f"  {str(cls):10s}: {cnt:7,d}  ({100*cnt/total:.1f}%)")

    return nodes, edges


def make_temporal_split(nodes: pd.DataFrame) -> dict:
    """
    P0.3.2: Create fixed temporal train/val/test split.
    Returns a dict with lists of txIds for each split.
    Only labeled (non-unknown) nodes are in the split.
    """
    labeled = nodes[nodes["label"].notna()]
    train = labeled[labeled["time_step"] <= TRAIN_MAX]["txId"].tolist()
    val   = labeled[(labeled["time_step"] > TRAIN_MAX) & (labeled["time_step"] <= VAL_MAX)]["txId"].tolist()
    test  = labeled[labeled["time_step"] > VAL_MAX]["txId"].tolist()

    log.info("\nTemporal split (labeled nodes only):")
    log.info(f"  Train (steps 1–{TRAIN_MAX}):  {len(train):,} nodes")
    log.info(f"  Val   (steps {TRAIN_MAX+1}–{VAL_MAX}): {len(val):,} nodes")
    log.info(f"  Test  (steps {VAL_MAX+1}–49):  {len(test):,} nodes")

    # Verify no seed/txId overlap between splits
    assert not set(train) & set(val),  "LEAKAGE: train ∩ val is non-empty"
    assert not set(train) & set(test), "LEAKAGE: train ∩ test is non-empty"
    assert not set(val)   & set(test), "LEAKAGE: val ∩ test is non-empty"
    log.info("  ✓ No overlap between train/val/test sets")

    # Illicit ratio per split
    tx_label = nodes.set_index("txId")["label"]
    for name, ids in [("train", train), ("val", val), ("test", test)]:
        labels = tx_label.loc[ids].dropna()
        illicit_pct = 100 * labels.sum() / len(labels) if len(labels) > 0 else 0
        log.info(f"  {name}: illicit ratio = {illicit_pct:.1f}%")

    return {"train": train, "val": val, "test": test}


def build_pyg_data(nodes: pd.DataFrame, edges: pd.DataFrame, split: dict) -> Data:
    """
    Build a PyTorch Geometric Data object for the Elliptic graph.

    Node features: 94 columns (f0–f93)
    Labels: 0=licit, 1=illicit, -1=unknown (unknown participates in message
            passing but is excluded from loss and evaluation)
    Edge index: directed, from txId1 → txId2
    Train/val/test masks: boolean tensors over all nodes
    """
    log.info("\nBuilding PyG Data object …")

    # Node ordering — we need a consistent integer index for all 203k nodes
    feat_cols = [c for c in nodes.columns if c.startswith("f")]  # f0–f93
    assert len(feat_cols) == 165, f"Expected 165 feature cols, got {len(feat_cols)}"

    # Sort by txId for deterministic ordering
    nodes = nodes.sort_values("txId").reset_index(drop=True)
    txid_to_idx = {txid: i for i, txid in enumerate(nodes["txId"])}

    # Node feature matrix: (N, 94) float32
    x = torch.tensor(nodes[feat_cols].values, dtype=torch.float)
    assert not torch.isnan(x).any(), "NaN in Elliptic feature matrix"
    log.info(f"  Node feature matrix: {x.shape}")

    # Labels: illicit=1, licit=0, unknown=-1
    y_raw = nodes["label"].fillna(-1).astype(int)
    y = torch.tensor(y_raw.values, dtype=torch.long)
    log.info(f"  Labels: {(y==1).sum().item():,} illicit, "
             f"{(y==0).sum().item():,} licit, "
             f"{(y==-1).sum().item():,} unknown")

    # Edge index: map txIds → node indices, drop edges with unknown nodes
    valid_txids = set(txid_to_idx.keys())
    edges_filtered = edges[
        edges["txId1"].isin(valid_txids) & edges["txId2"].isin(valid_txids)
    ]
    src = edges_filtered["txId1"].map(txid_to_idx).values
    dst = edges_filtered["txId2"].map(txid_to_idx).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    log.info(f"  Edge index: {edge_index.shape}  ({edge_index.shape[1]:,} edges)")

    # Train / val / test masks
    train_set = set(split["train"])
    val_set   = set(split["val"])
    test_set  = set(split["test"])

    train_mask = torch.tensor(
        [txid in train_set for txid in nodes["txId"]], dtype=torch.bool
    )
    val_mask = torch.tensor(
        [txid in val_set for txid in nodes["txId"]], dtype=torch.bool
    )
    test_mask = torch.tensor(
        [txid in test_set for txid in nodes["txId"]], dtype=torch.bool
    )
    log.info(f"  Train mask: {train_mask.sum().item():,} nodes")
    log.info(f"  Val   mask: {val_mask.sum().item():,} nodes")
    log.info(f"  Test  mask: {test_mask.sum().item():,} nodes")

    # Assemble PyG Data object
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=len(nodes),
    )

    # Final shape check
    assert data.x.shape == (len(nodes), 165), \
        f"Feature shape wrong: {data.x.shape}"  # (N, 165)=94 local + 71 agg
    assert data.y.shape == (len(nodes),), \
        f"Label shape wrong: {data.y.shape}"

    log.info("\n  PyG Data summary:")
    log.info(f"    x.shape       = {data.x.shape}")
    log.info(f"    y.shape       = {data.y.shape}")
    log.info(f"    edge_index    = {data.edge_index.shape}")
    log.info(f"    train nodes   = {data.train_mask.sum().item():,}")
    log.info(f"    val   nodes   = {data.val_mask.sum().item():,}")
    log.info(f"    test  nodes   = {data.test_mask.sum().item():,}")

    return data


def main():
    nodes, edges = load_elliptic()
    split = make_temporal_split(nodes)

    # Save split
    split_path = OUT_DIR / "elliptic_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f)
    log.info(f"\nSplit saved → {split_path}")

    # Build and save PyG graph
    data = build_pyg_data(nodes, edges, split)
    graph_path = OUT_DIR / "elliptic_graph.pt"
    torch.save(data, graph_path)
    log.info(f"PyG graph saved → {graph_path}")

    log.info("\nP0.3.1/P0.3.2 COMPLETE ✓")


if __name__ == "__main__":
    main()
