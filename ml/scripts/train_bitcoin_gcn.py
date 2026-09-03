"""
ml/scripts/train_bitcoin_gcn.py
==================================
P0.4.5 — Bitcoin GCN (weak-supervision, fail-closed)

Architecture
------------
Same 2-layer GCN as the validated Elliptic model, adapted for 14
Bitcoin address features (vs 165 Elliptic features). All hyperparameters
are frozen to match the Elliptic GCN:
  hidden = 128, dropout = 0.5, Adam lr = 1e-3, epochs = 100

Weak-supervision design
-----------------------
Labels come from co-spend clustering + seed list (P0.4.4):
  FLAGGED    : address is in a cluster containing a known seed (train set)
  SUSPICIOUS : address is in a cluster adjacent (1 hop) to a FLAGGED cluster
  UNKNOWN    : everything else

Training mask = FLAGGED | SUSPICIOUS only.
UNKNOWN nodes participate in message passing (graph structure learning)
but are EXCLUDED from the loss and all evaluation metrics.
This is structurally identical to how Elliptic handled unknown-label nodes.

Fail-closed behaviour
---------------------
If FLAGGED + SUSPICIOUS count = 0, the script exits with code 1 and a clear
error message. It does NOT fall back to training on UNKNOWN-class nodes.
The frozen spec requires positive weak-supervision signal; silent degradation
to an alternative objective violates the architecture contract.

This will happen with the 10k snapshot (no seed addresses in 30-minute window).
It will resolve once the 50k snapshot (90-day window) is available.

Risk ranking output
-------------------
After training, every address receives a continuous score in [0, 1]
(P(high-risk) from softmax), which is linearly mapped to 0–100.
This is a prioritisation ranking, not a calibrated probability.
The score answers: "Which addresses should be examined first?"

Outputs (only written if training succeeds)
-------------------------------------------
  models/gcn_bitcoin_v1.pt          — trained model + metadata
  data/btc_risk_scores.parquet      — address, risk_score (0–100), risk_tier
  data/btc_gcn_train_log.json       — per-epoch training log
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("btc_gcn")

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH    = Path("data/address_features_10k.parquet")
GRAPH_PATH       = Path("data/graph_10k.pt")
WEAK_LABELS_PATH = Path("data/btc_weak_labels.parquet")
MODEL_DIR        = Path("models")
OUT_DIR          = Path("data")

# ── Frozen hyperparameters (identical to Elliptic GCN) ────────────────────────
HIDDEN_DIM      = 128
DROPOUT         = 0.5
LR              = 1e-3
N_EPOCHS        = 100
FLAGGED_WEIGHT  = 50.0   # same class-weight ratio as Elliptic (high-risk : unknown)
LOG_EPOCHS      = {1, 10, 25, 50, 75, 100}
SEED            = 42

# Risk tier thresholds (score 0–100)
TIER_THRESHOLDS = {
    "CRITICAL": 75,
    "HIGH":     50,
    "MEDIUM":   25,
    "LOW":       0,
}

LABEL_TO_INT = {"flagged": 1, "suspicious": 1, "high_risk": 1,
                "unknown": 0, "eval_reference": -1}


# ── Model (identical to Elliptic GCN, input dim = 14) ─────────────────────────

class BitcoinGCN(nn.Module):
    """
    2-layer GCN for Bitcoin address risk ranking.

    Input: 14 engineered address features (P0.2.2 / FEATURE_SPEC.md)
    Output: 2 logits → softmax → P(high-risk)

    Architecture is identical to EllipticGCN; only the input dimension
    changes (14 vs 165). This is intentional: P0.3 validated the
    architecture on labeled data; P0.4.5 applies it to weak-supervised Bitcoin.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 dropout: float):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_bitcoin_graph(features_path: Path, graph_path: Path,
                       weak_labels_path: Path) -> Data:
    """
    Build a PyG Data object from:
      - Address feature matrix (N, 14) from address_features parquet
      - Address-address projection edge index from graph_10k.pt
      - Weak labels from btc_weak_labels.parquet

    Returns Data with:
      x            : (N, 14) float32 feature matrix
      edge_index   : (2, E) long — address projection edges
      y            : (N,) long — 1=high-risk, 0=unknown, -1=eval-reference
      train_mask   : FLAGGED | SUSPICIOUS only (non-zero positive signal)
      n_flagged    : count of label=1 nodes in train_mask
    """
    log.info("Loading address features …")
    feat_df = pd.read_parquet(features_path)
    # address_hash is already a column (the SHA-256[:16] of the original address)
    assert "address_hash" in feat_df.columns, \
        "Expected 'address_hash' column in address_features parquet"
    feat_cols = [c for c in feat_df.columns if c != "address_hash"]
    log.info(f"  Feature matrix: {feat_df.shape}  ({len(feat_cols)} feature cols)")
    log.info(f"  Feature columns: {feat_cols}")

    log.info("Loading projection graph …")
    pyg_data = torch.load(graph_path, weights_only=False)
    log.info(f"  PyG graph: {pyg_data}")

    log.info("Loading weak labels …")
    labels_df = pd.read_parquet(weak_labels_path)
    log.info(f"  Weak labels: {len(labels_df):,} addresses")
    log.info(f"  Label distribution:\n{labels_df['weak_label'].value_counts().to_string()}")

    # ── Hash weak-label addresses to match address_features ────────────────────
    # address_features stores address_hash = SHA-256(full_address)[:16]
    # weak_labels stores the full Bitcoin address string
    def _hash(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    labels_df["address_hash"] = labels_df["address"].map(_hash)

    # ── Align on address_hash ordering ────────────────────────────────────────
    feat_df = feat_df.sort_values("address_hash").reset_index(drop=True)

    # Build label array aligned with feat_df (ordered by address_hash)
    labels_map = labels_df.set_index("address_hash")["weak_label"].to_dict()
    y_raw = []
    for ah in feat_df["address_hash"]:
        lbl = labels_map.get(ah, "unknown")
        if lbl in ("flagged", "suspicious", "high_risk"):
            y_raw.append(1)
        elif lbl in ("eval_reference",):
            y_raw.append(-1)
        else:
            y_raw.append(0)
    y = torch.tensor(y_raw, dtype=torch.long)

    # ── Feature matrix ─────────────────────────────────────────────────────────
    x_np = feat_df[feat_cols].values.astype(np.float32)
    # NaN check — fill with 0 if any (should not happen after build_graph.py)
    if np.isnan(x_np).any():
        log.warning("NaN found in feature matrix — filling with 0")
        x_np = np.nan_to_num(x_np, nan=0.0)
    x = torch.tensor(x_np, dtype=torch.float)

    # ── Edge index: extract address→address projection from HeteroData ─────────
    # graph_10k.pt is a HeteroData with node types 'address' and 'transaction'.
    # The address projection edges are stored under ('address','INPUT_TO','transaction')
    # and ('transaction','OUTPUT_TO','address'). For the GCN we build a homogeneous
    # address-address edge set: addr A and addr B are connected if they share a tx.
    # We reconstruct this from the bipartite edges.
    try:
        addr_input  = pyg_data["address", "INPUT_TO", "transaction"].edge_index   # (2, E1)
        tx_output   = pyg_data["transaction", "OUTPUT_TO", "address"].edge_index  # (2, E2)
        # addr_input[0] = address node index, addr_input[1] = tx node index
        # tx_output[0]  = tx node index,      tx_output[1]  = address node index
        # Build addr→addr via shared transaction: for each tx, connect all input addrs
        # with all output addrs (and vice versa) to form the projection.
        # Simpler: just use the two bipartite halves as a heterogeneous message-passing
        # proxy via a concatenated edge list. For a homogeneous GCN, we create
        # addr→addr edges: (input_addr, output_addr) for each tx.
        tx_to_in  = {}  # tx_idx → list of input addr indices
        tx_to_out = {}  # tx_idx → list of output addr indices
        for a, t in addr_input.t().tolist():
            tx_to_in.setdefault(t, []).append(a)
        for t, a in tx_output.t().tolist():
            tx_to_out.setdefault(t, []).append(a)

        src_list, dst_list = [], []
        for t in set(tx_to_in) & set(tx_to_out):
            for a_in in tx_to_in[t]:
                for a_out in tx_to_out[t]:
                    src_list.append(a_in)
                    dst_list.append(a_out)
                    src_list.append(a_out)  # undirected
                    dst_list.append(a_in)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        # Deduplicate
        edge_index = torch.unique(edge_index, dim=1)
        log.info(f"  Address projection edges reconstructed: {edge_index.shape[1]:,}")
    except Exception as e:
        log.error(f"Cannot extract edge_index from graph: {e}")
        sys.exit(1)

    # Clamp edge index to valid node range (N = len(feat_df))
    n_nodes = len(feat_df)
    valid_mask = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum().item()
        log.warning(f"  Dropping {n_dropped} edges with out-of-range node indices")
        edge_index = edge_index[:, valid_mask]

    # ── Training mask: FLAGGED | SUSPICIOUS (label == 1) only ─────────────────
    train_mask = (y == 1)
    n_flagged  = int(train_mask.sum().item())

    log.info(f"\n  Nodes          : {n_nodes:,}")
    log.info(f"  Edges          : {edge_index.shape[1]:,}")
    log.info(f"  Feature dim    : {x.shape[1]}")
    log.info(f"  High-risk (1)  : {n_flagged:,}   ← training signal")
    log.info(f"  Unknown   (0)  : {int((y==0).sum()):,}  ← message-passing only")
    log.info(f"  Eval-ref  (-1) : {int((y==-1).sum()):,}  ← held out")

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        num_nodes=n_nodes,
    )
    data.n_flagged = n_flagged

    return data


# ── Fail-closed guard ──────────────────────────────────────────────────────────

def assert_positive_labels(data: Data) -> None:
    """
    Exit with code 1 if there are no FLAGGED/SUSPICIOUS training labels.

    This is intentional fail-closed behaviour per the frozen spec.
    The Bitcoin GCN requires positive weak-supervision signal to train.
    Silently degrading to an UNKNOWN-majority objective is not permitted.

    Expected cause: 10k snapshot covers only ~30 minutes and does not
    contain any seed addresses. Will resolve with the 50k snapshot.
    """
    if data.n_flagged == 0:
        log.error("=" * 65)
        log.error("TRAINING ABORTED — zero positive weak-supervision labels")
        log.error("=" * 65)
        log.error("")
        log.error("  Cause: No FLAGGED or SUSPICIOUS addresses found in the")
        log.error("         current Bitcoin snapshot.")
        log.error("")
        log.error("  This snapshot covers approximately 30 minutes of Bitcoin")
        log.error("  history. The known seed addresses (Hydra, Garantex, etc.)")
        log.error("  did not transact in that window.")
        log.error("")
        log.error("  Resolution:")
        log.error("    1. Wait for BigQuery quota reset (midnight IST).")
        log.error("    2. Run: python scripts/build_graph.py --n-tx 50000")
        log.error("    3. Run: python ml/scripts/build_weak_labels.py")
        log.error("    4. Re-run this script.")
        log.error("")
        log.error("  The frozen spec prohibits falling back to UNKNOWN-majority")
        log.error("  training. Exit code 1.")
        log.error("=" * 65)
        sys.exit(1)


# ── Training ───────────────────────────────────────────────────────────────────

def train_step(model, data, optimizer, class_weights, train_mask):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        logits[train_mask],
        data.y[train_mask],
        weight=class_weights,
    )
    loss.backward()
    optimizer.step()
    return loss, logits


def check_finite(epoch, loss, model, logits):
    if not torch.isfinite(loss):
        raise RuntimeError(f"Epoch {epoch}: NaN/Inf loss")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"Epoch {epoch}: NaN/Inf in logits")
    total_grad_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5
    log.info(f"    ✓ Loss finite: {loss.item():.6f}")
    log.info(f"    ✓ Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    log.info(f"    ✓ Gradient norm (L2): {total_grad_norm:.6f}")


# ── Risk ranking (P0.4.6 integrated) ──────────────────────────────────────────

@torch.no_grad()
def build_risk_ranking(model, data, feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    P0.4.6: Convert model output to 0–100 risk ranking.

    score = P(high-risk) from softmax, scaled to [0, 100].
    This is an investigator prioritisation ranking, NOT a calibrated probability.
    Tier boundaries: CRITICAL ≥ 75, HIGH ≥ 50, MEDIUM ≥ 25, LOW < 25.
    """
    model.eval()
    logits = model(data.x, data.edge_index)
    probs  = F.softmax(logits, dim=1)[:, 1].numpy()   # P(high-risk) per address

    risk_score = (probs * 100).clip(0, 100)

    def assign_tier(score):
        if score >= TIER_THRESHOLDS["CRITICAL"]: return "CRITICAL"
        if score >= TIER_THRESHOLDS["HIGH"]:     return "HIGH"
        if score >= TIER_THRESHOLDS["MEDIUM"]:   return "MEDIUM"
        return "LOW"

    risk_df = pd.DataFrame({
        "address_hash": feat_df["address_hash"].values,
        "risk_score":   risk_score,
        "risk_tier":    [assign_tier(s) for s in risk_score],
    })

    tier_counts = risk_df["risk_tier"].value_counts()
    log.info("\nRisk tier distribution:")
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        cnt = tier_counts.get(tier, 0)
        log.info(f"  {tier:10s}: {cnt:,} addresses")

    log.info(f"\nTop 10 highest-risk addresses:")
    top10 = risk_df.nlargest(10, "risk_score")[["address", "risk_score", "risk_tier"]]
    for _, row in top10.iterrows():
        log.info(f"  {row['address'][:40]:<42} score={row['risk_score']:5.1f}  {row['risk_tier']}")

    return risk_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data = load_bitcoin_graph(FEATURES_PATH, GRAPH_PATH, WEAK_LABELS_PATH)
    feat_df = pd.read_parquet(FEATURES_PATH).sort_values("address_hash").reset_index(drop=True)

    # ── Fail-closed: abort if no positive labels ───────────────────────────────
    assert_positive_labels(data)

    # ── (Execution reaches here only when n_flagged > 0) ──────────────────────
    log.info(f"\n✓ Positive training signal confirmed: {data.n_flagged} labelled nodes")

    # Class weights: high-risk weight = 50, unknown = 1
    class_weights = torch.tensor([1.0, FLAGGED_WEIGHT], dtype=torch.float)
    log.info(f"Class weights: unknown={class_weights[0]:.1f}, "
             f"high-risk={class_weights[1]:.1f}")

    # ── Model ──────────────────────────────────────────────────────────────────
    in_channels = data.x.shape[1]   # 14
    model = BitcoinGCN(in_channels=in_channels, hidden=HIDDEN_DIM,
                       out_channels=2, dropout=DROPOUT)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"\nModel: BitcoinGCN(in={in_channels}, hidden={HIDDEN_DIM}, "
             f"out=2, dropout={DROPOUT})")
    log.info(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Pre-training checks
    assert not torch.isnan(data.x).any(), "NaN in feature matrix"
    assert (data.y[data.train_mask] >= 0).all(), "Unknown label in train mask"
    log.info("Pre-training checks passed")

    # ── Training loop ──────────────────────────────────────────────────────────
    log.info(f"\nTraining for {N_EPOCHS} epochs …")
    epoch_log  = []
    t0         = time.time()
    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        loss, logits = train_step(model, data, optimizer,
                                  class_weights, data.train_mask)

        if epoch in LOG_EPOCHS:
            log.info(f"\nEpoch {epoch:>3d}/{N_EPOCHS}")
            check_finite(epoch, loss, model, logits)

            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            epoch_log.append({"epoch": epoch, "loss": float(loss.item())})

    elapsed = time.time() - t0
    log.info(f"\nTraining complete — {elapsed:.1f}s  ({elapsed/N_EPOCHS:.2f}s/epoch)")
    log.info(f"Best loss: {best_loss:.6f}")

    # ── Risk ranking (P0.4.6) ──────────────────────────────────────────────────
    log.info("\n── P0.4.6 Risk Ranking ──")
    model.load_state_dict(best_state)
    risk_df = build_risk_ranking(model, data, feat_df)
    risk_df.to_parquet(OUT_DIR / "btc_risk_scores.parquet", index=False)
    log.info(f"Risk scores saved → data/btc_risk_scores.parquet")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "in_channels": in_channels,
            "hidden":      HIDDEN_DIM,
            "out_channels": 2,
            "dropout":     DROPOUT,
        },
        "hyperparams": {
            "lr":             LR,
            "n_epochs":       N_EPOCHS,
            "flagged_weight": FLAGGED_WEIGHT,
            "seed":           SEED,
        },
        "training_info": {
            "n_flagged_train": data.n_flagged,
            "best_loss":       best_loss,
            "snapshot":        str(FEATURES_PATH),
        },
        "feature_dim":  in_channels,
        "feature_names": [c for c in feat_df.columns if c != "address"],
        "note": (
            "Bitcoin risk ranking model. Output is a 0–100 prioritisation score. "
            "NOT a calibrated probability. NOT a deterministic illicit classification. "
            "Trained with weak supervision from co-spend cluster seed propagation."
        ),
    }
    ckpt_path = MODEL_DIR / "gcn_bitcoin_v1.pt"
    torch.save(checkpoint, ckpt_path)
    log.info(f"Checkpoint saved → {ckpt_path}")

    full_log = {
        "model":      "gcn_bitcoin_v1",
        "config":     checkpoint["config"],
        "hyperparams": checkpoint["hyperparams"],
        "training_info": checkpoint["training_info"],
        "epoch_log":  epoch_log,
    }
    with open(OUT_DIR / "btc_gcn_train_log.json", "w") as f:
        json.dump(full_log, f, indent=2)
    log.info("Training log saved → data/btc_gcn_train_log.json")

    log.info("\nP0.4.5 + P0.4.6 COMPLETE ✓")


if __name__ == "__main__":
    main()
