"""
ml/scripts/train_sage_elliptic.py
====================================
P0.3.6 — GraphSAGE training on the Elliptic dataset

Everything is kept identical to the GCN run (P0.3.5) so that the
architecture comparison is fair:
  - Same temporal split (train/val/test masks)
  - Same 165-feature input
  - Same labels and unknown-node masking
  - Same class weights (illicit:licit = 50:1)
  - Same optimizer (Adam, lr=1e-3)
  - Same epoch count (100)
  - Same checkpoint epochs (1, 10, 25, 50, 75, 100)
  - Same evaluation code

Only the graph-convolution layer changes:
  GCN  → GCNConv  (spectral, normalised adjacency)
  SAGE → SAGEConv  (spatial, mean neighbor aggregation — inductive)

GraphSAGE aggregation (mean variant):
  h_v^(k) = W · concat(h_v^(k-1),  mean_{u ∈ N(v)} h_u^(k-1))

This is inductive: node representations are computed by sampling and
aggregating features from a node's local neighbourhood rather than
from the global graph Laplacian, making it better suited to
transductive tasks where the graph structure may change at inference.

Outputs:
  models/sage_elliptic_v1.pt      — trained model weights + metadata
  data/sage_train_log.json        — per-epoch training log
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import SAGEConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sage_train")

GRAPH_PATH = Path("data/elliptic_graph.pt")
MODEL_DIR  = Path("models")
LOG_PATH   = Path("data/sage_train_log.json")

# Frozen hyperparameters — identical to GCN
HIDDEN_DIM     = 128
DROPOUT        = 0.5
LR             = 1e-3
N_EPOCHS       = 100
ILLICIT_WEIGHT = 50.0
LOG_EPOCHS     = {1, 10, 25, 50, 75, 100}
SEED           = 42


# ── Model ──────────────────────────────────────────────────────────────────────

class EllipticSAGE(nn.Module):
    """
    Two-layer GraphSAGE for node classification.

    Architecture:
        Input (165) → SAGEConv (mean) → ReLU → Dropout
                    → SAGEConv (mean) → output logits (2)

    SAGEConv (mean aggregator) computes:
        h^(k) = W · concat(h_self, mean(h_neighbours))
    followed by L2 normalisation.

    Why GraphSAGE vs GCN?
    GCN uses the global normalised adjacency matrix (spectral approach).
    GraphSAGE samples and aggregates neighbours explicitly (spatial approach).
    GraphSAGE is inductive: it learns an aggregation function rather than
    fixed node embeddings, which generalises better to unseen graph structures.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 dropout: float):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden, aggr="mean")
        self.conv2   = SAGEConv(hidden, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # raw logits, shape (N, 2)


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: EllipticSAGE, data, mask: torch.Tensor,
             split_name: str) -> dict:
    """
    Evaluate on labeled nodes selected by `mask`.
    Returns split-specific metrics including split prevalence for PR-AUC reference.
    """
    model.eval()
    logits = model(data.x, data.edge_index)

    y_true  = data.y[mask].numpy()
    logits_ = logits[mask]
    probs   = F.softmax(logits_, dim=1)[:, 1].numpy()
    y_pred  = (probs >= 0.5).astype(int)

    unique_true = np.unique(y_true)
    if len(unique_true) < 2:
        log.warning(f"  [{split_name}] Only one class in y_true")

    prec   = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    prauc  = average_precision_score(y_true, probs) if len(unique_true) > 1 else 0.0
    rocauc = roc_auc_score(y_true, probs) if len(unique_true) > 1 else 0.5

    # Split-specific prevalence (correct no-skill PR-AUC reference for this split)
    prevalence = float(y_true.mean())

    return {
        "precision":  float(prec),
        "recall":     float(recall),
        "f1":         float(f1),
        "pr_auc":     float(prauc),
        "roc_auc":    float(rocauc),
        "prevalence": prevalence,          # split-specific no-skill PR-AUC reference
    }


def check_finite(epoch: int, loss: torch.Tensor, model: EllipticSAGE,
                 logits: torch.Tensor):
    """Verify loss, logits, and gradient norms are all finite."""
    if not torch.isfinite(loss):
        raise RuntimeError(f"Epoch {epoch}: NaN/Inf loss — training aborted")
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"Epoch {epoch}: NaN/Inf in logits — training aborted")

    total_grad_norm = 0.0
    any_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            any_grad = True
            norm = param.grad.data.norm(2).item()
            if not np.isfinite(norm):
                raise RuntimeError(f"Epoch {epoch}: NaN/Inf gradient in '{name}'")
            total_grad_norm += norm ** 2
    total_grad_norm = total_grad_norm ** 0.5

    log.info(f"    ✓ Loss finite: {loss.item():.6f}")
    log.info(f"    ✓ Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    if any_grad:
        log.info(f"    ✓ Gradient norm (L2): {total_grad_norm:.6f}")


# ── Training ───────────────────────────────────────────────────────────────────

def train_step(model: EllipticSAGE, data, optimizer,
               class_weights: torch.Tensor,
               train_mask: torch.Tensor) -> tuple:
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading Elliptic PyG graph …")
    data = torch.load(GRAPH_PATH, weights_only=False)
    log.info(f"  Nodes: {data.num_nodes:,}  |  Edges: {data.edge_index.shape[1]:,}")
    log.info(f"  Feature dim: {data.x.shape[1]}")
    log.info(f"  Train: {data.train_mask.sum().item():,}  "
             f"Val: {data.val_mask.sum().item():,}  "
             f"Test: {data.test_mask.sum().item():,}")

    # Log split-specific prevalences (correct PR-AUC no-skill references)
    for name, mask in [("train", data.train_mask),
                       ("val",   data.val_mask),
                       ("test",  data.test_mask)]:
        y = data.y[mask].numpy()
        prev = y.mean()
        log.info(f"  {name} illicit prevalence: {100*prev:.1f}%  "
                 f"(no-skill PR-AUC ref ≈ {prev:.4f})")

    # ── Class weights ──────────────────────────────────────────────────────────
    class_weights = torch.tensor([1.0, ILLICIT_WEIGHT], dtype=torch.float)
    log.info(f"\nClass weights: licit={class_weights[0]:.1f}, "
             f"illicit={class_weights[1]:.1f}  (ratio 1:{int(ILLICIT_WEIGHT)})")

    # ── Model ──────────────────────────────────────────────────────────────────
    in_channels = data.x.shape[1]
    model = EllipticSAGE(
        in_channels=in_channels,
        hidden=HIDDEN_DIM,
        out_channels=2,
        dropout=DROPOUT,
    )
    log.info("\nModel architecture:")
    log.info(f"  EllipticSAGE(in={in_channels}, hidden={HIDDEN_DIM}, "
             f"out=2, dropout={DROPOUT}, aggr=mean)")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Pre-training checks ────────────────────────────────────────────────────
    assert not torch.isnan(data.x).any(), "NaN in input features"
    assert (data.y[data.train_mask] >= 0).all(), "Unknown label in train mask"
    assert (data.y[data.val_mask]   >= 0).all(), "Unknown label in val mask"
    assert (data.y[data.test_mask]  >= 0).all(), "Unknown label in test mask"
    log.info("\nPre-training checks passed")

    # ── Training loop ──────────────────────────────────────────────────────────
    log.info(f"\nTraining for {N_EPOCHS} epochs …")
    log.info(f"  Checkpoints at epochs: {sorted(LOG_EPOCHS)}")

    epoch_log  = []
    t0         = time.time()
    best_prauc = 0.0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        loss, logits = train_step(model, data, optimizer,
                                  class_weights, data.train_mask)

        if epoch in LOG_EPOCHS:
            log.info(f"\nEpoch {epoch:>3d}/{N_EPOCHS}")
            check_finite(epoch, loss, model, logits)

            val_m = evaluate(model, data, data.val_mask, "val")
            log.info(f"    Val  PR-AUC={val_m['pr_auc']:.4f}  "
                     f"(no-skill ref≈{val_m['prevalence']:.4f})  "
                     f"ROC-AUC={val_m['roc_auc']:.4f}  "
                     f"F1={val_m['f1']:.4f}")

            if val_m["pr_auc"] > best_prauc:
                best_prauc = val_m["pr_auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            epoch_log.append({
                "epoch": epoch,
                "loss":  float(loss.item()),
                "val":   val_m,
            })

    elapsed = time.time() - t0
    log.info(f"\nTraining complete — {elapsed:.1f}s  ({elapsed/N_EPOCHS:.2f}s/epoch)")
    log.info(f"Best val PR-AUC during training: {best_prauc:.4f}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    log.info("\n── Final evaluation on TEST set ──")
    model.load_state_dict(best_state)
    test_m  = evaluate(model, data, data.test_mask, "test")
    train_m = evaluate(model, data, data.train_mask, "train")

    log.info(f"  Precision   : {test_m['precision']:.4f}")
    log.info(f"  Recall      : {test_m['recall']:.4f}")
    log.info(f"  F1          : {test_m['f1']:.4f}")
    log.info(f"  PR-AUC      : {test_m['pr_auc']:.4f}  "
             f"(no-skill ref ≈ {test_m['prevalence']:.4f})")
    log.info(f"  ROC-AUC     : {test_m['roc_auc']:.4f}")
    log.info(f"\n  Train PR-AUC (final): {train_m['pr_auc']:.4f}  "
             f"(vs val best: {best_prauc:.4f})  "
             f"(vs test: {test_m['pr_auc']:.4f})")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "in_channels":  in_channels,
            "hidden":       HIDDEN_DIM,
            "out_channels": 2,
            "dropout":      DROPOUT,
            "aggr":         "mean",
        },
        "hyperparams": {
            "lr":             LR,
            "n_epochs":       N_EPOCHS,
            "illicit_weight": ILLICIT_WEIGHT,
            "seed":           SEED,
        },
        "metrics": {
            "best_val_pr_auc": best_prauc,
            "test":            test_m,
            "train_final":     train_m,
        },
        "feature_dim": in_channels,
        "n_classes":   2,
        "dataset":     "elliptic",
        "purpose":     "methodology_validation_only",
        "note": (
            "GraphSAGE trained on Elliptic for architecture comparison (P0.3.6). "
            "Not used for Bitcoin risk ranking. "
            "Bitcoin uses an independent 14-feature space and weak supervision."
        ),
    }
    ckpt_path = MODEL_DIR / "sage_elliptic_v1.pt"
    torch.save(checkpoint, ckpt_path)
    log.info(f"\nCheckpoint saved → {ckpt_path}")

    full_log = {
        "model":      "sage_elliptic_v1",
        "config":     checkpoint["config"],
        "hyperparams": checkpoint["hyperparams"],
        "epoch_log":  epoch_log,
        "test":       test_m,
        "best_val_pr_auc": best_prauc,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(full_log, f, indent=2)
    log.info(f"Training log saved → {LOG_PATH}")

    log.info("\nP0.3.6 GraphSAGE COMPLETE ✓")


if __name__ == "__main__":
    main()
