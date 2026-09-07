"""
ml/scripts/train_gcn_elliptic.py
==================================
P0.3.5 — GCN training on the Elliptic dataset

Frozen configuration:
  Architecture : 2 GCN layers, hidden=128, dropout=0.5
  Loss         : class-weighted cross-entropy (illicit weight = 50, licit weight = 1)
  Optimizer    : Adam, lr=1e-3
  Epochs       : 100

Unknown-label nodes (y == -1) participate in message passing (their node
embeddings are computed and passed to neighbours) but are masked out of
both the training loss and all evaluation metrics.

Evaluation checkpoints (detailed logging):
  Epochs: 1, 10, 25, 50, 75, 100
  At each checkpoint: loss, logit range, gradient norms, val PR-AUC.

Outputs:
  models/gcn_elliptic_v1.pt   — trained model weights + metadata
  data/gcn_train_log.json     — per-epoch training log
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
from torch_geometric.nn import GCNConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gcn_train")

GRAPH_PATH  = Path("data/elliptic_graph.pt")
MODEL_DIR   = Path("models")
LOG_PATH    = Path("data/gcn_train_log.json")

# Frozen hyperparameters
HIDDEN_DIM    = 128
DROPOUT       = 0.5
LR            = 1e-3
N_EPOCHS      = 100
ILLICIT_WEIGHT = 50.0   # class-weight ratio: illicit : licit = 50 : 1
LOG_EPOCHS    = {1, 10, 25, 50, 75, 100}

SEED = 42


# ── Model ──────────────────────────────────────────────────────────────────────

class EllipticGCN(nn.Module):
    """
    Two-layer GCN for node classification.

    Architecture:
        Input (165) → GCNConv → ReLU → Dropout
                    → GCNConv → output logits (2)

    Why 2 layers?
    Each GCN layer aggregates information from 1 hop of neighbours.
    2 layers = each node sees its 2-hop neighbourhood, which is
    sufficient for the Elliptic transaction graph structure.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 dropout: float):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: aggregate 1-hop neighbourhood features
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: aggregate 2-hop neighbourhood (via 1-hop of previous layer)
        h = self.conv2(h, edge_index)
        return h  # raw logits, shape (N, 2)


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: EllipticGCN, data, mask: torch.Tensor,
             split_name: str) -> dict:
    """
    Evaluate on nodes selected by `mask`.
    Unknown nodes (y == -1) are never in any mask.
    """
    model.eval()
    logits = model(data.x, data.edge_index)

    # Only labeled nodes in this split
    y_true  = data.y[mask].numpy()
    logits_ = logits[mask]

    # Probabilities via softmax; column 1 = P(illicit)
    probs   = F.softmax(logits_, dim=1)[:, 1].numpy()
    y_pred  = (probs >= 0.5).astype(int)

    # Guard against degenerate cases (all same prediction)
    unique_true = np.unique(y_true)
    if len(unique_true) < 2:
        log.warning(f"  [{split_name}] Only one class in y_true — metrics may be degenerate")

    prec   = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    prauc  = average_precision_score(y_true, probs) if len(unique_true) > 1 else 0.0
    rocauc = roc_auc_score(y_true, probs) if len(unique_true) > 1 else 0.5

    return {
        "precision": float(prec),
        "recall":    float(recall),
        "f1":        float(f1),
        "pr_auc":    float(prauc),
        "roc_auc":   float(rocauc),
    }


def check_finite(epoch: int, loss: torch.Tensor, model: EllipticGCN,
                 logits: torch.Tensor):
    """
    At logged checkpoints, verify loss, logits, and gradient norms are finite.
    Raises RuntimeError if NaN/Inf is detected.
    """
    # Loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Epoch {epoch}: NaN/Inf loss detected — training aborted")

    # Logits
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"Epoch {epoch}: NaN/Inf in logits — training aborted")

    # Gradient norms
    total_grad_norm = 0.0
    any_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            any_grad = True
            norm = param.grad.data.norm(2).item()
            if not np.isfinite(norm):
                raise RuntimeError(
                    f"Epoch {epoch}: NaN/Inf gradient in param '{name}'"
                )
            total_grad_norm += norm ** 2
    total_grad_norm = total_grad_norm ** 0.5

    log.info(f"    ✓ Loss finite: {loss.item():.6f}")
    log.info(f"    ✓ Logit range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    if any_grad:
        log.info(f"    ✓ Gradient norm (L2): {total_grad_norm:.6f}")


# ── Training loop ──────────────────────────────────────────────────────────────

def train(model: EllipticGCN, data, optimizer, class_weights: torch.Tensor,
          train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single training step.
    Loss is computed ONLY on labeled training nodes (train_mask).
    Unknown-label nodes are in the graph but excluded from loss.
    """
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    # Apply training mask — only labeled nodes
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

    # ── Class weights ──────────────────────────────────────────────────────────
    # illicit (1) → weight 50, licit (0) → weight 1
    # Ordered by class index: [weight_licit, weight_illicit]
    class_weights = torch.tensor([1.0, ILLICIT_WEIGHT], dtype=torch.float)
    log.info(f"\nClass weights: licit={class_weights[0]:.1f}, "
             f"illicit={class_weights[1]:.1f}  (ratio 1:{int(ILLICIT_WEIGHT)})")

    # ── Model ──────────────────────────────────────────────────────────────────
    in_channels = data.x.shape[1]   # 165
    model = EllipticGCN(
        in_channels=in_channels,
        hidden=HIDDEN_DIM,
        out_channels=2,
        dropout=DROPOUT,
    )
    log.info("\nModel architecture:")
    log.info(f"  EllipticGCN(in={in_channels}, hidden={HIDDEN_DIM}, "
             f"out=2, dropout={DROPOUT})")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Verify data before training ────────────────────────────────────────────
    assert not torch.isnan(data.x).any(), "NaN in input features"
    assert (data.y[data.train_mask] >= 0).all(), "Unknown label in train mask"
    assert (data.y[data.val_mask]   >= 0).all(), "Unknown label in val mask"
    assert (data.y[data.test_mask]  >= 0).all(), "Unknown label in test mask"
    log.info("\nPre-training checks passed (no NaN features, no unknown in masks)")

    # ── Training loop ──────────────────────────────────────────────────────────
    log.info(f"\nTraining for {N_EPOCHS} epochs …")
    log.info(f"  Checkpoints at epochs: {sorted(LOG_EPOCHS)}")

    epoch_log  = []
    t0         = time.time()
    best_prauc = 0.0
    best_state = None

    for epoch in range(1, N_EPOCHS + 1):
        loss, logits = train(model, data, optimizer, class_weights, data.train_mask)

        log_this = (epoch in LOG_EPOCHS)
        if log_this:
            log.info(f"\nEpoch {epoch:>3d}/{N_EPOCHS}")
            check_finite(epoch, loss, model, logits)

            val_metrics = evaluate(model, data, data.val_mask, "val")
            log.info(f"    Val  PR-AUC={val_metrics['pr_auc']:.4f}  "
                     f"ROC-AUC={val_metrics['roc_auc']:.4f}  "
                     f"F1={val_metrics['f1']:.4f}")

            # Track best val PR-AUC model for checkpoint
            if val_metrics["pr_auc"] > best_prauc:
                best_prauc = val_metrics["pr_auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            epoch_log.append({
                "epoch":     epoch,
                "loss":      float(loss.item()),
                "val":       val_metrics,
            })

    elapsed = time.time() - t0
    log.info(f"\nTraining complete — {elapsed:.1f}s  ({elapsed/N_EPOCHS:.2f}s/epoch)")
    log.info(f"Best val PR-AUC during training: {best_prauc:.4f}")

    # ── Final evaluation on test set ───────────────────────────────────────────
    log.info("\n── Final evaluation on TEST set ──")
    # Load best checkpoint weights
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data, data.test_mask, "test")
    log.info(f"  Precision : {test_metrics['precision']:.4f}")
    log.info(f"  Recall    : {test_metrics['recall']:.4f}")
    log.info(f"  F1        : {test_metrics['f1']:.4f}")
    log.info(f"  PR-AUC    : {test_metrics['pr_auc']:.4f}")
    log.info(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")

    train_final = evaluate(model, data, data.train_mask, "train")
    log.info(f"\n  Train PR-AUC (final): {train_final['pr_auc']:.4f}  "
             f"(vs val best: {best_prauc:.4f})  "
             f"(vs test: {test_metrics['pr_auc']:.4f})")

    # ── Save model checkpoint ──────────────────────────────────────────────────
    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "in_channels":  in_channels,
            "hidden":       HIDDEN_DIM,
            "out_channels": 2,
            "dropout":      DROPOUT,
        },
        "hyperparams": {
            "lr":             LR,
            "n_epochs":       N_EPOCHS,
            "illicit_weight": ILLICIT_WEIGHT,
            "seed":           SEED,
        },
        "metrics": {
            "best_val_pr_auc": best_prauc,
            "test":            test_metrics,
            "train_final":     train_final,
        },
        "feature_dim":   in_channels,
        "n_classes":     2,
        "dataset":       "elliptic",
        "purpose":       "methodology_validation_only",
        "note": (
            "Trained on Elliptic labeled data for methodology validation. "
            "Not used for Bitcoin risk ranking. "
            "Bitcoin uses an independent 14-feature space and weak supervision."
        ),
    }
    ckpt_path = MODEL_DIR / "gcn_elliptic_v1.pt"
    torch.save(checkpoint, ckpt_path)
    log.info(f"\nCheckpoint saved → {ckpt_path}")

    # ── Save training log ──────────────────────────────────────────────────────
    full_log = {
        "model":      "gcn_elliptic_v1",
        "config":     checkpoint["config"],
        "hyperparams": checkpoint["hyperparams"],
        "epoch_log":  epoch_log,
        "test":       test_metrics,
        "best_val_pr_auc": best_prauc,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(full_log, f, indent=2)
    log.info(f"Training log saved → {LOG_PATH}")

    log.info("\nP0.3.5 GCN COMPLETE ✓")


if __name__ == "__main__":
    main()
