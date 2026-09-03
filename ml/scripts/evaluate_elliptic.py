"""
ml/scripts/evaluate_elliptic.py
==================================
P0.3.7 — Four-model comparison on the fixed temporal test split
P0.3.8 — Elliptic GCN calibration (GCN only, per frozen spec)
P0.3.9 — Checkpoint reload verification

Every row in the final table is computed fresh on the same test mask
and using the same metric functions. No values are copied from earlier runs.

Split-specific no-skill PR-AUC references:
  train prevalence ≈ 0.1158
  val   prevalence ≈ 0.0862
  test  prevalence ≈ 0.0461   ← used as the no-skill reference in the table

Outputs:
  data/elliptic_eval_report.json  — machine-readable results
  (reliability diagram logged inline; Brier score printed)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval")

GRAPH_PATH = Path("data/elliptic_graph.pt")
GCN_CKPT   = Path("models/gcn_elliptic_v1.pt")
SAGE_CKPT  = Path("models/sage_elliptic_v1.pt")
ELLIPTIC_DIR = Path("data/elliptic/elliptic_bitcoin_dataset")
OUT_PATH   = Path("data/elliptic_eval_report.json")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Model definitions (must match training scripts exactly) ────────────────────

class EllipticGCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden)
        self.conv2   = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class EllipticSAGE(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden, aggr="mean")
        self.conv2   = SAGEConv(hidden, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


# ── Shared metric computation ──────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray) -> dict:
    """
    All metrics computed fresh. Positive class = illicit (1).

    y_true  : ground-truth labels (0=licit, 1=illicit)
    y_pred  : hard predictions at threshold 0.5
    y_score : continuous illicit probability / score (higher = more illicit)
    """
    prec   = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    # Guard: PR-AUC and ROC-AUC require at least 2 classes in y_true
    unique = np.unique(y_true)
    prauc  = average_precision_score(y_true, y_score) if len(unique) > 1 else float("nan")
    rocauc = roc_auc_score(y_true, y_score)            if len(unique) > 1 else 0.5

    return {
        "precision":  float(prec),
        "recall":     float(recall),
        "f1":         float(f1),
        "pr_auc":     float(prauc),
        "roc_auc":    float(rocauc),
        "n_predicted_positive": int(y_pred.sum()),
        "n_actual_positive":    int(y_true.sum()),
        "n_total":              int(len(y_true)),
        "prevalence":           float(y_true.mean()),
    }


# ── P0.3.7 helpers ─────────────────────────────────────────────────────────────

def evaluate_majority(data) -> dict:
    """
    P0.3.7 row 1: Majority — always predict licit.
    Score = 0.0 (constant) for all nodes.
    """
    mask   = data.test_mask
    y_true = data.y[mask].numpy()
    y_pred  = np.zeros(len(y_true), dtype=int)
    y_score = np.zeros(len(y_true), dtype=float)
    return compute_metrics(y_true, y_pred, y_score)


def evaluate_heuristic(data) -> dict:
    """
    P0.3.7 row 2: Heuristic — degree + flow proxy.
    Same logic as baselines_elliptic.py, recomputed on the test mask.
    Threshold K = number of actual positives in test split (matches prevalence).
    """
    mask    = data.test_mask
    test_idx = torch.where(mask)[0].numpy()
    y_true  = data.y[mask].numpy()
    n_nodes = data.num_nodes

    # Degree
    edge_index = data.edge_index.numpy()
    degree = np.zeros(n_nodes, dtype=int)
    np.add.at(degree, edge_index[0], 1)
    np.add.at(degree, edge_index[1], 1)
    degree_test = degree[test_idx].astype(float)

    # Flow proxy: first feature column (column 0 of x)
    flow_test = np.abs(data.x[test_idx, 0].numpy())

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    score  = norm01(degree_test) + norm01(flow_test)
    k      = max(1, int(np.round(y_true.mean() * len(y_true))))
    thresh = np.sort(score)[::-1][k - 1]
    y_pred = (score >= thresh).astype(int)

    return compute_metrics(y_true, y_pred, score)


@torch.no_grad()
def evaluate_model(model, data, mask) -> dict:
    """
    P0.3.7 rows 3 and 4: GCN / GraphSAGE.
    Score = P(illicit) from softmax.
    Threshold = 0.5.
    """
    model.eval()
    logits = model(data.x, data.edge_index)
    probs  = F.softmax(logits[mask], dim=1)[:, 1].numpy()
    y_true = data.y[mask].numpy()
    y_pred = (probs >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, probs)


# ── P0.3.8 — GCN Calibration (frozen spec: GCN only, not SAGE) ────────────────

@torch.no_grad()
def get_logits_and_labels(model, data, mask):
    """Extract raw logits and true labels for a given mask."""
    model.eval()
    logits = model(data.x, data.edge_index)
    y_true  = data.y[mask].numpy()
    y_logit = logits[mask].numpy()      # shape (N, 2)
    return y_logit, y_true


def calibrate_gcn(gcn_model, data) -> dict:
    """
    P0.3.8: Calibrate GCN outputs using Platt scaling (LogisticRegression).

    Frozen spec:
      C = 1.0, max_iter = 1000
      Fit on validation logits → validation labels
      Evaluate on test logits  → calibrated test probabilities

    Returns dict with Brier score before/after calibration and
    reliability diagram bin data.
    """
    log.info("\n── P0.3.8 GCN Calibration ──")
    log.info("  Fitting Platt scaling on validation logits …")

    # ── Val: fit calibrator ────────────────────────────────────────────────────
    val_logits, val_labels = get_logits_and_labels(gcn_model, data, data.val_mask)
    # For Platt scaling: use the raw logit of the positive class (col 1)
    val_scores_raw = val_logits[:, 1].reshape(-1, 1)
    val_probs_raw  = torch.sigmoid(torch.tensor(val_logits[:, 1])).numpy()

    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    calibrator.fit(val_scores_raw, val_labels)
    log.info(f"  Calibrator fitted on {len(val_labels):,} val nodes")
    log.info(f"  Calibrator intercept: {calibrator.intercept_[0]:.4f}  "
             f"coef: {calibrator.coef_[0][0]:.4f}")

    # ── Test: apply calibrated probabilities ───────────────────────────────────
    test_logits, test_labels = get_logits_and_labels(gcn_model, data, data.test_mask)
    test_scores_raw = test_logits[:, 1].reshape(-1, 1)
    test_probs_raw  = torch.sigmoid(torch.tensor(test_logits[:, 1])).numpy()
    test_probs_cal  = calibrator.predict_proba(test_scores_raw)[:, 1]

    # ── Brier score before vs after ────────────────────────────────────────────
    # Brier score = mean((prob - true_label)^2); lower is better
    brier_before = brier_score_loss(test_labels, test_probs_raw)
    brier_after  = brier_score_loss(test_labels, test_probs_cal)

    log.info(f"\n  Brier score (raw sigmoid):   {brier_before:.6f}")
    log.info(f"  Brier score (calibrated):    {brier_after:.6f}")
    improvement = (brier_before - brier_after)
    log.info(f"  Δ Brier (positive = better): {improvement:+.6f}")

    # ── PR-AUC after calibration ────────────────────────────────────────────────
    prauc_before = average_precision_score(test_labels, test_probs_raw)
    prauc_after  = average_precision_score(test_labels, test_probs_cal)
    log.info(f"\n  Test PR-AUC (raw):        {prauc_before:.4f}")
    log.info(f"  Test PR-AUC (calibrated): {prauc_after:.4f}")

    # ── Reliability diagram (10-bin) ───────────────────────────────────────────
    log.info("\n  Reliability diagram (10 bins, calibrated probabilities):")
    log.info(f"  {'Bin centre':>12}  {'Predicted':>10}  {'Actual':>10}  {'Count':>6}")
    log.info(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*6}")

    frac_pos_raw, mean_pred_raw = calibration_curve(
        test_labels, test_probs_raw, n_bins=10, strategy="uniform"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        test_labels, test_probs_cal, n_bins=10, strategy="uniform"
    )

    reliability_bins = []
    for pred, actual in zip(mean_pred_cal, frac_pos_cal):
        log.info(f"  {pred:>12.4f}  {pred:>10.4f}  {actual:>10.4f}")
        reliability_bins.append({"pred_prob": float(pred), "actual_frac": float(actual)})

    log.info("\n  Interpretation: if calibration is good, predicted prob ≈ actual fraction.")
    log.info("  Perfect calibration = points fall on the diagonal y=x.")

    return {
        "brier_before":    float(brier_before),
        "brier_after":     float(brier_after),
        "delta_brier":     float(improvement),
        "prauc_before":    float(prauc_before),
        "prauc_after":     float(prauc_after),
        "calibrator_coef": float(calibrator.coef_[0][0]),
        "calibrator_intercept": float(calibrator.intercept_[0]),
        "reliability_diagram_calibrated": reliability_bins,
        "note": (
            "Calibration per P0.3.8 frozen spec: "
            "LR(C=1.0) fit on val logits, evaluated on test. "
            "GCN only — not applied to GraphSAGE or Bitcoin pipeline."
        ),
    }


# ── P0.3.9 — Checkpoint reload verification ────────────────────────────────────

def verify_checkpoint(ckpt_path: Path, ModelClass, model_name: str,
                      data) -> dict:
    """
    P0.3.9: Load checkpoint in isolation, rebuild model from config,
    run one inference pass, verify output shape and finite values.
    """
    log.info(f"\n── P0.3.9 Checkpoint verification: {model_name} ──")
    ckpt = torch.load(ckpt_path, weights_only=False)

    cfg = ckpt["config"]
    model = ModelClass(
        in_channels=cfg["in_channels"],
        hidden=cfg["hidden"],
        out_channels=cfg["out_channels"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    assert logits.shape == (data.num_nodes, cfg["out_channels"]), \
        f"Unexpected logit shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "NaN/Inf in reloaded model output"

    log.info(f"  ✓ Checkpoint reloaded successfully")
    log.info(f"  ✓ Config: {cfg}")
    log.info(f"  ✓ Output shape: {logits.shape}  (finite: True)")
    log.info(f"  ✓ Stored metrics: {ckpt['metrics']}")

    return {
        "model":    model_name,
        "path":     str(ckpt_path),
        "config":   cfg,
        "output_shape": list(logits.shape),
        "all_finite":   True,
        "stored_metrics": ckpt["metrics"],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Load graph ─────────────────────────────────────────────────────────────
    log.info("Loading Elliptic PyG graph …")
    data = torch.load(GRAPH_PATH, weights_only=False)

    # Log split-specific prevalences — these are the no-skill PR-AUC references
    log.info("\nSplit-specific no-skill PR-AUC references:")
    for name, mask in [("train", data.train_mask),
                       ("val",   data.val_mask),
                       ("test",  data.test_mask)]:
        y = data.y[mask].numpy()
        log.info(f"  {name}: prevalence = {y.mean():.4f}  "
                 f"({int(y.sum())} illicit / {len(y)} labeled)")

    # ── Load models from checkpoints ───────────────────────────────────────────
    log.info("\nLoading model checkpoints …")

    gcn_ckpt  = torch.load(GCN_CKPT, weights_only=False)
    gcn_cfg   = gcn_ckpt["config"]
    gcn_model = EllipticGCN(gcn_cfg["in_channels"], gcn_cfg["hidden"],
                             gcn_cfg["out_channels"], gcn_cfg["dropout"])
    gcn_model.load_state_dict(gcn_ckpt["model_state_dict"])
    log.info(f"  GCN loaded: {gcn_cfg}")

    sage_ckpt  = torch.load(SAGE_CKPT, weights_only=False)
    sage_cfg   = sage_ckpt["config"]
    sage_model = EllipticSAGE(sage_cfg["in_channels"], sage_cfg["hidden"],
                               sage_cfg["out_channels"], sage_cfg["dropout"])
    sage_model.load_state_dict(sage_ckpt["model_state_dict"])
    log.info(f"  SAGE loaded: {sage_cfg}")

    # ── P0.3.7 — Four-model evaluation on TEST split ───────────────────────────
    log.info("\n\n" + "═" * 75)
    log.info("P0.3.7 — FOUR-MODEL COMPARISON  (test split, illicit = positive class)")
    log.info("═" * 75)

    rows = []

    log.info("\n[1/4] Majority baseline …")
    m_maj = evaluate_majority(data)
    rows.append(("Majority (always licit)", m_maj))

    log.info("[2/4] Heuristic baseline …")
    m_heu = evaluate_heuristic(data)
    rows.append(("Heuristic (degree+flow)", m_heu))

    log.info("[3/4] GCN …")
    m_gcn = evaluate_model(gcn_model, data, data.test_mask)
    rows.append(("GCN (2-layer, hidden=128)", m_gcn))

    log.info("[4/4] GraphSAGE …")
    m_sage = evaluate_model(sage_model, data, data.test_mask)
    rows.append(("GraphSAGE (2-layer, mean)", m_sage))

    # ── Print table ────────────────────────────────────────────────────────────
    test_prevalence = data.y[data.test_mask].numpy().mean()
    log.info(f"\n{'─'*75}")
    log.info(f"{'Model':<28} {'Prec':>7} {'Recall':>7} {'F1':>7} {'PR-AUC':>8} {'ROC-AUC':>9}")
    log.info(f"{'─'*75}")
    for name, m in rows:
        prec_str = f"{m['precision']:>7.4f}" if m["n_predicted_positive"] > 0 else "     N/A"
        log.info(
            f"{name:<28}"
            f"{prec_str}"
            f"  {m['recall']:>6.4f}"
            f"  {m['f1']:>6.4f}"
            f"  {m['pr_auc']:>7.4f}"
            f"  {m['roc_auc']:>8.4f}"
        )
    log.info(f"{'─'*75}")
    log.info(f"No-skill PR-AUC reference (test prevalence): {test_prevalence:.4f}")
    log.info(f"{'═'*75}")

    log.info("\nNotes:")
    log.info("  - Precision/Recall/F1 use threshold = 0.5 for GCN and GraphSAGE.")
    log.info("  - PR-AUC and ROC-AUC evaluate ranking across all thresholds.")
    log.info("  - This evaluation is on the Elliptic benchmark only.")
    log.info("  - It does not constitute real-world Bitcoin illicit detection.")

    # ── P0.3.8 — GCN Calibration ───────────────────────────────────────────────
    cal_results = calibrate_gcn(gcn_model, data)

    # ── P0.3.9 — Checkpoint verification ──────────────────────────────────────
    ckpt_gcn  = verify_checkpoint(GCN_CKPT,  EllipticGCN,  "gcn_elliptic_v1",  data)
    ckpt_sage = verify_checkpoint(SAGE_CKPT, EllipticSAGE, "sage_elliptic_v1", data)

    log.info("\n── P0.3.9 Summary ──")
    log.info(f"  ✓ gcn_elliptic_v1.pt  — reloaded and verified")
    log.info(f"  ✓ sage_elliptic_v1.pt — reloaded and verified")

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "evaluation_split": "test (steps 42–49)",
        "test_prevalence":  float(test_prevalence),
        "noskill_prauc_ref": float(test_prevalence),
        "comparison_table": [
            {"model": name, **m} for name, m in rows
        ],
        "calibration": cal_results,
        "checkpoint_verification": [ckpt_gcn, ckpt_sage],
        "notes": [
            "All rows computed fresh on identical test mask and metric functions.",
            "Threshold = 0.5 for Precision/Recall/F1 of GCN and GraphSAGE.",
            "PR-AUC and ROC-AUC are threshold-independent ranking metrics.",
            "No-skill PR-AUC reference = test-split illicit prevalence.",
            "This is Elliptic methodology validation (Pipeline A) only.",
            "Bitcoin risk ranking uses an independent 14-feature pipeline (Pipeline B).",
        ],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nFull report saved → {OUT_PATH}")

    log.info("\nP0.3.7 + P0.3.8 + P0.3.9 COMPLETE ✓")


if __name__ == "__main__":
    main()
