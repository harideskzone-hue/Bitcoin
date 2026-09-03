"""
ml/scripts/baselines_elliptic.py
==================================
P0.3.3 — Majority-class baseline
P0.3.4 — Heuristic baseline (degree + total flow)

Both baselines operate only on LABELED nodes (class ∈ {1, 2}).
Unknown nodes are excluded from all metrics, exactly as the GCN will be.

Outputs
-------
Prints a metric table and saves:
  data/elliptic_baselines.json   — machine-readable results dict

Metrics reported (positive class = illicit = 1):
  Precision, Recall, F1, PR-AUC, ROC-AUC
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("baselines")

ELLIPTIC_DIR = Path("data/elliptic/elliptic_bitcoin_dataset")
OUT_DIR      = Path("data")
GRAPH_PATH   = OUT_DIR / "elliptic_graph.pt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray, name: str) -> dict:
    """
    Compute Precision, Recall, F1, PR-AUC, ROC-AUC for the illicit class.
    y_true  : 1-D array of ground-truth labels (0=licit, 1=illicit)
    y_pred  : 1-D array of hard predictions (0 or 1)
    y_score : 1-D array of continuous scores for the illicit class (higher = more illicit)
    """
    # Guard: if the model predicts zero positives, precision is ill-defined
    # sklearn returns 0 with a warning; we override with None for clarity.
    has_positives = y_pred.sum() > 0

    prec   = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    prauc  = average_precision_score(y_true, y_score)
    rocauc = roc_auc_score(y_true, y_score)

    prec_display = f"{prec:.4f}" if has_positives else "N/A (0 positives predicted)"

    log.info(f"\n── {name} ──")
    log.info(f"  Precision : {prec_display}")
    log.info(f"  Recall    : {recall:.4f}")
    log.info(f"  F1        : {f1:.4f}")
    log.info(f"  PR-AUC    : {prauc:.4f}")
    log.info(f"  ROC-AUC   : {rocauc:.4f}")

    return {
        "name":    name,
        "precision": float(prec),
        "recall":    float(recall),
        "f1":        float(f1),
        "pr_auc":    float(prauc),
        "roc_auc":   float(rocauc),
        "n_predicted_positive": int(y_pred.sum()),
        "n_actual_positive":    int(y_true.sum()),
        "n_total":              int(len(y_true)),
    }


# ── P0.3.3 Majority-Class Baseline ─────────────────────────────────────────────

def majority_baseline(data: "torch_geometric.data.Data") -> dict:
    """
    Always predict LICIT (0) for every labeled node.

    Expected results:
      Recall    = 0.0   (we never predict illicit)
      F1        = 0.0   (harmonic mean of prec=undef and recall=0)
      PR-AUC    > 0     (sklearn PR-AUC with a constant score is baseline-rate)
      ROC-AUC   = 0.5   (random classifier)
    """
    log.info("P0.3.3 — Majority-class baseline")

    # Operate only on labeled nodes (label != -1)
    labeled_mask = (data.y != -1)
    y_true  = data.y[labeled_mask].numpy()

    # Prediction: always 0 (licit), score: always 0.0
    y_pred  = np.zeros(len(y_true), dtype=int)
    y_score = np.zeros(len(y_true), dtype=float)

    log.info(f"  Total labeled nodes : {len(y_true):,}")
    log.info(f"  Illicit in labeled  : {y_true.sum():,} ({100*y_true.mean():.1f}%)")
    log.info(f"  Predicted positive  : 0  (always predicts licit)")

    return compute_metrics(y_true, y_pred, y_score, "Majority (always licit)")


# ── P0.3.4 Heuristic Baseline ──────────────────────────────────────────────────

def heuristic_baseline(data: "torch_geometric.data.Data",
                       nodes_df: pd.DataFrame,
                       edges_df: pd.DataFrame) -> dict:
    """
    Simple heuristic: flag the top-K transactions by combined score:

        heuristic_score = degree (in the transaction graph) + normalised total flow

    "Degree" here = number of directed edges touching the transaction node
    in the edge list (in-degree + out-degree).

    "Total flow" = sum of absolute edge values. Since Elliptic features are
    already normalized (z-scored), we proxy total flow using a feature that
    represents transaction volume. Feature f1 (col index 2 in the original CSV,
    i.e. feature f0 after stripping txId+timestep) is the first local feature —
    empirically this captures transaction volume information. We use it as a
    signed value; we take abs() so high-magnitude nodes score high.

    Threshold is set so the number of predicted positives matches the
    overall illicit prevalence in the labeled set (~10.6%) — this gives
    a fair comparison with the GCN.

    Why does this work as a comparison point?
    -----------------------------------------
    If the heuristic already achieves, say, F1=0.40 with zero training,
    the GCN needs to clearly beat that to justify its complexity.
    If the heuristic gets F1=0.05, the GCN improvement is much more
    meaningful relative to an already hard problem.
    """
    log.info("\nP0.3.4 — Heuristic baseline (degree + flow)")

    labeled_mask = (data.y != -1)
    labeled_idx  = torch.where(labeled_mask)[0].numpy()
    y_true       = data.y[labeled_mask].numpy()

    # ── Degree: count edges per node ──────────────────────────────────────────
    n_nodes = data.num_nodes
    edge_index = data.edge_index.numpy()

    # Each appearance of a node as src or dst counts toward its degree
    degree = np.zeros(n_nodes, dtype=int)
    np.add.at(degree, edge_index[0], 1)   # out-degree
    np.add.at(degree, edge_index[1], 1)   # in-degree
    degree_labeled = degree[labeled_idx]

    # ── Flow proxy: f0 (first local feature, after txId and timestep removed) ─
    # data.x has shape (N, 165); column 0 is the first normalized local feature
    flow_proxy_labeled = np.abs(data.x[labeled_idx, 0].numpy())

    # ── Combine and normalize to [0, 1] ───────────────────────────────────────
    def norm01(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    score = norm01(degree_labeled.astype(float)) + norm01(flow_proxy_labeled)

    # ── Threshold: match the labeled-set illicit prevalence ───────────────────
    illicit_rate = y_true.mean()
    k = max(1, int(np.round(illicit_rate * len(y_true))))
    threshold_val = np.sort(score)[::-1][k - 1]  # k-th largest value
    y_pred = (score >= threshold_val).astype(int)

    log.info(f"  Labeled nodes       : {len(y_true):,}")
    log.info(f"  Illicit prevalence  : {100*illicit_rate:.1f}%")
    log.info(f"  Threshold K         : {k:,}")
    log.info(f"  Predicted positive  : {int(y_pred.sum()):,}")

    return compute_metrics(y_true, y_pred, score, "Heuristic (degree + flow)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading PyG graph …")
    data = torch.load(GRAPH_PATH, weights_only=False)
    log.info(f"  Graph: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges")

    # Load the raw tables for the heuristic (we need the edge list)
    edges_df = pd.read_csv(ELLIPTIC_DIR / "elliptic_txs_edgelist.csv")
    nodes_df = pd.read_csv(
        ELLIPTIC_DIR / "elliptic_txs_features.csv",
        header=None, usecols=[0, 1],
        names=["txId", "time_step"],
    )

    results = []
    results.append(majority_baseline(data))
    results.append(heuristic_baseline(data, nodes_df, edges_df))

    # ── Summary table ──────────────────────────────────────────────────────────
    log.info("\n\n" + "═" * 65)
    log.info("BASELINE COMPARISON (illicit = positive class)")
    log.info("═" * 65)
    header = f"{'Model':<28} {'Prec':>7} {'Recall':>7} {'F1':>7} {'PR-AUC':>8} {'ROC-AUC':>9}"
    log.info(header)
    log.info("─" * 65)
    for r in results:
        row = (
            f"{r['name']:<28}"
            f"  {r['precision']:>6.4f}"
            f"  {r['recall']:>6.4f}"
            f"  {r['f1']:>6.4f}"
            f"  {r['pr_auc']:>7.4f}"
            f"  {r['roc_auc']:>8.4f}"
        )
        log.info(row)
    log.info("═" * 65)
    log.info("\nInterpretation:")
    log.info("  PR-AUC = illicit class prevalence rate is the no-skill baseline for PR-AUC.")
    log.info(f"  Illicit prevalence (labeled): "
             f"{100 * results[0]['n_actual_positive'] / results[0]['n_total']:.1f}%")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = OUT_DIR / "elliptic_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved → {out_path}")

    log.info("\nP0.3.3 + P0.3.4 COMPLETE ✓")
    log.info("GCN must beat PR-AUC > {:.4f} to justify the neural approach.".format(
        max(r["pr_auc"] for r in results)
    ))


if __name__ == "__main__":
    main()
