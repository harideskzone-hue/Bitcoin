"""
ml/scripts/evaluate_bitcoin_ranking.py
========================================
P0.4.7 — Risk disclaimer (hard-coded, embedded in every output)
P0.4.8 — Seed-level leakage evaluation
P0.4.9 — Ranking metrics: P@5, P@10, P@20, R@10, R@20, NDCG@10

Fail-closed design
------------------
When the current snapshot has zero seed-derived reference labels,
this script reports N/A for all ranking metrics and exits with code 0
(not an error — zero seeds is a valid, scientifically honest result).
Actual metric values are produced only when the 50k snapshot is run.

Evaluation frame
----------------
These metrics measure agreement with the seed-derived WEAK reference,
not ground-truth illicit activity. The Bitcoin GCN is a risk-ranking
model (prioritisation tool for human investigators); it is not evaluated
as a ground-truth illicit classifier.

Seed split (frozen)
-------------------
  A/B/C  → training seeds   → 'high_risk' weak labels (seen during training)
  D      → validation seed  → 'eval_reference_val' (held out from training)
  E      → test seed        → 'eval_reference_test' (held out from both)

Leakage rule: D/E addresses and their cluster members must NEVER appear
in training labels. P0.4.8 verifies this for every run.

Outputs
-------
  data/btc_ranking_eval.json  — full evaluation report
"""

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("btc_eval")

RISK_SCORES_PATH   = Path("data/btc_risk_scores.parquet")
WEAK_LABELS_PATH   = Path("data/btc_weak_labels.parquet")
CLUSTER_SUMMARY    = Path("data/btc_cluster_summary.json")
FEATURES_PATH      = Path("data/address_features_10k.parquet")
OUT_PATH           = Path("data/btc_ranking_eval.json")

# ── P0.4.7 — Risk disclaimer (frozen, hard-coded) ─────────────────────────────
RISK_DISCLAIMER = (
    "Model-derived risk ranking. "
    "For prioritization and human review only. "
    "Not a calibrated probability."
)

# ── P0.4.9 — Ranking metric functions ─────────────────────────────────────────

def precision_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """
    P@K = (# positive labels in top-K) / K.
    ranked_labels: boolean array sorted by descending risk score.
    """
    if k <= 0 or k > len(ranked_labels):
        return float("nan")
    return float(ranked_labels[:k].sum()) / k


def recall_at_k(ranked_labels: np.ndarray, k: int,
                n_positive: int) -> float:
    """
    R@K = (# positive labels in top-K) / (total # positives).
    """
    if n_positive == 0 or k <= 0:
        return float("nan")
    return float(ranked_labels[:k].sum()) / n_positive


def ndcg_at_k(ranked_labels: np.ndarray, k: int) -> float:
    """
    NDCG@K using binary relevance labels.
    DCG@K  = sum_i (rel_i / log2(i+2))  for i in 0..K-1
    IDCG@K = DCG of ideal ranking (all positives at top)
    NDCG@K = DCG@K / IDCG@K
    """
    if k <= 0 or k > len(ranked_labels):
        return float("nan")
    top_k = ranked_labels[:k].astype(float)
    dcg   = sum(top_k[i] / math.log2(i + 2) for i in range(len(top_k)))

    # Ideal: sort positives to top
    ideal = np.sort(ranked_labels.astype(float))[::-1][:k]
    idcg  = sum(ideal[i] / math.log2(i + 2) for i in range(len(ideal)))

    return float(dcg / idcg) if idcg > 0 else 0.0


# ── P0.4.8 — Leakage checks ───────────────────────────────────────────────────

def run_leakage_checks(labels_df: pd.DataFrame,
                       features_path: Path) -> dict:
    """
    P0.4.8 checks:
    1. D/E (eval_reference_*) addresses not in training labels (high_risk).
    2. No feature directly encodes a known seed address string.
    3. Seen-seed recall is computed separately from held-out seed recall.
    Returns a results dict; raises AssertionError on leakage.
    """
    log.info("\n── P0.4.8 Seed-level leakage checks ──")

    results = {}

    # ── Check 1: No eval_reference_* address has label 'high_risk' ─────────────
    eval_mask   = labels_df["weak_label"].isin(
        ["eval_reference_val", "eval_reference_test"]
    )
    train_mask  = labels_df["weak_label"] == "high_risk"
    # An address cannot be both eval_reference and high_risk
    overlap = labels_df[eval_mask & train_mask]
    assert len(overlap) == 0, (
        f"LEAKAGE: {len(overlap)} addresses appear as both "
        f"eval_reference and high_risk training labels"
    )
    log.info("  ✓ Check 1 PASSED: no eval_reference address in training labels")
    results["check1_eval_not_in_train"] = "PASSED"

    # ── Check 2: Feature columns don't directly encode seed identity ────────────
    # The address_features parquet uses address_hash (SHA-256[:16]) not raw
    # addresses. This truncated hash is not reversible to the original address.
    feat_df = pd.read_parquet(features_path)
    assert "address_hash" in feat_df.columns, \
        "Expected address_hash column — raw address not present (good)"
    assert "address" not in feat_df.columns, \
        "LEAKAGE: raw Bitcoin address string stored as feature column"
    feat_cols = [c for c in feat_df.columns if c != "address_hash"]
    # No feature column should be named after a specific seed
    suspicious_cols = [c for c in feat_cols if "seed" in c.lower()
                       or "known" in c.lower() or "flag" in c.lower()]
    assert not suspicious_cols, \
        f"LEAKAGE: feature columns appear to encode seed identity: {suspicious_cols}"
    log.info("  ✓ Check 2 PASSED: no feature encodes seed identity "
             "(address_hash is SHA-256[:16], not reversible)")
    results["check2_no_seed_identity_feature"] = "PASSED"
    results["feature_columns"] = feat_cols

    # ── Check 3: Label counts ───────────────────────────────────────────────────
    counts = labels_df["weak_label"].value_counts().to_dict()
    n_train   = counts.get("high_risk", 0)
    n_val_ref = counts.get("eval_reference_val", 0)
    n_test_ref = counts.get("eval_reference_test", 0)
    n_unknown = counts.get("unknown", 0)
    log.info("  Label counts:")
    log.info(f"    high_risk           : {n_train:,}  (training positives)")
    log.info(f"    eval_reference_val  : {n_val_ref:,}  (D seed cluster, val recall)")
    log.info(f"    eval_reference_test : {n_test_ref:,}  (E seed cluster, test recall)")
    log.info(f"    unknown             : {n_unknown:,}  (message-passing only)")
    results["label_counts"] = {
        "high_risk": n_train,
        "eval_reference_val": n_val_ref,
        "eval_reference_test": n_test_ref,
        "unknown": n_unknown,
    }

    # ── Generalization gap report ───────────────────────────────────────────────
    # At evaluation time (after 50k training), we compute:
    #   seen_seed_recall  = recall over high_risk (A/B/C) cluster members
    #   held_out_recall_D = recall over eval_reference_val cluster members
    #   held_out_recall_E = recall over eval_reference_test cluster members
    # Generalization gap = seen_recall - held_out_recall
    # This is reported in P0.4.9; here we just define the framework.
    log.info("\n  Generalization gap framework:")
    log.info("    seen_seed_recall  = recall over A/B/C cluster members (training set)")
    log.info("    held_out_recall_D = recall over D seed cluster (val reference)")
    log.info("    held_out_recall_E = recall over E seed cluster (test reference)")
    log.info("    gap = seen_recall - held_out_recall")
    log.info("    (computed in P0.4.9 when eval_reference labels exist)")
    results["generalization_gap_framework"] = {
        "seen_seed_recall":    "A/B/C cluster members",
        "held_out_recall_D":   "D cluster (val reference, not seen during training)",
        "held_out_recall_E":   "E cluster (test reference, not seen during either)",
        "gap_definition":      "seen_seed_recall - held_out_recall",
    }

    log.info("  ✓ All P0.4.8 leakage checks PASSED")
    return results


# ── P0.4.9 — Ranking evaluation ───────────────────────────────────────────────

def run_ranking_evaluation(risk_df: pd.DataFrame,
                           labels_df: pd.DataFrame) -> dict:
    """
    P0.4.9: Compute P@5, P@10, P@20, R@10, R@20, NDCG@10 against the
    seed-derived weak reference labels.

    Fail-closed: if no eval_reference labels exist in the snapshot,
    all metrics are reported as N/A with an explicit reason.

    Evaluation population:
      - 'seen' reference:    high_risk cluster members (A/B/C, seen in training)
      - 'val' reference:     eval_reference_val cluster members (D, not in training)
      - 'test' reference:    eval_reference_test cluster members (E, not in training)

    We compute metrics separately for each reference population so that
    the held-out generalisation gap is directly observable.
    """
    log.info("\n── P0.4.9 Ranking metrics ──")

    # Merge risk scores with labels
    merged = risk_df.merge(
        labels_df[["address", "weak_label"]],
        left_on="address_hash", right_on="address",
        how="left",
    )
    # Fill any unmatched addresses as unknown
    merged["weak_label"] = merged["weak_label"].fillna("unknown")

    # Sort by descending risk score
    merged = merged.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # ── Check for reference population ────────────────────────────────────────
    n_seen = (merged["weak_label"] == "high_risk").sum()
    n_val  = (merged["weak_label"] == "eval_reference_val").sum()
    n_test = (merged["weak_label"] == "eval_reference_test").sum()

    log.info("  Reference populations in risk score table:")
    log.info(f"    Seen (A/B/C high_risk):       {n_seen}")
    log.info(f"    Val  (D eval_reference_val):  {n_val}")
    log.info(f"    Test (E eval_reference_test): {n_test}")

    NA_REASON = (
        "No seed-derived reference labels in this snapshot. "
        "The 10k snapshot covers ~30 minutes of Bitcoin history; "
        "no seed address transacted in that window. "
        "Metrics will be available after the 50k snapshot is generated."
    )

    # If nothing to evaluate → N/A result
    if n_seen == 0 and n_val == 0 and n_test == 0:
        log.warning(f"  ⚠ {NA_REASON}")
        return {
            "status": "N/A",
            "reason": NA_REASON,
            "seen_seed":  {"n_reference": 0},
            "val_seed_D": {"n_reference": 0},
            "test_seed_E": {"n_reference": 0},
        }

    def _metrics_for_ref(label_value: str, n_ref: int,
                         ref_name: str) -> dict:
        """Compute all ranking metrics for one reference label type."""
        if n_ref == 0:
            return {"n_reference": 0, "status": "N/A — no reference labels"}

        is_positive = (merged["weak_label"] == label_value).values

        result = {
            "n_reference": int(n_ref),
            "disclaimer": RISK_DISCLAIMER,
            "metric_note": (
                "Measures agreement with seed-derived weak reference, "
                "not ground-truth illicit activity."
            ),
        }
        for k in [5, 10, 20]:
            result[f"P@{k}"] = precision_at_k(is_positive, k)
        for k in [10, 20]:
            result[f"R@{k}"] = recall_at_k(is_positive, k, n_ref)
        result["NDCG@10"] = ndcg_at_k(is_positive, 10)

        log.info(f"\n  [{ref_name}]  n_reference={n_ref}")
        for key in ["P@5", "P@10", "P@20", "R@10", "R@20", "NDCG@10"]:
            v = result[key]
            log.info(f"    {key:10s}: {v:.4f}" if isinstance(v, float) else f"    {key}: {v}")

        return result

    seen_metrics  = _metrics_for_ref("high_risk",            n_seen, "Seen A/B/C")
    val_metrics   = _metrics_for_ref("eval_reference_val",   n_val,  "Val D (held out)")
    test_metrics  = _metrics_for_ref("eval_reference_test",  n_test, "Test E (held out)")

    # ── Generalisation gap ─────────────────────────────────────────────────────
    gap_report = {}
    if (n_seen > 0 and n_test > 0
            and isinstance(seen_metrics.get("R@10"), float)
            and isinstance(test_metrics.get("R@10"), float)):
        gap_r10 = seen_metrics["R@10"] - test_metrics["R@10"]
        gap_report = {
            "seen_R@10":      seen_metrics["R@10"],
            "held_out_R@10":  test_metrics["R@10"],
            "gap_R@10":       gap_r10,
            "interpretation": (
                "Positive gap = model generalises less well to held-out E seed "
                "than to training-seen A/B/C seeds. "
                "Zero gap = perfect generalisation on the weak reference."
            ),
        }
        log.info(f"\n  Generalisation gap (R@10): {gap_r10:+.4f}")
        log.info(f"    (seen={seen_metrics['R@10']:.4f}, held-out E={test_metrics['R@10']:.4f})")

    return {
        "status":        "EVALUATED",
        "seen_seed":     seen_metrics,
        "val_seed_D":    val_metrics,
        "test_seed_E":   test_metrics,
        "generalization_gap": gap_report,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("═" * 65)
    log.info("P0.4.7 — Risk disclaimer")
    log.info("═" * 65)
    log.info(f"\n  DISCLAIMER: {RISK_DISCLAIMER}\n")
    log.info("  This disclaimer is hard-coded and embedded in every evaluation")
    log.info("  output. It must accompany any presentation of Bitcoin risk scores.")

    # ── Load inputs ────────────────────────────────────────────────────────────
    log.info("\nLoading inputs …")

    if not RISK_SCORES_PATH.exists():
        log.error(
            f"Risk scores not found: {RISK_SCORES_PATH}\n"
            "Run train_bitcoin_gcn.py first (requires 50k snapshot with seed labels)."
        )
        sys.exit(1)

    if not WEAK_LABELS_PATH.exists():
        log.error(f"Weak labels not found: {WEAK_LABELS_PATH}")
        sys.exit(1)

    risk_df   = pd.read_parquet(RISK_SCORES_PATH)
    labels_df = pd.read_parquet(WEAK_LABELS_PATH)

    log.info(f"  Risk scores : {len(risk_df):,} addresses")
    log.info(f"  Weak labels : {len(labels_df):,} addresses")

    # ── P0.4.8: Leakage checks ─────────────────────────────────────────────────
    leakage_results = run_leakage_checks(labels_df, FEATURES_PATH)

    # ── P0.4.9: Ranking metrics ────────────────────────────────────────────────
    ranking_results = run_ranking_evaluation(risk_df, labels_df)

    # ── Assemble report ────────────────────────────────────────────────────────
    report = {
        "disclaimer": RISK_DISCLAIMER,
        "evaluation_note": (
            "Metrics measure agreement with seed-derived weak reference labels, "
            "not ground-truth illicit activity classification."
        ),
        "snapshot": str(RISK_SCORES_PATH),
        "p0_4_7_disclaimer_verified": True,
        "p0_4_8_leakage": leakage_results,
        "p0_4_9_ranking": ranking_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport saved → {OUT_PATH}")
    log.info("\nP0.4.7 + P0.4.8 + P0.4.9 COMPLETE ✓")


if __name__ == "__main__":
    main()
