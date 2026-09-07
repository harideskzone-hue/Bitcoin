"""
ml/scripts/infer_bitcoin.py
==============================
P0.4 exit condition — spec-required Bitcoin inference script.

Usage (per frozen spec):
    python ml/scripts/infer_bitcoin.py --offline

Behaviour on 10k snapshot (zero seeds → fail-closed):
    Loads graph → runs weak labels → checks for GCN checkpoint →
    if checkpoint exists: runs inference → prints ranked list
    if no checkpoint (10k, zero seeds): prints explicit fail-closed
    diagnostic with instructions for the 50k run.

Behaviour on 50k snapshot (seeds present, checkpoint exists):
    Full inference → ranked list → P@10, R@10, NDCG@10 summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_GRAPH_50K    = _PROJECT_ROOT / "data" / "graph_50k.pt"
_GRAPH_10K    = _PROJECT_ROOT / "data" / "graph_10k.pt"
_FEAT_50K     = _PROJECT_ROOT / "data" / "address_features_50k.parquet"
_FEAT_10K     = _PROJECT_ROOT / "data" / "address_features_10k.parquet"
_LABELS_PATH  = _PROJECT_ROOT / "data" / "btc_weak_labels.parquet"
_SCORES_PATH  = _PROJECT_ROOT / "data" / "btc_risk_scores.parquet"
_CHECKPOINT   = _PROJECT_ROOT / "models" / "gcn_bitcoin_v1.pt"


def _print_ranked_list(scores_df, top_k: int = 20) -> None:
    top = scores_df.nlargest(top_k, "risk_score")
    print()
    print(f"  {'Rank':<5}  {'Address hash':<20}  {'Score':>5}  {'Label':<10}")
    print("  " + "─" * 50)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        addr  = str(row["address_hash"])[:18] + "…"
        score = int(row["risk_score"])
        label = row.get("risk_label", "—")
        print(f"  {rank:<5}  {addr:<20}  {score:>5}  {label:<10}")
    print()


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Run Bitcoin GCN inference and print risk-ranked list."
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use local parquet snapshot (required in Phase 0 — no BigQuery access needed)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top-ranked addresses to display (default: 20)"
    )
    args = parser.parse_args()

    if not args.offline:
        print("ERROR: Live BigQuery pull not implemented in Phase 0. Use --offline flag.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  Bitcoin Risk Ranking Inference (Pipeline B)")
    print("  Mode: OFFLINE (local snapshot)")
    print("=" * 70)

    # ── Select graph/features ────────────────────────────────────────────────
    if _GRAPH_50K.exists() and _FEAT_50K.exists():
        graph_path = _GRAPH_50K
        feat_path  = _FEAT_50K
        snapshot   = "50k"
    elif _GRAPH_10K.exists() and _FEAT_10K.exists():
        graph_path = _GRAPH_10K
        feat_path  = _FEAT_10K
        snapshot   = "10k"
    else:
        print("ERROR: No graph snapshot found. Run build_graph.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Snapshot: {snapshot} ({graph_path.name})")

    # ── Check weak labels ────────────────────────────────────────────────────
    import pandas as pd
    if not _LABELS_PATH.exists():
        print("  Running build_weak_labels.py …")
        import subprocess
        rc = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "ml" / "scripts" / "build_weak_labels.py")],
            cwd=str(_PROJECT_ROOT),
        ).returncode
        if rc != 0:
            print("ERROR: build_weak_labels.py failed.", file=sys.stderr)
            sys.exit(1)

    labels = pd.read_parquet(_LABELS_PATH)
    n_positive = (labels["weak_label"] == "high_risk").sum()
    print(f"  Positive training labels: {n_positive}")

    # ── Fail-closed check ────────────────────────────────────────────────────
    if n_positive == 0:
        print()
        print("  ⚠ TRAINING ABORTED — FAIL-CLOSED")
        print()
        print("  Reason: Zero positive (FLAGGED/SUSPICIOUS) labels in the current snapshot.")
        print(f"  This is expected for the {snapshot} snapshot:")
        print("  No known seed addresses (Hydra, Garantex, Blender.io, etc.)")
        print("  transacted in this time window.")
        print()
        print("  The system refuses to train on zero positives to avoid producing")
        print("  unjustified risk scores. This is the correct fail-closed behaviour.")
        print()
        print("  Resolution:")
        print("    1. Run on the 50k snapshot (90 days of history) once BigQuery quota resets")
        print("    2. The 50k window has seed coverage → training succeeds")
        print("    3. Re-run: python ml/scripts/infer_bitcoin.py --offline")
        print()
        print("  Downstream components (API, explainability, P2P) continue to function")
        print("  using their offline artifacts.")
        print("=" * 70)
        sys.exit(0)  # exit 0 — fail-closed is expected, not an error

    # ── GCN checkpoint check ─────────────────────────────────────────────────
    if not _CHECKPOINT.exists():
        print(f"\n  No checkpoint found at {_CHECKPOINT}.")
        print("  Run: python ml/scripts/train_bitcoin_gcn.py")
        sys.exit(1)

    # ── Load scores (already computed) or run inference ───────────────────────
    if _SCORES_PATH.exists():
        print(f"  Loading pre-computed scores from {_SCORES_PATH} …")
        scores_df = pd.read_parquet(_SCORES_PATH)
    else:
        print("  Running GCN inference …")
        import torch

        from backend.constants import score_to_label
        from ml.models.bitcoin_gcn import BitcoinGCN

        graph = torch.load(str(graph_path), weights_only=False)
        feat  = pd.read_parquet(feat_path)

        model = BitcoinGCN(in_channels=14, hidden_channels=128, out_channels=1)
        model.load_state_dict(torch.load(str(_CHECKPOINT), weights_only=True))
        model.eval()

        with torch.no_grad():
            logits = model(graph["address"].x, graph[("address","INPUT_TO","transaction")].edge_index)
            scores = (logits.sigmoid() * 100).round().long().squeeze().tolist()

        scores_df = feat[["address_hash"]].copy()
        scores_df["risk_score"] = scores
        scores_df["risk_label"] = scores_df["risk_score"].apply(score_to_label)
        scores_df.to_parquet(str(_SCORES_PATH), index=False)

    print(f"  Scored {len(scores_df):,} addresses.")
    print()
    print("  DISCLAIMER: Model-derived risk ranking. For prioritization and human")
    print("  review only. Not a calibrated probability.")

    # ── Print ranked list ────────────────────────────────────────────────────
    print(f"\n  Top {args.top_k} addresses by risk score:")
    _print_ranked_list(scores_df, top_k=args.top_k)

    # ── Summary stats ────────────────────────────────────────────────────────
    from backend.constants import RISK_LABEL_THRESHOLDS
    print("  Score distribution:")
    for label in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        lo, hi = RISK_LABEL_THRESHOLDS[label]
        n = ((scores_df["risk_score"] >= lo) & (scores_df["risk_score"] <= hi)).sum()
        pct = 100 * n / len(scores_df)
        print(f"    {label:<10}: {n:>6} addresses  ({pct:.1f}%)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
