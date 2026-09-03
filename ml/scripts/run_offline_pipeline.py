"""
ml/scripts/run_offline_pipeline.py
=====================================
P0.8.1 — Offline end-to-end pipeline runner

Executes every stage of the pipeline in sequence and writes a structured
dry-run report to data/dry_run_report.json.

Critical design principle (P0.8.1 constraint):
  The 10k snapshot contains zero weak-supervision seeds. This script does
  NOT fabricate a trained GCN or mock risk scores. Instead, it explicitly
  demonstrates the fail-closed behaviour and continues through the API,
  explainability, and P2P simulation paths using their real artifacts.

  This is the correct SIH demo posture: a system that fails safely rather
  than producing unjustified risk scores.

Pipeline stages:
  Stage 1: Snapshot inspection
  Stage 2: Feature extraction + graph build (checks existing artifacts)
  Stage 3: Co-spend clustering + weak labels
  Stage 4: GCN training attempt → fail-closed on zero seeds (expected)
  Stage 5: Explainability Layer 1 (runs on features, no GCN required)
  Stage 6: P2P signal generation (already complete)
  Stage 7: API readiness check (import + route enumeration)

Output:
  data/dry_run_report.json  — machine-readable stage results
  (printed to stdout as a human-readable summary)
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so `ml` and `backend` packages resolve
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Stage result helpers ───────────────────────────────────────────────────────

PASS    = "PASS"
FAIL    = "FAIL"
EXPECTED_FAIL = "EXPECTED_FAIL"   # fail-closed triggered correctly
SKIP    = "SKIP"


def stage(name: str, result: str, detail: str, duration_s: float = 0.0) -> dict:
    return {
        "stage":      name,
        "result":     result,
        "detail":     detail,
        "duration_s": round(duration_s, 2),
    }


def _run(cmd: list[str], cwd: str = ".") -> tuple[int, str, str]:
    """Run subprocess, return (returncode, stdout, stderr)."""
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=cwd,
        capture_output=True, text=True,
        timeout=300,
    )
    elapsed = time.time() - t0
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def _check_file(path: Path, label: str) -> dict | None:
    """Return a FAIL stage if file is missing, else None."""
    if not path.exists():
        return stage(label, FAIL, f"File not found: {path}")
    return None


# ── Stage implementations ─────────────────────────────────────────────────────

def stage1_snapshot(results: list[dict]) -> None:
    """Verify snapshot exists and report basic stats."""
    t0 = time.time()
    snap_path = Path("data/btc_snapshot_10k.parquet")
    if not snap_path.exists():
        results.append(stage("1_snapshot", FAIL,
                             f"Snapshot not found: {snap_path}", time.time()-t0))
        return
    import pandas as pd
    df = pd.read_parquet(snap_path)
    n_tx   = df["tx_hash"].nunique()
    n_addr = len(set(df.get("input_address", pd.Series()).dropna()) |
                 set(df.get("output_address", pd.Series()).dropna()))
    detail = (
        f"Snapshot OK — {n_tx:,} transactions, ~{n_addr:,} unique addresses. "
        f"Window: ~30 minutes of Bitcoin history."
    )
    results.append(stage("1_snapshot", PASS, detail, time.time()-t0))


def stage2_graph(results: list[dict]) -> None:
    """Check that graph and feature artifacts exist."""
    t0 = time.time()
    missing = []
    for p in [Path("data/graph_10k.pt"), Path("data/address_features_10k.parquet")]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        results.append(stage("2_graph", FAIL,
                             f"Missing artifacts: {missing}", time.time()-t0))
        return
    import torch, pandas as pd
    g    = torch.load("data/graph_10k.pt", weights_only=False)
    feat = pd.read_parquet("data/address_features_10k.parquet")
    n_addr = g["address"].num_nodes
    n_tx   = g["transaction"].num_nodes
    n_feat = len([c for c in feat.columns if c != "address_hash"])
    detail = (
        f"Graph OK — {n_addr:,} address nodes, {n_tx:,} transaction nodes, "
        f"{n_feat} features per address. "
        f"Edge types: INPUT_TO, OUTPUT_TO."
    )
    results.append(stage("2_graph", PASS, detail, time.time()-t0))


def stage3_weak_labels(results: list[dict]) -> None:
    """Run build_weak_labels.py and check output."""
    t0 = time.time()
    rc, out, err, elapsed = _run(
        [sys.executable, "ml/scripts/build_weak_labels.py"]
    )
    if rc != 0:
        results.append(stage("3_weak_labels", FAIL,
                             f"build_weak_labels.py exited {rc}: {err[:300]}", elapsed))
        return
    import pandas as pd
    labels = pd.read_parquet("data/btc_weak_labels.parquet")
    counts = labels["weak_label"].value_counts().to_dict()
    n_high_risk = counts.get("high_risk", 0)
    detail = (
        f"Weak labels built — {len(labels):,} addresses. "
        f"Label distribution: {counts}. "
        f"high_risk (training positives): {n_high_risk}."
    )
    results.append(stage("3_weak_labels", PASS, detail, elapsed))


def stage4_gcn(results: list[dict]) -> None:
    """
    Attempt GCN training. Expect exit code 1 (fail-closed) on 10k snapshot.
    This is the correct and expected behaviour — not a pipeline error.
    """
    t0 = time.time()
    rc, out, err, elapsed = _run(
        [sys.executable, "ml/scripts/train_bitcoin_gcn.py"]
    )
    combined = out + err

    if rc == 1 and "TRAINING ABORTED" in combined and "zero positive" in combined.lower():
        detail = (
            "GCN fail-closed correctly (exit 1). "
            "Reason: 10k snapshot covers ~30 minutes of Bitcoin history — "
            "no known seed addresses (Hydra, Garantex, etc.) transacted in that window. "
            "Training positives: 0. "
            "The frozen spec prohibits falling back to UNKNOWN-majority training. "
            "Resolution: run the 50k snapshot after BigQuery quota resets."
        )
        results.append(stage("4_gcn_training", EXPECTED_FAIL, detail, elapsed))

    elif rc == 0:
        # Unexpected success — checkpoint must exist
        ckpt = Path("models/gcn_bitcoin_v1.pt")
        detail = (
            f"GCN training completed (unexpected on 10k — seeds found). "
            f"Checkpoint: {ckpt} ({'exists' if ckpt.exists() else 'MISSING'})."
        )
        results.append(stage("4_gcn_training", PASS, detail, elapsed))

    else:
        results.append(stage("4_gcn_training", FAIL,
                             f"Unexpected failure (exit {rc}): {combined[:400]}", elapsed))


def stage5_explainability(results: list[dict]) -> None:
    """
    Run Layer 1 deterministic explanation on the 10k addresses.
    Layer 1 does not require the GCN — it only needs features + graph context.
    """
    t0 = time.time()
    try:
        import pandas as pd
        from ml.explain.deterministic import GraphContext, explain_batch

        feat_df   = pd.read_parquet("data/address_features_10k.parquet")
        labels_df = pd.read_parquet("data/btc_weak_labels.parquet")

        # Minimal graph context (no flagged nodes on 10k — BFS produces empty hops_map)
        ctx = GraphContext(
            hops_map={},
            flagged_clusters=set(),
            cluster_sizes={},
        )

        rows = feat_df.to_dict(orient="records")
        results_map = explain_batch(rows, ctx, max_reasons=3)

        n_with = sum(1 for v in results_map.values() if v)
        # Layer 1 rules that can fire without flagged context:
        # burst, reuse, rapid_tx, change_address
        fired_sample = [
            (k, v) for k, v in results_map.items() if v
        ][:3]

        detail = (
            f"Layer 1 deterministic rules ran on {len(results_map):,} addresses. "
            f"Addresses with ≥1 reason: {n_with:,}. "
            f"Sample: {[(addr[:12]+'…', r) for addr, r in fired_sample]}. "
            f"Note: hops_to_flagged and cluster rules require seed coverage "
            f"(available after 50k snapshot)."
        )
        results.append(stage("5_explainability_L1", PASS, detail, time.time()-t0))

    except Exception as e:
        results.append(stage("5_explainability_L1", FAIL, str(e), time.time()-t0))


def stage6_p2p(results: list[dict]) -> None:
    """Verify P2P simulation artifacts exist and pass the IP check."""
    t0 = time.time()
    sig_path = Path("data/p2p_signals_simulated.json")
    ann_path = Path("data/p2p_node_annotations.json")
    if not sig_path.exists() or not ann_path.exists():
        results.append(stage("6_p2p_simulation", FAIL,
                             "P2P artifacts missing. Run generate_p2p_signals.py.", time.time()-t0))
        return

    import re, json
    signals = json.loads(sig_path.read_text())
    n       = len(signals)
    all_sim = all(s["signal_source"] == "SIMULATED" for s in signals)
    payload = json.dumps(signals)
    ip_hits = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', payload)
    clusters = sorted({s["ip_cluster_id"] for s in signals})

    if ip_hits:
        results.append(stage("6_p2p_simulation", FAIL,
                             f"IP strings found: {ip_hits}", time.time()-t0))
        return

    detail = (
        f"P2P simulation OK — {n} signals, all SIMULATED. "
        f"IP clusters: {clusters}. "
        f"No real IP strings (P0.7.3 PASS). "
        f"Disclaimer present on all records: "
        f"'{signals[0]['disclaimer']}'"
    )
    results.append(stage("6_p2p_simulation", PASS, detail, time.time()-t0))


def stage7_api(results: list[dict]) -> None:
    """Verify FastAPI app imports cleanly and all 5 routes are registered."""
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from backend.app.main import app
        routes = [
            r.path for r in app.routes
            if hasattr(r, "methods") and "GET" in r.methods
        ]
        expected = {
            "/api/v1/health",
            "/api/v1/model/info",
            "/api/v1/alerts",
            "/api/v1/wallets/{address_hash}",
            "/api/v1/graph",
        }
        missing = expected - set(routes)
        if missing:
            results.append(stage("7_api_readiness", FAIL,
                                 f"Routes missing: {missing}", time.time()-t0))
            return
        detail = (
            f"API OK — {len(expected)} routes registered: {sorted(expected)}. "
            f"OpenAPI spec: backend/openapi.yaml. "
            f"Mock server: npx @stoplight/prism-cli mock backend/openapi.yaml"
        )
        results.append(stage("7_api_readiness", PASS, detail, time.time()-t0))
    except Exception as e:
        results.append(stage("7_api_readiness", FAIL, str(e), time.time()-t0))


# ── Report builder ─────────────────────────────────────────────────────────────

def print_report(stages: list[dict], total_s: float) -> None:
    icons = {PASS: "✅", FAIL: "❌", EXPECTED_FAIL: "⚠ ", SKIP: "⏭ "}

    print()
    print("=" * 70)
    print("  P0.8.1 — Offline Pipeline Dry Run Report")
    print(f"  Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print("=" * 70)

    for s in stages:
        icon   = icons.get(s["result"], "?")
        result = s["result"]
        name   = s["stage"]
        dur    = f"({s['duration_s']:.1f}s)"
        print(f"\n{icon}  [{result}] Stage {name}  {dur}")
        # Word-wrap detail at 72 chars
        detail = s["detail"]
        import textwrap
        for line in textwrap.wrap(detail, width=68, subsequent_indent="     "):
            print(f"     {line}")

    print()
    print("─" * 70)
    n_pass   = sum(1 for s in stages if s["result"] == PASS)
    n_efail  = sum(1 for s in stages if s["result"] == EXPECTED_FAIL)
    n_fail   = sum(1 for s in stages if s["result"] == FAIL)

    print(f"  PASS:           {n_pass}")
    print(f"  EXPECTED_FAIL:  {n_efail}  (fail-closed, correct behaviour)")
    print(f"  FAIL:           {n_fail}")
    print(f"  Total time:     {total_s:.1f}s")
    print()

    if n_fail == 0:
        print("  ✅  ALL GATES PASSED — demo-ready on 10k snapshot")
        print("      (50k snapshot unblocks empirical GCN metrics)")
    else:
        print(f"  ❌  {n_fail} UNEXPECTED FAILURE(S) — resolve before SIH demo")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    stages:  list[dict] = []

    print("Running P0.8.1 offline pipeline dry run …")
    stage1_snapshot(stages)
    stage2_graph(stages)
    stage3_weak_labels(stages)
    stage4_gcn(stages)
    stage5_explainability(stages)
    stage6_p2p(stages)
    stage7_api(stages)

    total_s = time.time() - t_start

    # ── Console report ─────────────────────────────────────────────────────────
    print_report(stages, total_s)

    # ── JSON report ────────────────────────────────────────────────────────────
    report = {
        "generated_at":  datetime.now(tz=timezone.utc).isoformat(),
        "total_duration_s": round(total_s, 2),
        "snapshot":      "10k (30-minute window)",
        "gcn_status":    "fail-closed — zero positive seeds (expected)",
        "stages":        stages,
        "summary": {
            "pass":          sum(1 for s in stages if s["result"] == PASS),
            "expected_fail": sum(1 for s in stages if s["result"] == EXPECTED_FAIL),
            "fail":          sum(1 for s in stages if s["result"] == FAIL),
        },
    }
    out = Path("data/dry_run_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {out}")

    # Exit 1 only on unexpected failures
    n_fail = report["summary"]["fail"]
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
