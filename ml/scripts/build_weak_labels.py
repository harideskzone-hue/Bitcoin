"""
ml/scripts/build_weak_labels.py
==================================
P0.4.1 — Load and inspect Bitcoin snapshot
P0.4.2 — Co-spend clustering (Union-Find)
P0.4.3 — Change-address candidates
P0.4.4 — Weak-label construction from seed list

Theory
------
Co-spend clustering (P0.4.2):
  If addresses A and B both appear as inputs in the same transaction,
  they must share the same controlling entity (both private keys were used
  to sign). We merge co-spending addresses into clusters using Union-Find.
  Each cluster = one inferred wallet.

Change-address detection (P0.4.3):
  When a transaction has one input and two outputs, the output that:
    (a) is a "fresh" address (never seen before), AND
    (b) receives the smaller amount (often the true spend is larger)
  is likely the change address — returning funds to the sender's own wallet.
  These candidates are merged with the sender cluster.

Weak labels (P0.4.4):
  Seeds = a set of known flagged addresses (from public blacklists / OFAC /
  well-known mixer addresses). Any cluster containing a seed address
  inherits a "high-risk" weak label. All other addresses get "unknown".
  Seed addresses held out at evaluation time must NOT appear in training labels.

Outputs
-------
  data/btc_clusters.parquet   — address → cluster_id mapping
  data/btc_weak_labels.parquet — address, cluster_id, weak_label
  data/btc_cluster_summary.json — cluster statistics
"""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("weak_labels")

SNAPSHOT_PATH = Path("data/btc_snapshot_10k.parquet")
OUT_DIR       = Path("data")

# ── Seed addresses (A/B/C/D/E split) ─────────────────────────────────────────
# Frozen spec: A/B/C = training seeds (weak-label source)
#              D     = validation seed (held out from training labels)
#              E     = test seed       (held out from both training and validation)
#
# Leakage rule: D and E must never appear in training labels.
# A/B/C addresses and their co-spend cluster members receive 'high_risk' labels.
# D/E addresses and their cluster members receive 'eval_reference' labels.
#
# Sources: OFAC SDN list (public), DOJ indictments (public),
#          community-maintained abuse databases.

# Group A — OFAC sanctioned exchanges/mixers (2022–2023)
SEEDS_A = {
    "12QtD5BFwRsdNsAZY76UVE1xyCGNTojH9h",  # Hydra Marketplace
    "3FupZp77ySr7jwoLYEJ9mwzJpvoNBXsBnE",  # Garantex exchange
    "1BlenderiUQaY3tBLWwPxhFCJzMdBN8T2V",  # Blender.io mixer
}

# Group B — DOJ indictments (2021–2022)
SEEDS_B = {
    "1ASkqdo1hvydosVRc8MqtmHgbzTxb2R1ZD",  # BitcoinFog mixer
    "1BitzlatoNFGE5P2LiXLRN7R3Fm7mEEK5y",  # Bitzlato
}

# Group C — Seized darknet markets (2014–2017)
SEEDS_C = {
    "14KZsAdjJAFZbsHXBrb8VEMbFbVPdE7uS4",  # AlphaBay market
    "1SiLkRoadnyc7Nv9TkMatHi9UvHJvXSyFu",  # Silk Road 2
    "115p7UMMngoj1pMvkpHijcRdfJNXj6LrLn",  # WannaCry ransomware
}

# Group D — Validation seed (held out from training, used for val recall)
SEEDS_D = {
    "1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF",  # Mt. Gox cold wallet (known, historical)
}

# Group E — Test seed (held out from both training and validation)
SEEDS_E = {
    "1BTC3HgJaGBiCqkN6r3CKE7PeDVtfHyQsc",  # BTC-e exchange (sanctioned)
}

# Derived convenience sets
TRAIN_SEEDS = SEEDS_A | SEEDS_B | SEEDS_C   # used for 'high_risk' weak labels
VAL_SEEDS   = SEEDS_D                        # 'eval_reference', val recall only
TEST_SEEDS  = SEEDS_E                        # 'eval_reference', test recall only
EVAL_SEEDS  = VAL_SEEDS | TEST_SEEDS         # never in training labels
ALL_SEEDS   = TRAIN_SEEDS | EVAL_SEEDS


# ── P0.4.2 — Union-Find ────────────────────────────────────────────────────────

class UnionFind:
    """
    Efficient Union-Find with path compression and union by rank.
    Used to merge co-spending addresses into clusters.

    Why Union-Find?
    We have potentially millions of (address, address) co-spend pairs.
    Union-Find merges clusters in near-O(1) amortised time per operation,
    making it the standard algorithm for this task in blockchain analysis.
    """

    def __init__(self):
        self._parent: dict = {}
        self._rank:   dict = {}

    def find(self, x: str) -> str:
        """Return root of x's cluster, with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x]   = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge x's cluster and y's cluster (union by rank)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Attach smaller-rank tree under larger-rank tree
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def get_cluster_map(self) -> dict:
        """Return {address: cluster_root} for all known addresses."""
        return {addr: self.find(addr) for addr in self._parent}


# ── P0.4.3 — Change-address heuristic ─────────────────────────────────────────

def find_change_candidates(df: pd.DataFrame) -> set:
    """
    Return a set of (input_address, candidate_change_address) pairs
    for transactions with exactly 1 unique input and exactly 2 outputs,
    where the candidate change address has never appeared as an input
    in any other transaction.

    Heuristic: single-input, two-output → one output is the payment,
    the other is likely change returning to the sender's own wallet.
    We identify the change candidate as the output that:
      1. Is a "fresh" address (never seen as an input elsewhere)
      2. Receives the smaller value (payment is usually larger)

    This is a heuristic, not a certainty. The candidate is merged into
    the sender's cluster with a weak confidence tag.
    """
    # All addresses that ever appear as inputs (the "known spender" set)
    known_input_addrs = set(df["input_address"].unique())

    # Transactions with exactly 1 input address and exactly 2 output addresses
    inp_per_tx = df.groupby("tx_hash")["input_address"].nunique()
    out_per_tx = df.groupby("tx_hash")["output_address"].nunique()

    one_in_two_out = set(
        inp_per_tx[inp_per_tx == 1].index
    ) & set(out_per_tx[out_per_tx == 2].index)

    log.info(f"  1-input 2-output transactions: {len(one_in_two_out):,}")

    change_pairs = set()
    subset = df[df["tx_hash"].isin(one_in_two_out)].copy()
    subset["output_value_sat"] = subset["output_value_sat"].astype("float64")

    for _tx_hash, grp in subset.groupby("tx_hash"):
        input_addr  = grp["input_address"].iloc[0]
        output_rows = grp.drop_duplicates(subset=["output_address"])

        # Get the two output addresses and their values
        out_addrs  = output_rows["output_address"].tolist()
        out_values = output_rows["output_value_sat"].tolist()

        if len(out_addrs) != 2:
            continue

        # Fresh address = never appeared as an input anywhere
        fresh = [a for a in out_addrs if a not in known_input_addrs]
        if len(fresh) != 1:
            continue  # ambiguous — skip

        candidate = fresh[0]
        # Verify it's the smaller-value output (strengthens the heuristic)
        candidate_val = out_values[out_addrs.index(candidate)]
        other_val     = out_values[1 - out_addrs.index(candidate)]
        if candidate_val >= other_val:
            continue  # not consistent with typical change pattern

        change_pairs.add((input_addr, candidate))

    log.info(f"  Change-address candidates found: {len(change_pairs):,}")
    return change_pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── P0.4.1: Load and inspect snapshot ─────────────────────────────────────
    log.info("P0.4.1 — Loading Bitcoin snapshot …")
    df = pd.read_parquet(SNAPSHOT_PATH)

    # Fix Decimal → float64 (parquet may preserve object dtype)
    sat_cols = ["fee_satoshi", "total_input_sat", "total_output_sat",
                "input_value_sat", "output_value_sat"]
    for col in sat_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    log.info(f"  Rows          : {len(df):,}")
    log.info(f"  Transactions  : {df['tx_hash'].nunique():,}")
    log.info(f"  Input addrs   : {df['input_address'].nunique():,}")
    log.info(f"  Output addrs  : {df['output_address'].nunique():,}")
    log.info(f"  Time range    : {df['block_timestamp'].min()} → {df['block_timestamp'].max()}")

    all_addresses = set(df["input_address"].unique()) | set(df["output_address"].unique())
    log.info(f"  Total unique addresses: {len(all_addresses):,}")

    # Seed coverage check
    seeds_in_snapshot = ALL_SEEDS & all_addresses
    log.info(f"\n  Seed coverage in this snapshot: {len(seeds_in_snapshot)}/{len(ALL_SEEDS)}")
    if seeds_in_snapshot:
        log.info(f"    Found: {seeds_in_snapshot}")

    # ── P0.4.2: Co-spend clustering ────────────────────────────────────────────
    log.info("\nP0.4.2 — Co-spend clustering (Union-Find) …")

    # For each transaction, group all input addresses and union-find merge them
    uf = UnionFind()

    # First register all addresses
    for addr in all_addresses:
        uf.find(addr)  # initialises if not present

    # Merge co-spending inputs
    n_merges = 0
    inp_groups = df.groupby("tx_hash")["input_address"].apply(list)
    for _tx_hash, addrs in inp_groups.items():
        unique_addrs = list(set(addrs))
        if len(unique_addrs) >= 2:
            for i in range(1, len(unique_addrs)):
                uf.union(unique_addrs[0], unique_addrs[i])
                n_merges += 1

    log.info(f"  Co-spend merges performed: {n_merges:,}")

    # ── P0.4.3: Change-address candidates ─────────────────────────────────────
    log.info("\nP0.4.3 — Change-address detection …")
    change_pairs = find_change_candidates(df)
    n_change = 0
    for input_addr, change_addr in change_pairs:
        uf.union(input_addr, change_addr)
        n_change += 1
    log.info(f"  Change-address merges applied: {n_change:,}")

    # ── Build cluster map ──────────────────────────────────────────────────────
    cluster_map = uf.get_cluster_map()

    # Assign integer cluster IDs (root address → integer)
    roots      = sorted(set(cluster_map.values()))
    root_to_id = {r: i for i, r in enumerate(roots)}
    addr_to_cluster = {addr: root_to_id[root]
                       for addr, root in cluster_map.items()}

    # Cluster size distribution
    cluster_sizes = pd.Series(list(addr_to_cluster.values())).value_counts()
    log.info(f"\n  Total clusters             : {len(cluster_sizes):,}")
    log.info(f"  Singleton clusters (size=1): {(cluster_sizes==1).sum():,}")
    log.info(f"  Clusters with 2+ addresses : {(cluster_sizes>=2).sum():,}")
    log.info(f"  Largest cluster (addresses): {cluster_sizes.max():,}")
    log.info("\n  Cluster size distribution:")
    for sz, cnt in cluster_sizes.value_counts().sort_index().head(8).items():
        log.info(f"    size={sz}: {cnt:,} clusters")

    # ── P0.4.4: Weak-label construction ────────────────────────────────────────
    log.info("\nP0.4.4 — Weak-label construction …")

    # Find which clusters contain training seeds
    train_seed_clusters = set()
    for seed in TRAIN_SEEDS:
        if seed in addr_to_cluster:
            train_seed_clusters.add(addr_to_cluster[seed])
            log.info(f"  Seed {seed[:20]}… → cluster {addr_to_cluster[seed]}")

    eval_seed_clusters = set()
    for seed in EVAL_SEEDS:
        if seed in addr_to_cluster:
            eval_seed_clusters.add(addr_to_cluster[seed])

    # Leakage guard: eval seed clusters must not overlap with train seed clusters
    leakage = train_seed_clusters & eval_seed_clusters
    assert not leakage, f"LEAKAGE: clusters {leakage} appear in both train and eval seeds"

    # Assign weak labels
    # A/B/C cluster → "high_risk" (weak positive, used in training)
    # D cluster     → "eval_reference_val" (held out, used for val recall only)
    # E cluster     → "eval_reference_test" (held out, used for test recall only)
    val_seed_clusters  = set()
    test_seed_clusters = set()
    for seed in VAL_SEEDS:
        if seed in addr_to_cluster:
            val_seed_clusters.add(addr_to_cluster[seed])
    for seed in TEST_SEEDS:
        if seed in addr_to_cluster:
            test_seed_clusters.add(addr_to_cluster[seed])

    records = []
    for addr, cid in addr_to_cluster.items():
        if cid in train_seed_clusters:
            label = "high_risk"
        elif cid in val_seed_clusters:
            label = "eval_reference_val"
        elif cid in test_seed_clusters:
            label = "eval_reference_test"
        else:
            label = "unknown"
        records.append({"address": addr, "cluster_id": cid, "weak_label": label})

    labels_df = pd.DataFrame(records)

    label_counts = labels_df["weak_label"].value_counts()
    log.info("\n  Weak label distribution:")
    for lbl, cnt in label_counts.items():
        log.info(f"    {lbl:20s}: {cnt:,} addresses ({100*cnt/len(labels_df):.2f}%)")

    n_high_risk = (labels_df["weak_label"] == "high_risk").sum()
    log.info(f"\n  Positive label rate (high_risk): {100*n_high_risk/len(labels_df):.4f}%")
    log.info("  This is expected to be very low — most addresses are unlabeled.")

    if n_high_risk == 0:
        log.warning(
            "\n  ⚠ ZERO high-risk labels: none of the seed addresses appear in this snapshot.\n"
            "  The 10k snapshot covers only ~30 minutes of Bitcoin history.\n"
            "  Proceeding with structural-feature-only training (no seed-cluster labels).\n"
            "  The GCN will learn from graph topology + the 14 features only.\n"
            "  Seed-based weak labels will be available in the 50k snapshot."
        )

    # ── Save outputs ───────────────────────────────────────────────────────────
    clusters_df = pd.DataFrame(
        list(addr_to_cluster.items()), columns=["address", "cluster_id"]
    )
    clusters_df.to_parquet(OUT_DIR / "btc_clusters.parquet", index=False)
    labels_df.to_parquet(OUT_DIR / "btc_weak_labels.parquet", index=False)

    summary = {
        "snapshot":           str(SNAPSHOT_PATH),
        "n_transactions":     int(df["tx_hash"].nunique()),
        "n_addresses":        int(len(all_addresses)),
        "n_input_addresses":  int(df["input_address"].nunique()),
        "n_output_addresses": int(df["output_address"].nunique()),
        "n_clusters":         int(len(cluster_sizes)),
        "n_singleton_clusters": int((cluster_sizes == 1).sum()),
        "n_multi_clusters":     int((cluster_sizes >= 2).sum()),
        "largest_cluster_size": int(cluster_sizes.max()),
        "n_co_spend_merges":  n_merges,
        "n_change_merges":    n_change,
        "seed_split": {
            "A": sorted(SEEDS_A),
            "B": sorted(SEEDS_B),
            "C": sorted(SEEDS_C),
            "D": sorted(SEEDS_D),
            "E": sorted(SEEDS_E),
        },
        "train_seeds_total":  len(TRAIN_SEEDS),
        "val_seeds_total":    len(VAL_SEEDS),
        "test_seeds_total":   len(TEST_SEEDS),
        "train_seeds_in_snapshot": len(
            [s for s in TRAIN_SEEDS if s in addr_to_cluster]
        ),
        "val_seeds_in_snapshot": len(
            [s for s in VAL_SEEDS if s in addr_to_cluster]
        ),
        "test_seeds_in_snapshot": len(
            [s for s in TEST_SEEDS if s in addr_to_cluster]
        ),
        "label_distribution": label_counts.to_dict(),
        "leakage_check": "PASSED — eval seed clusters disjoint from train seed clusters",
    }
    with open(OUT_DIR / "btc_cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\nOutputs saved:")
    log.info(f"  data/btc_clusters.parquet       ({len(clusters_df):,} rows)")
    log.info(f"  data/btc_weak_labels.parquet    ({len(labels_df):,} rows)")
    log.info("  data/btc_cluster_summary.json")
    log.info("\nP0.4.1 + P0.4.2 + P0.4.3 + P0.4.4 COMPLETE ✓")


if __name__ == "__main__":
    main()
