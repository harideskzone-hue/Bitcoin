# Bitcoin Feature Specification
## `/data/FEATURE_SPEC.md`

**Project:** SIH26146 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic  
**Pipeline:** Pipeline B — Bitcoin Graph Risk Ranking  
**Required by:** P0.2.7 (acceptance gate before hackathon)  
**Sign-off required from:** ML lane owner + Backend lane owner  

> This file defines the exact mathematical formula for every feature before a single line of feature-engineering code is written. Implementation must match these definitions. Tests in `ml/tests/test_features.py` must verify each formula against hand-computed fixtures.

---

## Graph Definitions

All features are computed on the **address-address projection** unless stated otherwise. The projection is derived from the bipartite source graph as follows:

```
Source graph (bipartite):
  Nodes: Address ∪ Transaction
  Edges: (a, t) ∈ INPUT_TO  if address a is an input to transaction t
         (t, a) ∈ OUTPUT_TO if transaction t has address a as an output

Projection (address-address) — SIMPLE UNDIRECTED GRAPH:
  Edge (a₁, a₂) exists iff ∃ transaction t such that
    (a₁, t) ∈ INPUT_TO  and  (t, a₂) ∈ OUTPUT_TO
    AND a₁ ≠ a₂  ← self-loops excluded from the structural graph
  Edge weight w(a₁, a₂) = Σ_t  value_btc(t → a₂) for all qualifying t

Self-loop rule: when an address appears as both input and output of the same
transaction (e.g., addr_A in tx_1 of the fixture), the bipartite edges
(addr_A, tx_1) ∈ INPUT_TO and (tx_1, addr_A) ∈ OUTPUT_TO exist, but the
resulting projected edge (addr_A, addr_A) is DROPPED before any structural
metric is computed. Flow weight for self-loops is still captured by
`address_reuse_count` and `total_received_btc`; it is not added to the
edge-weight matrix of the projection.
```

**Node set notation:**
- `A` = set of all address nodes in the snapshot
- `T(a)` = set of all transactions in which address `a` appears (as input or output)
- `T_in(a)` = set of transactions where `a` appears as an input
- `T_out(a)` = set of transactions where `a` appears as an output
- `ts(t)` = UNIX timestamp of transaction `t` (block timestamp, seconds)
- `snapshot_now` = timestamp of the most recent transaction in the snapshot

---

## Address Node Features (14 total)

### Feature 1 — `address_degree`

| Field | Value |
|---|---|
| **Symbol** | `deg(a)` |
| **Type** | `int ≥ 0` |
| **Computed on** | Address-address projection |

**Formula:**
```
deg(a) = |{ a' ∈ A : (a, a') ∈ E_projection  OR  (a', a) ∈ E_projection }|
```

Undirected degree on the projection. Counts distinct neighbour addresses, not edge multiplicity.

**Edge case:** Isolated address (no transactions in snapshot) → `deg(a) = 0`.

**Implementation note:** Build the projection as `networkx.Graph` (simple undirected) with `G_projection.remove_edges_from(nx.selfloop_edges(G_projection))` called immediately after projection construction, before any metric is computed. Then use `networkx.degree(G_projection, a)`. The mini-graph fixture verifies that `addr_A` (which has a self-loop candidate from `tx_1`) still has `address_degree` counting only its distinct *neighbour* addresses.

---

### Feature 2 — `total_sent_btc`

| Field | Value |
|---|---|
| **Symbol** | `sent(a)` |
| **Type** | `float ≥ 0.0`, denominated in BTC |
| **Computed on** | Bipartite source graph |

**Formula:**
```
sent(a) = Σ_{t ∈ T_in(a)}  Σ_{a' ∈ outputs(t)}  value_btc(t, a')
```

Sum of total transaction outflow for all transactions where `a` participates as an input. **Note on attribution:** in multi-input transactions, the full transaction output value is associated with every input address — this is an input-associated transaction-flow exposure measure, not an exact per-address attribution of BTC sent. If address A and address B are both inputs to a transaction with 10 BTC of outputs, both A and B receive `sent = 10`. This is intentional: the feature captures the flow magnitude an address is exposed to as an input participant.

**Propagates to `avg_tx_value_btc`:** because `avg_val(a) = (sent(a) + recv(a)) / n(a)`, the same attribution convention applies there.

**Edge case:** Address that never appears as an input → `sent(a) = 0.0`.

**Implementation note:** Raw BigQuery column `outputs.value` is in satoshi. Divide by `1e8` to convert to BTC before storing.

---

### Feature 3 — `total_received_btc`

| Field | Value |
|---|---|
| **Symbol** | `recv(a)` |
| **Type** | `float ≥ 0.0`, denominated in BTC |
| **Computed on** | Bipartite source graph |

**Formula:**
```
recv(a) = Σ_{t ∈ T_out(a)}  value_btc(t, a)
```

Sum of all output values directed to `a`. This is the total BTC that arrived at the address.

**Edge case:** Address that never appears as an output → `recv(a) = 0.0`.

---

### Feature 4 — `tx_count`

| Field | Value |
|---|---|
| **Symbol** | `n(a)` |
| **Type** | `int ≥ 0` |
| **Computed on** | Bipartite source graph |

**Formula:**
```
n(a) = |T(a)| = |T_in(a) ∪ T_out(a)|
```

Count of distinct transactions in which the address appears in any role.

**Edge case:** Address with no transactions in the snapshot → `n(a) = 0`.

---

### Feature 5 — `avg_tx_value_btc`

| Field | Value |
|---|---|
| **Symbol** | `avg_val(a)` |
| **Type** | `float ≥ 0.0`, denominated in BTC |
| **Computed on** | Bipartite source graph |

**Formula:**
```
        sent(a) + recv(a)
avg_val(a) = ─────────────────    if n(a) > 0
                  n(a)

avg_val(a) = 0.0                  if n(a) = 0
```

Average BTC value moved per transaction. Uses total flow (sent + received) divided by transaction count.

---

### Feature 6 — `time_since_last_tx_hrs`

| Field | Value |
|---|---|
| **Symbol** | `Δt(a)` |
| **Type** | `float ≥ 0.0`, hours |
| **Computed on** | Bipartite source graph |

**Formula:**
```
last_tx(a)  = max_{t ∈ T(a)} ts(t)
Δt(a)       = (snapshot_now - last_tx(a)) / 3600.0    if n(a) > 0
Δt(a)       = NaN → imputed to global median           if n(a) = 0
```

**Implementation note:** Impute missing values with the median of all non-null `Δt` values in the snapshot before passing to the GCN. Do not use 0 as the imputed value — that would falsely imply a very recent transaction.

---

### Feature 7 — `clustering_coefficient`

| Field | Value |
|---|---|
| **Symbol** | `C(a)` |
| **Type** | `float ∈ [0.0, 1.0]` |
| **Computed on** | Address-address projection (undirected, unweighted) |

**Formula (standard local clustering coefficient):**
```
N(a) = { a' : (a, a') ∈ E_projection }    (open neighbourhood, excluding a)

        |{ (u,v) : u ∈ N(a), v ∈ N(a), (u,v) ∈ E_projection }|
C(a) = ─────────────────────────────────────────────────────────   if |N(a)| ≥ 2
                    |N(a)| × (|N(a)| − 1) / 2

C(a) = 0.0    if |N(a)| < 2
```

Fraction of the address's neighbours that are also connected to each other. High clustering can indicate participation in tight transaction clusters (mixing or coordinated activity).

**Implementation note:** `networkx.clustering(G_projection, a)` returns the correct value **only on a simple graph with self-loops already removed**. Because NetworkX's clustering function is undefined on multigraphs and treats self-loops incorrectly, always call `G_projection.remove_edges_from(nx.selfloop_edges(G_projection))` before computing any structural metrics. Self-loop removal must happen once, immediately after projection construction, not separately for each metric.

---

### Feature 8 — `input_count`

| Field | Value |
|---|---|
| **Symbol** | `in_count(a)` |
| **Type** | `int ≥ 0` |
| **Computed on** | Bipartite source graph |

**Formula:**
```
input_count(a) = |T_in(a)|
```

Number of distinct transactions where `a` is an input (i.e., `a` is spending). Not the same as `address_degree` — this counts transaction nodes in the bipartite graph, not address neighbours in the projection.

---

### Feature 9 — `output_count`

| Field | Value |
|---|---|
| **Symbol** | `out_count(a)` |
| **Type** | `int ≥ 0` |
| **Computed on** | Bipartite source graph |

**Formula:**
```
output_count(a) = |T_out(a)|
```

Number of distinct transactions where `a` receives an output.

---

### Feature 10 — `is_script_hash`

> **Renamed from `is_multisig`:** P2SH and P2WSH output types indicate script-hash encumbrance, not confirmed multisignature. The underlying script may be multisig, but may also encode timelocks, hash pre-images, or other conditions. Calling this feature `is_multisig` would overstate what the `type` field establishes.

| Field | Value |
|---|---|
| **Symbol** | `sh(a)` |
| **Type** | `bool` → encoded as `int ∈ {0, 1}` |
| **Computed on** | BigQuery `outputs.type` column |

**Formula:**
```
sh(a) = 1  if any output in T_out(a) has type ∈ {'scripthash', 'witness_v0_scripthash'}
sh(a) = 0  otherwise
```

Detects whether an address has ever received a script-hash output (P2SH or P2WSH). These output types are structurally more complex than standard P2PKH/P2WPKH and include most multisig wallets — but also Lightning channel outputs, threshold schemes, and other script patterns. The feature captures **script complexity**, not confirmed multisignature.

**Implementation note:** Pull `type` in the BigQuery query. Do not attempt to decode raw `script_asm` — the `type` field is pre-parsed. If actual multisig confirmation is needed in a later phase, script-level decoding against `script_asm` is required.

---

### Feature 11 — `address_reuse_count`

| Field | Value |
|---|---|
| **Symbol** | `reuse(a)` |
| **Type** | `int ≥ 0` |
| **Computed on** | Bipartite source graph |

**Formula:**
```
reuse(a) = |T_in(a) ∩ T_out(a)|
```

**Precise definition:** Number of transactions in which address `a` appears **simultaneously as both an input and an output** within the same transaction. This is a **same-transaction input/output co-occurrence indicator** — it captures the pattern of change outputs returning to the spending address within a single transaction.

This does **not** count how many times an address is reused across distinct transactions (i.e., it is not a count of total spend events by a previously-receiving address). The name is retained for API/model-feature consistency, but the implementation and unit tests must match this exact definition.

**Edge case:** An address that always sends to fresh addresses and never receives change back → `reuse(a) = 0`.

**Note:** This is a per-address metric, not a wallet-level metric. Do not conflate with the Meiklejohn co-spend cluster.

---

### Feature 12 — `tx_burst_score`

| Field | Value |
|---|---|
| **Symbol** | `burst(a)` |
| **Type** | `float`, Z-score (can be negative) |
| **Computed on** | Transaction timestamps in `T(a)` |

**Formula:**
```
For address a, partition T(a) by hour-of-day (UTC):
  count_h(a) = |{ t ∈ T(a) : hour_of_day(ts(t)) = h }|   for h ∈ {0, 1, …, 23}

  μ_a = mean({count_h(a) : h ∈ 0..23})          (over all 24 buckets)
  σ_a = std({count_h(a) : h ∈ 0..23}, ddof=0)   (population std)

  peak_h(a) = argmax_{h} count_h(a)

            count_{peak_h}(a) − μ_a
burst(a) = ──────────────────────────    if σ_a > 0
                     σ_a

burst(a) = 0.0    if σ_a = 0  (uniform distribution or single transaction)
```

Z-score of the address's busiest hour relative to its own 24-hour distribution. Address-self-normalized so naturally high-volume addresses are not penalized.

**Edge case:** `|T(a)| < 24` → some hour buckets will be 0. This is expected; the formula handles it correctly.

---

### Feature 13 — `weekday_vs_weekend_ratio`

| Field | Value |
|---|---|
| **Symbol** | `wk_ratio(a)` |
| **Type** | `float ≥ 0.0` |
| **Computed on** | Transaction timestamps in `T(a)` |

**Formula:**
```
weekday_count(a) = |{ t ∈ T(a) : dayofweek_utc(ts(t)) ∈ {0,1,2,3,4} }|
weekend_count(a) = |{ t ∈ T(a) : dayofweek_utc(ts(t)) ∈ {5,6} }|

                      weekday_count(a)
wk_ratio(a) = ────────────────────────────────   if weekend_count(a) > 0
                      weekend_count(a)

wk_ratio(a) = float(weekday_count(a))            if weekend_count(a) = 0 and weekday_count(a) > 0
wk_ratio(a) = 1.0                                if both counts = 0
```

`dayofweek_utc`: ISO convention, 0=Monday…4=Friday (weekday), 5=Saturday, 6=Sunday (weekend). All timestamps in UTC.

---

### Feature 14 — `avg_time_between_tx_hrs`

| Field | Value |
|---|---|
| **Symbol** | `avg_ibi(a)` |
| **Type** | `float ≥ 0.0`, hours |
| **Computed on** | Transaction timestamps in `T(a)` |

**Formula:**
```
Sort T(a) by ts: t₁ ≤ t₂ ≤ … ≤ tₙ

If n ≥ 2:
  Δᵢ = (ts(t_{i+1}) − ts(tᵢ)) / 3600.0   for i = 1…n-1
  avg_ibi(a) = mean(Δ₁, Δ₂, …, Δ_{n-1})

If n < 2:
  avg_ibi(a) = NaN → imputed to global median
```

Mean inter-transaction interval. Very small values indicate automated chaining or layering.

**Imputation:** Same strategy as `time_since_last_tx_hrs` — global median, never 0.

---

## Change-Address Candidate Confidence Formula

Output of P0.4.3: `{ address: { candidate_change: bool, confidence: float ∈ [0,1] } }`

Score each output of a transaction:

| Condition | Points |
|---|---|
| Output value is strictly less than all other non-OP_RETURN outputs of the same transaction | +2 |
| Output address has no prior appearance in any `T_in` set (fresh address) | +2 |
| Output address falls in the same co-spend cluster as ≥1 input address of the same transaction | +2 |
| Output is the only non-OP_RETURN output besides the main payment (single-change pattern) | +1 |
| Output address reuse count ≥ 1 (address was already seen — penalise) | −1 |

```
raw_score(a, t) = clamp(sum_of_conditions, 0, 7)
confidence(a, t) = raw_score(a, t) / 7.0
candidate_change(a, t) = True if confidence(a, t) ≥ 0.4
```

**Important:** `confidence = 1.0` means all heuristics fired simultaneously. It is **not** a probability of confirmed change ownership. Investigators must not treat this as ground truth.

---

## Co-spend Cluster Confidence Assignment

After Union-Find clustering (P0.4.2), assign to each cluster:

| Condition | Confidence |
|---|---|
| Cluster contains ≥ 1 address from the known-bad seed list | `HIGH` |
| Cluster size ≥ 3 addresses with ≥ 2 co-spend transactions each | `MEDIUM` |
| All other clusters | `LOW` |

Confidence is assigned to the cluster then propagated to all member addresses. It represents strength of co-spend evidence, not probability of illicit activity.

---

## Feature Matrix Summary

| # | Feature | Type | Range | Source |
|---|---|---|---|---|
| 1 | `address_degree` | int | ≥ 0 | Projection |
| 2 | `total_sent_btc` | float | ≥ 0 | Bipartite |
| 3 | `total_received_btc` | float | ≥ 0 | Bipartite |
| 4 | `tx_count` | int | ≥ 0 | Bipartite |
| 5 | `avg_tx_value_btc` | float | ≥ 0 | Derived |
| 6 | `time_since_last_tx_hrs` | float | ≥ 0, imputed | Bipartite |
| 7 | `clustering_coefficient` | float | [0, 1] | Projection |
| 8 | `input_count` | int | ≥ 0 | Bipartite |
| 9 | `output_count` | int | ≥ 0 | Bipartite |
| 10 | `is_script_hash` | {0, 1} | — | BigQuery `type` |
| 11 | `address_reuse_count` | int | ≥ 0 | Bipartite |
| 12 | `tx_burst_score` | float | Z-score | Timestamps |
| 13 | `weekday_vs_weekend_ratio` | float | ≥ 0 | Timestamps |
| 14 | `avg_time_between_tx_hrs` | float | ≥ 0, imputed | Timestamps |

---

## Normalisation

Apply **RobustScaler** (median / IQR) rather than standard z-score. Justification: `total_sent_btc`, `total_received_btc`, and `avg_tx_value_btc` follow power-law distributions in Bitcoin; standard scaling gives extreme-value addresses disproportionate feature magnitudes.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
# Persist: joblib.dump(scaler, 'ml/checkpoints/feature_scaler.pkl')
X_val_scaled   = scaler.transform(X_val_raw)    # never refit
X_test_scaled  = scaler.transform(X_test_raw)   # never refit
```

**Critical:** Fit scaler on training node set only. Apply the same fitted scaler at inference time.

---

## Acceptance Test Fixtures (for `ml/tests/test_features.py`)

Minimum one hand-computed fixture per feature. Use the following mini-graph:

```
Addresses: addr_A, addr_B, addr_C
Transactions:
  tx_1  ts=Monday 02:00 UTC   addr_A -[INPUT 1.0 BTC]-> tx_1
                               tx_1 -[OUTPUT 0.9 BTC]-> addr_B
                               tx_1 -[OUTPUT 0.1 BTC]-> addr_A   ← reuse

  tx_2  ts=Monday 03:00 UTC   addr_B -[INPUT 0.9 BTC]-> tx_2
                               tx_2 -[OUTPUT 0.85 BTC]-> addr_C

  tx_3  ts=Tuesday 14:00 UTC  addr_A -[INPUT 0.1 BTC]-> tx_3
                               tx_3 -[OUTPUT 0.09 BTC]-> addr_C

snapshot_now = ts(tx_3)
```

Expected values (compute by hand before coding):

| Feature | addr_A | addr_B | addr_C |
|---|---|---|---|
| `tx_count` | 2 | 2 | 2 |
| `total_sent_btc` | 1.1 | 0.9 | 0.0 |
| `total_received_btc` | 0.1 | 0.9 | 0.94 |
| `input_count` | 2 | 1 | 0 |
| `output_count` | 1 | 1 | 2 |
| `address_reuse_count` | 1 | 0 | 0 |
| `avg_time_between_tx_hrs` | 36.0 | 1.0 | 35.0 |
| `time_since_last_tx_hrs` | 0.0 | 11.0 | 0.0 |

All 14 features must have at least one passing fixture test. Tests must be deterministic and require no network access.
