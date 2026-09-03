# Bitcoin BigQuery Schema Reference
## `/data/SCHEMA.md`

**Dataset:** `bigquery-public-data.crypto_bitcoin`  
**Access:** Public — no credentials required for read-only queries.  
**Required by:** P0.2.1 (schema documentation before any query is written)

---

## Tables Used

We join three tables to reconstruct the full input→transaction→output flow:

### 1. `transactions`

| Column | Type | Description |
|---|---|---|
| `hash` | STRING | Transaction hash (unique identifier) |
| `block_timestamp` | TIMESTAMP | When the block containing this tx was mined |
| `fee` | INT64 | Miner fee in **satoshi** (divide by 1e8 for BTC) |
| `input_count` | INT64 | Number of input addresses |
| `output_count` | INT64 | Number of output addresses |
| `input_value` | INT64 | Total value of all inputs, in satoshi |
| `output_value` | INT64 | Total value of all outputs, in satoshi |
| `is_coinbase` | BOOL | True if this is a block reward transaction (no real inputs) |

### 2. `inputs`

Each row is one input of one transaction (a transaction can have many inputs).

| Column | Type | Description |
|---|---|---|
| `transaction_hash` | STRING | FK → `transactions.hash` |
| `spent_transaction_hash` | STRING | Hash of the tx whose output is being spent |
| `spent_output_index` | INT64 | Which output of the spent tx is being consumed |
| `value` | INT64 | Value being spent, in satoshi |
| `addresses` | ARRAY<STRING> | Sending address(es) — usually one, can be multi-sig |
| `type` | STRING | Script type of the input being spent |

### 3. `outputs`

Each row is one output of one transaction.

| Column | Type | Description |
|---|---|---|
| `transaction_hash` | STRING | FK → `transactions.hash` |
| `index` | INT64 | Output position within this transaction |
| `value` | INT64 | Value received, in satoshi |
| `addresses` | ARRAY<STRING> | Receiving address(es) |
| `type` | STRING | Script type: `pubkeyhash`, `scripthash`, `witness_v0_keyhash`, `witness_v0_scripthash`, `nulldata` |

---

## Key Join Pattern

```sql
-- Standard join to reconstruct full transaction flow
SELECT
    t.hash              AS tx_hash,
    t.block_timestamp,
    t.fee,
    t.input_count,
    t.output_count,
    t.input_value,
    t.output_value,
    t.is_coinbase,
    i.addresses         AS input_addresses,
    i.value             AS input_value_sat,
    o.addresses         AS output_addresses,
    o.value             AS output_value_sat,
    o.type              AS output_type,
    o.index             AS output_index
FROM
    `bigquery-public-data.crypto_bitcoin.transactions` t
JOIN
    `bigquery-public-data.crypto_bitcoin.inputs`  i ON i.transaction_hash = t.hash
JOIN
    `bigquery-public-data.crypto_bitcoin.outputs` o ON o.transaction_hash = t.hash
WHERE
    t.block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
    AND t.is_coinbase = FALSE     -- exclude coinbase: they have no real inputs
LIMIT 50000
```

---

## Feature → Column Mapping

This maps every feature in `FEATURE_SPEC.md` to its BigQuery source column:

| Feature | Source table | Source column | Notes |
|---|---|---|---|
| `address_degree` | Derived | — | Computed from projection after join |
| `total_sent_btc` | `outputs` | `value` | Sum of `outputs.value` for txs where address is in `inputs.addresses`. Divide by 1e8. |
| `total_received_btc` | `outputs` | `value` | Sum of `outputs.value` where address is in `outputs.addresses`. Divide by 1e8. |
| `tx_count` | Both | — | Count of distinct `tx_hash` where address appears in inputs or outputs |
| `avg_tx_value_btc` | Derived | — | `(total_sent_btc + total_received_btc) / tx_count` |
| `time_since_last_tx_hrs` | `transactions` | `block_timestamp` | `(snapshot_now - max(block_timestamp)) / 3600` |
| `clustering_coefficient` | Derived | — | Computed on address projection after build |
| `input_count` | `inputs` | `addresses` | Count of distinct txs where address is in `inputs.addresses` |
| `output_count` | `outputs` | `addresses` | Count of distinct txs where address is in `outputs.addresses` |
| `is_script_hash` | `outputs` | `type` | `1` if any `output.type` ∈ `{scripthash, witness_v0_scripthash}` |
| `address_reuse_count` | Both | — | `|T_in(a) ∩ T_out(a)|` — same-transaction co-occurrence |
| `tx_burst_score` | `transactions` | `block_timestamp` | Z-score of peak hour vs. 24-hour distribution |
| `weekday_vs_weekend_ratio` | `transactions` | `block_timestamp` | Weekday count / weekend count |
| `avg_time_between_tx_hrs` | `transactions` | `block_timestamp` | Mean interval between consecutive transactions |

---

## Important Implementation Notes

1. **`addresses` is an ARRAY** — use `UNNEST(addresses)` in SQL to explode it into one row per address.
2. **All monetary values are in satoshi** — divide by `1e8` to convert to BTC.
3. **`is_coinbase = TRUE` rows have no real inputs** — always exclude them from address-level features.
4. **Addresses can appear in both inputs and outputs of the same transaction** — this is exactly what `address_reuse_count` captures.
5. **`nulldata` outputs** (OP_RETURN) have no receiving address — filter them out when building the output address list.

---

## Offline Snapshot Strategy

| File | Rows | Used when |
|---|---|---|
| `data/btc_snapshot_50k.parquet` | 50,000 transactions | Primary demo |
| `data/btc_snapshot_10k.parquet` | 10,000 transactions | Emergency fallback if 50k is slow |

Both files are produced by `scripts/build_graph.py` and committed to Git LFS (they are excluded from regular git tracking via `.gitignore`).
