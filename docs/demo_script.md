# SIH26146 — 5-Minute Demo Script

> **P0.8.4 — Click-by-click walkthrough for the SIH event**
> Timed target: **≤ 5 minutes end-to-end**
> Practice until every section is automatic.

---

## Before you start (setup — done before judges arrive)

```bash
# Terminal 1: start API (offline mode)
cd /path/to/Bitcoin
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: start Prism mock (if API DB not connected)
npx @stoplight/prism-cli mock backend/openapi.yaml

# Verify pipeline is ready
python ml/scripts/run_offline_pipeline.py
```

Expected output: `✅ ALL GATES PASSED — demo-ready on 10k snapshot`

---

## Demo flow (5 minutes)

### 0:00–0:45 — Problem framing (45 seconds)

**Say:**
> "Bitcoin transactions are public but pseudonymous. Investigators face two problems:
> 17 million daily transactions to triage, and no ground-truth labels for most addresses.
> Our system uses graph neural networks and weak supervision to rank addresses by risk
> for human investigators — not to make enforcement decisions."

**Click:** Show the architecture diagram (or describe it):
```
Bitcoin blockchain
  → co-spend clustering (ownership heuristic)
  → seed-derived weak labels (A/B/C training, D/E held out)
  → GCN risk ranking (0–100 score)
  → API + dashboard for investigators
```

---

### 0:45–1:30 — Data pipeline (45 seconds)

**Open terminal. Run:**
```bash
python ml/scripts/run_offline_pipeline.py
```

**Point to each stage as it prints:**
- `✅ Snapshot OK — 10,000 transactions, 17,660 addresses`
- `✅ Graph OK — 14 features, INPUT_TO/OUTPUT_TO edges`
- `✅ Weak labels — 0 high-risk labels` ← **highlight this**
- `⚠ GCN fail-closed` ← **explain this explicitly**

**Say:**
> "Notice Stage 4. This 30-minute window contains zero known seed addresses,
> so the model refuses to train. This is the correct behaviour — the system
> fails safely rather than producing unjustified risk scores. The 50k snapshot
> covering 90 days would contain seed coverage."

---

### 1:30–2:15 — Elliptic methodology validation (45 seconds)

**Say:**
> "To validate our GNN approach before deploying on Bitcoin, we tested on the
> publicly labelled Elliptic dataset — 200k Bitcoin transactions with ground truth."

**Show (or print):**
```
Model         PR-AUC    ROC-AUC
─────────────────────────────────
Majority      0.0461    0.5000
Heuristic     0.0375    0.4342
GCN           0.1087    0.7726
GraphSAGE     0.3347    0.8152   ← best
```

**Say:**
> "Both GNNs significantly outperform the dumb baselines on a fixed temporal test split.
> GraphSAGE PR-AUC 0.3347 vs. majority-class baseline 0.0461 — a 7x improvement.
> This validates the GNN architecture before we use it on Bitcoin."

---

### 2:15–3:00 — API + risk scores (45 seconds)

**Open browser → http://127.0.0.1:4010/api/v1/alerts**

```json
{
  "score_disclaimer": "Model-derived risk ranking. For prioritization and human review only. Not a calibrated probability.",
  "alerts": [...],
  "total": ...
}
```

**Point to `score_disclaimer`. Say:**
> "Every single API response that contains a risk score also carries this disclaimer.
> It's structural — the schema enforces it. There is no way to return a score without it."

**Click → http://127.0.0.1:4010/api/v1/wallets/00016fa6fea049c0**

**Point to `reasons` array. Say:**
> "Layer 1 deterministic explanations always run — they're fast (<100ms) and need no
> model. Layer 2 GNNExplainer runs on the top-5 flagged addresses for deeper attribution."

---

### 3:00–3:45 — P2P simulation (45 seconds)

**Open terminal:**
```bash
cat data/p2p_signals_simulated.json | python3 -m json.tool | head -30
```

**Point to fields. Say:**
> "All P2P data is clearly marked SIMULATED. The `signal_source` field is always
> 'SIMULATED', the `disclaimer` field says 'Not real network observations',
> and we use IP_CLUSTER_XX identifiers — no real IP addresses anywhere in the file.
> We verified this with a grep check."

**Show grep verification:**
```bash
grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' data/p2p_signals_simulated.json \
  && echo "FAIL" || echo "PASS: no real IPs"
```

> ⚠ **SIMULATED DATA — NOT REAL NETWORK TELEMETRY**

---

### 3:45–4:30 — Co-spend clustering (45 seconds)

**Say:**
> "We use a Union-Find co-spend heuristic: if two addresses appear as inputs
> to the same transaction, we infer they're controlled by the same entity.
> On the 10k snapshot, we found 3,857 co-spend merges and 1,013 change-address
> candidates. This is an ownership/control *heuristic*, not proof of common ownership."

**Show summary:**
```bash
cat data/btc_cluster_summary.json | python3 -m json.tool
```

**Point to:** `largest_cluster_size`, `n_multi_clusters`, `leakage_check: PASSED`

---

### 4:30–5:00 — Closing (30 seconds)

**Say:**
> "What we've built is a complete weak-supervision risk-ranking pipeline:
> graph construction, co-spend clustering, seed-derived labels with a
> train/validation/test split, a GNN that fails closed when there's no
> supervision signal, two-layer explainability, and an API with structural
> disclaimer enforcement.
>
> On the Elliptic benchmark, GraphSAGE PR-AUC 0.3347 vs. 0.0461 baseline.
> On Bitcoin: ready to train and evaluate once the 50k snapshot is available."

---

## P0.8.5 — P2P panel required wording

Wherever P2P signals are displayed in the UI, show this prominently:

```
⚠ SIMULATED DATA — NOT REAL NETWORK TELEMETRY
```

The `disclaimer` field in every P2P record reads:
> `"Simulated/replayed demonstration data only. Not real network observations."`

The frontend must display this before showing any P2P signal details.

---

## Timing checklist

| Section | Target | Notes |
|---|---|---|
| Problem framing | 0:45 | Keep to 3 sentences |
| Data pipeline | 0:45 | Let the terminal output do the talking |
| Elliptic results | 0:45 | Show the 4-row table, highlight PR-AUC |
| API + explanations | 0:45 | Browser + `score_disclaimer` visual |
| P2P simulation | 0:45 | Terminal + grep + "NOT REAL" wording |
| Co-spend clustering | 0:45 | JSON summary |
| Closing | 0:30 | Memorise the two numbers: 0.3347 vs 0.0461 |
| **Total** | **5:00** | |

---

## Emergency recoveries

| Problem | Recovery |
|---|---|
| API won't start | Switch to Prism: `npx @stoplight/prism-cli mock backend/openapi.yaml` |
| Dry run shows FAIL | Pre-generated `data/dry_run_report.json` is already committed |
| Judge asks about 50k | "BigQuery quota resets daily. The 50k run unblocks P0.4 empirical metrics, which is the correct next step after today's demo." |
| Judge asks "is this a real IP?" | "No — grep verifies it. All P2P signals are synthetic, clearly labelled SIMULATED." |
| Judge asks about false positives | "That's exactly why every score carries the disclaimer and routes to human review." |
