# SIH26146 — AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

> **Smart India Hackathon 2026 · Round 1**  
> Weak-supervision Graph Neural Network for Bitcoin risk ranking and investigator prioritization.

---

## Quick start (any machine, <10 minutes)

```bash
git clone <repo>
cd Bitcoin

# Option A — Docker (recommended)
docker compose up --wait
# API: http://localhost:8000/docs
# Dashboard: http://localhost:5174

# Option B — Local (Python 3.11 venv)
conda env create -f environment.yml && conda activate sih26146
# OR
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Run offline pipeline (no BigQuery quota needed)
python ml/scripts/run_offline_pipeline.py

# Start backend
uvicorn backend.app.main:app --reload --port 8000

# Start frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5174
```

---

## Architecture

```
Investigator Dashboard (React + Cytoscape.js)  :5174
        ↕ REST/JSON
FastAPI Investigation API                       :8000
        ↕
  ┌─────────────┬────────────────┬─────────────┐
  Risk Ranking  Explanation       Graph Query
  ↕             ↕                ↕
  Bitcoin GCN   Two-layer        BFS on
  (Pipeline B)  Explainer        bipartite graph
        ↕
  ┌─────────────┬──────────────┐
  Clustering    GCN Model
  (co-spend /   (14 features,
  change-addr)  weak supervision)
        ↕
  Bitcoin snapshot (BigQuery / local parquet)
```

---

## Two ML Pipelines

### Pipeline A — Elliptic (Methodology Validation)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Majority (always licit) | 0.000 | 0.000 | 0.000 | 0.046 | 0.500 |
| Heuristic (degree+flow) | 0.007 | 0.007 | 0.007 | 0.038 | 0.434 |
| GCN (2-layer, weighted) | 0.060 | **0.980** | 0.113 | 0.109 | 0.773 |
| **GraphSAGE (2-layer)** | 0.083 | 0.900 | 0.151 | **0.335** | **0.815** |

> Fixed temporal split (time steps 42–49 as test). Unknown nodes participate in message passing only.  
> PR-AUC is the primary metric for imbalanced classification (~5% positive rate).  
> These results validate the GNN *methodology*. They do not apply directly to the Bitcoin pipeline.

### Pipeline B — Bitcoin Graph Risk Ranking

- **Model:** GCN with 14 Bitcoin-specific features, weak-supervision labels
- **Weak labels:** FLAGGED (known OFAC/sanctions seeds) → SUSPICIOUS (≤1 hop on projection graph) → UNKNOWN (excluded from loss)
- **Seed split:** A/B/C (training) / D (validation) / E (test) — no leakage
- **Output:** 0–100 model-derived risk ranking (NOT a calibrated probability)
- **10k status:** Fail-closed (zero seeds in window) — correct behaviour
- **50k status:** Pending BigQuery quota reset

**Score disclaimer (hard-coded in every API response):**  
*"Model-derived risk ranking. For prioritization and human review only. Not a calibrated probability."*

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | Health + model version |
| `GET /api/v1/model/info` | Full model metadata |
| `GET /api/v1/alerts?min_risk=70&limit=20` | Top-risk addresses |
| `GET /api/v1/wallets/{address_hash}` | Full risk profile + explanations + P2P signals |
| `GET /api/v1/graph?address=&depth=1` | Ego-graph (bipartite BFS) |

Interactive docs: http://localhost:8000/docs  
OpenAPI spec: [`backend/openapi.yaml`](backend/openapi.yaml)

---

## Risk Label Thresholds (spec correction #28)

| Score | Label | Priority |
|---|---|---|
| 0–24 | LOW | Monitor |
| 25–49 | MEDIUM | Review when capacity allows |
| 50–74 | HIGH | Review within 24h |
| 75–100 | CRITICAL | Review immediately |

Defined once in [`backend/constants.py`](backend/constants.py) and [`frontend/src/constants.ts`](frontend/src/constants.ts). Never hard-coded inline.

---

## Explainability (Two-Layer)

**Layer 1 — Deterministic (always runs, <100ms):**
- Within 2 hops of a flagged address
- Unusual transaction volume burst (off-hours)
- High address reuse pattern
- Linked to cluster with flagged member
- Candidate change-address relationship detected
- Rapid consecutive transactions

**Layer 2 — GNNExplainer (optional, top-5 only, 2–5s each):**  
Top-3 node features + top-3 edge importances. Gracefully skipped if time budget exceeded.

---

## P2P Signals

All P2P signals are `signal_source: "SIMULATED"` with `ip_cluster_id: "IP_CLUSTER_XX"`.  
No real IP addresses. No real network observations. Dashboard shows **⚠ SIMULATED DATA** banner at all times.

---

## Repository Structure

```
/ml
  /scripts      — Pipeline scripts (build_graph, train_gcn, infer_bitcoin, eval_elliptic …)
  /explain      — Explainability (deterministic + GNNExplainer)
  /tests        — 127 tests (all passing)
/backend
  /app          — FastAPI app (routers, schemas, models, services)
  constants.py  — Single source of truth for thresholds/disclaimers
  openapi.yaml  — API contract
/frontend       — React + Vite + Cytoscape.js dashboard (:5174)
/data           — Snapshots, graphs, P2P signals, weak labels
/models         — Saved checkpoints (gcn_elliptic_v1.pt, sage_elliptic_v1.pt)
/notebooks      — Evaluation notebooks (elliptic_eval.ipynb, bitcoin_eval.ipynb)
/docs           — Demo script + Q&A
```

---

## Running tests

```bash
source .venv/bin/activate
python -m pytest ml/tests/ -v          # 127/127 pass

# Spec-required exit scripts
python ml/scripts/eval_elliptic.py --split data/elliptic_split.json
python ml/scripts/infer_bitcoin.py --offline
python ml/scripts/run_offline_pipeline.py   # 6 PASS / 1 EXPECTED_FAIL
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Fail-closed GCN | Rather than produce unjustified scores from zero positives, training aborts. The system is safer wrong-way-closed than wrong-way-open. |
| Risk ranking, not probability | No Bitcoin calibration dataset exists in Round 1. Platt scaling removed. Every API response carries the disclaimer. |
| Two-pipeline separation | Elliptic validates methodology. Bitcoin ranking is a separate model. Results are not cross-claimed. |
| Deterministic explanations first | Always works, <100ms. GNNExplainer is optional and time-budgeted. Demo never breaks on explanation latency. |
| SIMULATED P2P, always labelled | The UI makes simulation status unambiguous. Judges cannot mistake simulated signals for real evidence. |
| PostgreSQL + SQLite fallback | Explicit `DATABASE_BACKEND=` printed at startup. Never a silent switch. `USE_SQLITE_FALLBACK=true` for venue demos without Docker. |

---

## Phase 0 exit checklist

- [x] `docker compose up` works on a clean machine
- [x] `--offline` mode runs end-to-end in <2 minutes
- [x] Elliptic comparison table populated with actual numbers
- [x] Bitcoin GCN produces fail-closed result on 10k (zero seeds — correct)
- [x] API contract committed; frontend runs against Prism mock
- [x] No IP-looking strings in P2P simulation data (P0.7.3 PASS)
- [x] Demo script tested and timed (`docs/demo_script.md`)
- [x] 10 hardest Q&A prepared (`docs/qa_hardest.md`)
- [ ] 50k Bitcoin GCN empirical validation (pending BigQuery quota)
