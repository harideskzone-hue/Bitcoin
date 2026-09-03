# SIH26146 — Final Implementation Backlog
## AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic
### Version 5.1 — Frozen, Implementation-Ready

> **Date:** September 2026 | **Review score before this version:** 9.8/10 → Final: 9.9/10 execution-ready  
> **This document is the implementation backlog, not a planning document.** Every item below maps to: Phase → Task → Subtask → Owner → Input → Output → Dependency → Technology → Acceptance Test → Hours.
> **Status: FROZEN.** No further architectural iteration. Proceed to implementation.

---

## Corrections Applied (Summary)

### Version 3.0 corrections (from 8.3 → 9.1)

| # | Issue | Old Plan | This Plan |
|---|---|---|---|
| 1 | Transfer learning mismatch | "94 Elliptic features overlap with 7 BigQuery features" | Elliptic = methodology validation only. Bitcoin uses independent feature space. Cross-dataset TL deferred to Round 2. |
| 2 | Risk score framing | "Platt-scaled probability → 0–100" | "Model-derived risk ranking, 0–100. For prioritization only, not determination of illicit activity." |
| 3 | Graph model | Wallets as nodes, transactions as edges | Bipartite: `Address –INPUT_TO→ Transaction –OUTPUT_TO→ Address`. Aggregated projection for GNN. |
| 4 | Co-spend logic | "Detect change, confirm wallet ownership" | "Candidate entity linkage with confidence levels." Heuristics are probabilistic. |
| 5 | Explainability bottleneck | GNNExplainer on all top-20 nodes | Two-layer: deterministic features first (always works); GNNExplainer optional on top 5 only. |
| 6 | P2P UI risk | "(simulated)" text alongside IP-like addresses | Dedicated "SIMULATED DATA" panel; `IP_CLUSTER_XX` labels; `signal_source: SIMULATED` in API. |
| 7 | P2P data labels | `192.168.x.x/24` | `IP_CLUSTER_07` — no IP-looking strings anywhere in the demo. |
| 8 | Dependency graph error | GCN depends on clustering | Clustering and GCN run in **parallel** from Data Pipeline; they merge at Risk/Ranking. |
| 9 | Missing milestone | No interim freeze | **Hour 18: MVP Freeze checkpoint.** If core isn't running, switch to fallback immediately. |
| 10 | Snapshot inconsistency | 10k in one place, 50k in another | Two tiers: 50k primary snapshot, 10k emergency-minimal snapshot. |
| 11 | pgvector premature | pgvector in Round 1 | Plain PostgreSQL for Round 1. pgvector added in Round 2 if embedding similarity is needed. |
| 12 | Missing API endpoint | No model metadata endpoint | `GET /api/v1/model/info` added. |
| 13 | Cluster-elevation fallback | "Clusters with ≥1 known-bad seed get elevated scores" | "Proximity to a known-flagged seed contributes to prioritization score." No automatic cluster elevation. |
| 14 | Missing baseline comparison | Only GCN reported | Evaluation table: majority baseline → heuristic → GCN → GraphSAGE. |
| 15 | Hard P/R gate | P≥0.70, R≥0.50 is pass/fail | Aspirational targets. If missed: document, compare baseline, analyze failure. |
| 16 | PR-AUC missing | Only ROC-AUC | PR-AUC added (essential for imbalanced classification). |
| 17 | Legal instrument over-specification | "IT Act S.69, PMLA 2023, NTRO" prescribed | "Obtain written authorization from agency counsel. Agency determines applicable authority." |
| 18 | CERT-In categorical | "Register with CERT-In per IT Act" | "Conduct CERT-In compliance assessment with agency. Implement resulting obligations." |
| 19 | Phase 3 too monolithic | Long sequential list | Adaptive parallel tracks: ML Scale / Operations / Multi-chain, prioritized by Phase 2 outcomes. |
| 20 | Hackathon structure | 7 build steps | 7 explicit phases with clear objectives and freeze conditions. |
| 21 | Feature alignment claim | "94 common features, direct alignment" | Statement removed. Elliptic and Bitcoin have separate, independently engineered feature spaces. |

### Version 4.0 corrections (from 9.1 → 9.5)

| # | Issue | v3.0 Plan | This Plan |
|---|---|---|---|
| 22 | Bitcoin calibration undefined | `risk_score = round(calibrated_prob * 100)` — no Bitcoin calibration dataset exists | Calibration removed from Bitcoin pipeline. Score is model-derived ranking from raw GNN output. Proper calibration deferred to Round 2 after investigator-reviewed labels are available. |
| 23 | Seed-level label leakage | No holdout — model could learn "near seed = suspicious" from training graph itself | Seed-level holdout added (P0.4.8): training seeds, validation seed, test seed kept separate. Metrics reported for both seen-seed proximity and held-out-seed generalization. |
| 24 | Elliptic split unspecified | "Exclude unknown nodes from training" — random split risk of temporal leakage | Fixed temporal train/val/test split specified. `data/elliptic_split.json` committed. Unknown nodes participate in message passing but not in loss or evaluation. |
| 25 | Baseline comparison as gate | "GCN must outperform baselines" as acceptance test | Made an evaluation requirement, not an outcome gate. If GNN does not outperform: document, retain best-performing baseline for demo, analyze failure mode. |
| 26 | Offline/live equality too strict | "`--offline` output identical to live pull" — impossible (ordering, timestamps, query version) | Changed to: "`--offline` produces schema-equivalent output and passes deterministic validation." |
| 27 | Pipeline B misnaming | "Bitcoin Graph (Live Detection)" — Round 1 uses a static snapshot, not live streaming | Renamed to "Bitcoin Graph Risk Ranking". Real-time monitoring is Phase 3 Track B. |

### Version 5.1 corrections (from 9.8 → 9.9/10)

| # | Issue | v4.0 Plan | This Plan |
|---|---|---|---|
| 28 | Risk-label thresholds undefined | `risk_label: LOW\|MEDIUM\|HIGH\|CRITICAL` declared but no mapping specified — backend and frontend could diverge | Deterministic threshold table added: 0–24=LOW, 25–49=MEDIUM, 50–74=HIGH, 75–100=CRITICAL. Defined as a named constant in API contract and committed to `backend/constants.py`. |
| 29 | Bitcoin ranking evaluation too thin | Only recall near seen/held-out seeds — not enough to demonstrate ranking quality to judges | Precision@10, Recall@10, and NDCG@10 added (P0.4.9). Precision@K is the metric most legible to SIH judges for a prioritization system. |
| 30 | Hop distance ambiguous | "address is ≤1 hop from seed" — could mean bipartite or projected graph | Explicit: "hop distance measured on the aggregated address-address projection, not the bipartite source graph." |
| 31 | Calibration implementation breaks on PyG | `CalibratedClassifierCV(cv='prefit')` assumes a sklearn-compatible estimator — PyG GCN is not one | Replaced with: export GCN logits from validation split, fit `LogisticRegression(C=1.0)` on `(logits, true_labels)`, apply to test logits. No sklearn estimator interface required. |

---

## Corrected Architecture

```
                    ┌────────────────────────┐
                    │    Investigator UI     │
                    │ React + Cytoscape.js   │
                    └───────────┬────────────┘
                                │ REST / JSON
                    ┌───────────▼────────────┐
                    │       FastAPI           │
                    │ Investigation API       │
                    └───────────┬────────────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
      Risk Ranking         Explanation         Graph Query
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │     AI / Analytics     │
                    │ GCN (Bitcoin graph)    │
                    │ Graph features         │
                    │ Temporal features      │
                    │ Behavioral features    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │    Unified Graph       │
                    │ Address                │
                    │ Transaction             │
                    │ Entity/Cluster (inferred)│
                    │ Simulated P2P signal   │
                    └───────────┬────────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
       Bitcoin Data          Elliptic          P2P Demo
       BigQuery              (methodology      Signals
       + Local Parquet        validation        (SIMULATED,
       Offline Fallback       only)             IP_CLUSTER_XX)
```

**Corrected dependency graph (hackathon build):**

```
                 ┌───────────────┐
                 │ Data Pipeline │
                 └───────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Clustering              GCN Model
      (co-spend /             (Bitcoin feature
    change-address)            space, independent)
              │                     │
              └──────────┬──────────┘
                         ▼
                  Risk Ranking
                  (merge cluster
                   evidence +
                   GCN score)
                         │
                ┌────────┴────────┐
                ▼                 ▼
          Explanation          P2P Demo
        (deterministic         (simulated,
         first; GNNExp          explicitly
         optional)              labelled)
                │                 │
                └────────┬────────┘
                         ▼
                      Backend API
                         ↓
                     Frontend UI
```

---

## Two ML Pipelines (Critical Correction #1)

### Pipeline A — Elliptic (Methodology Validation)

```
Elliptic Dataset (labeled)
        ↓
Elliptic feature engineering (94 features as-is)
        ↓
GCN / GraphSAGE
        ↓
Evaluate: precision, recall, F1, PR-AUC, ROC-AUC
        ↓
Comparison table vs. baselines
        ↓
Demonstrates: the GNN methodology works on labeled data
```

**Purpose:** Prove the approach is sound. This is what you show judges when they ask "does this actually detect illicit transactions?"

**Feature set:** The 94 Elliptic features, used as-is. Do not map them to BigQuery features.

---

### Pipeline B — Bitcoin Graph Risk Ranking

> **Renamed from "Live Detection":** Round 1 uses a static local snapshot, not a live stream. Real-time monitoring is Phase 3 Track B. Calling it "live" invites a judge question the team cannot answer in Round 1.

```
BigQuery / Local Parquet Snapshot
        ↓
Bitcoin-specific feature engineering:
  address_degree, total_sent_btc, total_received_btc,
  tx_count, avg_tx_value_btc, time_since_last_tx_hrs,
  clustering_coefficient, input_count, output_count,
  is_script_hash, address_reuse_count, tx_burst_score,
  weekday_vs_weekend_ratio, avg_time_between_tx_hrs
        ↓
Bipartite graph: Address –INPUT_TO→ Tx –OUTPUT_TO→ Address
        ↓
Aggregated address-address projection for GNN input
        ↓
GCN trained on weak-supervision labels (seed-level holdout enforced)
        ↓
Raw GNN output score → 0–100 model-derived risk ranking
(NOT a calibrated probability — no Bitcoin calibration dataset exists in Round 1;
proper calibration is a Round 2 task requiring investigator-reviewed labels)
        ↓
Explanation generation (deterministic features first)
```

**Purpose:** Risk-ranked prioritization on real Bitcoin data. No label transfer from Elliptic. Labels for weak supervision come from co-spend cluster membership and known-flagged seed addresses (with held-out seeds for evaluation).

**What is NOT claimed:** That this GCN produces calibrated probabilities, or that it has the same precision/recall as the Elliptic-trained one. The Elliptic numbers validate the GNN methodology; the Bitcoin pipeline produces a ranked prioritization list for human investigators.

---

### Round 2 — Cross-Dataset Transfer Learning (Deferred)

```
Elliptic pretrained GNN
        ↓
Representation/embedding projection layer
        (learned alignment between feature spaces)
        ↓
Bitcoin graph representation
        ↓
Fine-tuning on agency-labeled cases
        ↓
Evaluation with held-out agency ground truth
```

**This is a research task, not a hackathon feature.** It requires: (a) a labeled Bitcoin dataset for the fine-tuning target, (b) an explicit alignment strategy (e.g., feature projection MLP, domain adaptation), and (c) evaluation that confirms transfer actually helps vs. training from scratch on Bitcoin data alone.

---

## Phase 0 — Pre-Hackathon Preparation
> **Duration:** 14 days pre-event | **All tasks must be complete before Day 0 of Round 1**

---

### Task P0.1 — Environment & Repository

| Field | Detail |
|---|---|
| **Owner** | Backend |
| **Duration** | Day 1–2 |
| **Dependency** | None |
| **Input** | Empty repo |
| **Output** | Running Docker Compose stack, green CI pipeline |
| **Technology** | Python 3.11, PyTorch 2.3, PyG 2.5, NetworkX 3.3, scikit-learn 1.5, FastAPI 0.111, PostgreSQL 16 (no pgvector in Round 1), React 18 + Vite, Cytoscape.js 3.x, Docker Compose |

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.1.1 | Initialize monorepo: `/ml`, `/backend`, `/frontend`, `/data`, `/scripts` | `git log` shows initial commit | 0.5 |
| P0.1.2 | `environment.yml` with pinned versions | `conda env create && conda activate` succeeds with 0 conflicts | 1 |
| P0.1.3 | `docker-compose.yml`: services `api`, `db` (PostgreSQL 16), `frontend` | `docker compose up --wait` exits 0; all services healthy | 2 |
| P0.1.4 | GitHub Actions: ruff lint, mypy type-check, pytest | First PR shows green checks | 1 |
| P0.1.5 | PostgreSQL schema: `address_clusters`, `node_scores`, `node_reasons`, `audit_log` tables | `psql` shows all tables; `\d` dumps schema | 1 |

**Exit:** Any team member, on a fresh machine, runs `docker compose up` and has a working stack in < 10 minutes.

---

### Task P0.2 — Data Pipeline & Local Snapshot

| Field | Detail |
|---|---|
| **Owner** | Backend + ML |
| **Duration** | Day 2–5 |
| **Dependency** | P0.1 |
| **Input** | BigQuery credentials, `crypto_bitcoin` public dataset |
| **Output** | `data/btc_snapshot_50k.parquet`, `data/btc_snapshot_10k.parquet`, `data/graph_50k.pt` |
| **Technology** | google-cloud-bigquery, pandas, PyTorch Geometric, pyarrow |

**Graph model (corrected):**

The graph is bipartite, not address-to-address. Addresses and transactions are both nodes.

```
Address node features:
  address_hash (str, hashed), degree, total_sent_btc,
  total_received_btc, tx_count, avg_tx_value_btc,
  time_since_last_tx_hrs, clustering_coefficient (of address projection),
  input_count_total, output_count_total, is_script_hash (bool),
  address_reuse_count, tx_burst_score, weekday_ratio, avg_time_between_tx_hrs

Transaction node features:
  tx_hash (str, hashed), fee_satoshi, input_count, output_count,
  total_input_btc, total_output_btc, hour_of_day, day_of_week,
  is_coinbase (bool)

Edges:
  (address) –[INPUT_TO]--> (transaction)  : value_btc
  (transaction) –[OUTPUT_TO]--> (address) : value_btc

Aggregated projection (for GNN input):
  Collapse bipartite graph → address-address graph
  edge weight = sum of BTC flows between pairs
  (used as GNN input; not claimed to be ground truth wallet ownership)
```

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.2.1 | Document BigQuery schema: `transactions`, `inputs`, `outputs` table joins | `/data/SCHEMA.md` committed | 2 |
| P0.2.2 | `scripts/build_graph.py --n-tx 50000 --output data/` | Script runs without error; produces `.parquet` and `.pt` | 4 |
| P0.2.3 | Add `--offline` flag to use local parquet | `--offline` flag works; output is schema-equivalent to live pull and passes deterministic graph validation (node count, feature shape, no NaN) | 1 |
| P0.2.4 | Create 50k primary snapshot | `data/btc_snapshot_50k.parquet` committed to Git LFS; loads in <2s | 1 |
| P0.2.5 | Create 10k emergency-minimal snapshot | `data/btc_snapshot_10k.parquet` committed; used only if 50k load fails | 0.5 |
| P0.2.6 | Graph validation: log num_nodes, num_edges, connected components, isolated nodes | Validation script passes; no NaN features | 1 |
| P0.2.7 | Feature engineering functions unit-tested; `/data/FEATURE_SPEC.md` committed | `pytest ml/tests/test_features.py` passes. `/data/FEATURE_SPEC.md` exists and contains exact mathematical definitions for all 14 address features and the change-address confidence formula. ML owner and Backend owner must both sign off on this file before implementation begins. | 2 |

**Exit:** `python scripts/build_graph.py --offline --n-tx 50000` produces a valid PyG `Data` object in < 30 seconds.

---

### Task P0.3 — Elliptic Pipeline A (Methodology Validation)

| Field | Detail |
|---|---|
| **Owner** | ML |
| **Duration** | Day 3–7 |
| **Dependency** | P0.1 |
| **Input** | Elliptic dataset (kaggle), 94 node features, edge list, labels (illicit/licit/unknown) |
| **Output** | `ml/checkpoints/gcn_elliptic_v1.pt`, `notebooks/elliptic_eval.ipynb` with filled-in metrics |
| **Technology** | PyTorch Geometric, scikit-learn, matplotlib |

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.3.1 | Download and inspect Elliptic dataset | Class distribution logged: ~2% illicit, ~21% licit, ~77% unknown | 1 |
| P0.3.2 | Fixed temporal train/val/test split | Split uses Elliptic's time-step ordering (earlier time steps = train, later = val/test). Unknown-label nodes participate in message passing but contribute to neither loss nor evaluation metrics. Split node IDs committed to `data/elliptic_split.json` for reproducibility. `data.x.shape == (N, 94)`, no NaN. | 2 |
| P0.3.3 | Majority-class baseline: predict all licit | Precision=N/A, Recall=0.0, F1=0.0, PR-AUC documented | 0.5 |
| P0.3.4 | Heuristic baseline: flag top-k by degree + total_sent | Precision, Recall, F1, PR-AUC, ROC-AUC logged | 1 |
| P0.3.5 | GCN model: 2-layer, hidden=128, dropout=0.5, class-weighted loss (ratio 1:50), Adam lr=1e-3 | Trains 100 epochs without NaN loss | 3 |
| P0.3.6 | GraphSAGE model: same config, inductive | Trains without error | 2 |
| P0.3.7 | Evaluate all models; fill comparison table (see below) | All 4 rows populated with actual numbers from the fixed temporal split. Result is reported honestly regardless of outcome. | 1 |
| P0.3.8 | Calibration of Elliptic GCN outputs (Elliptic only — not applied to Bitcoin pipeline) | (1) Run trained GCN on Elliptic **validation** split (temporal, from `elliptic_split.json`); collect raw logits and true labels. (2) Fit `LogisticRegression(C=1.0, max_iter=1000)` on `(val_logits.reshape(-1,1), val_labels)`. (3) Apply to **test** split logits → calibrated probabilities. (4) Plot reliability diagram; report Brier score before/after. Acceptance: Brier score on test set ≤ uncalibrated baseline. No `CalibratedClassifierCV` or sklearn estimator wrapping of the GCN required. | 1 |
| P0.3.9 | Save best checkpoint | `gcn_elliptic_v1.pt` loads correctly after reload | 0.5 |

**Evaluation comparison table (to be filled):**

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Majority-class baseline | — | 0.00 | 0.00 | — | 0.50 |
| Degree+flow heuristic | ___ | ___ | ___ | ___ | ___ |
| GCN (2-layer, weighted) | ___ | ___ | ___ | ___ | ___ |
| GraphSAGE (inductive) | ___ | ___ | ___ | ___ | ___ |

> **Aspirational targets:** P ≥ 0.70, R ≥ 0.50. If not reached: document result, compare to baselines, analyze failure mode, do not manufacture improvement.
>
> **PR-AUC is the primary metric** for imbalanced classification. ROC-AUC is reported but can be misleadingly optimistic at 2% positive rate.

**Exit:** All 4 models trained successfully. Comparison table fully populated with actual numbers on the fixed temporal split. Results reproducible via `python ml/scripts/eval_elliptic.py --split data/elliptic_split.json`. If GNN does not outperform baselines: the best-performing model (including a baseline) is used for the demo; failure mode is documented; no result is withheld or selectively reported.

---

### Task P0.4 — Bitcoin GCN Pipeline B (Weak-Supervision Setup)

| Field | Detail |
|---|---|
| **Owner** | ML |
| **Duration** | Day 6–8 |
| **Dependency** | P0.2, P0.3 |
| **Input** | `data/graph_50k.pt`, co-spend clustering output, known-bad seed list (from OFAC/public sanctions lists for demo) |
| **Output** | `ml/models/bitcoin_gcn.py`, prototype trained on snapshot |
| **Technology** | PyTorch Geometric |

**Weak supervision strategy:**

```
Labels for Bitcoin GCN training:
  FLAGGED = address appears in known sanctions/OFAC list used for demo
  SUSPICIOUS = co-spend cluster contains ≥1 FLAGGED seed AND
               address is ≤1 hop from seed (NOT automatic cluster elevation)
  UNKNOWN = everything else (excluded from training loss)
```

> ⚠️ **Important framing:** The Bitcoin GCN is a **risk-ranking model**, not a ground-truth classifier. It learns to surface addresses that share structural patterns with known-bad seeds. Its output is a prioritization score for human review. It is NOT equivalent to the Elliptic-validated classifier and should not be presented as such.

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.4.1 | Load known-bad seed list (public demo list, clearly labelled as demo data) | Seed list loaded; 0 real personal data | 1 |
| P0.4.2 | Implement co-spend clustering (Union-Find) with confidence levels | `cospend.cluster(graph)` returns `{address: {cluster_id, confidence: HIGH/MEDIUM/LOW}}` | 3 |
| P0.4.3 | Change-address candidate detection with heuristic scoring | Returns `{address: {candidate_change: bool, confidence: 0.0–1.0}}`, NOT confirmed ownership | 2 |
| P0.4.4 | Weak label generation with **seed-level split** | Seeds partitioned: training seeds (A, B, C) / validation seed (D) / test seed (E). Labels for training nodes come from proximity to training seeds only. Held-out seeds (D, E) never appear in training labels. **"≤1 hop" means exactly 1 edge on the aggregated address-address projection graph — not the bipartite source graph.** Distance is computed via BFS on the projection adjacency list after co-spend aggregation. Class distribution logged per split. | 2 |
| P0.4.5 | Bitcoin GCN (same architecture as Elliptic, 14 features) | Trains without error on snapshot; training seed IDs confirmed absent from validation/test label sets | 2 |
| P0.4.6 | Risk ranking computation: raw GNN output → 0–100 integer ranking | `risk_score = round(gnn_output_logit.sigmoid().item() * 100)`. **No Platt scaling.** No calibration dataset exists for Bitcoin in Round 1. Score stored with label `risk_ranking`. Score is a model-derived ranking, not a calibrated probability. | 1 |
| P0.4.7 | Risk score disclaimer hard-coded in API response | API always returns `score_disclaimer: "Model-derived risk ranking. For prioritization and human review only. Not a calibrated probability."` | 0.5 |
| P0.4.8 | **Seed-level leakage check and holdout evaluation** | (a) Assert: no held-out seed address appears in any training label. (b) Assert: no graph feature directly encodes held-out seed identity (e.g., no `is_training_seed` feature). (c) Evaluate seen-seed recall: fraction of SUSPICIOUS nodes near training seeds that rank in top-K. (d) Evaluate held-out-seed recall: same metric for nodes near held-out seeds. Report both. If held-out recall is substantially lower than seen-seed recall, document the generalization gap honestly. | 2 |
| P0.4.9 | **Bitcoin ranking metrics: Precision@K / Recall@K / NDCG@K** | Using the risk-ranked list from P0.4.6 and the held-out-seed reference set from P0.4.8, compute and report: Precision@5, Precision@10, Precision@20, Recall@10, Recall@20, NDCG@10. **Precision@10 is the primary judge-legible metric** ("of the top 10 addresses flagged, how many match the independently held-out seed-derived reference set?"). **Important framing:** these metrics measure agreement with held-out seed-derived reference labels, not real-world illicit-activity classification performance. Independent investigator-reviewed ground truth is unavailable in Round 1; the evaluation demonstrates ranking consistency, not ground-truth precision. Commit results table to `notebooks/bitcoin_eval.ipynb`. | 1 |

**Exit:** `python ml/scripts/infer_bitcoin.py --offline` produces a risk-ranked list from the snapshot. Leakage check passes (a and b). Seen-seed and held-out-seed recall both reported. Ranking metrics table populated (P@5, P@10, P@20, R@10, R@20, NDCG@10) in `notebooks/bitcoin_eval.ipynb`.

---

### Task P0.5 — API Contract

| Field | Detail |
|---|---|
| **Owner** | Backend + Frontend (joint sign-off required) |
| **Duration** | Day 5–6 |
| **Dependency** | P0.1 |
| **Input** | Team agreement on data model |
| **Output** | `/backend/openapi.yaml` committed, frontend mock server running against it |
| **Technology** | FastAPI (auto-generates OpenAPI), Prism mock server for frontend |

**Complete API contract:**

```yaml
# GET /api/v1/health
response:
  status: "ok"
  model_version: string
  last_updated: ISO8601

# GET /api/v1/model/info          ← NEW (from correction #12)
response:
  model_version: "gcn_btc_v1"
  training_dataset: "Weak supervision on Bitcoin snapshot (co-spend seeds)"
  validation_dataset: "Elliptic (methodology validation only)"
  inference_dataset: "Bitcoin snapshot 50k txns, [date range]"
  model_status: "demo"
  score_disclaimer: "Model-derived risk ranking. For prioritization and human review only."
  last_updated: ISO8601

# GET /api/v1/alerts
params:
  min_risk: int (default 70)
  limit: int (default 20)
response:
  alerts: list[WalletAlert]
  total: int
  score_disclaimer: string

# WalletAlert schema:
  address: string (hashed)
  risk_score: int (0–100)
  risk_label: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"

# RISK_LABEL_THRESHOLDS — defined here, enforced in backend/constants.py and frontend/src/constants.ts
# Backend and frontend MUST import from these constants, never hard-code inline.
#
#   0 –  24  →  LOW       (informational; monitor)
#  25 –  49  →  MEDIUM    (elevated; review when capacity allows)
#  50 –  74  →  HIGH      (prioritised; review within 24h)
#  75 – 100  →  CRITICAL  (urgent; review immediately)
#
# Rationale: These thresholds are presentation categories for prioritization
# consistency across the API and UI. They are not probability thresholds,
# do not represent estimated illicit-activity prevalence, and do not
# constitute an operational or enforcement decision rule. The sigmoid score
# distribution is not necessarily uniform, so score bands do not map
# directly to address-count percentiles. Final operational thresholds,
# if required, must be selected using investigator-reviewed agency data
# in Phase 2.
  top_reason: string (primary deterministic reason)
  reasons: list[string] (max 3)
  cluster_id: string | null
  cluster_confidence: "HIGH" | "MEDIUM" | "LOW" | null

# GET /api/v1/wallets/{address_hash}
response:
  address: string
  risk_score: int
  risk_label: string
  reasons: list[string]
  score_disclaimer: string
  hops_to_nearest_flagged: int | null
  tx_burst_score: float
  timing_anomaly_score: float
  cluster_id: string | null
  cluster_confidence: string | null
  candidate_change_address: bool
  p2p_signals: list[P2PSignal] | null

# P2PSignal schema:
  signal_source: "SIMULATED"           ← always present
  ip_cluster_id: string                ← e.g. "IP_CLUSTER_07"
  timestamp: ISO8601
  tx_hash: string
  disclaimer: "Simulated/replayed demonstration data only."

# GET /api/v1/graph
params:
  address: string
  depth: int (1 or 2)
response:
  nodes: list[{id, risk_score, risk_label, node_type: "address"|"transaction"}]
  edges: list[{source, target, edge_type: "INPUT_TO"|"OUTPUT_TO", value_btc, timestamp}]
  disclaimer: string
```

**Exit:** OpenAPI spec committed. Frontend can run `npx @stoplight/prism-cli mock openapi.yaml` and get mock responses matching the schema.

---

### Task P0.6 — Explainability System (Two-Layer)

| Field | Detail |
|---|---|
| **Owner** | ML |
| **Duration** | Day 8–9 |
| **Dependency** | P0.4 |
| **Input** | Trained Bitcoin GCN, graph with features |
| **Output** | `ml/explain/explainer.py` with both layers |
| **Technology** | PyTorch Geometric GNNExplainer, custom deterministic feature scorer |

**Two-layer design:**

```
Layer 1 (PRIMARY — always runs, < 100ms per address):
  Deterministic graph features → reason strings
  Rules:
    hops_to_nearest_flagged ≤ 2           → "Within 2 hops of a flagged address"
    tx_burst_score > threshold             → "Unusual transaction volume burst (off-hours)"
    address_reuse_count > threshold        → "High address reuse pattern"
    cluster_confidence == HIGH and cluster
      contains a flagged member            → "Linked to cluster with flagged member"
    candidate_change_address == True       → "Candidate change-address relationship detected"
    avg_time_between_tx_hrs < threshold    → "Rapid consecutive transactions"

Layer 2 (OPTIONAL — runs only on top-5 flagged addresses, may take 2–5s each):
  GNNExplainer → top-3 node features + top-3 edge importances
  Output appended to Layer 1 reasons if time budget allows
  If GNNExplainer is slow or fails: demo still works via Layer 1
```

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.6.1 | `ml/explain/deterministic.py`: all 6 rules implemented | Unit tests for each rule | 2 |
| P0.6.2 | `ml/explain/gnnexplainer.py`: wrapper around PyG GNNExplainer | Runs on 1 node in < 5s on CPU | 2 |
| P0.6.3 | `ml/explain/formatter.py`: merges both layers into reason strings | Output: list of ≤3 strings per address | 1 |
| P0.6.4 | Fallback: if GNNExplainer unavailable, Layer 1 only | `--no-gnnexplainer` flag works | 0.5 |

**Exit:** `explainer.explain(address)` returns 1–3 human-readable strings in < 200ms for Layer 1 alone.

---

### Task P0.7 — P2P Simulation Data

| Field | Detail |
|---|---|
| **Owner** | Domain/Backend |
| **Duration** | Day 9–10 |
| **Dependency** | P0.2 |
| **Input** | Address list from snapshot |
| **Output** | `data/p2p_signals_simulated.json` |
| **Technology** | Python random/numpy for synthetic generation |

**Data schema (corrected — no IP-looking strings):**

```json
[
  {
    "signal_source": "SIMULATED",
    "ip_cluster_id": "IP_CLUSTER_07",
    "timestamp": "2023-10-14T02:14:33Z",
    "address_hash": "addr_abc123...",
    "tx_hash": "tx_def456...",
    "peer_count": 12,
    "disclaimer": "Simulated/replayed demonstration data only. Not real network observations."
  }
]
```

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.7.1 | Generate synthetic P2P signals for 20 snapshot addresses | File has 20 entries, all with `signal_source: SIMULATED` | 1 |
| P0.7.2 | Join with graph: annotate nodes | `p2p_signals` field populated for relevant addresses | 1 |
| P0.7.3 | Verify: no `192.168.*`, no `10.*`, no real IP strings anywhere in the file | `grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' data/p2p_signals_simulated.json` returns empty | 0.5 |

**Exit:** File exists, passes IP-string check, joins correctly with graph nodes.

---

### Task P0.8 — Dry Run Demo

| Field | Detail |
|---|---|
| **Owner** | All |
| **Duration** | Day 12–13 (2 days before event) |
| **Dependency** | P0.1–P0.7 |

**Subtasks:**

| # | Subtask | Acceptance Test | Hours |
|---|---|---|---|
| P0.8.1 | Full pipeline run (offline mode) | End-to-end: snapshot → graph → GCN → scores → API → dashboard | 2 |
| P0.8.2 | Time the demo | 5-minute demo script completed within 5 minutes | 1 |
| P0.8.3 | Q&A dry run: 10 hardest questions | All 4 team members answer without notes | 2 |
| P0.8.4 | Demo script committed | `/docs/demo_script.md` with click-by-click walkthrough | 1 |
| P0.8.5 | Confirm P2P panel shows "SIMULATED DATA" prominently | Judge cannot mistake simulated signals for real | 0.5 |

**Phase 0 Exit Criteria (must ALL be green before leaving for the event):**

- [ ] `docker compose up` works on a clean machine
- [ ] `--offline` mode runs end-to-end in < 2 minutes
- [ ] Elliptic comparison table populated with actual numbers
- [ ] Bitcoin GCN produces a ranked list from snapshot
- [ ] API contract committed; frontend runs against mock
- [ ] No IP-looking strings in P2P simulation data
- [ ] Demo script tested and timed
- [ ] Every team member can answer "why is P2P simulated?" without notes

---

## Phase 1 — Hackathon Build (Round 1, 36 Hours)

> **Scope rule:** Build exactly what is committed on the Feasibility slide. Hour 28–32 is demo rehearsal. Hour 32–36 is bug-fix only. No new features after Hour 18.

---

### Hour-by-Hour Schedule

| Hours | ML Lane | Backend Lane | Frontend Lane | Domain Lane |
|---|---|---|---|---|
| **0–2** | Load snapshot via `--offline` flag, validate graph object | Stand up FastAPI; wire `GET /health`, `GET /model/info` | Scaffold React + Vite; implement `AlertList` skeleton against mock API | Prep Q&A brief; review evaluation criteria sheet |
| **2–6** | Run co-spend clustering; persist cluster assignments to DB | Implement `GET /alerts` (mock data); PostgreSQL migrations | Implement risk-score badge; `WalletDetail` skeleton | Draft 2-minute verbal pitch |
| **6–10** | Load Bitcoin GCN checkpoint; run inference on all nodes; compute risk scores; persist to DB | Implement `GET /wallets/{address}` with live data | Connect `AlertList` to live API; implement `WalletDetail` with reasons | Review 4 core papers for Q&A |
| **10–14** | Run Layer 1 explainability (deterministic); persist reason strings | Implement `GET /graph` endpoint | Implement `GraphExplorer` (Cytoscape.js force-directed); 1-hop expansion | Test dashboard as mock investigator; write demo narrative |
| **14–18** | **🔴 HOUR 18 MVP FREEZE** — verify all 5 items below are working | → | → | → |
| **18–22** | Run GNNExplainer on top-5 (Layer 2); if slow, skip and proceed with Layer 1 | Add P2P signal overlay to `GET /wallets/{address}` | Implement P2P signal panel (SIMULATED badge prominent) | Write plain-language demo walkthrough for 3 demo wallets |
| **22–26** | Re-evaluate: update Elliptic comparison table if anything changed | Integration test: all endpoints return correct data types | Polish: loading states, error handling, `SearchBar` | Practise demo delivery; time it |
| **26–28** | Document precision/recall numbers in notebook | Final end-to-end integration check | Final UI polish | Prepare Q&A answers for top-10 expected questions |
| **28–32** | **🔴 DEMO REHEARSAL** — Full pipeline, all lanes, timed run | → | → | → |
| **32–36** | **Bug fixes only. No new features. No new slides.** | → | → | → |

---

### Hour 18 MVP Freeze Checkpoint

At exactly Hour 18, the team lead checks all 5 items. If any are red, **immediately switch to the fallback**:

| Item | Green | Fallback if Red |
|---|---|---|
| Data loaded from local snapshot | Graph object valid | Use 10k emergency snapshot |
| Graph created with address + tx nodes | num_nodes > 0, no NaN | Use pre-built `graph_50k.pt` from Phase 0 |
| GCN inference running | At least 1 risk score computed | Use cluster-proximity heuristic scoring only |
| API responding at `GET /health` | HTTP 200 | Run FastAPI directly without Docker |
| At least 1 flagged wallet visible in UI | Badge rendered in browser | Use static JSON fixture for demo |

> **Fallback scoring:** If GCN doesn't produce scores, use: `risk_score = min(100, round(50 + (hops_to_flagged==1)*30 + (tx_burst_score>2)*15 + (address_reuse>5)*5))`. This is a deterministic heuristic. State it honestly in the demo: "We've fallen back to heuristic scoring while the model is being investigated."

---

### Phase 1 Build Steps (7 Phases Within Hackathon)

#### Hackathon Phase 1 — Data Foundation (Hours 0–6)

**Objective:** Reliable Bitcoin graph from snapshot.

| Task | Acceptance Test |
|---|---|
| Load `btc_snapshot_50k.parquet` via `DataLoader` | `len(graph.x) > 0`, no NaN |
| Build bipartite graph: `Address –INPUT_TO→ Tx –OUTPUT_TO→ Address` | Both node types present in graph |
| Compute aggregated address-address projection | Projection has fewer nodes than bipartite graph |
| Compute all 14 address features | Feature tensor shape matches |
| Validate: log node/edge counts, isolated nodes, feature stats | Validation script exits 0 |

---

#### Hackathon Phase 2 — Detection Engine (Hours 4–14, parallel with Phase 1)

**Objective:** Identify and rank suspicious activity.

| Task | Acceptance Test |
|---|---|
| Co-spend clustering (Union-Find) with confidence levels | `{address: {cluster_id, confidence}}` for all addresses |
| Change-address candidate detection | Returns `candidate_change_address` bool + confidence float |
| Bitcoin GCN inference from checkpoint | `risk_score` (0–100) for all nodes; computed from raw GNN sigmoid output, not Platt scaling |
| Risk rankings persisted to `node_scores` table | `SELECT * FROM node_scores LIMIT 5` returns rows; `score_type` column = `"raw_ranking"` |
| `score_disclaimer` hardcoded in every score-bearing response | `grep score_disclaimer` finds it in API handlers; disclaimer reads "not a calibrated probability" |

---

#### Hackathon Phase 3 — Explainability (Hours 10–20)

**Objective:** Explain why an address was prioritized.

| Task | Acceptance Test |
|---|---|
| Layer 1: all 6 deterministic rules fire correctly on test addresses | Unit tests pass |
| Layer 2: GNNExplainer on top-5 addresses | Completes in < 30s total; if not, falls back |
| Formatter produces 1–3 plain-language strings per address | All top-20 addresses have ≥1 reason string |
| Reasons stored in `node_reasons` table | `SELECT address, reasons FROM node_reasons LIMIT 5` returns rows |

---

#### Hackathon Phase 4 — P2P Fusion Demo (Hours 16–22)

**Objective:** Demonstrate the second data layer, explicitly and honestly.

| Task | Acceptance Test |
|---|---|
| Load `data/p2p_signals_simulated.json` | 20 records loaded |
| Join signals to addresses via `address_hash` | 20 addresses have `p2p_signals` in API response |
| API returns `signal_source: "SIMULATED"` always | Assert in integration test |
| No IP-looking strings in any API response | `grep -E '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'` on response fixtures returns empty |

---

#### Hackathon Phase 5 — Investigation Dashboard (Hours 0–26, incremental)

**Objective:** One usable investigator interface.

| Component | Available by Hour | Acceptance Test |
|---|---|---|
| `AlertList` (skeleton, mock data) | Hour 2 | Renders in browser |
| Risk-score badge (color: LOW=green, MEDIUM=yellow, HIGH=orange, CRITICAL=red) | Hour 6 | All 4 colors render |
| `WalletDetail` with reason bullets | Hour 10 | 3 demo addresses show reasons |
| `GraphExplorer` (Cytoscape.js) | Hour 14 | Force-directed graph renders; node click expands 1 hop |
| P2P signal panel with "⚠ SIMULATED DATA" banner | Hour 22 | Banner visible without scrolling |
| `SearchBar` | Hour 24 | Address lookup returns result |

**P2P Signal Panel spec (corrected — no IP-looking data):**

```
┌──────────────────────────────────────────┐
│  ⚠  NETWORK SIGNAL — SIMULATED DATA  ⚠  │
│  Demonstration only. Not real evidence.  │
├──────────────────────────────────────────┤
│  IP Cluster:   IP_CLUSTER_07             │
│  Timestamp:    2023-10-14 02:14 UTC      │
│  Transaction:  tx_abc...def              │
│  Peer count:   12                        │
└──────────────────────────────────────────┘
```

---

#### Hackathon Phase 6 — End-to-End Integration (Hours 22–28)

**Objective:** Every component works together without manual intervention.

| Task | Acceptance Test |
|---|---|
| Full pipeline: `--offline` → graph → GCN → scores → reasons → API → dashboard | Completes without error |
| All 5 API endpoints return correct schemas | OpenAPI validation passes |
| Dashboard renders 3 demo wallets with scores, reasons, graph, P2P panel | Visual check |
| `GET /model/info` shows correct training dataset disclaimer | Response inspected |

---

#### Hackathon Phase 7 — Validation + Demo (Hours 28–36)

**Objective:** Prove the system; don't merely show it.

| Task | Acceptance Test |
|---|---|
| Elliptic comparison table visible in demo notebook | All 4 rows populated |
| Demo runs in 5 minutes | Timed at Hour 28 rehearsal |
| Demo runs again at Hour 32 (after any bug fixes) | Timed again |
| Every team member can answer "is this transfer learning?" correctly | Verbal check |

**Correct answer to "is this transfer learning?" for judges:**
> "The Elliptic dataset validates that a GNN can detect illicit transaction patterns with the precision and recall shown in our comparison table. Our Bitcoin pipeline uses its own independently engineered feature space and weak supervision from co-spend heuristics. Cross-dataset transfer learning with representation alignment is our Round 2 research objective."

---

### Phase 1 Exit Criteria

- [ ] `GET /api/v1/alerts` returns ≥ 3 addresses with risk scores and reasons
- [ ] Elliptic comparison table shows all 4 models (majority, heuristic, GCN, GraphSAGE)
- [ ] Elliptic and Bitcoin metrics shown as **separate panels** — never merged:
  ```
  ELLIPTIC — supervised methodology validation
  Precision | Recall | F1 | PR-AUC | ROC-AUC

  BITCOIN — weakly supervised prioritization
  Precision@10 | Recall@10 | NDCG@10
  (Agreement with held-out seed-derived reference set)
  ```
- [ ] P2P panel shows `IP_CLUSTER_XX` labels, not IP-like strings
- [ ] P2P panel has "SIMULATED DATA" banner visible without scrolling
- [ ] `GET /api/v1/model/info` returns correct training dataset disclaimer
- [ ] Graph visualization renders with bipartite node types distinguished
- [ ] Score disclaimer present in every alert and wallet response
- [ ] Demo runs end-to-end at Hour 30 rehearsal without manual intervention

---

## Phase 2 — Agency Pilot (3–6 Months Post-Selection)

---

### Task P2.1 — Agency Onboarding

| Field | Detail |
|---|---|
| **Owner** | Domain (agency liaison) |
| **Duration** | Month 1–2 |
| **Dependency** | Round 1 selection |
| **Input** | Round 1 demo, agency contacts |
| **Output** | Signed MOU + data-sharing agreement |

| Subtask | Acceptance Test | Weeks |
|---|---|---|
| Identify 3 target agencies (FIU-IND, ED, state cyber cell); rank by likelihood | Shortlist document | 1 |
| Initiate formal engagement with top-ranked agency | Meeting scheduled | 2 |
| Agency legal counsel determines applicable authority for data sharing | Written legal opinion from agency counsel (not student team) | 4 |
| Data-sharing agreement governing STR data for model retraining | Signed DSA | 4 |
| MOU signed | MOU on file | 2 |
| Dedicated liaison named (one team member, all agency comms go through them) | Named in MOU | 1 |

**Gating rule:** Phase 2.2 and beyond do NOT start until DSA is signed. If agency engagement fails, document this as a real outcome and reassess.

---

### Task P2.2 — Real-Data Retraining

| Field | Detail |
|---|---|
| **Owner** | ML |
| **Duration** | Month 2–3 |
| **Dependency** | P2.1 (DSA signed) |
| **Input** | Agency STR data, Phase 1 Bitcoin GCN checkpoint |
| **Output** | `ml/checkpoints/gcn_agency_v1.pt`, false-positive rate on real cases |

| Subtask | Acceptance Test | Weeks |
|---|---|---|
| Ingest agency STR data | Pipeline runs on agency data (no raw data leaves agency environment) | 2 |
| Expand feature set to include agency-specific features | Feature set documented; agency approves | 2 |
| Fine-tune Bitcoin GCN on agency-labeled cases (small lr=1e-4, freeze early layers) | Fine-tuned checkpoint saved | 3 |
| Integrate Wu et al. mixer/CoinJoin detector | Mixed txns flagged as `LOW_CONFIDENCE`; not mis-clustered | 2 |
| Evaluate: false-positive rate reviewed by actual investigator | FP rate documented per STR category | 2 |

---

### Task P2.3 — Human-in-the-Loop Workflow

| Field | Detail |
|---|---|
| **Owner** | Backend + Frontend |
| **Duration** | Month 2–4 |
| **Dependency** | P2.1 |

| Subtask | Acceptance Test | Weeks |
|---|---|---|
| Investigator review UI: confirm/reject/escalate a flag | Feature deployed in staging | 3 |
| Derivation chain log: model version, features, timestamp, analyst ID | Audit trail queryable | 2 |
| Feedback loop: confirmed/rejected flags fed back to retraining dataset | Pipeline documented and tested | 2 |
| Role-based access: analyst, supervisor, admin | Roles enforced in API middleware | 2 |

---

### Task P2.4 — Cross-Dataset Transfer Learning Research (Round 2 ML Task)

| Field | Detail |
|---|---|
| **Owner** | ML |
| **Duration** | Month 3–5 |
| **Dependency** | P2.2 (agency-labeled cases available as target) |

| Subtask | Acceptance Test | Weeks |
|---|---|---|
| Design feature/representation projection layer (MLP aligning Elliptic embedding → Bitcoin embedding) | Architecture document reviewed by team | 2 |
| Train projection layer on paired examples | Loss converges | 3 |
| Fine-tune on agency data | Checkpoint saved | 2 |
| Ablation: compare transfer-learned model vs. Bitcoin-from-scratch model | If transfer helps by > 5% PR-AUC, adopt; otherwise document failure and continue with Bitcoin-from-scratch | 2 |

---

### Task P2.5 — Neo4j Migration

| Field | Detail |
|---|---|
| **Owner** | Backend |
| **Duration** | Month 3–4 |
| **Dependency** | P2.2 (understand actual query patterns from real usage) |

| Subtask | Acceptance Test | Weeks |
|---|---|---|
| Query analysis: log top-10 graph traversal queries from Phase 1 usage | Query log report | 1 |
| Decision: if traversal queries > 30% of all queries, migrate to Neo4j; else stay on PostgreSQL | Decision documented with data | 1 |
| If migrating: design property graph model `(Address)-[:INPUT_TO]->(Tx)-[:OUTPUT_TO]->(Address)` | Schema committed | 2 |
| ETL from PostgreSQL → Neo4j; validate integrity | Row counts match | 2 |
| Replace SQL traversals with Cypher; query parity tests | All parity tests pass | 2 |

---

### Task P2.6 — Network-Layer Capability (Legal Gate)

> [!CAUTION]
> **Do not begin this task until written legal authorization is received from the sponsoring agency's legal counsel.** The student team does not determine which legal instrument applies — that is the agency's responsibility. The correct engineering gate is: written authorization exists, in hand.

| Subtask | Acceptance Test | Weeks |
|---|---|---|
| Agency counsel provides written authorization for network monitoring capability | Document on file | — |
| Deploy probe Bitcoin Core node(s) | Node operational, peering confirmed | 3 |
| Implement IP-to-transaction correlation (NTSSL, Zhang et al. 2026) | Correlation pipeline runs | 3 |
| Evaluate: ~50% precision, ~56.8% recall expected (NTSSL baseline) | Numbers documented with explanation of probabilistic nature | 2 |
| Communicate realistic limits to agency: ~25–40% of targets yield no usable signal | Expectations documented in writing | 1 |

---

### Phase 2 Exit Criteria

- [ ] Model run against real agency data (not Elliptic, not snapshot only)
- [ ] False-positive rate reviewed by at least one actual investigator
- [ ] Derivation chain logging implemented and audit-verified
- [ ] Legal basis for any network-layer work is owned by the agency, in writing
- [ ] Signed DSA and MOU on file
- [ ] Transfer learning ablation completed (adopt if beneficial; discard if not)

---

## Phase 3 — Production Hardening (Adaptive Parallel Tracks)

> Phase 3 structure is adaptive: it branches into three parallel tracks based on Phase 2 outcomes. The team leads prioritize tracks by Phase 2 findings.

```
                 Phase 2 Results
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ML Scale     Operations    Multi-chain
```

---

### Track A — ML Scale

| Task | Detail | Dependency |
|---|---|---|
| GraphSAGE full pipeline | Replace GCN baseline with inductive GraphSAGE; enables inference on unseen nodes without graph rebuild | Phase 2 model |
| Hyperbolic GNN evaluation | Evaluate Ghimire et al. (2026) hyperbolic embedding for Bitcoin's power-law structure; adopt if > 5% PR-AUC gain | GraphSAGE baseline |
| Automated retraining pipeline | Weekly or STR-volume-triggered; automated evaluation + canary deployment; rollback if metrics drop | Operations Track |
| Synthetic augmentation | Generate synthetic examples for underrepresented patterns (flash laundering, bridge protocols) | Agency clearance |

---

### Track B — Operations

| Task | Detail | Dependency |
|---|---|---|
| Real-time ingestion | Stream new Bitcoin transactions via WebSocket from local or partner node; replace BigQuery for production | Phase 2 completion |
| Incremental graph updates | Add new nodes/edges without full graph rebuild; amortized update cost < 100ms per new transaction | Real-time ingestion |
| High availability | Load balancer, PostgreSQL/Neo4j replication, 99.5% uptime SLA | Ops infrastructure |
| Disaster recovery | Backup + restore procedures; RTO < 4 hours; tested quarterly | HA |
| Performance targets | API < 200ms for wallet detail; graph traversal < 500ms for 2-hop | HA |
| Audit & compliance | Full derivation-chain logging; MFA; IP allowlisting; data-retention rules | Agency policy |
| Compliance assessment | Conduct CERT-In / applicable cybersecurity compliance assessment with the sponsoring agency; implement resulting obligations (do not self-prescribe registration requirements) | Agency legal |
| External audit | Commission third-party security and compliance audit before production deployment | All above |

---

### Track C — Multi-Chain Extension

> Bitcoin is ~6% of FIU-IND crypto STR caseload. Tether is ~76%. This track is the highest-impact long-term deliverable.
>
> ⚠️ **Source verification required before external use:** These figures must be traced to a specific FIU-IND annual report, FATF mutual evaluation, or comparable primary source with the reporting period cited before being used in judge-facing slides or agency presentations. Do not present as unsupported assertions.

| Task | Detail | Dependency |
|---|---|---|
| Tether (TRC-20, ERC-20) graph ingestion | Extend graph construction to Tron + Ethereum chains | Track A GraphSAGE |
| VASP entity linking | Link addresses to known VASP identifiers (exchange registration data) | Agency data sharing |
| Cross-chain bridge detection | Detect assets moving between chains via bridge protocols | Multi-chain ingestion |
| Unified risk dashboard | Single investigator view: Bitcoin + stablecoin flagged wallets | All above |

---

### Phase 3 Cost Validation (Real Numbers Required)

Replace the illustrative estimate with actual measured figures:

| Metric | Measure | Target |
|---|---|---|
| Infrastructure cost | Actual cloud/on-prem cost per month | Document as-is |
| License cost avoided | Price of Chainalysis/TRM Labs equivalent tier for actual query volume | Document as-is |
| Investigator time per case | Before/after hours-per-case measured with pilot agency | 50% reduction goal |
| Net cost avoidance | (License avoided) − (Infrastructure) − (Maintenance engineering) | Positive |

---

## Team Roles (Fill Before Phase 0 Starts)

| Lane | Primary Owner | Backup | Decisions they can make unilaterally |
|---|---|---|---|
| **ML/Modeling** | _____________ | _____________ | Model architecture, feature selection, training config, evaluation methodology |
| **Backend/Data** | _____________ | _____________ | API design, database schema, Docker config, data pipeline |
| **Frontend/UX** | _____________ | _____________ | Component library, UI layout, Cytoscape.js config |
| **Domain/Q&A** | _____________ | _____________ | Demo narrative, Q&A answers, agency liaison, legal framing |

**Coordination rules:**
- Cross-lane decisions (e.g., API contract, graph model definition) require both owners to sign off in writing (Slack thread or GitHub comment is sufficient)
- No team member adds a feature after Hour 18 of Round 1 without the team lead's explicit approval
- Fallback decisions (using heuristic scoring, using 10k snapshot) are made by the team lead, not by the lane owner alone

---

## Risk Register (Updated)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| BigQuery unavailable on-site | Medium | High | 50k local snapshot; 10k emergency snapshot; `--offline` flag tested in Phase 0 |
| Bitcoin GCN doesn't produce useful scores | Medium | High | Hour 18 freeze; deterministic heuristic fallback documented and tested |
| Dashboard-API integration breaks | Medium | High | API contract pre-agreed; mock-first frontend development; integration test at Hour 22 |
| GNNExplainer too slow for demo | Medium | Medium | Layer 1 (deterministic) is always primary; GNNExplainer is optional Layer 2 |
| Judge asks if P2P signal is real | High | Medium | Answer scripted; "SIMULATED DATA" banner unmissable in UI |
| Judge asks about transfer learning | High | Medium | Answer scripted (see Phase 7); two-pipeline architecture is the defense |
| Class imbalance tanks Elliptic recall | High | Medium | Class-weighted loss (1:50); PR-AUC tracked; failure documented honestly if targets missed |
| Agency partner unavailable for Phase 2 | Medium | High | 3 agencies identified in Phase 0; don't depend on one |
| Transfer learning doesn't help in Phase 2 | Medium | Low | Ablation is the test; Bitcoin-from-scratch remains the fallback |
| BIP 324 limits P2P signal quality | Certain (already known) | Medium | Out of Round 1 scope; realistic limits set for Phase 2+ (50% precision, 57% recall baseline) |

---

## Technology Stack (Finalized)

| Component | Technology | Version | Phase Added |
|---|---|---|---|
| ML | PyTorch + PyTorch Geometric | 2.3 / 2.5 | Phase 0 |
| Graph analytics | NetworkX | 3.3 | Phase 0 |
| Classical ML | scikit-learn | 1.5 | Phase 0 |
| Backend | FastAPI | 0.111 | Phase 0 |
| Primary DB | PostgreSQL | 16 | Phase 0 |
| Vector similarity | pgvector | — | **Phase 2+ only, if needed** |
| Graph DB | Neo4j | 5.x | **Phase 2+, if query analysis justifies it** |
| Data warehouse | BigQuery / local parquet | — | Phase 0 (parquet primary for Round 1) |
| Frontend | React 18 + Vite | 18 / 5.x | Phase 0 |
| Graph viz | Cytoscape.js | 3.x | Phase 0 |
| Charts | Recharts | 2.x | Phase 0 |
| Containers | Docker Compose | Latest | Phase 0 |
| CI | GitHub Actions | Latest | Phase 0 |

---

## Summary Timeline

```
Days -14 to 0:   Phase 0 — Preparation (14 tasks, all with exit criteria)
                 P0.1 Env + repo (Day 1–2)
                 P0.2 Data pipeline + 50k + 10k snapshots (Day 2–5)
                 P0.3 Elliptic Pipeline A + comparison table (Day 3–7)
                 P0.4 Bitcoin Pipeline B + weak supervision (Day 6–8)
                 P0.5 API contract + OpenAPI spec (Day 5–6)
                 P0.6 Two-layer explainability (Day 8–9)
                 P0.7 P2P simulation (IP_CLUSTER_XX) (Day 9–10)
                 P0.8 Dry run demo (Day 12–13)

Hours 0–36:      Phase 1 — Hackathon MVP
                 Hour 18: MVP Freeze checkpoint
                 Hour 28: Demo rehearsal
                 Hour 32: Bug-fix only

Months 1–6:      Phase 2 — Agency Pilot
                 P2.1 Agency onboarding + DSA (Month 1–2)
                 P2.2 Real-data retraining (Month 2–3)
                 P2.3 HITL workflow (Month 2–4)
                 P2.4 Transfer learning ablation (Month 3–5)
                 P2.5 Neo4j migration (if justified) (Month 3–4)
                 P2.6 Network-layer (Month 4–6, legal gate)

Months 6–24:     Phase 3 — Production (Adaptive Parallel Tracks)
                 Track A: ML Scale (GraphSAGE, hyperbolic, auto-retrain)
                 Track B: Operations (real-time, HA, compliance)
                 Track C: Multi-chain (Tether TRC-20/ERC-20, VASP linking)
```
