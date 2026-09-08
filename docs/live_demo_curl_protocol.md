# Live Demo — `curl` Test Protocol

> **Purpose**: Defeat the "is this just a mockup?" question before it's asked.  
> Run these commands in a visible terminal, in order, during the live evaluation. Every command
> hits the locally-running Docker stack — no external internet dependency required.

---

## Pre-Demo Checklist

```bash
# 1. Confirm stack is running
docker compose ps

# 2. Confirm API health and model version
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool

# 3. Confirm frontend is reachable
curl -s -o /dev/null -w "Frontend HTTP %{http_code}\n" http://localhost:5174/
```

Expected health response:
```json
{
    "status": "ok",
    "model_version": "gcn_btc_v1",
    "offline_mode": true
}
```

---

## Test 1 — CRITICAL Burst Address (Case A)

**What it proves**: Full risk profile is computed, and `score_disclaimer` is strictly enforced in payload.

```bash
curl -s http://localhost:8000/api/v1/wallets/17ebdd724dbfe21e | python3 -m json.tool
```

**Key Signals to Highlight**:
- `"risk_score": 98` and `"risk_label": "CRITICAL"`
- `"score_disclaimer"` field — present in every single response payload
- `"reasons"` array: human-readable evidentiary signals, not raw unexplainable floats
- `"tx_burst_score"` and `"timing_anomaly_score"`: features driving the score

**Terminal Assertion Check**:
```bash
curl -s http://localhost:8000/api/v1/wallets/17ebdd724dbfe21e \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert 'score_disclaimer' in d, 'FAIL: score_disclaimer missing'
assert d['risk_score'] >= 80,   'FAIL: expected CRITICAL score'
assert len(d['reasons']) > 0,   'FAIL: reasons empty'
print(f'PASS  risk_score={d[\"risk_score\"]}  label={d[\"risk_label\"]}  disclaimer_present=True')
"
```

---

## Test 2 — HIGH Co-Spend Address (Case B)

**What it proves**: Multi-tier risk grading and entity linkage via co-spend graph clustering.

```bash
curl -s http://localhost:8000/api/v1/wallets/761d189d93e7bac2 \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Address    :', d['address'])
print('Risk Score :', d['risk_score'])
print('Risk Label :', d['risk_label'])
print('Cluster ID :', d.get('cluster_id'))
print('Score Type :', d.get('score_type'))
print('Disclaimer :', d['score_disclaimer'][:65] + '...')
"
```

---

## Test 3 — P2P Simulation Node (Case C)

**What it proves**: P2P network telemetry panel integration with mandatory `SIMULATED DATA` disclosure banner.

```bash
curl -s http://localhost:8000/api/v1/wallets/558e45a2f878bd09 \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
p2p = d.get('p2p_signals') or []
print('Risk Score  :', d['risk_score'])
print('P2P Signals :', len(p2p), 'signals detected')
if p2p:
    print('First Signal Disclaimer:', p2p[0].get('disclaimer'))
"
```

---

## Test 4 — API Contract Enforcement: 422 on Invalid Input

**What it proves**: Strict schema validation at the gateway level — will not accept arbitrary or unvalidated inputs.

```bash
# Must return HTTP 422 (Unprocessable Entity)
curl -s -w "\nHTTP Status: %{http_code}\n" \
  "http://localhost:8000/api/v1/wallets/risk-list?min_risk=999"
```

```bash
# View structured validation error response
curl -s "http://localhost:8000/api/v1/wallets/risk-list?min_risk=999" | python3 -m json.tool
```

---

## Test 5 — Dynamic Risk List Query

**What it proves**: Dynamic filtering and ranked retrieval across snapshot data.

```bash
# Query top 5 high-risk addresses (risk >= 70)
curl -s "http://localhost:8000/api/v1/wallets/risk-list?min_risk=70&limit=5" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Total Qualified Addresses: {d[\"total\"]}')
for idx, a in enumerate(d['addresses'], 1):
    print(f' {idx}. {a[\"address\"]} | Score: {a[\"risk_score\"]} | Tier: {a[\"risk_label\"]}')
"
```

---

## Test 6 — Live Ego-Network Traversal

**What it proves**: Live sub-graph extraction around the focal address.

```bash
curl -s "http://localhost:8000/api/v1/graph?address=17ebdd724dbfe21e&depth=1" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
nodes, edges = d.get('nodes', []), d.get('edges', [])
print(f'Graph Traversal: {len(nodes)} Nodes, {len(edges)} Edges')
focal = next((n for n in nodes if n['id'] == '17ebdd724dbfe21e'), None)
print(f'Focal Node Score in Subgraph: {focal[\"risk_score\"] if focal else \"Not Found\"}')
"
```

---

## Test 7 — FIU-IND Investigation Report Export

**What it proves**: Complete audit-ready payload export format.

```bash
curl -s http://localhost:8000/api/v1/wallets/17ebdd724dbfe21e/report \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Report Generated Keys:', list(d.keys()))
print('Mandatory Disclaimer Present:', 'score_disclaimer' in d)
"
```

---

## Judge Defense Cheat Sheet

| Question / Skepticism | Direct Evidence Command | What Screen Shows |
|---|---|---|
| *"Is this just static mock JSON?"* | Run Test 4 (`min_risk=999`) | Pydantic 422 runtime validation schema error |
| *"Are risk scores arbitrary?"* | Run Test 1 & check `reasons` | Explainability breakdown (burst, intervals, degree) |
| *"Can you trace related nodes?"* | Run Test 2 (`cluster_id`) | Shared co-spend cluster identification |
| *"Where did P2P IP come from?"* | Run Test 3 (`disclaimer`) | Clear `SIMULATED DATA` disclaimer displayed |
| *"Is the graph static pre-rendered SVG?"* | Run Test 6 (`/api/v1/graph`) | Dynamic BFS ego-graph expansion JSON |
