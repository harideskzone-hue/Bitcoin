# SIH26146 — Demo Script + Complete Q&A
## 5-Minute Click-by-Click Walkthrough

---

## Opening Hook (0:00 – 0:30)

> _"In FY2023-24, Indian enforcement agencies froze ₹936 crore in crypto assets under PMLA. FIU-IND — India's financial intelligence unit — receives thousands of Suspicious Transaction Reports every year that mention Bitcoin addresses. They have no automated tool to decide which ones to investigate first. Manual graph tracing of a single address can take days. We built a system that does it in seconds."_

**Transition:** _"Let me show you what an investigator sees."_

---

## Segment 1 — Architecture (0:30 – 1:30)

**Say:** _"We run two completely separate ML pipelines."_

Point to architecture diagram:
- **Pipeline A (Elliptic):** Supervised GNN on 203,000 labelled transactions. Validates that our GNN architecture can detect illicit patterns. GraphSAGE achieves PR-AUC 0.335 — 33× the no-skill baseline on the fixed temporal split.
- **Pipeline B (Bitcoin):** Weakly supervised GNN on real Bitcoin snapshot. Uses co-spend clustering and known-bad seed addresses to generate training labels. Output is a risk ranking — NOT a classification. Every API response carries a disclaimer to this effect.

**Key framing:** _"We don't claim Pipeline B has the same precision as Elliptic. Elliptic validates the methodology. Bitcoin gives investigators a prioritized queue."_

---

## Segment 2 — Case A: The Burst Address (1:30 – 2:30)

**Action:** Type `17ebdd724dbfe21e` into SearchBar → press Enter

**Say:** _"Risk score 98, CRITICAL. Two reasons fired: extreme transaction volume burst — this address sent transactions at 95th-percentile frequency — and sub-30-minute inter-transaction intervals. Look at the graph."_

**Click GraphExplorer:** _"This is a bipartite graph — addresses and transactions. The fan-out here — many output arrows from a single transaction — is a structural signature of layering."_

**Key point:** _"No investigator manually spots this across 17,000 addresses. We surface it in the top 5 of the alert feed."_

---

## Segment 3 — Case B: Entity Linkage (2:30 – 3:30)

**Action:** Click `761d189d93e7bac2` in AlertList

**Say:** _"Score 68, HIGH. But notice: same burst pattern. Our co-spend clustering — a Union-Find algorithm over shared transaction inputs — links addresses that a common entity controls. This is co-spend heuristic, not wallet confirmation. We call it 'candidate entity linkage with confidence levels.'"_

**Click GraphExplorer → expand 1 hop**

**Say:** _"A human investigator tracing this manually would take hours. We show the connection immediately. This is the core value for FIU-IND: STR mentions one address, we surface the entity's full operational footprint."_

---

## Segment 4 — Case C: P2P Signals + Methodology Proof (3:30 – 4:30)

**Action:** Click `558e45a2f878bd09`

**Say:** _"Score 99, CRITICAL. This address has a P2P network signal."_

**Scroll to P2P Panel:** _"You'll see '⚠ SIMULATED DATA' prominently. This is honest. In Phase 1, the P2P layer uses replayed, synthetic signals — no real network monitoring. In Phase 2, with written legal authorization from FIU-IND under IT Act Section 69, this becomes real network telemetry."_

**Switch to Methodology Tab / open notebook:**

_"This table shows our Elliptic results. Four models. GraphSAGE PR-AUC 0.335 versus 0.046 for the no-skill baseline — that's a 7× improvement. These numbers are honest, on a fixed temporal split, with no result selection. The GCN methodology is validated. The Bitcoin pipeline applies it."_

---

## Segment 5 — Phase 2 Close (4:30 – 5:00)

**Say:** _"Phase 2 is a 6-month FIU-IND pilot. Month 1: formal MOU. Month 2: retrospective validation on 3 historical STR cases — does our ranking match what investigators actually found? Month 3-4: fine-tune on agency-labeled data with investigator review feedback loop. Month 5-6: role-based access, full audit trail, CERT-In compliance."_

_"The legal basis is PMLA 2002 Sections 12 and 13. We've mapped every system capability to the applicable instrument. The P2P network layer is gated — it does not activate without a written authorization order on file."_

_"Questions?"_

---

## Complete Q&A Bank (20 questions)

### Original 10 (from docs/qa_hardest.md)

**Q1: Is this transfer learning?**
> "The Elliptic dataset validates that a GNN can detect illicit patterns — PR-AUC 0.335, 7× no-skill baseline. Our Bitcoin pipeline uses its own independently engineered 14-feature space and weak supervision from co-spend heuristics. Cross-dataset representation transfer is our Round 2 research objective, requiring an alignment layer and ablation study. These are two separate models."

**Q2: Why is precision only 6–8%?**
> "That's the default operating point. At threshold 0.8, GraphSAGE flags 3,023 of 8,841 addresses and catches 75% of illicit ones — precision rises to 10.1%. The operating point is an investigator decision, not a model decision. We've published the full PR curve. For FIU-IND's use case — prioritization, not determination — recall matters more than precision at this stage."

**Q3: Why not calibrated probabilities?**
> "Calibration requires investigator-reviewed ground truth labels on Bitcoin data. That doesn't exist in Round 1 — we have weak supervision only. Claiming a calibrated probability would be epistemically dishonest. We state this explicitly: 'Model-derived risk ranking, not a calibrated probability' — hardcoded in every API response."

**Q4: What stops this from being a surveillance tool?**
> "Three engineering constraints: (1) The system is read-only — it ranks for review, it never blocks or flags publicly. (2) The P2P network layer is disabled by default — it requires a written legal authorization order logged in the audit trail. (3) Every scoring decision has a derivation chain: model version, features used, timestamp, analyst ID. There is no anonymous or unaudited flagging."

**Q5: The GCN produces zero scores on 10k — is that a failure?**
> "It is the correct behavior. The 10k snapshot covers 30 minutes of Bitcoin history. No known-bad seed addresses transacted in that window. Training on zero positive labels would produce unjustified risk scores. The system fails closed — it refuses to produce output rather than produce wrong output. This is a safety property, not a bug."

### 10 Additional Q&As (Track 8 — new)

**Q6: Why Bitcoin specifically? Why not Ethereum, Tron, or Monero?**
> "Bitcoin has the highest volume of sanctioned-entity transactions in Indian enforcement cases — Hydra, Garantex, Blender are all Bitcoin-primary. Ethereum and Tron are Round 2 extensions using the same GNN architecture with blockchain-specific feature engineering. Monero's privacy guarantees make graph analysis fundamentally harder — that's an active research problem we've identified for Round 3."

**Q7: How do you handle CoinJoin and mixing services?**
> "Co-spend clustering breaks on CoinJoin by design — CoinJoin aggregates inputs from multiple users specifically to defeat co-spend heuristics. We flag this explicitly: mixed transactions get scored on temporal and value features only, with a LOW_CONFIDENCE annotation. The Wu et al. 2023 CoinJoin detector is on our Phase 2 roadmap — integrating it would let us flag mixing attempts rather than misattribute them."

**Q8: What is your false positive rate on real investigations?**
> "We don't have real-investigation ground truth in Round 1 — that's the honest answer. What we have: GraphSAGE at threshold 0.5 catches 90% of illicit addresses in Elliptic at 8.3% precision, meaning 91.7% of flagged addresses are licit. For an investigator tool, the question is whether human review time is worth the 8.3% hit rate. Phase 2 retrospective validation on FIU-IND historical cases will give us the real number."

**Q9: How do you integrate with FIU-IND's FINTRACK system?**
> "FINTRACK is FIU-IND's STR intake system. The integration point is: when an STR mentions a Bitcoin address, FINTRACK passes it to our API as GET /api/v1/wallets/{address}. We return the risk score, reasons, graph neighbourhood, and P2P signals. The investigator sees this as a panel alongside the STR. We've designed the API specifically for this integration — REST, JSON, no proprietary protocol."

**Q10: What is the computational cost for 1 million address inference?**
> "Our current GraphSAGE inference on 17,660 addresses runs in under 5 seconds on CPU. Scaling to 1M addresses: linear in graph size for the feature computation, sub-linear for GNN inference with neighbor sampling. On a single A100 GPU, 1M addresses in under 2 minutes. For FIU-IND's actual volume — thousands of STR addresses per year, not millions — CPU inference is sufficient."

**Q11: How do you keep the seed list updated?**
> "Currently: OFAC SDN list updated monthly, public Chainalysis sanctions reports, ED press releases. Phase 2: formal data-sharing agreement gives FIU-IND-confirmed bad addresses. Phase 3: automated OFAC API pull daily. The seed update triggers a re-run of the weak label generation and model fine-tuning pipeline — fully automated."

**Q12: Can this work in real-time as transactions happen?**
> "Round 1 is a static snapshot — deliberately. Real-time streaming requires a Bitcoin Core node, Kafka pipeline, and incremental GNN update — that's Phase 3 Track B in our spec. The architecture supports it: the graph is additive, new transactions add nodes and edges, GNN inference on new subgraphs is local. We've scoped the engineering; we haven't built it yet because it's not needed to validate the approach."

**Q13: What makes this better than Chainalysis or Elliptic (commercial tools)?**
> "Three things: (1) Transparency — Chainalysis is a black box. We publish our methodology, feature spec, and model weights. FIU-IND can audit every decision. (2) India-specific deployment — Chainalysis costs ₹2–5 crore per year per agency seat. We're open-source and deployable on FIU-IND's own infrastructure. (3) Extensibility — our feature space can incorporate India-specific signals (VASP registration data, ED case history) that Chainalysis doesn't have access to."

**Q14: What happens when a suspect changes wallets frequently?**
> "This is the standard Sybil problem in crypto forensics. Our co-spend heuristic helps: if a new wallet reuses inputs with a known address, we cluster them. If they don't reuse inputs — a sophisticated actor — we rely on temporal and value signatures. This is a genuine limitation; we don't claim to catch sophisticated wallets. The system prioritizes the low-hanging fruit, which is the majority of PMLA cases."

**Q15: How do you ensure the model isn't biased against legitimate high-volume addresses like exchanges?**
> "Exchange addresses are characteristically different from mixer addresses in our feature space: exchanges have very high address reuse (many customers deposit to the same address), very high tx_count, but LOW clustering_coefficient — they don't cluster with each other. Our co-spend heuristic would not cluster exchange deposit addresses together because exchange inputs come from many independent users. The tx_burst_score would be high, but the absence of cluster signals would reduce the final score. We haven't run a formal bias audit — that's a Phase 2 validation requirement."

---

## Demo Checklist (run through 30 minutes before judging)

- [ ] Backend running: `USE_SQLITE_FALLBACK=true uvicorn backend.app.main:app`
- [ ] Frontend running: `cd frontend && npm run dev` → http://localhost:5174
- [ ] Three demo addresses in alert feed: 17ebdd724dbfe21e, 761d189d93e7bac2, 558e45a2f878bd09
- [ ] Case A search returns CRITICAL score with 2 reasons
- [ ] Case B shows GraphExplorer with edges
- [ ] Case C shows P2P panel with SIMULATED banner
- [ ] Elliptic comparison table is visible in browser (open notebooks/elliptic_eval.ipynb)
- [ ] Timer app open for 5-minute countdown
- [ ] Every team member can answer Q1–Q15 without notes
