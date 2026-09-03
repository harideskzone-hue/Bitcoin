# SIH26146 — 10 Hardest Q&A

> **P0.8.3** — Model answers for the most technically challenging questions
> expected from SIH judges. All four team members should be able to answer
> any of these without notes.

---

### Q1. "Your model produces a risk score. What is the probability that an address with score 75 is actually doing something illicit?"

**A:** We deliberately don't answer that question, and the API makes it
structurally impossible to present a score as a probability. Every response
containing a score includes the field:

> `"score_disclaimer": "Model-derived risk ranking. For prioritization and human review only. Not a calibrated probability."`

The score is a **ranking** — a 75 means "prioritise this address above a 50 for human review." It does not mean 75% probability of illicit activity. The sigmoid output distribution is not calibrated, and the positive-class rate in the training data (~10% in Elliptic) does not represent ground-truth illicit prevalence in the real Bitcoin network.

---

### Q2. "Your Elliptic validation gave GraphSAGE PR-AUC 0.3347. Why should we trust that this transfers to real Bitcoin?"

**A:** We explicitly don't claim it transfers. The Elliptic result is **methodology
validation** — it demonstrates that the GNN architecture (graph structure + node
features) learns signal beyond dumb baselines on a real-world labelled dataset
with similar characteristics to our Bitcoin pipeline.

The Bitcoin GCN is trained on **weak labels derived from co-spend clustering
and public seed addresses** — a completely separate pipeline (Pipeline B) from
the Elliptic evaluation (Pipeline A). The Elliptic numbers are cited with the
correct framing: "This validates the GNN approach; the Bitcoin pipeline is
evaluated on its own held-out seed split."

---

### Q3. "What is co-spend clustering? Doesn't that assume something you can't prove?"

**A:** Correct — and we're transparent about it. Co-spend clustering uses the
Union-Find heuristic: if two addresses appear as *inputs* to the same transaction,
they were likely controlled by the same private key at signing time. We call this
an **ownership/control heuristic**, not proof of common ownership.

Two limitations we acknowledge:
1. CoinJoin transactions deliberately mix co-spenders — they would create false
   clusters. We don't yet have CoinJoin detection.
2. Payment batching by exchanges can merge unrelated users. We size-cap our
   "high-confidence" cluster label.

The heuristic is well-established in the blockchain analysis literature
(Meiklejohn et al. 2013, Androulaki et al. 2013).

---

### Q4. "In Stage 4 of your demo, the GCN fails. How can you submit a GCN system that doesn't train?"

**A:** The fail-closed behaviour **is the system working correctly**. The 10k
snapshot covers 30 minutes of Bitcoin history. None of the known seed addresses
(Hydra, Garantex, Blender.io, BitcoinFog, etc.) transacted in that 30-minute window.

Training on zero positive labels would produce a model that assigns equal score to
every address — a model that is worse than useless because it would appear to work
while providing no signal.

The frozen specification explicitly prohibits falling back to UNKNOWN-majority
training. The 50k snapshot covering 90 days of history provides seed coverage
and unblocks training. This is not a bug — it is a safety property.

---

### Q5. "Your P2P signals are all labelled SIMULATED. What would real P2P monitoring look like?"

**A:** Real P2P monitoring would require a **live Bitcoin network listener** — a node
that logs which IP addresses propagate which transactions. That data is:
1. Not publicly available (requires running instrumented nodes 24/7)
2. Subject to privacy/legal considerations
3. Outside the scope of a 36-hour hackathon

The frozen spec required us to simulate this as a **design placeholder** so that
the frontend and API can demonstrate what the data would look like. The simulation
is clearly labelled with `signal_source: "SIMULATED"` and `ip_cluster_id: "IP_CLUSTER_XX"`
(no real IP addresses). The disclaimer is on every record. A judge cannot
mistake it for real telemetry.

---

### Q6. "You use seed addresses from OFAC/DOJ lists. Can an adversary who knows your seeds poison your model?"

**A:** Good question. Two defences:

1. **A/B/C/D/E seed split with leakage checks.** The D (validation) and E (test)
   seeds never appear in training labels. A poisoning attack on A/B/C seeds would
   not fool the D/E evaluation. We explicitly check that no eval seed appears in
   training labels and report this as a PASS in the dry run.

2. **Weak labels are cluster-derived, not direct.** We label entire co-spend
   clusters, not just the seed address itself. An attacker would need to control
   the cluster — not just a single known address.

Real adversarial robustness would require Phase 2 work with investigator-reviewed
labels, which we have scoped out explicitly.

---

### Q7. "Your heuristic baseline got PR-AUC 0.0375, below the no-skill baseline of 0.0461. Doesn't that mean your heuristic is worse than random?"

**A:** Yes — that specific degree+flow heuristic performs poorly on the Elliptic
test split. We report this honestly, as the methodology requires.

The important result is that the **GCN (0.1087) and GraphSAGE (0.3347) both
outperform all baselines**, including the no-skill baseline. The heuristic
result is informative: it tells us that simple degree-centrality ranking doesn't
generalise to the temporal test split, which is exactly why we use learned GNN
representations instead.

---

### Q8. "Why GraphSAGE over GCN? Couldn't you just use one?"

**A:** We trained both on the same fixed temporal split so the comparison is
controlled. GraphSAGE's mean-aggregation approach (Prop. 4.1, Hamilton et al. 2017)
learns *inductive* representations — it aggregates neighbour features rather than
learning a fixed spectral filter. On the Elliptic evaluation, this produced a
significantly higher PR-AUC (0.3347 vs 0.1087).

For the Bitcoin pipeline, we use the **same 2-layer GCN architecture** as the
Elliptic model (14 Bitcoin features vs 165 Elliptic features). GraphSAGE is
documented as a **potential future upgrade** — it's not silently substituted.

---

### Q9. "How do you prevent the same address from appearing in both your training labels and your evaluation reference?"

**A:** The leakage check is implemented in `build_weak_labels.py` and run
automatically every time the pipeline executes:

1. Training seeds (A/B/C groups) produce `high_risk` labels.
2. Validation seed (D) produces `eval_reference_val` labels.
3. Test seed (E) produces `eval_reference_test` labels.
4. We assert that `train_seed_clusters ∩ eval_seed_clusters == ∅` before writing
   the parquet file. If this assertion fails, the script aborts.

The dry run explicitly shows `leakage_check: "PASSED"` in the cluster summary.

---

### Q10. "Your system says 'for human review only.' Who is that human, and what do they do with this output?"

**A:** In the intended deployment (Phase 2, beyond this hackathon), the human is
a **financial intelligence investigator** at an agency such as an FIU or law
enforcement unit. The workflow is:

1. Daily batch run produces risk scores for the latest snapshot.
2. Investigator opens the dashboard and sees the top-50 HIGH/CRITICAL addresses.
3. For each address, they see:
   - Risk score (0–100, clearly labelled as ranking, not probability)
   - Up to 3 explanation reasons (Layer 1 deterministic, Layer 2 optional)
   - Co-spend cluster membership
   - P2P network context (simulated in Phase 0)
4. The investigator decides whether to escalate to a formal investigation.
   **The model never makes that decision.**

The audit log records every query for compliance purposes. Phase 2 would add
MFA, investigator identity, and formal escalation workflow.
