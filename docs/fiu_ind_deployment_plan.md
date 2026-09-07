# FIU-IND Deployment Plan — SIH26146
## 6-Month Pilot from Round 1 Selection to Agency Handover

---

## Indian Regulatory Context

### Why this tool is needed now

| Event | Impact |
|---|---|
| MHA crypto seizures FY2023-24 | ₹936 crore in crypto assets frozen under PMLA |
| FIU-IND mandate (PMLA 2002 S.12) | VASPs must file STRs; FIU-IND has no automated graph tool to prioritize them |
| RBI Circular Nov 2022 on VDAs | Banks must report suspicious crypto transactions — volume is overwhelming manual review |
| CERT-In IT (Amendment) Rules 2022 | Crypto exchanges must report incidents within 6 hours — creates a real-time data requirement |
| ED crypto investigations 2023-24 | 15+ formal cases involving Bitcoin address tracing — all done manually |

### Applicable legal instruments

| Instrument | Scope for this system |
|---|---|
| **PMLA 2002 Section 12** | FIU-IND receives STRs from VASPs. This tool automates the ranking of Bitcoin addresses mentioned in STRs. |
| **PMLA 2002 Section 13** | FIU-IND can direct VASPs to provide transaction records. System identifies which addresses to request records for. |
| **FEMA 1999** | Cross-border Bitcoin flows are covered. System flags addresses with large outbound BTC consistent with capital flight. |
| **IT Act 2000 Section 69** | Network monitoring requires written authorization from competent authority. P2P layer is GATED behind this — not enabled without written order. |
| **CERT-In Rules 2022** | Crypto exchange incident reports feed into the system's P2P signal layer (Phase 2). |
| **Prevention of Corruption Act** | Investigation support — not primary use case, but relevant for DA cases involving crypto. |

> ⚠️ **The student team does not determine which legal instrument applies.** That is FIU-IND's legal counsel's responsibility. The system is engineered so that network-layer capability (P2P real data) is disabled until a written authorization is provided and logged in the audit trail.

---

## 6-Month Pilot Plan

### Month 1 — Formal Engagement

| Task | Owner | Deliverable |
|---|---|---|
| Formal letter to FIU-IND Director General requesting pilot MOU | Domain lead | Letter sent |
| Meet FIU-IND Technology Division | Domain + Backend lead | Meeting notes, requirements captured |
| Identify 3 historical STR cases for retrospective validation | FIU-IND + ML lead | 3 anonymized case files |
| Legal counsel review of data-sharing scope | FIU-IND counsel (not student team) | Written legal opinion |

**Contact path:** FIU-IND Technology Division, Ministry of Finance, North Block, New Delhi. Director General currently: as per MoF annual report. SIH sponsoring ministry can facilitate introduction.

---

### Month 2 — Data Sharing Agreement + Retrospective Validation

| Task | Owner | Deliverable |
|---|---|---|
| Data Sharing Agreement (DSA) signed | Both parties | Signed DSA on file |
| Ingest 3 historical STR Bitcoin addresses into system | ML lead | Risk scores generated for STR addresses |
| Compare system ranking against investigator's actual finding | FIU-IND analyst | Validation report: did our system flag the right addresses? |
| False positive review by actual investigator | FIU-IND analyst | FP rate per STR category |

> **Gating rule:** Phase 2.3 onwards does NOT start until DSA is signed.

---

### Month 3–4 — Model Fine-Tuning on Agency Data

| Task | Deliverable |
|---|---|
| Expand feature set with agency-specific fields (STR category, VASP name, jurisdiction) | `data/agency_features.parquet` |
| Fine-tune Bitcoin GCN on agency-labeled cases (small lr=1e-4, freeze early layers) | `models/gcn_agency_v1.pt` |
| Integrate with FIU-IND's FINTRACK system (read-only API) | API connector |
| Evaluate: false-positive rate vs. manual baseline | Report to FIU-IND |

---

### Month 5–6 — Human-in-the-Loop Workflow + Hardening

| Task | Deliverable |
|---|---|
| Investigator review UI: confirm / reject / escalate a flag | Feature in staging |
| Derivation chain log: model version, features used, timestamp, analyst ID | Full audit trail |
| Feedback loop: confirmed/rejected flags fed back to retraining | Pipeline documented |
| Role-based access: analyst / supervisor / admin | Roles in API middleware |
| CERT-In compliance assessment | Assessment report |
| Phase 2 exit: system running on real FIU-IND STR data | Deployment sign-off |

---

## What makes FIU-IND the right first pilot partner

1. **Volume problem is real:** FIU-IND processes thousands of STRs mentioning Bitcoin addresses annually, with no automated graph tool
2. **Legal framework exists:** PMLA 2002 already gives FIU-IND authority to analyze transaction data
3. **No real-data dependency for Phase 1:** Our Round 1 system uses public Bitcoin data — we don't need agency data to demonstrate the approach
4. **Low deployment risk:** System is read-only (no transaction blocking), investigator-in-the-loop, with full audit trail
5. **SIH alignment:** Ministry of Finance is the SIH problem setter — FIU-IND is under MoF

---

## How this system does NOT work (important to state for judges)

- ❌ Does NOT block transactions
- ❌ Does NOT constitute a legal finding
- ❌ Does NOT use real P2P network data without written legal authorization
- ❌ Does NOT replace investigator judgment
- ✅ Prioritizes which addresses human investigators should review first
- ✅ Surfaces evidence (graph structure, timing, clustering) that humans can verify
- ✅ Every scoring decision is explainable and auditable
