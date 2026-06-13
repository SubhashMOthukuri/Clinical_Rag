# MedReconcile AI — Phase 2 Plan

**Status:** Phase 1 complete and baselined (mean e2e = 3363ms, p95 = 4171ms, 100% SUCCESS)  
**Phase 2 Goal:** Clinical accuracy. A nurse waiting 5–6 seconds for a precise, cited, pharmacist-grade answer is better than getting a fast, vague one. Latency is not a priority. Correctness is.

---

## The Reframe

Phase 1 proved the pipeline works end-to-end. Phase 2 answers a harder question: **is the output actually right?**

Right now we do not know. We have no labelled drug pairs, no ground truth severity, and a known silent bug that makes every citation potentially fake. We cannot measure accuracy of a system we haven't instrumented for accuracy.

Everything in Phase 2 flows from fixing that.

---

## The 4 Things That Can Fail Independently

The system has 4 separate accuracy dimensions. They fail independently and depend on each other in a strict chain. Measuring the wrong one first gives you meaningless numbers.

```
#4  Interaction detection recall
    Did interaction_checker FIND the pair at all?
    A missed pair → no retrieval, no LLM, no warning. Silent miss.

        ↓ only if #4 finds the pair

#2  Citation accuracy  (currently BROKEN)
    Does the chunk_id the LLM cites match a chunk that was actually retrieved?
    If broken → every response silently downgrades to FDA_LABEL fallback.
    You cannot measure #1 or #3 until this is fixed.

        ↓ only if #2 is working

#1  Retrieval accuracy
    Do the returned chunks actually explain THIS pair's interaction?
    "warfarin mechanism" ≠ "warfarin + aspirin bleeding risk."
    Measure: Recall@K — is the known-relevant chunk in the top-K?

        ↓ only if #1 returns the right chunks

#3  Severity classification accuracy
    Does RED/YELLOW/GREEN match what a pharmacist would assign?
    This is the clinical output. The thing that matters to the nurse.
    Only meaningful if #4, #2, and #1 are all working.
```

**The dependency chain is strict.** Fix #2 first. Then measure #4 and #1. Then evaluate #3. In that order.

---

## Priority 1 — Unblock Measurement

Nothing else can be evaluated until these two things are done.

### 1.1 Fix the citation bug (#2)

**What:** The prompt wraps chunk IDs as `[chunk_id: X]` but the validator checks raw IDs. A format mismatch silently rejects all citations and every response falls back to `FDA_LABEL`. The LLM may be reasoning correctly from retrieved chunks — we cannot tell, because the citation check always fails.

**Why this is first:** Until this is fixed, #1 and #3 scores on any eval set are noise. You'd be measuring "how often does the system fall back to FDA" not "how accurate is the RAG pipeline."

**Fix:** One integration test — send a real request, assert `DrugWarning.citation` contains actual chunk IDs from what was retrieved, not `["FDA_LABEL"]`. This test does not exist. Write it, watch it fail, fix the format, watch it pass.

**This is a binary bug fix. No eval set needed. Just the test.**

### 1.2 Build the eval set

**What:** 50–100 labelled drug pairs covering all 4 accuracy dimensions.

For each pair, record:
- Ground-truth severity: RED / YELLOW / GREEN (pharmacist-assigned)
- Whether interaction_checker should find it: yes/no (#4 label)
- Which StatPearls chunk IDs are relevant (#1 label)
- Expected action: STOP / MONITOR / CONSULT_DOCTOR (#3 label)

**Coverage requirements:**
- At least 15 RED pairs (dangerous interactions — highest stakes)
- At least 20 YELLOW pairs (monitoring required)
- At least 10 GREEN pairs (no significant interaction — false positive risk)
- At least 10 pairs where interaction_checker should find nothing (tests #4 specificity)
- Include brand names, misspellings, and generic names for the same drug (tests #4 robustness)

**Source for ground truth:** Clinical pharmacist review, or cross-reference against established interaction databases (Drugs.com, Lexicomp, FDA REMS documents).

### 1.3 Establish Phase 1 baseline scores

Run the eval set against the current Phase 1 system before changing anything. Record:

| Axis | Metric | Phase 1 baseline |
|---|---|---|
| #4 Interaction detection | Recall (% of labelled pairs found) | TBD |
| #4 Interaction detection | Precision (% of found pairs that are real) | TBD |
| #1 Retrieval | Recall@3 (relevant chunk in top 3?) | TBD |
| #1 Retrieval | MRR (mean reciprocal rank of first relevant chunk) | TBD |
| #3 Severity | Accuracy vs pharmacist label | TBD |
| #3 Severity | RED recall (% of true RED pairs classified RED) | TBD — most critical |

**RED recall is the most important single number.** A missed RED is a patient safety failure. It matters more than any latency number.

Every subsequent change must hold or improve these scores. If a change improves latency but drops RED recall — it does not ship.

---

## Priority 2 — Fix Interaction Detection (#4)

The interaction_checker is a keyword scanner over FDA label text. It finds pairs where one drug's FDA label mentions the other by name. This approach has known gaps.

### 2.1 Understand the current miss rate

Before fixing anything, measure it. Run the eval set. Find every pair in the labelled set where:
- Ground truth says: this is an interaction
- interaction_checker says: no evidence found

Those are the silent misses. Categorise why each one was missed:
- Drug mentioned by brand name only (checker uses generic)
- Drug mentioned by drug class, not name ("anticoagulants" not "warfarin")
- Interaction evidence is in StatPearls but not FDA label text
- FDA label uses abbreviation or INN name

### 2.2 Extend checker to catch class-level interactions

Many dangerous interactions are stated at the drug class level in FDA labels. "Avoid concurrent use with anticoagulants" covers warfarin, apixaban, rivaroxaban — but a name-match checker misses all of them unless it knows the drug class mapping.

**Fix:** Add drug class → member drug mapping. If the FDA label mentions "anticoagulants" and the patient is on warfarin, that counts as evidence.

### 2.3 Hybrid search as a second-pass (#4 + #1)

Pure semantic search misses exact drug name matches. `ciprofloxacin` and `cipro` may not be close in embedding space. Brand names, abbreviations, and INN names for the same compound create gaps.

**Implementation:** Pinecone supports sparse-dense hybrid natively. Add BM25 sparse vectors at ingest time, combine with dense at query time using Reciprocal Rank Fusion (RRF).

**Gate:** Recall@3 must improve on the eval set. Do not merge if it doesn't.

---

## Priority 3 — Fix Retrieval Quality (#1)

### 3.1 Replace the reranker with a biomedical-tuned model

**Current reranker:** `ms-marco-MiniLM-L-6-v2` — trained on general web search queries. Does not understand `QT prolongation`, `CYP3A4 inhibition`, `serotonin syndrome`. Its chunk reordering for clinical text is not meaningfully better than Pinecone's ANN order.

**Replace with:** A cross-encoder fine-tuned on biomedical text. Model candidates:
- `ncats/NRSA-cross-encoder`
- A cross-encoder fine-tuned on PubMed pairs
- A custom model fine-tuned on StatPearls drug interaction pairs (highest value, most effort)

**Deploy:** Separate FastAPI service. Retriever calls it over HTTP with 200ms timeout. On timeout → fall back to Pinecone ANN order. This decouples it from the app pod — no more 917ms CPU block per request.

**Gate:** Recall@3 and MRR must improve over the Phase 1 baseline before this ships. If it scores the same as Pinecone ANN order, it does not ship.

### 3.2 Tune retrieval parameters against the eval set

`score_threshold=0.5`, `retrieve_k=10`, `rerank_n=3` are all Phase 1 placeholders. Now that there is a labelled set, tune them properly.

- Lower `score_threshold` if relevant chunks are being filtered out (hurts Recall@K)
- Raise `retrieve_k` if the relevant chunk ranks below 10 (before reranking)
- Adjust `rerank_n` based on where relevant chunks land after biomedical reranker

**Gate:** Every parameter change must show improvement on Recall@K or MRR. No blind tuning.

### 3.3 Include dose and unit in the retrieval query

Current query: `"warfarin aspirin drug interaction mechanism clinical management"`  
Better query: `"warfarin 5mg aspirin 81mg drug interaction mechanism clinical management"`

Dose-specific interactions exist (warfarin at therapeutic vs supratherapeutic doses behave differently). Including dose in the query may pull more relevant chunks.

**Gate:** Recall@3 must improve. If it doesn't, revert.

---

## Priority 4 — Fix Generation Quality (#3)

These only matter after #4, #2, and #1 are working. Improving the LLM prompt while retrieval is broken just produces confident-sounding wrong answers.

### 4.1 Include dose and unit in the LLM prompt

`warfarin 1mg` and `warfarin 10mg` carry very different clinical risk profiles. The current prompt sends drug names only. Pass `evidence.drug_a.dose`, `.unit`, `evidence.drug_b.dose`, `.unit` into `build_user_prompt()`.

**Gate:** Severity accuracy on eval set must hold or improve.

### 4.2 LLM provider — correctness first

Switching from Gemini to Groq (llama-3.3-70b) is a model swap, not a hardware swap. A different model may classify severity differently or cite different chunks.

**This change is only justified if:** the eval set shows Groq's severity accuracy matches or exceeds Gemini's. Run both models against the full eval set before deciding. If Gemini is more accurate, Gemini stays primary regardless of latency.

**How to evaluate:** Shadow mode — run both providers on every eval set request, compare severity classification and citation accuracy side by side. Only promote Groq if it wins on RED recall specifically.

### 4.3 Prompt versioning

Move `SYSTEM_PROMPT` to `prompts/v1/system.txt`. Log `prompt_version` in every `DrugWarning`. Every prompt change gets a version number. If a new prompt drops eval scores, roll back by version ID.

### 4.4 Few-shot examples in the prompt

Add one worked RED example and one worked GREEN example inside the system prompt. This anchors the model's calibration for severity classification without relying entirely on retrieved chunks.

**Gate:** Severity accuracy on eval set must improve.

---

## Priority 5 — Safety & Compliance (required before any real patient data)

### 5.1 HIPAA audit log middleware

Every `/reconcile` request logged immutably: `correlation_id`, `nurse_id`, `patient_id`, `medications[]`, `warnings[]`, `computed_at`, `response_time_ms`.
- Append-only — never updated or deleted
- nurse_id and patient_id from JWT, not request body
- Written before response returns, not after

### 5.2 Authentication middleware

JWT validation on every `/reconcile` call. Fields from token: `nurse_id`, `clinic_id`, `role`. FastAPI dependency on the router — `/health` and `/metrics` stay unauthenticated.

### 5.3 PII redaction in logs

`structlog` processor scrubs PHI fields from all log output. Exception: the HIPAA audit log, which must contain full data in a compliant store.

### 5.4 Hard request timeout

Single `asyncio.wait_for` around the full pipeline. Timeout: 10s (generous — accuracy matters more than speed). On expiry: PARTIAL response, `confidence=0`, `data_source=FDA_FALLBACK`.

### 5.5 Rate limiting

`slowapi`, 10 req/min per `nurse_id`. Returns HTTP 429 with `Retry-After`.

---

## Priority 6 — Observability

### 6.1 Prometheus /metrics endpoint

Replace in-process snapshot. Key alerts — all accuracy-oriented:
- `red_recall < 0.90` → failing to catch dangerous interactions, page immediately
- `fda_fallback_rate > 10%` → citation bug may have regressed
- `retrieval_empty_results > 5%` → interaction_checker or retriever is missing pairs

### 6.2 Structured logging

Replace `logging.basicConfig` with `structlog`. JSON lines queryable in CloudWatch / Datadog / Loki.

### 6.3 OpenTelemetry tracing

Spans across `enrich → embed → pinecone → rerank → llm`. Waterfall trace per request.

### 6.4 Graceful shutdown + secret rotation

`asyncio.Event` shutdown hook drains in-flight requests on SIGTERM. API keys in AWS/GCP Secrets Manager, reloadable without restart.

---

## Phase 2 Target State

| Dimension | Phase 1 | Phase 2 target |
|---|---|---|
| Citation accuracy (#2) | Unknown (bug present) | 100% — every response cites real retrieved chunks |
| Interaction detection recall (#4) | Unknown | >95% of labelled pairs found |
| Retrieval Recall@3 (#1) | Unknown | >85% — relevant chunk in top 3 |
| RED severity recall (#3) | Unknown | >90% — missing a RED is a patient safety failure |
| GREEN precision (#3) | Unknown | >80% — false positives erode nurse trust |
| Latency | 3363ms mean | Not a target. Accuracy gates every change. |
| Auth | None | JWT per nurse |
| Audit log | None | Append-only, HIPAA-compliant |

---

## Delivery Order

```
Week 1  → 1.1 Fix citation bug (integration test, watch it fail, fix it, watch it pass)
          1.2 Build eval set (50-100 labelled pairs, pharmacist-reviewed)
          1.3 Establish Phase 1 baseline scores on all 4 axes

Week 2  → 2.1 Measure interaction detection miss rate (#4 analysis)
          2.2 Extend checker with drug class mapping
          2.3 Hybrid search BM25 + semantic
          Gate: #4 recall must improve over Phase 1 baseline

Week 3  → 3.1 Biomedical reranker as hosted service
          3.2 Tune score_threshold, retrieve_k, rerank_n against eval set
          3.3 Dose + unit in retrieval query
          Gate: Recall@3 and MRR must improve

Week 4  → 4.1 Dose + unit in LLM prompt
          4.2 LLM shadow mode — Gemini vs Groq on eval set, pick winner
          4.3 Prompt versioning
          4.4 Few-shot examples in prompt
          Gate: severity accuracy and RED recall must hold or improve

Week 5  → 5.1 HIPAA audit log + 5.2 JWT auth + 5.3 PII redaction
          Security review. No patient data before this week clears.

Week 6  → 6.1 Prometheus + 6.2 structlog + 6.3 OTel
          5.4 Hard timeout + 5.5 rate limiting + 6.4 graceful shutdown
          Load test: 50 RPS sustained, all circuit breakers verified
          Phase 2 sign-off: RED recall >90%, citation accuracy 100%
```

---

## What Changed From The Original Plan And Why

| Original | Now | Why |
|---|---|---|
| Priority 1: latency | Latency removed as a priority | A precise slow answer beats a fast vague one |
| <1000ms target | No latency target | Was never validated against real nurse workflow |
| Week 1: Groq + remove reranker | Week 1: fix citation bug + build eval set | Can't measure anything until citation bug is fixed |
| "Accuracy" as one thing | 4 separate measurable dimensions (#4, #2, #1, #3) | Each fails independently, each needs its own metric |
| Groq as primary for speed | Groq only if it wins on RED recall | Provider choice is a correctness decision, not a latency decision |
| Speculative embedding | Removed entirely | Not safe without verified drug names |
