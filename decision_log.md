# Decision Log — MedReconcile AI

Architectural decisions that shaped this system. Captured so future contributors understand not just what was chosen, but why — and what the alternatives were.

---

## Decision 1: FastAPI over Flask/Django

**Chosen:** FastAPI  
**Rejected:** Flask, Django REST Framework

FastAPI is async-native. Every external call in this pipeline — Pinecone, OpenAI, Gemini, Groq, FDA API, RxNorm — is an I/O wait. With Flask, each of those waits would block a thread. FastAPI + uvicorn handles all of them concurrently on a single event loop, which matters at clinic scale (multiple nurses submitting reconciliation requests simultaneously). The automatic Pydantic-based request validation and OpenAPI doc generation are bonuses for a healthcare API where contract clarity is mandatory.

---

## Decision 2: Pinecone (managed vector DB) over self-hosted pgvector

**Chosen:** Pinecone  
**Rejected:** pgvector (PostgreSQL extension), Weaviate, Qdrant

Pinecone's free tier supports a 768-dim index with no infrastructure to operate. For an MVP targeting small specialty practices, not managing a separate vector database server is the right call. pgvector requires tuning HNSW parameters, managing index bloat, and benchmarking under load — overhead that adds no clinical value in Phase 1. The Pinecone SDK wraps all retry and connection pool logic; we expose our own circuit breaker on top for fail-fast behavior when the service is degraded.

---

## Decision 3: StatPearls as the knowledge corpus

**Chosen:** StatPearls (NIH/NCBI open-access clinical articles)  
**Rejected:** UpToDate (paywalled), Drugs.com scraping (ToS risk), general PubMed

StatPearls is NCBI-published, open-access, and structured (title + sections + article type). It covers pharmacology, drug mechanisms, and clinical interactions in a format that chunks cleanly. UpToDate is the clinical gold standard but is behind a license wall. PubMed abstracts are too short for dense retrieval chunks. StatPearls gives drug-interaction-relevant content at the right granularity without legal risk.

---

## Decision 4: FDA Drug Label API for interaction data, not RxNorm Interaction API

**Chosen:** FDA openFDA Drug Label API  
**Rejected:** RxNorm Drug Interaction API

RxNorm's drug interaction API was discontinued in January 2024. FDA is the authoritative source anyway — every approved drug's label is filed with the FDA and updated with each new adverse event or interaction report. FDA labels are updated monthly; RxNorm interaction data lagged behind. The FDA endpoint gives us raw interaction text that becomes evidence for the LLM to reason over, rather than a pre-computed severity score we'd have to trust blindly.

---

## Decision 5: Two-layer drug cache (Redis → PostgreSQL) with 7-day stale refresh

**Chosen:** Redis (24h TTL, hot path) in front of PostgreSQL (durable store, 7-day refresh)  
**Rejected:** In-memory dict, single-layer PostgreSQL, Redis only

Drug RXCUI lookups are expensive (one RxNorm API call per drug). A nurse submits 5–12 drugs per reconciliation; without caching, every request calls RxNorm in real time. Redis gives sub-millisecond reads for hot drugs (frequently prescribed formularies). PostgreSQL is the durable source of truth that survives Redis restarts. The 7-day background refresh catches FDA monthly updates — stale interaction data in a healthcare system is a patient safety risk, not just a data quality issue.

---

## Decision 6: Embedder as a Protocol (structural typing), not an abstract base class

**Chosen:** Python `Protocol` with `OpenAIEmbedder` and `GeminiEmbedder` implementations  
**Rejected:** ABC (abstract base class) with forced inheritance

Protocol-based design means any object with the right methods is a valid embedder — no inheritance required. This is critical for testability: unit tests inject a `MockEmbedder` that satisfies the protocol without inheriting from anything. It also decouples the retrieval pipeline from OpenAI or Gemini specifics — swapping in a `MedCPTEmbedder` (domain-fine-tuned model) in Phase 2 requires zero changes to the retriever. The same pattern applies to `VectorStore` and `VectorStore` protocol.

---

## Decision 7: Circuit breaker on every external service client

**Chosen:** Custom `CircuitBreaker` (threshold + cooldown) on FDA, RxNorm, OpenAI, Gemini, Groq, Pinecone  
**Rejected:** No circuit breaker, letting exceptions propagate, using `tenacity` retry-only

If OpenAI is degraded, every concurrent reconciliation request would sit waiting for a 30s timeout before failing. With a circuit breaker, after 5 consecutive failures the breaker trips and subsequent calls fail immediately for 30 seconds. The nurse gets the FDA fallback response in milliseconds instead of a 30s hang. Rate-limit errors (429) explicitly do not trip the breaker — rate limits are a throttling signal, not a service outage.

---

## Decision 8: Dual LLM provider (Gemini primary → Groq fallback) with independent breakers

**Chosen:** Gemini Flash as primary, Groq (Llama-3 or Mixtral) as fallback, independent `CircuitBreaker` per provider  
**Rejected:** Single provider, synchronous fallback, shared breaker

A single LLM provider makes the generation step a single point of failure. In a clinical tool where a nurse is waiting at the bedside, a silent LLM outage must degrade gracefully. Gemini and Groq have different infrastructure dependencies, so a Gemini outage rarely coincides with a Groq outage. Independent circuit breakers mean a Groq rate storm doesn't trip the Gemini breaker and vice versa. Temperature is set to 0 on both — medication interaction reporting must be deterministic, not creative.

---

## Decision 9: LLM output validated against retrieved chunk IDs (hallucination guardrail)

**Chosen:** `validate_llm_response` checks every LLM-cited source against `allowed_citation_sources` (the set of chunk IDs actually retrieved)  
**Rejected:** Trust LLM citations, post-hoc human review, no citation requirement

The LLM can only cite sources the retrieval step actually returned. If the model invents a citation ("FDA_2023_ibuprofen_interaction"), it will not be in the retrieved chunk set — the response is rejected and the FDA fallback is returned instead. This is the concrete implementation of the dual-model guardrail philosophy: the LLM generates, the validator acts as the second model that checks every claim. In healthcare, an uncited drug interaction warning is indistinguishable from a hallucination.

---

## Decision 10: Five-stage validation pipeline with typed `StageValidationError`

**Chosen:** Sequential stages (1: input sanitization → 2: RxNorm → 3: FDA → 4: retrieval → 5: LLM output), each raising `StageValidationError(stage, message, details)`  
**Rejected:** Single validation pass, untyped exceptions, nested try/except soup

Each stage has a known failure mode. Tagging errors with a `ValidationStage` enum (e.g., `STAGE_2_RXNORM`) makes every failure observable as a Prometheus metric (`validation_errors_total{stage="STAGE_2_RXNORM"}`). Without stage labels, all failures collapse into one counter and you cannot diagnose whether errors spike at input cleaning (bad nurse data) or at RxNorm (upstream outage). Stages 1–4 fail closed on first error (cheap to abort). Stage 5 collects all LLM issues before returning a result (compute was already spent — get the full diagnostic).

---

## Decision 11: Sentence-boundary chunking with overlap (512 char max, 80 char overlap)

**Chosen:** Sentence-boundary split at 512 chars with 80-char overlap  
**Rejected:** Fixed-size split, paragraph split, 1024-char chunks, no overlap

512 characters fits comfortably within token limits for both `text-embedding-3-small` (8191 tokens) and `gemini-embedding-001` (2048 tokens) while keeping chunks semantically dense. The sentence-boundary heuristic (`rfind(".")`) avoids cutting a drug interaction sentence mid-phrase, which would make the chunk semantically incomplete. The 80-char overlap ensures that sentences straddling a chunk boundary appear in at least one chunk fully — critical when a dosage qualifier or contraindication appears at the end of one sentence and the beginning of the next.

---

## Decision 12: PostgreSQL for durable drug master data (not a document store or SQLite)

**Chosen:** PostgreSQL with `INSERT … ON CONFLICT DO UPDATE` for drug master  
**Rejected:** SQLite (no concurrent writes), MongoDB (overkill schema flexibility), Redis only

The drug master table is written by the background refresh job and read by every reconciliation request — concurrent access requires proper MVCC, which SQLite doesn't handle safely. `ON CONFLICT DO UPDATE` (upsert) means re-ingesting a known drug safely bumps `lookup_count` and refreshes `rxcui` without losing history or requiring a read-before-write. PostgreSQL's JSONB columns give schema flexibility for future drug metadata without a migration. The asyncpg driver provides async connection pooling that fits the FastAPI event loop.

---

## Decision 13: Strict input sanitization with regex + injection scan

**Chosen:** Drug names validated against `^[A-Za-z0-9][A-Za-z0-9 \-/().]*$` (literal space, no `\s`) plus secondary scan for prompt-injection phrases  
**Rejected:** Free-text drug names, whitelist-only approach, client-side validation

This system sends drug names directly into LLM prompts. A nurse could inadvertently (or a malicious actor deliberately) submit `"Ignore previous instructions and..."` as a drug name. The regex rejects tabs, newlines, RTL overrides, zero-width characters, and null bytes at Stage 1 — before any data reaches an API. The secondary scan catches known injection patterns on `name.lower()`. This combination passed 55/55 adversarial inputs in pen testing (homoglyph, zero-width, RTL override, null byte, newline-smuggling).

---

## Decision 14: `generate_one` and `generate_many` never raise — always return a `DrugWarning`

**Chosen:** FDA fallback `DrugWarning` on any failure inside the generation step  
**Rejected:** Propagate exceptions to the route handler, return `None`, partial results

The nurse endpoint must always return a usable response. If the LLM is down, the nurse still needs to know about the interaction — they just get an FDA-label-derived warning instead of an LLM-reasoned one. `generate_many` uses `asyncio.gather(return_exceptions=True)` so one failed drug pair never cancels the others. The fallback includes a `confidence=0.5` flag and a note in `nurse_summary_to_doctor` indicating degraded mode, so the nurse knows to verify with the physician rather than treating the output as fully reasoned.

---

## Decision 15: Service separation into `src/` (domain logic) and `api/` (HTTP layer)

**Chosen:** `src/` for all domain services (embedder, retriever, generator, chunker, validators), `api/` for routes, middleware, exception handlers  
**Rejected:** Monolithic `app.py`, mixing business logic in route handlers

Route handlers that contain business logic cannot be tested without spinning up HTTP. Keeping domain logic in `src/` means every service is independently testable with plain `pytest` — no test client, no HTTP overhead. The `api/` layer is thin by design: it validates the HTTP contract, calls `Depends(get_xxx)` to pull components from `app.state`, and delegates to domain services. This boundary also makes it possible to add a CLI interface or a background worker that calls the same domain services without duplicating logic.

---

*Decisions documented at project inception and updated as the design evolved.*  
*Last updated: 2026-06-01*
