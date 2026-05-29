# MedReconcile AI — Phase 1

![MedReconcile AI](docs/MedReconcile%20AI_.png)

**Clinical medication reconciliation powered by RAG + LLM.**  
A nurse submits a list of medications. The system checks every drug pair for interactions, retrieves grounding evidence from FDA labels and StatPearls clinical articles, and returns a structured warning with severity, recommended action, and cited sources.

---

## What it does

1. **Normalise** — each drug name is resolved to an RxCUI via the RxNorm API
2. **Enrich** — FDA label interaction text is fetched for each drug
3. **Check** — every pair of drugs is scanned for known interactions using keyword-based severity pre-filtering
4. **Retrieve** — for each interaction, the top-10 StatPearls chunks are pulled from Pinecone and re-ranked to top-3 using a cross-encoder
5. **Generate** — Gemini (primary) or Groq (fallback) produces a structured `DrugWarning` grounded only in the retrieved evidence
6. **Validate** — every LLM response is checked: JSON schema, citation integrity, drug hallucination detection
7. **Respond** — a `ReconciliationResponse` is returned with severity, action, cited sources, and a nurse-facing summary

If the LLM is unavailable or its output fails validation, the system degrades to an FDA-label-only response — the nurse always gets an answer.

---

## Architecture

```
POST /reconcile
        │
        ▼
┌─────────────────┐
│  Stage 1: Input  │  validate_input() — allowlist, injection scan, dedup
│  Validation      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Step 2: Enrich (parallel)           │
│  RxNormClient  ──►  rxcui            │
│  FDAClient     ──►  interaction text │
└────────────────────┬────────────────┘
                     │
                     ▼
         ┌───────────────────┐
         │  Step 3: Pairwise  │
         │  InteractionChecker│
         └─────────┬─────────┘
                   │
                   ▼
    ┌──────────────────────────┐
    │  Step 4: Retrieve (par.) │
    │  OpenAI embed query      │
    │  Pinecone top-10 ANN     │
    │  CrossEncoder rerank → 3 │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │  Step 5: Generate (par.) │
    │  Gemini → Groq fallback  │
    │  validate_llm_response() │
    │  → DrugWarning           │
    └────────────┬─────────────┘
                 │
                 ▼
    ┌──────────────────────────┐
    │  Step 6: Stage 5 Check   │
    │  validate_response()     │
    │  → ReconciliationResponse│
    └──────────────────────────┘
```

---

## Project structure

```
.
├── api/
│   ├── routes/
│   │   ├── health.py           # GET /health
│   │   └── reconcile.py        # POST /reconcile — full pipeline
│   ├── middleware/
│   │   └── correlation_id.py   # X-Correlation-ID on every request
│   ├── dependencies.py         # FastAPI Depends() helpers
│   └── exception_handlers.py   # StageValidationError → 422
│
├── src/
│   ├── main.py                 # FastAPI app + lifespan wiring
│   ├── config/
│   │   └── config.py           # Env var loading (fails fast on missing keys)
│   ├── chunking/
│   │   └── chunker.py          # TextChunker — character-based sliding window
│   ├── embedding/
│   │   └── embedder.py         # OpenAIEmbedder + GeminiEmbedder (Protocol)
│   ├── generation/
│   │   ├── generator.py        # LLM gateway — Gemini→Groq with circuit breakers
│   │   └── prompt_template.py  # SYSTEM_PROMPT + build_user_prompt()
│   ├── ingestion/
│   │   ├── fda_client.py       # FDA Label API client
│   │   ├── rxnorm_client.py    # RxNorm API client
│   │   ├── drug_cache_store.py # Redis-backed drug cache
│   │   └── statpearls-processor.py  # Two-phase StatPearls article processor
│   ├── resilience/
│   │   └── circuit_breaker.py  # CircuitBreaker — threshold + cooldown
│   ├── retrieval/
│   │   ├── interaction_checker.py  # Keyword-based pairwise interaction scanner
│   │   ├── pinecone_store.py        # Pinecone upsert + ANN query
│   │   └── retrieval.py             # Retriever — embed → query → rerank
│   ├── utils/
│   │   ├── schema.py           # Pydantic models: DrugWarning, ReconciliationRequest/Response
│   │   └── validators.py       # 5-stage validation pipeline
│   └── exceptions/             # Typed exceptions per component
│
├── scripts/
│   ├── 01_extract_statpearls.py    # Download + extract StatPearls XML
│   ├── 01b_filter_articles.py      # (optional) filter by drug keyword
│   ├── 02_chunk_statpearls.py      # Chunk articles → JSONL
│   ├── 03_ingest_statpearls.py     # Embed + upsert to Pinecone (resumable)
│   └── 04_eval_retrieval.py        # Retrieval quality eval
│
└── tests/
    └── units/                  # Unit tests for every component (pytest)
```

---

## API

### `POST /reconcile`

**Request**
```json
{
  "medications": [
    { "name": "warfarin", "dose": 5.0, "unit": "mg" },
    { "name": "ibuprofen", "dose": 400.0, "unit": "mg" }
  ]
}
```

**Response**
```json
{
  "medications": [...],
  "warnings": [
    {
      "drugs_involved": ["warfarin", "ibuprofen"],
      "severity": "RED",
      "reaction_result": "Concomitant use increases bleeding risk ...",
      "action": "MONITOR",
      "citation": ["article-31296_chunk_0037", "article-31294_chunk_0001"],
      "nurse_summary_to_doctor": "Monitor INR and bleeding risk closely",
      "confidence": 0.9,
      "data_source": "FRESH_FDA",
      "computed_at": "2026-05-29T18:11:54Z"
    }
  ],
  "unverified_drugs": [],
  "status": "SUCCESS",
  "response_time_ms": 8649,
  "total_medications": 2,
  "total_warnings": 1,
  "critical_warnings": 1
}
```

**Status values**
| Value | Meaning |
|---|---|
| `SUCCESS` | LLM-generated response with StatPearls citations |
| `PARTIAL` | FDA-label fallback used (LLM unavailable or output rejected) |
| `FAILED` | Pipeline error (rare — all components have soft-failure paths) |

**Severity / Action**
| Severity | Action |
|---|---|
| `RED` | `STOP` or `MONITOR` — critical interaction |
| `YELLOW` | `MONITOR` or `CONSULT_DOCTOR` — moderate interaction |
| `GREEN` | `MONITOR` — low / no known interaction |

### `GET /health`

Returns component liveness: `retriever`, `generator`, `rxnorm`, `fda`.

---

## Safety guarantees

- `generate_one` **never raises** — always returns a `DrugWarning`
- `generate_many` **never raises** — always returns `list[DrugWarning]` of the same length as input
- Every LLM response is validated: JSON schema + citation cross-check + drug hallucination check
- `extra="forbid"` on all Pydantic models — unknown fields from the LLM raise `ValidationError`
- Medication name allowlist (`^[A-Za-z0-9][A-Za-z0-9\-/().]*$`) + prompt injection scan on every input
- Circuit breakers on Gemini, Groq, Pinecone, FDA, RxNorm — unhealthy dependencies fail fast

---

## Setup

### Prerequisites

- Python 3.13+
- API keys for: OpenAI, Gemini, Groq, Pinecone

### Install

```bash
git clone https://github.com/SubhashMOthukuri/Clinical_Rag.git
cd Clinical_Rag
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn[standard] openai google-genai groq pinecone \
    pydantic sentence-transformers httpx python-dotenv tiktoken pytest pytest-asyncio
```

### Configure

```bash
cp .env.example .env
# Fill in your keys:
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...
# GROQ_API_KEY=...
# PINECONE_API_KEY=...
# PINECONE_INDEX_NAME=medreconcile-clinical-rag
# GEMINI_MODEL=gemini-2.0-flash
# GROQ_MODEL=llama-3.3-70b-versatile
```

### Run

```bash
python -m uvicorn src.main:app --reload --port 8000
```

### Test a query

```bash
curl -X POST http://localhost:8000/reconcile \
  -H "Content-Type: application/json" \
  -d '{
    "medications": [
      {"name": "warfarin", "dose": 5.0, "unit": "mg"},
      {"name": "ibuprofen", "dose": 400.0, "unit": "mg"}
    ]
  }'
```

---

## StatPearls knowledge base (Pinecone)

The retrieval layer uses 9,634 StatPearls clinical articles chunked and stored in Pinecone.

**Ingest pipeline** (one-time setup):

```bash
# Step 1 — extract articles from StatPearls XML
python scripts/01_extract_statpearls.py

# Step 2 — chunk into JSONL (512 char max, 80 char overlap)
python scripts/02_chunk_statpearls.py

# Step 3 — embed + upsert to Pinecone (resumable — safe to re-run after interruption)
python scripts/03_ingest_statpearls.py \
  --chunks data/processed/statpearls/chunks_v1.jsonl \
  --namespace full_v1 \
  --checkpoint data/processed/statpearls/chunks_v1_full_checkpoint.json \
  --resume
```

**Current status**

| Namespace | Chunks | Status |
|---|---|---|
| `full_v1` | 440,800 / 534,760 (82.4%) | Writes paused — Pinecone Starter WU limit hit. Resuming June 1, 2026. |
| `smoke_v1–v4` | ~200 each | Eval-only namespaces, to be deleted after WU reset |

---

## Validation pipeline (5 stages)

| Stage | Function | Behaviour on failure |
|---|---|---|
| 1 — Input | `validate_input()` | Raises `StageValidationError` → 422 |
| 2 — RxNorm | `validate_rxnorm_response()` | Drug marked `unverified`, pipeline continues |
| 3 — FDA | `validate_fda_response()` | FDA data skipped, interaction checker uses fallback |
| 4 — LLM | `validate_llm_response()` | LLM response rejected → FDA fallback `DrugWarning` |
| 5 — Response | `validate_response()` | Issues logged, response still returned |

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.136 + Uvicorn |
| LLM (primary) | Gemini 2.0 Flash (`google-genai`) |
| LLM (fallback) | Llama 3.3 70B via Groq |
| Embeddings | OpenAI `text-embedding-3-small` (768 dims) |
| Vector DB | Pinecone Starter (serverless) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Schema validation | Pydantic v2 |
| Drug normalisation | RxNorm API (NIH) |
| Drug label data | FDA openFDA Label API |
| Resilience | Custom CircuitBreaker |
| Python | 3.13 |

---

## Roadmap

### Phase 2 — Reliability & Observability
- [ ] Retry with exponential backoff before tripping circuit breaker
- [ ] Half-open circuit breaker state (test call before fully re-closing)
- [ ] `generate_many` concurrency cap (semaphore to prevent rate-limit storms)
- [ ] OpenTelemetry spans around every external call
- [ ] Prometheus `/metrics` endpoint
- [ ] Structured JSON logging (replace `basicConfig` with `structlog`)
- [ ] JWT / mTLS authentication middleware
- [ ] Rate limiting per nurse/clinic
- [ ] Alert if FDA fallback rate exceeds threshold (LLM quality degraded)
- [ ] Resume full StatPearls ingest (93,960 remaining chunks, June 1 2026)

### Phase 3 — Production Hardening
- [ ] HIPAA audit log middleware (every request logged with `cid` + `nurse_id`)
- [ ] PII redaction in logs
- [ ] Kubernetes deployment manifests (see `k8s/`)
- [ ] Grafana dashboards + Prometheus alerting (see `monitoring/`)
- [ ] Docker Compose for local development
- [ ] A/B prompt testing (shadow mode — compare Gemini vs Groq output quality)
- [ ] Frontend (web first, Android later) — after Phase 2 auth is in place

---

## Running tests

```bash
pytest tests/units/ -v
```

Unit tests cover: chunker, embedder, FDA client, RxNorm client, generator, interaction checker, Pinecone store, retriever, prompt template, validators (13 test files).
