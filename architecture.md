# MedReconcile AI — Architecture

```mermaid
flowchart TD
    Client(["🏥 Nurse Client\nPOST /reconcile\n{medications[]}"])

    subgraph API ["API Layer — FastAPI"]
        MW["Correlation ID Middleware\nassigns cid to every request"]
        Stage1["Stage 1 · Input Validation\nregex · injection scan · dose checks\nfail closed → 422"]
    end

    subgraph Enrich ["Medication Enrichment  ← parallel per drug →"]
        RxNorm["RxNorm API\nname → rxcui\ningredient_rxcui"]
        FDA["FDA openFDA API\ndrug label\ninteraction evidence text"]
    end

    subgraph Cache ["Drug Cache"]
        Redis[("Redis\nhot path · 24 h TTL")]
        PG[("PostgreSQL\ndrug_master\n7-day stale refresh")]
        Redis <-->|"miss → write-through"| PG
    end

    IC["InteractionChecker\npairwise FDA evidence scan\nproduces InteractionEvidence per pair"]

    subgraph RAG ["RAG Retrieval  ← parallel per drug pair →"]
        direction TB
        Embed["OpenAI Embedder\ntext-embedding-3-small\nquery → 768-dim vector"]
        Pinecone[("Pinecone\nVector DB\nANN top_k = 10")]
        Rerank["CrossEncoder Reranker\nms-marco-MiniLM\ntop 3 chunks"]
        Embed --> Pinecone --> Rerank
    end

    subgraph KB ["Knowledge Base  (offline ingest)"]
        SP["StatPearls Articles\nNIH open-access\n512 char chunks · 80 overlap"]
        GE["Gemini Embedder\ngemini-embedding-001\nbatch ingest"]
        SP --> GE --> Pinecone
    end

    subgraph Gen ["LLM Generation  ← parallel per drug pair →"]
        direction TB
        Gemini["Gemini Flash\nprimary · temp = 0\ncircuit breaker"]
        Groq["Groq / Llama-3\nfallback · temp = 0\ncircuit breaker"]
        CitVal["Citation Validator\nrejects uncited claims\nchunk IDs only"]
        FallbackFDA["FDA-only Fallback\nconfidence = 0.5\nstatus = PARTIAL"]
        Gemini -->|"breaker open or failed"| Groq
        Groq -->|"both fail"| FallbackFDA
        Gemini --> CitVal
        Groq --> CitVal
        CitVal -->|"hallucination detected"| FallbackFDA
    end

    Stage5["Stage 5 · Response Validation\ncollects all issues · never raises\nlogs errors to observability"]

    Response(["ReconciliationResponse\nDrugWarning list · severity RED/YELLOW/GREEN\nstatus SUCCESS / PARTIAL · cid · latency_ms"])

    %% --- Main request flow ---
    Client --> MW --> Stage1
    Stage1 -->|"valid"| Enrich
    Stage1 -.->|"invalid → 422"| Client

    %% --- Enrichment + cache ---
    RxNorm <-->|"read/write"| Redis
    FDA    <-->|"read/write"| Redis

    %% --- Pipeline ---
    Enrich --> IC
    IC -->|"N drug pairs"| RAG
    RAG --> Gen
    Gen --> Stage5
    Stage5 --> Response --> Client

    %% --- Styling ---
    classDef api     fill:#1e3a5f,color:#fff,stroke:#4a90d9
    classDef enrich  fill:#1a4731,color:#fff,stroke:#4caf50
    classDef cache   fill:#3b2a1a,color:#fff,stroke:#ff9800
    classDef rag     fill:#2a1a4f,color:#fff,stroke:#9c27b0
    classDef gen     fill:#4f1a1a,color:#fff,stroke:#f44336
    classDef io      fill:#263238,color:#fff,stroke:#90a4ae,rx:20
    classDef kb      fill:#1a3040,color:#fff,stroke:#29b6f6

    class MW,Stage1 api
    class RxNorm,FDA enrich
    class Redis,PG cache
    class Embed,Pinecone,Rerank rag
    class Gemini,Groq,CitVal,FallbackFDA gen
    class Client,Response io
    class SP,GE,IC,Stage5 kb
```

## Request Path (left to right)

| Step | Component | What happens |
|---|---|---|
| 1 | **Correlation ID Middleware** | Assigns a `cid` to every request for end-to-end tracing |
| 2 | **Stage 1 Validation** | Regex + injection scan + Pydantic constraints. Fail closed → 422 |
| 3 | **Medication Enrichment** | RxNorm → rxcui. FDA → interaction evidence text. Parallel per drug |
| 4 | **Drug Cache** | Redis (24h) in front of PostgreSQL. Writes back on miss. 7-day background refresh |
| 5 | **InteractionChecker** | Builds all drug pairs from FDA evidence. Pure text scan |
| 6 | **RAG Retrieval** | Embed query → Pinecone ANN (top 10) → CrossEncoder rerank → top 3 chunks |
| 7 | **LLM Generation** | Gemini Flash → Groq fallback → citation validator → FDA fallback if all fail |
| 8 | **Stage 5 Validation** | Collects all response issues. Never raises — logs for observability |

## Failure modes

```
Stage 1 fails       → 422 immediately, no downstream calls
RxNorm/FDA fails    → drug goes into unverified_drugs, pipeline continues
Retrieval fails     → empty chunks, generator uses FDA-only path
Gemini fails        → circuit breaker trips, Groq takes over
Groq fails          → FDA-only DrugWarning, status = PARTIAL
Citation hallucination → response rejected, FDA fallback returned
Stage 5 fails       → logged, response still returned
```
