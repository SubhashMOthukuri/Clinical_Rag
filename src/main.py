"""FastAPI app — lifespan and wiring only.

Creates every long-lived component at startup, stores them on app.state,
and closes them cleanly at shutdown.  Routes, middleware, and exception
handlers live in the api/ package and are registered here.

Pipeline (per request) — see api/routes/reconcile.py for details.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.exception_handlers import stage_validation_handler
from api.middleware.correlation_id import correlation_id_middleware
from api.routes.health import router as health_router
from api.routes.reconcile import router as reconcile_router
from src.config.config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from src.embedding.embedder import OpenAIEmbedder
from src.generation.generator import Generator
from src.ingestion.fda_client import FDAClient
from src.ingestion.rxnorm_client import RxNormClient
from src.retrieval.interaction_checker import InteractionChecker
from src.retrieval.pinecone_store import PineconeStore
from src.retrieval.retrieval import Retriever
from src.utils.validators import StageValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN — create components once, close them on shutdown
# ============================================================================
# Why lifespan and not module-level globals:
#  - Components are created ONCE (reranker model load is slow, ~3-5 s)
#  - Stored on app.state so any route can access them via Depends(get_xxx)
#  - Cleanly closed on SIGTERM — no leaked HTTP connections or half-finished work
#  - The pattern FastAPI recommends since v0.95+ (replaces @app.on_event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup.begin")

    embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)
    store = PineconeStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        dimensions=768,
    )
    retriever = Retriever(embedder=embedder, store=store, namespace="full_v1")
    rxnorm = RxNormClient()
    fda = FDAClient()
    interaction_checker = InteractionChecker(db_pool=None, redis_client=None)
    generator = Generator(
        gemini_api_key=GEMINI_API_KEY,
        groq_api_key=GROQ_API_KEY,
        gemini_model=GEMINI_MODEL,
        groq_model=GROQ_MODEL,
    )

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.rxnorm = rxnorm
    app.state.fda = fda
    app.state.interaction_checker = interaction_checker
    app.state.generator = generator

    logger.info("startup.complete")
    yield

    logger.info("shutdown.begin")
    await embedder.close()
    await store.close()

    await rxnorm.aclose()
    await fda.aclose()
    logger.info("shutdown.complete")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="MedReconcile AI",
    description="Clinical medication reconciliation with RAG-grounded LLM judgment",
    version="0.1.0",
    lifespan=lifespan,
)

app.middleware("http")(correlation_id_middleware)
app.add_exception_handler(StageValidationError, stage_validation_handler)
app.include_router(health_router)
app.include_router(reconcile_router)


# ============================================================================
# PHASE 2 TODOs
# ============================================================================
# [ ] Hard request timeout middleware (e.g. 5 s) — currently relies on per-component timeouts
# [ ] Rate limiting per nurse/clinic (slowapi or starlette-limiter)
# [ ] Prometheus /metrics endpoint
# [ ] Structured JSON logging (replace basicConfig with structlog)
# [ ] Authentication middleware — JWT/mTLS for service-to-service

# ============================================================================
# PROD TODOs
# ============================================================================
# [ ] CORS config if browser clients call directly (FastAPI CORSMiddleware)
# [ ] OpenTelemetry middleware for distributed tracing
# [ ] Graceful shutdown — drain in-flight requests on SIGTERM
# [ ] HIPAA audit log middleware — every request logged immutably with cid + nurse_id
# [ ] PII redaction in logs (med names + nurse_ids are PHI in some contexts)