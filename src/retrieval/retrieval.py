from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence
from sentence_transformers import CrossEncoder

from src.embedding.embedder import Embedder
from src.retrieval.pinecone_store import VectorStore, QueryResult
from src.retrieval.interaction_checker import InteractionEvidence
from src.exceptions.retrieval import (RerankerUnavailable,RetrievalError)
from src.exceptions.embedder import EmbedderError
from src.exceptions.pinecone import PineconeStoreError

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RetrievalResult:
    """One durg pair's intearction evidence + its retrieved clinical chunks."""
    evidence: InteractionEvidence
    chunks: list[QueryResult]

class Retriever:
    def __init__(
            self,
            embedder: Embedder,
            store: VectorStore,
            namespace: str ="statpearls",
            retrieve_k: int = 10,
            rerank_n: int = 3,
            score_threshold: float = 0.5,
            reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self._embedder = embedder
        self._store = store

        self._namespace = namespace
        self._retrieve_k = retrieve_k
        self._rerank_n = rerank_n
        self._score_threshold = score_threshold

        try:
            self._reranker = CrossEncoder(reranker_model)
        except Exception as e:
            raise RerankerUnavailable(f"Failed to load reranker: {e}") from e
        
        logger.info(
            "retriever.initialized",
            extra={"namespace": namespace,
                   "retrieve_k": retrieve_k,
                   "rerank_n": rerank_n,
                   "threshold": score_threshold
                   },
        )
    async def retrieve(
            self,
            evidence: InteractionEvidence,
            *, 
            correlation_id: str | None = None,
    ) -> list[QueryResult]:
        """Retrieve reranked clinical chunks for ONE drug pair."""
        log_ctx = {"cid": correlation_id, "pair": f"{evidence.drug_a.name} | {evidence.drug_b.name}"}


        query = (
            f"{evidence.drug_a.name} {evidence.drug_b.name} " 
            f"drug interaction mechanism clinical management."
            )
        try:
            vector = await self._embedder.embed(query, correlation_id=correlation_id)

        except EmbedderError as e:
            logger.warning("retriever.embed_failed", extra = {**log_ctx, "error": str(e)})
            return []
        
        try:
            candidates = await self._store.query(
                query_vector = vector, 
                top_k = self._retrieve_k,
                namespace=self._namespace,
                correlation_id = correlation_id
            )
        except PineconeStoreError as e:
            logger.warning("retriever.pinecone_failed", extra={**log_ctx, "error": str(e)})
            return []
        
        candidates = [c for c in candidates if c.score >= self._score_threshold]
        if not candidates:
            logger.info("retriever.no_candidates_above_threshold", extra=log_ctx)
            return []
        try:
            ranked = await self._rerank(query, candidates)
        except Exception as e:
            logger.warning("retriever.rerank_failed", extra= {**log_ctx, "error": str(e)})
            ranked = candidates[:self._rerank_n]
        return ranked
    
    async def _rerank(self, query:str, candidates: list[QueryResult])-> list[QueryResult]:
        """Score candidates with cross-encoder, return top N. Runs model in thread."""
        pairs = [[query, c.metadata.text] for c in candidates]
        scores = await asyncio.to_thread(self._reranker.predict, pairs)
        scored = sorted(zip(candidates, scores), key = lambda x: x[1], reverse =True)
        return [c for c, _ in scored[:self._rerank_n]]
    
    async def retrieve_many(self, evidences: Sequence[InteractionEvidence], *, correlation_id : str | None = None, )-> list[RetrievalResult]:
        """Retrieve chunks for many durgs paris in PARALLEL."""
        if not evidences: 
            return []
        
        tasks = [self.retrieve(ev, correlation_id=correlation_id) for ev in evidences]
        results = await asyncio.gather(*tasks)
        return [
            RetrievalResult(evidence = ev, chunks = chunks) 
            for ev, chunks in zip(evidences, results)
        ]
        

# ============================================================================
# PHASE 2 TODOs — Quality & Observability
# ============================================================================
# [ ] Concurrency semaphore — cap parallel retrieve() calls to avoid
#     embedder/pinecone rate-limit storms under high drug-pair counts
# [ ] Metrics (Prometheus):
#       - retriever_latency_seconds (histogram, p50/p95/p99)
#       - retriever_chunks_returned (histogram — detect frequent empty results)
#       - retriever_empty_results_total (how often falling back to FDA-only)
#       - retriever_rerank_failures_total
# [ ] Hybrid search — add BM25 keyword retrieval alongside semantic,
#     merge with Reciprocal Rank Fusion (RRF) to catch exact drug-name matches
# [ ] Swap reranker for a biomedical cross-encoder (ms-marco is general web)
# [ ] Redis cache for common drug-pair retrievals (short TTL, e.g. 1hr)
# [ ] Query expansion — generate query variations, merge results
# [ ] Build eval set (50-100 labeled pairs) + measure Recall@K, MRR
#     BEFORE tuning score_threshold, retrieve_k, rerank_n

# ============================================================================
# PHASE 3 TODOs — Production Hardening
# ============================================================================
# [ ] Move reranker to dedicated inference service (Triton/TorchServe) —
#     frees per-pod memory, scales reranking independently
# [ ] OpenTelemetry spans around embed -> query -> rerank for latency tracing
# [ ] Circuit breaker on reranker service — trip + fall back to Pinecone order
# [ ] Adaptive retrieve_k based on how many candidates pass score_threshold
# [ ] MMR (Maximal Marginal Relevance) — diverse top-N, not near-duplicate chunks
# [ ] Reranker A/B testing in shadow mode — promote winner by downstream answer quality
# [ ] Tune score_threshold from eval data (currently 0.5 placeholder)

# ============================================================================
# KNOWN LIMITATIONS (MVP)
# ============================================================================
# - Pure semantic search; no keyword/hybrid (misses some exact-name matches)
# - General-domain reranker (not biomedical-tuned)
# - score_threshold=0.5 is a placeholder, not validated against an eval set
# - No caching — repeated drug pairs re-query every time
# - No concurrency cap — many pairs can spike embedder/pinecone load
    