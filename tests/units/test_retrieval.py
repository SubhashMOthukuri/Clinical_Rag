"""
Unit tests for src/retrieval/retrieval.py

Covers:
- Retriever.__init__: happy path attributes, reranker load failure → RerankerUnavailable,
  default param values, logging on init
- Retriever.retrieve: happy path end-to-end, query string format, embed failure → [],
  pinecone failure → [], all candidates below threshold → [], rerank fallback on failure,
  correlation_id threaded through, no_candidates log, logging on embed/pinecone failure
- Retriever._rerank: top-N selection, highest-score first ordering, fewer candidates
  than rerank_n, scores tied
- Retriever.retrieve_many: empty input → [], parallel tasks, results zipped with evidence,
  partial failures (some pairs return [])
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.retrieval.retrieval import Retriever, RetrievalResult
from src.retrieval.pinecone_store import QueryResult, ChunkMetadata
from src.retrieval.interaction_checker import DrugContext, InteractionEvidence
from src.exceptions.retrieval import RerankerUnavailable
from src.exceptions.embedder import EmbedderUnavailable, EmbedderRateLimited, EmbedderTimeout
from src.exceptions.pinecone import PineconeUnavailable, PineconeTimeout

pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine.*never awaited:RuntimeWarning"
)

# ============================================================================
# Helpers
# ============================================================================

DIM = 768
DUMMY_VECTOR = [0.1] * DIM


def make_metadata(**overrides) -> ChunkMetadata:
    base = dict(
        text="Warfarin and aspirin interaction increases bleeding risk.",
        title="Drug Interactions",
        source="StatPearls",
        article_id="NBK001",
        article_type="general",
        token_count=12,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )
    base.update(overrides)
    return ChunkMetadata(**base)


def make_query_result(id_: str = "chunk_0", score: float = 0.9) -> QueryResult:
    return QueryResult(id=id_, score=score, metadata=make_metadata())


def make_drug_context(name: str = "warfarin") -> DrugContext:
    return DrugContext(
        name=name,
        dose=5.0,
        unit="mg",
        ingredient_rxcui="11289",
        drug_class="anticoagulant",
        fda_label_id="fda-001",
    )


def make_evidence(drug_a: str = "warfarin", drug_b: str = "aspirin") -> InteractionEvidence:
    return InteractionEvidence(
        drug_a=make_drug_context(drug_a),
        drug_b=make_drug_context(drug_b),
        evidence_text="Concurrent use increases bleeding risk.",
        source_drug=drug_a,
        estimated_severity="RED",
    )


def make_embedder(vector: list[float] | None = None) -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed.return_value = vector or DUMMY_VECTOR
    return embedder


def make_store(results: list[QueryResult] | None = None) -> AsyncMock:
    store = AsyncMock()
    store.query.return_value = results if results is not None else [make_query_result()]
    return store


def make_reranker(scores: list[float] | None = None) -> MagicMock:
    reranker = MagicMock()
    reranker.predict.return_value = scores or [0.9]
    return reranker


def make_retriever(
    embedder=None,
    store=None,
    reranker=None,
    retrieve_k: int = 10,
    rerank_n: int = 3,
    score_threshold: float = 0.5,
) -> Retriever:
    """Build a Retriever with mocked CrossEncoder to avoid downloading the model."""
    embedder = embedder or make_embedder()
    store = store or make_store()
    reranker = reranker or make_reranker()

    with patch("src.retrieval.retrieval.CrossEncoder", return_value=reranker):
        retriever = Retriever(
            embedder=embedder,
            store=store,
            retrieve_k=retrieve_k,
            rerank_n=rerank_n,
            score_threshold=score_threshold,
        )
    return retriever


# ============================================================================
# TestInit
# ============================================================================

class TestInit:

    def test_reranker_load_failure_raises(self):
        with patch("src.retrieval.retrieval.CrossEncoder", side_effect=OSError("model not found")):
            with pytest.raises(RerankerUnavailable, match="Failed to load reranker"):
                Retriever(embedder=make_embedder(), store=make_store())

    def test_happy_path_stores_embedder(self):
        embedder = make_embedder()
        r = make_retriever(embedder=embedder)
        assert r._embedder is embedder

    def test_happy_path_stores_store(self):
        store = make_store()
        r = make_retriever(store=store)
        assert r._store is store

    def test_default_namespace_is_statpearls(self):
        r = make_retriever()
        assert r._namespace == "statpearls"

    def test_default_retrieve_k(self):
        r = make_retriever()
        assert r._retrieve_k == 10

    def test_default_rerank_n(self):
        r = make_retriever()
        assert r._rerank_n == 3

    def test_default_score_threshold(self):
        r = make_retriever()
        assert r._score_threshold == 0.5

    def test_custom_params_stored(self):
        r = make_retriever(retrieve_k=20, rerank_n=5, score_threshold=0.7)
        assert r._retrieve_k == 20
        assert r._rerank_n == 5
        assert r._score_threshold == 0.7

    def test_init_logs(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="src.retrieval.retrieval"):
            make_retriever()
        assert "retriever.initialized" in caplog.text


# ============================================================================
# TestRetrieve
# ============================================================================

@pytest.mark.asyncio
class TestRetrieve:

    async def test_happy_path_returns_reranked_chunks(self):
        candidates = [make_query_result("c0", 0.9), make_query_result("c1", 0.8)]
        reranker = make_reranker(scores=[0.8, 0.9])  # c1 scores higher after reranking
        r = make_retriever(
            store=make_store(candidates),
            reranker=reranker,
            rerank_n=2,
        )
        result = await r.retrieve(make_evidence())
        assert len(result) == 2
        assert result[0].id == "c1"  # higher rerank score first

    async def test_query_string_contains_both_drug_names(self):
        embedder = make_embedder()
        r = make_retriever(embedder=embedder)
        evidence = make_evidence("warfarin", "aspirin")
        await r.retrieve(evidence)
        query_arg = embedder.embed.call_args[0][0]
        assert "warfarin" in query_arg
        assert "aspirin" in query_arg

    async def test_query_string_mentions_interaction(self):
        embedder = make_embedder()
        r = make_retriever(embedder=embedder)
        await r.retrieve(make_evidence())
        query_arg = embedder.embed.call_args[0][0]
        assert "interaction" in query_arg.lower()

    async def test_embed_called_with_correlation_id(self):
        embedder = make_embedder()
        r = make_retriever(embedder=embedder)
        await r.retrieve(make_evidence(), correlation_id="cid-123")
        embedder.embed.assert_called_once()
        assert embedder.embed.call_args.kwargs.get("correlation_id") == "cid-123"

    async def test_pinecone_called_with_retrieve_k(self):
        store = make_store()
        r = make_retriever(store=store, retrieve_k=7)
        await r.retrieve(make_evidence())
        store.query.assert_called_once()
        assert store.query.call_args.kwargs["top_k"] == 7

    async def test_pinecone_called_with_namespace(self):
        store = make_store()
        with patch("src.retrieval.retrieval.CrossEncoder", return_value=make_reranker()):
            r = Retriever(
                embedder=make_embedder(),
                store=store,
                namespace="custom-ns",
            )
        await r.retrieve(make_evidence())
        assert store.query.call_args.kwargs["namespace"] == "custom-ns"

    async def test_pinecone_called_with_correlation_id(self):
        store = make_store()
        r = make_retriever(store=store)
        await r.retrieve(make_evidence(), correlation_id="cid-xyz")
        assert store.query.call_args.kwargs["correlation_id"] == "cid-xyz"

    async def test_embed_failure_returns_empty_list(self):
        embedder = make_embedder()
        embedder.embed.side_effect = EmbedderUnavailable("OpenAI down")
        r = make_retriever(embedder=embedder)
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_embed_rate_limit_returns_empty_list(self):
        embedder = make_embedder()
        embedder.embed.side_effect = EmbedderRateLimited(retry_after_s=1.0)
        r = make_retriever(embedder=embedder)
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_embed_timeout_returns_empty_list(self):
        embedder = make_embedder()
        embedder.embed.side_effect = EmbedderTimeout("timed out")
        r = make_retriever(embedder=embedder)
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_embed_failure_logs_warning(self, caplog):
        import logging
        embedder = make_embedder()
        embedder.embed.side_effect = EmbedderUnavailable("OpenAI down")
        r = make_retriever(embedder=embedder)
        with caplog.at_level(logging.WARNING, logger="src.retrieval.retrieval"):
            await r.retrieve(make_evidence())
        assert "retriever.embed_failed" in caplog.text

    async def test_pinecone_failure_returns_empty_list(self):
        store = make_store()
        store.query.side_effect = PineconeUnavailable("down")
        r = make_retriever(store=store)
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_pinecone_timeout_returns_empty_list(self):
        store = make_store()
        store.query.side_effect = PineconeTimeout("timed out")
        r = make_retriever(store=store)
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_pinecone_failure_logs_warning(self, caplog):
        import logging
        store = make_store()
        store.query.side_effect = PineconeUnavailable("down")
        r = make_retriever(store=store)
        with caplog.at_level(logging.WARNING, logger="src.retrieval.retrieval"):
            await r.retrieve(make_evidence())
        assert "retriever.pinecone_failed" in caplog.text

    async def test_all_candidates_below_threshold_returns_empty(self):
        candidates = [make_query_result("c0", 0.3), make_query_result("c1", 0.2)]
        r = make_retriever(store=make_store(candidates), score_threshold=0.5)
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_all_candidates_below_threshold_logs(self, caplog):
        import logging
        candidates = [make_query_result("c0", 0.3)]
        r = make_retriever(store=make_store(candidates), score_threshold=0.5)
        with caplog.at_level(logging.INFO, logger="src.retrieval.retrieval"):
            await r.retrieve(make_evidence())
        assert "retriever.no_candidates_above_threshold" in caplog.text

    async def test_candidates_filtered_by_threshold(self):
        candidates = [
            make_query_result("pass", 0.8),
            make_query_result("fail", 0.3),
            make_query_result("pass2", 0.6),
        ]
        reranker = make_reranker(scores=[0.9, 0.7])
        r = make_retriever(store=make_store(candidates), reranker=reranker, score_threshold=0.5, rerank_n=5)
        result = await r.retrieve(make_evidence())
        ids = [c.id for c in result]
        assert "fail" not in ids
        assert "pass" in ids
        assert "pass2" in ids

    async def test_rerank_failure_falls_back_to_raw_order(self):
        candidates = [
            make_query_result("c0", 0.9),
            make_query_result("c1", 0.8),
            make_query_result("c2", 0.7),
        ]
        reranker = make_reranker()
        reranker.predict.side_effect = RuntimeError("model crash")
        r = make_retriever(store=make_store(candidates), reranker=reranker, rerank_n=2)
        result = await r.retrieve(make_evidence())
        # Falls back to candidates[:rerank_n] — first 2 in original order
        assert len(result) == 2
        assert result[0].id == "c0"

    async def test_rerank_failure_logs_warning(self, caplog):
        import logging
        candidates = [make_query_result("c0", 0.9)]
        reranker = make_reranker()
        reranker.predict.side_effect = RuntimeError("model crash")
        r = make_retriever(store=make_store(candidates), reranker=reranker)
        with caplog.at_level(logging.WARNING, logger="src.retrieval.retrieval"):
            await r.retrieve(make_evidence())
        assert "retriever.rerank_failed" in caplog.text

    async def test_pinecone_not_called_when_embed_fails(self):
        embedder = make_embedder()
        embedder.embed.side_effect = EmbedderUnavailable("down")
        store = make_store()
        r = make_retriever(embedder=embedder, store=store)
        await r.retrieve(make_evidence())
        store.query.assert_not_called()

    async def test_reranker_not_called_when_pinecone_fails(self):
        store = make_store()
        store.query.side_effect = PineconeUnavailable("down")
        reranker = make_reranker()
        r = make_retriever(store=store, reranker=reranker)
        await r.retrieve(make_evidence())
        reranker.predict.assert_not_called()

    async def test_empty_pinecone_results_returns_empty(self):
        r = make_retriever(store=make_store([]))
        result = await r.retrieve(make_evidence())
        assert result == []

    async def test_result_capped_at_rerank_n(self):
        candidates = [make_query_result(f"c{i}", 0.9 - i * 0.01) for i in range(8)]
        reranker = make_reranker(scores=[0.9 - i * 0.01 for i in range(8)])
        r = make_retriever(store=make_store(candidates), reranker=reranker, rerank_n=3)
        result = await r.retrieve(make_evidence())
        assert len(result) == 3


# ============================================================================
# TestRerank
# ============================================================================

@pytest.mark.asyncio
class TestRerank:

    async def test_highest_score_first(self):
        candidates = [
            make_query_result("low", 0.9),
            make_query_result("high", 0.9),
        ]
        reranker = make_reranker(scores=[0.3, 0.9])  # "high" gets 0.9 from reranker
        r = make_retriever(reranker=reranker, rerank_n=2)
        result = await r._rerank("test query", candidates)
        assert result[0].id == "high"

    async def test_top_n_returned(self):
        candidates = [make_query_result(f"c{i}", 0.9) for i in range(6)]
        reranker = make_reranker(scores=[float(i) for i in range(6)])
        r = make_retriever(reranker=reranker, rerank_n=3)
        result = await r._rerank("test query", candidates)
        assert len(result) == 3

    async def test_fewer_candidates_than_rerank_n_returns_all(self):
        candidates = [make_query_result("c0", 0.9)]
        reranker = make_reranker(scores=[0.8])
        r = make_retriever(reranker=reranker, rerank_n=5)
        result = await r._rerank("test query", candidates)
        assert len(result) == 1

    async def test_pairs_built_from_query_and_metadata_text(self):
        candidates = [make_query_result("c0", 0.9)]
        reranker = make_reranker(scores=[0.8])
        r = make_retriever(reranker=reranker, rerank_n=1)
        await r._rerank("my query", candidates)
        pairs = reranker.predict.call_args[0][0]
        assert pairs[0][0] == "my query"
        assert pairs[0][1] == candidates[0].metadata.text

    async def test_reranker_called_via_to_thread(self):
        candidates = [make_query_result("c0", 0.9)]
        reranker = make_reranker(scores=[0.8])
        r = make_retriever(reranker=reranker, rerank_n=1)
        with patch("src.retrieval.retrieval.asyncio.to_thread", wraps=asyncio.to_thread) as mock_thread:
            await r._rerank("query", candidates)
        mock_thread.assert_called_once()


# ============================================================================
# TestRetrieveMany
# ============================================================================

@pytest.mark.asyncio
class TestRetrieveMany:

    async def test_empty_input_returns_empty_list(self):
        r = make_retriever()
        result = await r.retrieve_many([])
        assert result == []

    async def test_returns_retrieval_result_instances(self):
        evidence = [make_evidence("warfarin", "aspirin")]
        r = make_retriever(store=make_store([make_query_result()]))
        results = await r.retrieve_many(evidence)
        assert all(isinstance(res, RetrievalResult) for res in results)

    async def test_evidence_zipped_with_chunks(self):
        ev = make_evidence("warfarin", "aspirin")
        chunk = make_query_result("c0", 0.9)
        r = make_retriever(store=make_store([chunk]))
        results = await r.retrieve_many([ev])
        assert results[0].evidence is ev
        assert results[0].chunks[0].id == "c0"

    async def test_multiple_evidences_all_processed(self):
        evidences = [
            make_evidence("warfarin", "aspirin"),
            make_evidence("metformin", "ibuprofen"),
        ]
        r = make_retriever(store=make_store([make_query_result()]))
        results = await r.retrieve_many(evidences)
        assert len(results) == 2

    async def test_correlation_id_passed_to_each_retrieve(self):
        evidences = [make_evidence(), make_evidence("metformin", "ibuprofen")]
        embedder = make_embedder()
        r = make_retriever(embedder=embedder)
        await r.retrieve_many(evidences, correlation_id="batch-cid")
        for call_ in embedder.embed.call_args_list:
            assert call_.kwargs.get("correlation_id") == "batch-cid"

    async def test_partial_failure_returns_empty_chunks_for_failed_pair(self):
        evidences = [make_evidence("warfarin", "aspirin"), make_evidence("x", "y")]
        call_count = 0
        original_retrieve = Retriever.retrieve

        async def side_effect(self_inner, ev, *, correlation_id=None):
            nonlocal call_count
            call_count += 1
            if ev.drug_a.name == "x":
                return []
            return [make_query_result()]

        r = make_retriever()
        with patch.object(Retriever, "retrieve", side_effect=side_effect, autospec=True):
            results = await r.retrieve_many(evidences)

        assert len(results) == 2
        assert len(results[0].chunks) == 1
        assert results[1].chunks == []

    async def test_runs_in_parallel_via_gather(self):
        evidences = [make_evidence(), make_evidence("x", "y")]
        r = make_retriever()
        with patch("src.retrieval.retrieval.asyncio.gather", wraps=asyncio.gather) as mock_gather:
            await r.retrieve_many(evidences)
        mock_gather.assert_called_once()
        assert len(mock_gather.call_args[0]) == 2

    async def test_output_order_matches_input_order(self):
        drug_pairs = [("warfarin", "aspirin"), ("metformin", "ibuprofen"), ("lisinopril", "potassium")]
        evidences = [make_evidence(a, b) for a, b in drug_pairs]
        r = make_retriever(store=make_store([make_query_result()]))
        results = await r.retrieve_many(evidences)
        for i, (a, b) in enumerate(drug_pairs):
            assert results[i].evidence.drug_a.name == a
            assert results[i].evidence.drug_b.name == b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
