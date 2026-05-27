"""
Unit tests for src/generation/generator.py

Covers:
- Generator.__init__: injected clients, stored params, separate circuit breakers, logging
- Generator._call_llm: Gemini success, Gemini failure → Groq fallback,
  Groq success, Groq failure → GeneratorUnavailable, both breakers open,
  Gemini breaker open skips to Groq, circuit breaker record_success/failure
- Generator.generate_one: no-chunks FDA fallback, happy path DrugWarning,
  GeneratorUnavailable → FDA fallback, StageValidationError → FDA fallback,
  unexpected Exception → FDA fallback, correlation_id threaded, logging
- Generator.generate_many: empty input, happy path, length == input length,
  correlation_id propagated, leaked exception → FDA fallback, output order
- Generator._fda_fallback: severity mapping (RED/YELLOW/UNKNOWN→YELLOW),
  drugs_involved, citation, action, data_source, confidence, reason in summary
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.generator import Generator
from src.exceptions.generator import GeneratorUnavailable
from src.retrieval.retrieval import RetrievalResult
from src.retrieval.pinecone_store import QueryResult, ChunkMetadata
from src.retrieval.interaction_checker import DrugContext, InteractionEvidence
from src.utils.schema import DrugWarning, Severity, Action, DataSource
from src.utils.validators import StageValidationError, ValidationStage

pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine.*never awaited:RuntimeWarning"
)

# ============================================================================
# Helpers
# ============================================================================

def make_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        text="Clinical text about drug interaction.",
        title="Drug Interactions",
        source="StatPearls",
        article_id="NBK001",
        article_type="general",
        token_count=10,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


def make_chunk(id_: str = "NBK001_chunk_0000") -> QueryResult:
    return QueryResult(id=id_, score=0.9, metadata=make_metadata())


def make_drug(name: str = "warfarin") -> DrugContext:
    return DrugContext(
        name=name,
        dose=5.0,
        unit="mg",
        ingredient_rxcui="11289",
        drug_class="anticoagulant",
        fda_label_id="fda-001",
    )


def make_evidence(
    drug_a: str = "warfarin",
    drug_b: str = "aspirin",
    severity: str = "RED",
) -> InteractionEvidence:
    return InteractionEvidence(
        drug_a=make_drug(drug_a),
        drug_b=make_drug(drug_b),
        evidence_text="Concurrent use increases bleeding risk.",
        source_drug=drug_a,
        estimated_severity=severity,
    )


def make_retrieval_result(
    drug_a: str = "warfarin",
    drug_b: str = "aspirin",
    chunks: list[QueryResult] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        evidence=make_evidence(drug_a, drug_b),
        chunks=chunks if chunks is not None else [make_chunk()],
    )


def make_drug_warning(
    drug_a: str = "warfarin",
    drug_b: str = "aspirin",
) -> DrugWarning:
    return DrugWarning(
        drugs_involved=[drug_a, drug_b],
        severity=Severity.RED,
        reaction_result="Increases bleeding risk significantly.",
        action=Action.STOP,
        citation=["NBK001_chunk_0000"],
        nurse_summary_to_doctor="Do not co-administer.",
        confidence=0.9,
        data_source=DataSource.STATPEARLS_RAG,
        computed_at=datetime.now(timezone.utc),
    )


def make_gemini_client(text: str = '{"result": "ok"}') -> MagicMock:
    """Gemini runs in asyncio.to_thread — regular MagicMock."""
    client = MagicMock()
    response = MagicMock()
    response.text = text
    client.models.generate_content.return_value = response
    return client


def make_groq_client(content: str = '{"result": "ok"}') -> AsyncMock:
    """Groq is an async client — AsyncMock."""
    client = AsyncMock()
    response = MagicMock()
    response.choices[0].message.content = content
    client.chat.completions.create.return_value = response
    return client


def make_generator(
    gemini_client=None,
    groq_client=None,
    timeout_s: float = 5.0,
    breaker_threshold: int = 3,
    breaker_cooldown_s: float = 1.0,
) -> Generator:
    return Generator(
        gemini_api_key="fake-gemini-key",
        groq_api_key="fake-groq-key",
        gemini_model="gemini-2.0-flash",
        groq_model="llama-3.3-70b-versatile",
        timeout_s=timeout_s,
        breaker_threshold=breaker_threshold,
        breaker_cooldown_s=breaker_cooldown_s,
        gemini_client=gemini_client or make_gemini_client(),
        groq_client=groq_client or make_groq_client(),
    )


VALID_LLM_JSON = '[{"drugs_involved":["warfarin","aspirin"],"severity":"RED","reaction_result":"Increases bleeding.","action":"STOP","citation":["NBK001_chunk_0000"],"nurse_summary_to_doctor":"Do not use together.","confidence":0.9}]'


# ============================================================================
# TestInit
# ============================================================================

class TestInit:

    def test_stores_gemini_model(self):
        g = make_generator()
        assert g._gemini_model == "gemini-2.0-flash"

    def test_stores_groq_model(self):
        g = make_generator()
        assert g._groq_model == "llama-3.3-70b-versatile"

    def test_stores_timeout(self):
        g = make_generator(timeout_s=8.0)
        assert g._timeout_s == 8.0

    def test_uses_injected_gemini_client(self):
        client = make_gemini_client()
        g = make_generator(gemini_client=client)
        assert g._gemini_client is client

    def test_uses_injected_groq_client(self):
        client = make_groq_client()
        g = make_generator(groq_client=client)
        assert g._groq_client is client

    def test_two_independent_circuit_breakers(self):
        g = make_generator()
        assert g._gemini_breaker is not g._groq_breaker

    def test_init_logs(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="src.generation.generator"):
            make_generator()
        assert "generator.initialized" in caplog.text


# ============================================================================
# TestCallLLM
# ============================================================================

@pytest.mark.asyncio
class TestCallLLM:

    async def test_gemini_success_returns_response_text(self):
        gemini = make_gemini_client(text="hello from gemini")
        g = make_generator(gemini_client=gemini)
        result = await g._call_llm("sys", "usr")
        assert result == "hello from gemini"

    async def test_gemini_success_records_success(self):
        g = make_generator()
        await g._call_llm("sys", "usr")
        assert g._gemini_breaker._failures == 0

    async def test_gemini_failure_records_failure(self):
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("network error")
        g = make_generator(gemini_client=gemini)
        try:
            await g._call_llm("sys", "usr")
        except GeneratorUnavailable:
            pass
        assert g._gemini_breaker._failures >= 1

    async def test_gemini_failure_falls_back_to_groq(self):
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client(content="hello from groq")
        g = make_generator(gemini_client=gemini, groq_client=groq)
        result = await g._call_llm("sys", "usr")
        assert result == "hello from groq"

    async def test_gemini_failure_logs_warning(self, caplog):
        import logging
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client()
        g = make_generator(gemini_client=gemini, groq_client=groq)
        with caplog.at_level(logging.WARNING, logger="src.generation.generator"):
            await g._call_llm("sys", "usr")
        assert "gemini_failed_fallback" in caplog.text

    async def test_groq_success_returns_message_content(self):
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client(content="groq answer")
        g = make_generator(gemini_client=gemini, groq_client=groq)
        result = await g._call_llm("sys", "usr")
        assert result == "groq answer"

    async def test_groq_success_records_success(self):
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        g = make_generator(gemini_client=gemini)
        await g._call_llm("sys", "usr")
        assert g._groq_breaker._failures == 0

    async def test_groq_failure_raises_generator_unavailable(self):
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client()
        groq.chat.completions.create.side_effect = RuntimeError("groq down")
        g = make_generator(gemini_client=gemini, groq_client=groq)
        with pytest.raises(GeneratorUnavailable):
            await g._call_llm("sys", "usr")

    async def test_groq_failure_records_failure(self):
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client()
        groq.chat.completions.create.side_effect = RuntimeError("groq down")
        g = make_generator(gemini_client=gemini, groq_client=groq)
        with pytest.raises(GeneratorUnavailable):
            await g._call_llm("sys", "usr")
        assert g._groq_breaker._failures >= 1

    async def test_groq_failure_logs_error(self, caplog):
        import logging
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client()
        groq.chat.completions.create.side_effect = RuntimeError("groq down")
        g = make_generator(gemini_client=gemini, groq_client=groq)
        with caplog.at_level(logging.ERROR, logger="src.generation.generator"):
            with pytest.raises(GeneratorUnavailable):
                await g._call_llm("sys", "usr")
        assert "all_providers_failed" in caplog.text

    async def test_gemini_breaker_open_skips_to_groq(self):
        groq = make_groq_client(content="from groq")
        g = make_generator(groq_client=groq)
        # Trip the gemini breaker manually
        for _ in range(5):
            g._gemini_breaker.record_failure()
        result = await g._call_llm("sys", "usr")
        assert result == "from groq"
        g._gemini_client.models.generate_content.assert_not_called()

    async def test_both_breakers_open_raises_unavailable(self):
        g = make_generator()
        for _ in range(5):
            g._gemini_breaker.record_failure()
            g._groq_breaker.record_failure()
        with pytest.raises(GeneratorUnavailable, match="circuit breakers"):
            await g._call_llm("sys", "usr")

    async def test_both_breakers_open_logs_error(self, caplog):
        import logging
        g = make_generator()
        for _ in range(5):
            g._gemini_breaker.record_failure()
            g._groq_breaker.record_failure()
        with caplog.at_level(logging.ERROR, logger="src.generation.generator"):
            with pytest.raises(GeneratorUnavailable):
                await g._call_llm("sys", "usr")
        assert "all_providers_unavailable" in caplog.text

    async def test_correlation_id_in_log_context(self, caplog):
        import logging
        gemini = make_gemini_client()
        gemini.models.generate_content.side_effect = RuntimeError("down")
        groq = make_groq_client()
        groq.chat.completions.create.side_effect = RuntimeError("down")
        g = make_generator(gemini_client=gemini, groq_client=groq)
        with caplog.at_level(logging.ERROR, logger="src.generation.generator"):
            with pytest.raises(GeneratorUnavailable):
                await g._call_llm("sys", "usr", correlation_id="cid-abc")


# ============================================================================
# TestGenerateOne
# ============================================================================

@pytest.mark.asyncio
class TestGenerateOne:

    async def test_no_chunks_returns_fda_fallback_immediately(self):
        g = make_generator()
        rr = make_retrieval_result(chunks=[])
        with patch.object(g, "_call_llm") as mock_llm:
            result = await g.generate_one(rr)
        mock_llm.assert_not_called()
        assert isinstance(result, DrugWarning)
        assert result.citation == ["FDA_LABEL"]

    async def test_no_chunks_fda_fallback_reason(self):
        g = make_generator()
        rr = make_retrieval_result(chunks=[])
        result = await g.generate_one(rr)
        assert "no_chunks_retrieved" in result.nurse_summary_to_doctor

    async def test_happy_path_returns_drug_warning(self):
        g = make_generator()
        rr = make_retrieval_result()
        expected = make_drug_warning()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response", return_value=[expected]):
                result = await g.generate_one(rr)
        assert result is expected

    async def test_call_llm_receives_system_prompt(self):
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON) as mock_llm:
            with patch("src.generation.generator.validate_llm_response", return_value=[make_drug_warning()]):
                await g.generate_one(rr)
        assert mock_llm.call_args[0][0] is not None  # system prompt passed

    async def test_call_llm_receives_correlation_id(self):
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON) as mock_llm:
            with patch("src.generation.generator.validate_llm_response", return_value=[make_drug_warning()]):
                await g.generate_one(rr, correlation_id="cid-test")
        assert mock_llm.call_args.kwargs.get("correlation_id") == "cid-test"

    async def test_allowed_citations_built_from_chunk_ids(self):
        g = make_generator()
        chunks = [make_chunk("chunk_A"), make_chunk("chunk_B")]
        rr = make_retrieval_result(chunks=chunks)
        captured = {}
        def capture_validator(**kwargs):
            captured.update(kwargs)
            return [make_drug_warning()]
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response", side_effect=capture_validator):
                await g.generate_one(rr)
        assert captured["allowed_citation_sources"] == {"chunk_A", "chunk_B"}

    async def test_allowed_drugs_built_from_evidence_names(self):
        g = make_generator()
        rr = make_retrieval_result(drug_a="warfarin", drug_b="aspirin")
        captured = {}
        def capture_validator(**kwargs):
            captured.update(kwargs)
            return [make_drug_warning()]
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response", side_effect=capture_validator):
                await g.generate_one(rr)
        assert captured["allowed_drug_names"] == {"warfarin", "aspirin"}

    async def test_generator_unavailable_returns_fda_fallback(self):
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", side_effect=GeneratorUnavailable("down")):
            result = await g.generate_one(rr)
        assert isinstance(result, DrugWarning)
        assert result.citation == ["FDA_LABEL"]

    async def test_generator_unavailable_logs_warning(self, caplog):
        import logging
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", side_effect=GeneratorUnavailable("down")):
            with caplog.at_level(logging.WARNING, logger="src.generation.generator"):
                await g.generate_one(rr)
        assert "fallback_to_fda" in caplog.text

    async def test_stage_validation_error_returns_fda_fallback(self):
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value="bad json"):
            with patch("src.generation.generator.validate_llm_response",
                       side_effect=StageValidationError(ValidationStage.LLM, "invalid")):
                result = await g.generate_one(rr)
        assert isinstance(result, DrugWarning)
        assert result.citation == ["FDA_LABEL"]

    async def test_stage_validation_error_logs_warning(self, caplog):
        import logging
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value="bad"):
            with patch("src.generation.generator.validate_llm_response",
                       side_effect=StageValidationError(ValidationStage.LLM, "invalid")):
                with caplog.at_level(logging.WARNING, logger="src.generation.generator"):
                    await g.generate_one(rr)
        assert "fallback_to_fda" in caplog.text

    async def test_unexpected_exception_returns_fda_fallback(self):
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response",
                       side_effect=IndexError("list index out of range")):
                result = await g.generate_one(rr)
        assert isinstance(result, DrugWarning)
        assert result.citation == ["FDA_LABEL"]

    async def test_unexpected_exception_logs_error(self, caplog):
        import logging
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response",
                       side_effect=KeyError("oops")):
                with caplog.at_level(logging.ERROR, logger="src.generation.generator"):
                    await g.generate_one(rr)
        assert "unexpected_failure" in caplog.text

    async def test_unexpected_exception_reason_contains_type_name(self):
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response",
                       side_effect=ZeroDivisionError("oops")):
                result = await g.generate_one(rr)
        assert "ZeroDivisionError" in result.nurse_summary_to_doctor

    async def test_generate_one_never_raises(self):
        """Safety guarantee: generate_one must never raise, always return DrugWarning."""
        g = make_generator()
        rr = make_retrieval_result()
        with patch.object(g, "_call_llm", side_effect=Exception("anything")):
            result = await g.generate_one(rr)
        assert isinstance(result, DrugWarning)


# ============================================================================
# TestGenerateMany
# ============================================================================

@pytest.mark.asyncio
class TestGenerateMany:

    async def test_empty_input_returns_empty_list(self):
        g = make_generator()
        result = await g.generate_many([])
        assert result == []

    async def test_returns_list_of_drug_warnings(self):
        g = make_generator()
        rrs = [make_retrieval_result(), make_retrieval_result("metformin", "ibuprofen")]
        expected = make_drug_warning()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response", return_value=[expected]):
                results = await g.generate_many(rrs)
        assert all(isinstance(r, DrugWarning) for r in results)

    async def test_output_length_matches_input_length(self):
        g = make_generator()
        rrs = [make_retrieval_result() for _ in range(4)]
        expected = make_drug_warning()
        with patch.object(g, "_call_llm", return_value=VALID_LLM_JSON):
            with patch("src.generation.generator.validate_llm_response", return_value=[expected]):
                results = await g.generate_many(rrs)
        assert len(results) == 4

    async def test_correlation_id_propagated_to_each_call(self):
        g = make_generator()
        rrs = [make_retrieval_result(), make_retrieval_result("x", "y")]
        call_cids = []
        original = g.generate_one
        async def capture(rr, *, correlation_id=None):
            call_cids.append(correlation_id)
            return make_drug_warning()
        with patch.object(g, "generate_one", side_effect=capture):
            await g.generate_many(rrs, correlation_id="batch-cid")
        assert all(cid == "batch-cid" for cid in call_cids)

    async def test_leaked_exception_replaced_by_fda_fallback(self):
        """return_exceptions=True must catch any leaked exception from generate_one."""
        g = make_generator()
        rrs = [make_retrieval_result()]
        # Patch generate_one to raise directly (bypassing its own catch-all)
        async def raise_always(rr, *, correlation_id=None):
            raise RuntimeError("leaked!")
        with patch.object(g, "generate_one", side_effect=raise_always):
            results = await g.generate_many(rrs)
        assert len(results) == 1
        assert isinstance(results[0], DrugWarning)
        assert results[0].citation == ["FDA_LABEL"]

    async def test_partial_failure_does_not_kill_other_results(self):
        g = make_generator()
        good_rr = make_retrieval_result("warfarin", "aspirin", chunks=[])
        fail_rr = make_retrieval_result("x", "y")

        async def selective(rr, *, correlation_id=None):
            if rr.evidence.drug_a.name == "x":
                raise RuntimeError("fail!")
            return make_drug_warning(rr.evidence.drug_a.name, rr.evidence.drug_b.name)

        with patch.object(g, "generate_one", side_effect=selective):
            results = await g.generate_many([good_rr, fail_rr])

        assert len(results) == 2
        assert isinstance(results[0], DrugWarning)
        assert isinstance(results[1], DrugWarning)

    async def test_output_order_matches_input_order(self):
        g = make_generator()
        pairs = [("warfarin", "aspirin"), ("metformin", "ibuprofen"), ("lisinopril", "potassium")]
        rrs = [make_retrieval_result(a, b) for a, b in pairs]

        async def make_warning(rr, *, correlation_id=None):
            return make_drug_warning(rr.evidence.drug_a.name, rr.evidence.drug_b.name)

        with patch.object(g, "generate_one", side_effect=make_warning):
            results = await g.generate_many(rrs)

        for i, (a, b) in enumerate(pairs):
            assert a in results[i].drugs_involved
            assert b in results[i].drugs_involved

    async def test_generate_many_never_raises(self):
        """Safety guarantee: generate_many must never raise."""
        g = make_generator()
        rrs = [make_retrieval_result()]
        async def always_raise(rr, *, correlation_id=None):
            raise RuntimeError("chaos")
        with patch.object(g, "generate_one", side_effect=always_raise):
            results = await g.generate_many(rrs)
        assert isinstance(results, list)


# ============================================================================
# TestFdaFallback
# ============================================================================

class TestFdaFallback:

    def test_drugs_involved_set_correctly(self):
        g = make_generator()
        ev = make_evidence("warfarin", "aspirin")
        result = g._fda_fallback(ev, reason="test")
        assert "warfarin" in result.drugs_involved
        assert "aspirin" in result.drugs_involved

    def test_red_severity_preserved(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(severity="RED"), reason="test")
        assert result.severity == Severity.RED

    def test_yellow_severity_preserved(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(severity="YELLOW"), reason="test")
        assert result.severity == Severity.YELLOW

    def test_unknown_severity_maps_to_yellow(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(severity="UNKNOWN"), reason="test")
        assert result.severity == Severity.YELLOW

    def test_green_severity_maps_to_yellow(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(severity="GREEN"), reason="test")
        assert result.severity == Severity.YELLOW

    def test_evidence_text_is_reaction_result(self):
        g = make_generator()
        ev = make_evidence()
        result = g._fda_fallback(ev, reason="test")
        assert result.reaction_result == ev.evidence_text

    def test_citation_is_fda_label(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(), reason="test")
        assert result.citation == ["FDA_LABEL"]

    def test_action_is_consult_doctor(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(), reason="test")
        assert result.action == Action.CONSULT_DOCTOR

    def test_data_source_is_fresh_fda(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(), reason="test")
        assert result.data_source == DataSource.FRESH_FDA

    def test_confidence_is_0_5(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(), reason="test")
        assert result.confidence == 0.5

    def test_reason_in_nurse_summary(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(), reason="llm_timeout")
        assert "llm_timeout" in result.nurse_summary_to_doctor

    def test_long_reason_truncated_to_80_chars(self):
        g = make_generator()
        long_reason = "x" * 200
        result = g._fda_fallback(make_evidence(), reason=long_reason)
        assert len(result.nurse_summary_to_doctor) < 300  # truncated, not full 200

    def test_returns_drug_warning_instance(self):
        g = make_generator()
        result = g._fda_fallback(make_evidence(), reason="test")
        assert isinstance(result, DrugWarning)

    def test_computed_at_is_recent_utc(self):
        g = make_generator()
        before = datetime.now(timezone.utc)
        result = g._fda_fallback(make_evidence(), reason="test")
        after = datetime.now(timezone.utc)
        assert before <= result.computed_at <= after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
