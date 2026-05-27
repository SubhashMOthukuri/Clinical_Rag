"""LLM gateway + clinical interaction generator.

Takes RetrievalResult (evidence + chunks), produces a validated DrugWarning.
Uses a two-provider gateway (Gemini → Groq) with independent circuit breakers.
On any failure (LLM down, malformed output, hallucinated citation), falls back
to a degraded FDA-only DrugWarning so the nurse always gets a response.

Safety guarantees:
- generate_one NEVER raises — always returns a DrugWarning
- generate_many NEVER raises — always returns list[DrugWarning], length = input length
- Citation/drug hallucinations rejected by validators.validate_llm_response
- Both LLM providers tracked with independent CircuitBreakers
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from google import genai
from groq import AsyncGroq

from src.resilience.circuit_breaker import CircuitBreaker
from src.retrieval.retrieval import RetrievalResult
from src.retrieval.interaction_checker import InteractionEvidence
from src.exceptions.generator import (GeneratorUnavailable)
from src.utils.validators import validate_llm_response, StageValidationError
from src.generation.prompt_template import SYSTEM_PROMPT, build_user_prompt
from datetime import datetime, timezone
from src.utils.schema import DrugWarning, Severity, Action, DataSource

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, gemini_api_key : str, groq_api_key: str, gemini_model : str, groq_model : str, timeout_s : float = 5.0, breaker_threshold: int = 3, breaker_cooldown_s: float = 30.0, gemini_client= None, groq_client= None):
        self._gemini_client = gemini_client or genai.Client(api_key=gemini_api_key)
        self._groq_client = groq_client or AsyncGroq(api_key=groq_api_key)
        self._gemini_model = gemini_model
        self._groq_model = groq_model
        self._timeout_s = timeout_s
        self._gemini_breaker = CircuitBreaker(
            threshold=breaker_threshold, cooldown_s=breaker_cooldown_s
        )
        self._groq_breaker = CircuitBreaker(
            threshold = breaker_threshold, cooldown_s=breaker_cooldown_s
        )

        logger.info(
            "generator.initialized", 
            extra ={"gemini_model": gemini_model, "groq_model": groq_model},
        )
    async def _call_llm(self, system_prompt: str, user_prompt: str, *, correlation_id : str | None = None)-> str:
        log_ctx = {"cid": correlation_id}
        if not  self._gemini_breaker.is_open():
            try: 
                response= await asyncio.wait_for(asyncio.to_thread(
                    self._gemini_client.models.generate_content,
                    model= self._gemini_model,
                    contents = user_prompt,
                    config={
                        "system_instruction": system_prompt,
                        "temperature": 0
                    },
                ), timeout=self._timeout_s
                )
                self._gemini_breaker.record_success()
                return response.text
            except Exception as e:
                self._gemini_breaker.record_failure()
                logger.warning("generator.gemini_failed_fallback", extra={**log_ctx, "error": str(e)},
                )
        if not self._groq_breaker.is_open():
            try:
                response = await asyncio.wait_for(
                    self._groq_client.chat.completions.create(
                        model= self._groq_model,
                        messages= [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ], temperature=0
                    ), timeout =self._timeout_s,
                )
                self._groq_breaker.record_success()
                return response.choices[0].message.content
            except Exception as e:
                self._groq_breaker.record_failure()
                logger.error("generator.all_providers_failed", extra ={**log_ctx, "error": str(e)},)
                raise GeneratorUnavailable("Both Gemini and Groq failed") from e
        logger.error("generator.all_providers_unavailable", extra= log_ctx)
        raise GeneratorUnavailable("Both provider circuit breakers are open.")

    async def generate_one(
        self,
        retrieval_result: RetrievalResult,
        *,
        correlation_id: str | None = None,
    ) -> DrugWarning:
        evidence = retrieval_result.evidence
        chunks = retrieval_result.chunks

        if not chunks:
            return self._fda_fallback(evidence, reason="no_chunks_retrieved")

        allowed_citations = {c.id for c in chunks}
        allowed_drugs = {evidence.drug_a.name, evidence.drug_b.name}
        user_prompt = build_user_prompt(evidence, chunks)

        try:
            raw = await self._call_llm(
                SYSTEM_PROMPT, user_prompt, correlation_id=correlation_id
            )
            warnings = validate_llm_response(
                raw_output=raw,
                allowed_drug_names=allowed_drugs,
                allowed_citation_sources=allowed_citations,
            )
            return warnings[0]
        except (GeneratorUnavailable, StageValidationError) as e:
            logger.warning(
                "generator.fallback_to_fda",
                extra={"cid": correlation_id, "reason": str(e)},
            )
            return self._fda_fallback(evidence, reason=str(e))
        except Exception as e:
            logger.error(
                "generator.unexpected_failure",
                extra={"cid": correlation_id, "error": str(e)},
            )
            return self._fda_fallback(evidence, reason=f"unexpected: {type(e).__name__}")

    async def generate_many(
        self,
        retrieval_results: Sequence[RetrievalResult],
        *,
        correlation_id: str | None = None,
    ) -> list[DrugWarning]:
        """Generate DrugWarnings for many interactions IN PARALLEL."""
        if not retrieval_results:
            return []
        tasks = [
            self.generate_one(r, correlation_id=correlation_id)
            for r in retrieval_results
        ]
        # gather preserves order: results[i] is the outcome of generate_one(retrieval_results[i])
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r if isinstance(r, DrugWarning)
            else self._fda_fallback(
                retrieval_results[i].evidence,
                reason=f"unexpected: {type(r).__name__}",
            )
            for i, r in enumerate(results)
        ]

    def _fda_fallback(
        self, evidence: InteractionEvidence, reason: str
    ) -> DrugWarning:
        hint = evidence.estimated_severity
        severity = Severity(hint) if hint in {"RED", "YELLOW"} else Severity.YELLOW
        return DrugWarning(
            drugs_involved=[evidence.drug_a.name, evidence.drug_b.name],
            severity=severity,
            reaction_result=evidence.evidence_text,
            action=Action.CONSULT_DOCTOR,
            citation=["FDA_LABEL"],
            nurse_summary_to_doctor=(
                f"Degraded mode — LLM unavailable ({reason[:80]}). "
                f"Verify with FDA evidence directly."
            ),
            confidence=0.5,
            data_source=DataSource.FRESH_FDA,
            computed_at=datetime.now(timezone.utc),
        )


# ============================================================================
# PHASE 2 TODOs
# ============================================================================
# [ ] LLMProvider Protocol — extract Gemini/Groq into separate provider classes
#     implementing a common Protocol (generate(system, user) -> str) so new
#     providers (Claude, GPT-4o) can be added without touching Generator logic
# [ ] Retry with exponential backoff on transient errors before tripping breaker
# [ ] Per-provider timeout config — Gemini and Groq have different p95 latencies
# [ ] Log token usage per call (input + output tokens) for cost tracking
# [ ] Structured logging: log severity + action from each DrugWarning for
#     monitoring distribution drift in production
# [ ] generate_many concurrency cap — semaphore to avoid rate-limit storms
#     when many drug pairs are processed simultaneously
# [ ] Verify allowed_drug_names case-handling — interaction_checker stores
#     drug names lowercased (.lower()); confirm allowed_drugs set matches
#     what the LLM will produce. Validator does case-insensitive compare.
# [ ] First integration test: confirm LLM cites bare chunk_id, not the
#     "[chunk_id: X]" wrapper format shown in the prompt — citation validator
#     checks against raw IDs; a format mismatch silently rejects all citations

# ============================================================================
# PROD TODOs
# ============================================================================
# [ ] Move model names + timeouts to config / env vars
# [ ] OpenTelemetry span around _call_llm for latency tracing per provider
# [ ] Alert if fda_fallback_rate exceeds threshold (LLM quality degraded)
# [ ] Shadow mode — run both providers in parallel, compare outputs, promote
#     the better one based on downstream answer quality metrics