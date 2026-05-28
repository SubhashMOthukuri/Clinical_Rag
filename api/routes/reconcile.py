"""Reconciliation route and private pipeline helpers.

Pipeline per request:
  1. Stage 1 input validation (validators.validate_input)
  2. RxNorm normalize + FDA fetch each medication (parallel)
  3. Pairwise interaction check → InteractionEvidence list
  4. Retrieve clinical chunks per evidence pair (parallel)
  5. Generate DrugWarning per evidence pair (parallel LLM calls)
  6. Assemble ReconciliationResponse
  7. Stage 5 response validation (collects issues, does not raise)
  8. Return JSON

Failure modes:
  - Stage 1 fails closed → StageValidationError → 422 via exception handler
  - RxNorm/FDA fail soft → unverified_drugs list populated, pipeline continues
  - Retriever fails soft → empty chunks, generator uses FDA-only fallback
  - Generator fails soft → FDA-only DrugWarning, confidence < 1.0 → PARTIAL status
  - Stage 5 failure logged for observability but never blocks the response
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from api.dependencies import (
    get_fda,
    get_generator,
    get_interaction_checker,
    get_retriever,
    get_rxnorm,
)
from src.generation.generator import Generator
from src.ingestion.fda_client import FDAClient, FDADrugData
from src.ingestion.rxnorm_client import RxNormClient, RxcuiFound, RxcuiUnverified
from src.retrieval.interaction_checker import InteractionChecker
from src.retrieval.retrieval import Retriever
from src.utils.schema import (
    DrugWarning,
    Medication,
    ReconciliationRequest,
    ReconciliationResponse,
    Severity,
    Status,
)
from src.utils.validators import validate_input, validate_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/reconcile", response_model=ReconciliationResponse)
async def reconcile(
    request: Request,
    payload: ReconciliationRequest,
    rxnorm: RxNormClient = Depends(get_rxnorm),
    fda: FDAClient = Depends(get_fda),
    checker: InteractionChecker = Depends(get_interaction_checker),
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> ReconciliationResponse:
    """End-to-end medication reconciliation pipeline.

    All upstream failures degrade gracefully — the nurse always gets a
    response, with degraded components clearly flagged in the status field.
    """
    cid = request.state.correlation_id
    t0 = time.perf_counter()
    logger.info("reconcile.begin", extra={"cid": cid, "med_count": len(payload.medications)})

    # Stage 1: validate input — fails closed, raises StageValidationError
    validate_input(payload)

    # Steps 2+3: enrich each medication with rxcui + FDA data in parallel
    enriched_meds, fda_map, unverified_drugs = await _enrich_medications(
        payload.medications, rxnorm, fda, cid
    )

    # Step 4: pairwise interaction check (sync, pure text scan — fast)
    evidences = checker.check(enriched_meds, fda_map)
    logger.info("reconcile.evidences_found", extra={"cid": cid, "count": len(evidences)})

    if not evidences:
        return _build_response(
            medications=enriched_meds,
            warnings=[],
            unverified=unverified_drugs,
            status=Status.SUCCESS,
            response_time_ms=(time.perf_counter() - t0) * 1000,
        )

    # Step 5: retrieve clinical chunks per evidence pair (parallel)
    retrieval_results = await retriever.retrieve_many(evidences, correlation_id=cid)

    # Step 6: generate DrugWarnings per evidence pair (parallel LLM calls)
    warnings = await generator.generate_many(retrieval_results, correlation_id=cid)

    # Step 7: assemble response — any FDA-only fallback warning → PARTIAL
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response_status = (
        Status.PARTIAL if any(w.confidence < 1.0 for w in warnings) else Status.SUCCESS
    )
    response = _build_response(
        medications=enriched_meds,
        warnings=warnings,
        unverified=unverified_drugs,
        status=response_status,
        response_time_ms=elapsed_ms,
    )

    # Step 8: Stage 5 response validation — collects issues, never raises
    result = validate_response(payload, response)
    if not result.ok:
        logger.error(
            "reconcile.response_validation_failed",
            extra={"cid": cid, "errors": result.errors},
        )
    if result.warnings:
        logger.warning(
            "reconcile.response_warnings",
            extra={"cid": cid, "resp_warnings": result.warnings},
        )

    logger.info(
        "reconcile.complete",
        extra={
            "cid": cid,
            "elapsed_ms": elapsed_ms,
            "warnings_count": len(warnings),
            "status": response_status.value,
        },
    )
    return response


# ============================================================================
# Private pipeline helpers
# ============================================================================

async def _enrich_medications(
    medications: list[Medication],
    rxnorm: RxNormClient,
    fda: FDAClient,
    cid: str,
) -> tuple[list[Medication], dict[str, FDADrugData], list[str]]:
    """Resolve rxcui + FDA data for every medication in parallel.

    Returns:
      enriched_meds  — medications with rxcui/ingredient_rxcui populated
      fda_map        — lowercase drug name → FDADrugData (for interaction checker)
      unverified     — drug names RxNorm could not recognise
    """
    async def _one(med: Medication) -> tuple[Medication, FDADrugData | None, bool]:
        rx_task = rxnorm.get_rxcui(med.name, correlation_id=cid)
        fda_task = fda.get_drug_data(med.name, correlation_id=cid)
        rx_result, fda_data = await asyncio.gather(rx_task, fda_task)

        is_unverified = isinstance(rx_result, RxcuiUnverified)
        rxcui = rx_result.rxcui if isinstance(rx_result, RxcuiFound) else None

        ingredient_rxcui = None
        if rxcui:
            ingredient_rxcui = await rxnorm.get_ingredient_rxcui(rxcui)

        enriched = med.model_copy(
            update={"rxcui": rxcui, "ingredient_rxcui": ingredient_rxcui, "verified": rxcui is not None}
        )
        return enriched, fda_data, is_unverified

    results = await asyncio.gather(*[_one(m) for m in medications])

    enriched_meds: list[Medication] = []
    fda_map: dict[str, FDADrugData] = {}
    unverified: list[str] = []

    for enriched, fda_data, is_unverified in results:
        enriched_meds.append(enriched)
        if fda_data is not None:
            fda_map[enriched.name.lower()] = fda_data
        if is_unverified:
            unverified.append(enriched.name)

    return enriched_meds, fda_map, unverified


def _build_response(
    *,
    medications: list[Medication],
    warnings: list[DrugWarning],
    unverified: list[str],
    status: Status,
    response_time_ms: float,
) -> ReconciliationResponse:
    """Assemble the final ReconciliationResponse with denormalised counts."""
    critical = sum(1 for w in warnings if w.severity == Severity.RED)
    return ReconciliationResponse(
        medications=medications,
        warnings=warnings,
        unverified_drugs=unverified,
        status=status,
        response_time_ms=response_time_ms,
        computed_at=datetime.now(timezone.utc),
        total_medications=len(medications),
        total_warnings=len(warnings),
        critical_warnings=critical,
    )
