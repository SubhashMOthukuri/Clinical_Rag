"""
validators.py — 5-stage validation pipeline for MedReconcile AI.

Each stage raises StageValidationError on failure, carrying structured
details for observability (stage label, message, details dict). Stage 5
returns a ValidationResult instead of raising, because we want to collect
ALL response-level issues for monitoring, not bail on the first.

Stages:
  1. INPUT    — validate ReconciliationRequest before any API calls
  2. RXNORM   — validate RxNorm normalization response
  3. FDA      — validate FDA Label API response
  4. LLM      — validate GPT-4o-mini output (JSON + semantic checks)
  5. RESPONSE — validate final ReconciliationResponse before returning
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from src.utils.schema import (
    DrugWarning,
    ReconciliationRequest,
    ReconciliationResponse,
    Severity,
)

logger = logging.getLogger(__name__)


# ---------- Constants ----------

MIN_DRUG_NAME_LEN = 3
MAX_DRUG_NAME_LEN = 100
MAX_MEDICATIONS_PER_REQUEST = 50
MAX_RESPONSE_LATENCY_MS = 300.0

# Strict character allowlist for medication names.
# Allowed: letters, digits, spaces, hyphens, forward slashes, parentheses, periods.
# Must start with alphanumeric. Rejects: quotes, angle brackets, braces,
# semicolons, backticks, ampersands, pipes — every common injection vector.
_DRUG_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-/().]*$")

# Secondary scan: prompt-injection phrases that can slip past the allowlist
# (letters and spaces only). Compared against name.lower().
_PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "forget previous",
    "forget everything",
    "system prompt",
    "new instructions",
    "you are now",
    "act as",
    "pretend to be",
    "jailbreak",
)


# ---------- Result types ----------

class ValidationStage(str, Enum):
    INPUT = "STAGE_1_INPUT"
    RXNORM = "STAGE_2_RXNORM"
    FDA = "STAGE_3_FDA"
    LLM = "STAGE_4_LLM"
    RESPONSE = "STAGE_5_RESPONSE"


class StageValidationError(Exception):
    """Raised when a pipeline stage fails validation.

    Attributes:
        stage: which pipeline stage failed (for metrics labels)
        message: human-readable failure reason
        details: structured payload for logs/observability
    """

    def __init__(
        self,
        stage: ValidationStage,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.message = message
        self.details = details or {}
        super().__init__(f"[{stage.value}] {message}")


@dataclass
class ValidationResult:
    """Non-raising result for Stage 5 (collects all issues)."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------- Stage 1: Input ----------

def validate_input(request: ReconciliationRequest) -> None:
    """Validate nurse input before any API calls. Raises on failure."""
    meds = request.medications

    if not meds:
        raise StageValidationError(
            ValidationStage.INPUT, "medications list is empty"
        )

    if len(meds) > MAX_MEDICATIONS_PER_REQUEST:
        raise StageValidationError(
            ValidationStage.INPUT,
            f"too many medications: {len(meds)} > {MAX_MEDICATIONS_PER_REQUEST}",
            {"count": len(meds)},
        )

    seen_names: set[str] = set()
    for idx, med in enumerate(meds):
        _validate_drug_name(med.name, idx)

        # Pydantic's gt=0 rejects negatives but NaN/inf handling is fuzzy;
        # re-check explicitly because a NaN dose reaching the LLM is dangerous.
        if not math.isfinite(med.dose):
            raise StageValidationError(
                ValidationStage.INPUT,
                f"medication[{idx}].dose is not a finite number",
                {"index": idx, "dose": str(med.dose)},
            )

        # Duplicate detection — case-insensitive. Same drug listed twice
        # is itself a reconciliation red flag; surface it early.
        key = med.name.strip().lower()
        if key in seen_names:
            raise StageValidationError(
                ValidationStage.INPUT,
                f"duplicate medication: {med.name!r}",
                {"index": idx, "name": med.name},
            )
        seen_names.add(key)


def _validate_drug_name(name: str, idx: int) -> None:
    """Strict allowlist + prompt-injection scan."""
    if len(name) < MIN_DRUG_NAME_LEN:
        raise StageValidationError(
            ValidationStage.INPUT,
            f"medication[{idx}].name too short (min {MIN_DRUG_NAME_LEN} chars)",
            {"index": idx, "name": name},
        )

    if len(name) > MAX_DRUG_NAME_LEN:
        raise StageValidationError(
            ValidationStage.INPUT,
            f"medication[{idx}].name too long (max {MAX_DRUG_NAME_LEN} chars)",
            {"index": idx, "length": len(name)},
        )

    if not _DRUG_NAME_PATTERN.match(name):
        raise StageValidationError(
            ValidationStage.INPUT,
            f"medication[{idx}].name contains disallowed characters",
            {"index": idx, "name": name},
        )

    lowered = name.lower()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern in lowered:
            logger.warning("prompt_injection_attempt idx=%d name=%r", idx, name)
            raise StageValidationError(
                ValidationStage.INPUT,
                f"medication[{idx}].name matches prompt-injection pattern",
                {"index": idx, "pattern": pattern},
            )


# ---------- Stage 2: RxNorm ----------

def validate_rxnorm_response(
    response: dict[str, Any],
    drug_name: str,
) -> str | None:
    """Validate RxNorm /rxcui.json response.

    Returns:
        rxcui string on success, or None if drug is unverified (idGroup empty).
    Raises:
        StageValidationError on malformed response.
    """
    if not isinstance(response, dict):
        raise StageValidationError(
            ValidationStage.RXNORM,
            "response is not a dict",
            {"drug": drug_name, "type": type(response).__name__},
        )

    id_group = response.get("idGroup")
    if id_group is None:
        raise StageValidationError(
            ValidationStage.RXNORM,
            "missing idGroup key",
            {"drug": drug_name},
        )

    if not isinstance(id_group, dict):
        raise StageValidationError(
            ValidationStage.RXNORM,
            "idGroup is not a dict",
            {"drug": drug_name},
        )

    rxnorm_ids = id_group.get("rxnormId")
    if not rxnorm_ids:
        # Empty idGroup = drug not in RxNorm. Not an error — caller flags
        # the drug as UNVERIFIED and the pipeline continues.
        logger.info("rxnorm_unverified drug=%r", drug_name)
        return None

    if not isinstance(rxnorm_ids, list) or not rxnorm_ids[0]:
        raise StageValidationError(
            ValidationStage.RXNORM,
            "rxnormId is not a non-empty list",
            {"drug": drug_name},
        )

    rxcui = str(rxnorm_ids[0])
    if not rxcui.isdigit():
        raise StageValidationError(
            ValidationStage.RXNORM,
            "rxcui is not numeric",
            {"drug": drug_name, "rxcui": rxcui},
        )

    return rxcui


# ---------- Stage 3: FDA ----------

def validate_fda_response(
    response: dict[str, Any],
    drug_name: str,
) -> list[dict[str, Any]]:
    """Validate FDA Label API response.

    Returns:
        results list on success.
    Raises:
        StageValidationError on error response or malformed payload.
        Caller should catch and trigger Level 2 fallback (skip RAG, use cache).
    """
    if not isinstance(response, dict):
        raise StageValidationError(
            ValidationStage.FDA,
            "response is not a dict",
            {"drug": drug_name, "type": type(response).__name__},
        )

    if "error" in response:
        raise StageValidationError(
            ValidationStage.FDA,
            "FDA API returned error — trigger fallback",
            {"drug": drug_name, "error": response["error"]},
        )

    results = response.get("results")
    if not isinstance(results, list) or len(results) == 0:
        raise StageValidationError(
            ValidationStage.FDA,
            "results missing or empty — trigger fallback",
            {"drug": drug_name},
        )

    return results


# ---------- Stage 4: LLM ----------

def validate_llm_response(
    raw_output: str,
    allowed_drug_names: set[str],
    allowed_citation_sources: set[str],
) -> list[DrugWarning]:
    """Validate GPT-4o-mini output.

    Checks:
      - valid JSON
      - conforms to DrugWarning schema
      - every warning has >= 1 citation (no uncited medical claims)
      - every citation is in allowed_citation_sources (no hallucinated URLs)
      - every drug in drugs_involved is in allowed_drug_names (no hallucinated drugs)
      - severity / action / data_source are valid enum values (enforced by schema)

    Returns:
        list of validated DrugWarning objects.
    Raises:
        StageValidationError on any failure. Guardrail rule: reject the
        entire response if any single warning is suspect. Healthcare
        hallucinations are life-threatening — fail closed, not open.
    """
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise StageValidationError(
            ValidationStage.LLM,
            "LLM output is not valid JSON",
            {"error": str(exc), "preview": raw_output[:200]},
        ) from exc

    if not isinstance(parsed, list):
        raise StageValidationError(
            ValidationStage.LLM,
            "LLM output is not a JSON list",
            {"type": type(parsed).__name__},
        )

    warnings_out: list[DrugWarning] = []
    allowed_drugs_lower = {d.lower() for d in allowed_drug_names}

    for idx, item in enumerate(parsed):
        try:
            warning = DrugWarning.model_validate(item)
        except PydanticValidationError as exc:
            raise StageValidationError(
                ValidationStage.LLM,
                f"warning[{idx}] failed schema validation",
                {"index": idx, "errors": exc.errors()},
            ) from exc

        # Schema enforces citation min_length=1 already, but re-check at this
        # boundary because LLM output is adversarial input.
        if not warning.citation:
            raise StageValidationError(
                ValidationStage.LLM,
                f"warning[{idx}] has no citation — uncited claim rejected",
                {"index": idx},
            )

        for citation in warning.citation:
            if citation not in allowed_citation_sources:
                raise StageValidationError(
                    ValidationStage.LLM,
                    f"warning[{idx}] cites unknown source — possible hallucination",
                    {"index": idx, "citation": citation},
                )

        for drug in warning.drugs_involved:
            if drug.lower() not in allowed_drugs_lower:
                raise StageValidationError(
                    ValidationStage.LLM,
                    f"warning[{idx}] references unknown drug — possible hallucination",
                    {"index": idx, "drug": drug},
                )

        warnings_out.append(warning)

    return warnings_out


# ---------- Stage 5: Response ----------

def validate_response(
    request: ReconciliationRequest,
    response: ReconciliationResponse,
) -> ValidationResult:
    """Validate final response. Collects all issues (does not raise).

    Unlike earlier stages, this one does not fail closed — we want to
    surface every problem at once for observability, then let the caller
    decide whether to return 500 or degrade gracefully.

    Returns:
        ValidationResult with .ok, .errors, .warnings.
    """
    result = ValidationResult(ok=True)

    # 5a. No silent drops — every input drug must appear either in
    # response.medications or response.unverified_drugs.
    input_names = {m.name.strip().lower() for m in request.medications}
    output_names = {m.name.strip().lower() for m in response.medications}
    unverified = {d.strip().lower() for d in response.unverified_drugs}
    missing = input_names - (output_names | unverified)
    if missing:
        result.ok = False
        result.errors.append(f"silent drops detected: {sorted(missing)}")

    # 5b. Every RED warning must carry a citation (defense in depth —
    # schema already enforces this, but we re-verify at the boundary).
    for idx, w in enumerate(response.warnings):
        if w.severity == Severity.RED and not w.citation:
            result.ok = False
            result.errors.append(f"warning[{idx}] is RED without citation")

    # 5c. Denormalized counts must match the actual lists.
    if response.total_medications != len(response.medications):
        result.ok = False
        result.errors.append(
            f"total_medications={response.total_medications} "
            f"!= len(medications)={len(response.medications)}"
        )

    if response.total_warnings != len(response.warnings):
        result.ok = False
        result.errors.append(
            f"total_warnings={response.total_warnings} "
            f"!= len(warnings)={len(response.warnings)}"
        )

    critical_actual = sum(1 for w in response.warnings if w.severity == Severity.RED)
    if response.critical_warnings != critical_actual:
        result.ok = False
        result.errors.append(
            f"critical_warnings={response.critical_warnings} "
            f"!= actual RED count={critical_actual}"
        )

    # 5d. Latency budget — warn only. Still return the payload; let the
    # caller (or Prometheus alert) decide what to do about SLO burn.
    if response.response_time_ms > MAX_RESPONSE_LATENCY_MS:
        result.warnings.append(
            f"response_time_ms={response.response_time_ms} "
            f"exceeds budget {MAX_RESPONSE_LATENCY_MS}"
        )

    return result
