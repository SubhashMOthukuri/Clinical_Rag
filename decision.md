## Target User Clarification

NOT targeting large hospital systems like Aurora, Mayo
They already have solutions

TARGETING small specialty practices using ModMed:
- Dermatology clinics
- Ophthalmology practices  
- Orthopedic surgery centers

Why they have the problem:
- No dedicated pharmacist
- No medication reconciliation team
- Nurse does everything manually
- Basic EHR with limited drug checking
- Doctor prescribes from memory

Real risk:
- Specialist prescribes new drug
- Doesn't know patient's full medication list
- Primary care doctor prescribed conflicting drug
- Nobody catches it
- Patient ends up in ER

Decision 1: RxNorm for normalization only
RxNorm interaction API discontinued January 2024. Use FDA Drug Label API for interactions instead. FDA is more authoritative anyway — official source updated with every drug approval.

Decision 2: Two-layer caching
Redis 24hr for query results. PostgreSQL 7 days for drug pair interactions. Weekly refresh every Sunday 2am to catch FDA monthly updates. Outdated interaction data in healthcare is dangerous.

Decision 3: Store dose (mg) in cache
Same drug + different dose = different severity. ibuprofen 200mg + lisinopril = YELLOW. ibuprofen 800mg + lisinopril = RED. Cache key includes dose value.

Decision 4: patient_id and nurse_id optional in MVP
No authentication system in MVP. Required in Phase 2 when auth added. IDs are always str not int — IDs can contain letters and dashes.

Decision 5: Guardrails dual-model system
GPT-4o-mini generates the report. Second smaller model checks every claim is cited from real source. If any uncited medical claim found — remove it. Never let uncited claims reach the nurse. Healthcare hallucinations are life-threatening.

Decision 6: Pydantic v2 with production ConfigDict on all schemas
All models use model_config = PROD_CONFIG with str_strip_whitespace=True, extra="forbid", validate_assignment=True. The extra="forbid" is the first defense against payload injection — unknown fields raise ValidationError instead of silently passing through.

Decision 7: All enums inherit from str
class Severity(str, Enum) not class Severity(Enum). Without str inheritance, FastAPI JSON serialization emits "Severity.RED" instead of "RED", breaking API contracts with downstream consumers.

Decision 8: Rename Warning → DrugWarning
Python has a built-in Warning class. Shadowing it causes silent collisions with import warnings and confuses static analyzers. Domain-prefixed names are safer in prod.

Decision 9: Field-level constraints on all schema fields
dose: Field(gt=0, le=10000), rxcui: Field(pattern=r"^\d{1,10}$"), confidence: Field(ge=0.0, le=1.0), etc. Schema-level validation is cheaper than runtime checks and documents the contract.

Decision 10: Unit as Enum, not free-text string
Nurses typing "mg", "mG", "milligram", "mgs" all get normalized to a fixed enum set {mg, mcg, g, mL, IU, unit}. Prevents downstream dose comparison bugs.
Decision 11: Five-stage validation pipeline with typed failures
Every stage raises StageValidationError(stage, message, details) with a ValidationStage enum label. Enables Prometheus metrics by stage (validation_errors_total{stage="STAGE_2_RXNORM"}) — generic exceptions destroy observability.

Decision 12: Stages 1–4 fail closed, Stage 5 collects all issues
Early stages raise on first failure (cheap to abort, trigger fallback). Stage 5 returns a ValidationResult(ok, errors, warnings) because once response compute is spent, we want the full diagnostic picture for logs — not just the first problem.

Decision 13: Dual-layer input sanitization
Drug names validated against strict regex ^[A-Za-z0-9][A-Za-z0-9 \-/().]*$ (literal space, not \s — tabs/newlines excluded) plus secondary scan for known prompt-injection phrases on name.lower(). Pressure-tested against homoglyph, zero-width, RTL override, null byte, and newline-smuggling attacks (55/55 pass).

Decision 14: Duplicate detection by drug name only (case-insensitive, whitespace-stripped)
Same-drug-different-dose entries (e.g., ibuprofen 200mg + ibuprofen 800mg) are rejected at Stage 1. Nurses must use the frequency field for split dosing. Revisit in Phase 2 if field usage data shows high rejection rates.

Decision 15: Explicit NaN/inf check on dose beyond Pydantic's gt=0
Pydantic's numeric constraints are fuzzy on non-finite floats. A NaN dose reaching GPT-4o-mini is undefined behavior in a safety system — belt-and-suspenders math.isfinite() check in Stage 1.
Decision 16: LLM output validation requires allowed_drug_names and allowed_citation_sources as dependency-injected sets
The LLM can only cite sources the RAG step actually retrieved, and can only warn about drugs the nurse actually submitted. Any deviation = hallucination = full response rejected. Implements the guardrail system from Decision 5.

Decision 17: Response-level invariants enforced at Stage 5
No silent drops (every input drug must appear in output or unverified_drugs), denormalized counts must match lists, every RED must have a citation, latency budget is warn-only (don't fail the response over an SLO breach — let Prometheus alert).