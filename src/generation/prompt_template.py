"""Prompt template"""

from src.retrieval.retrieval import RetrievalResult, QueryResult
from src.retrieval.interaction_checker import InteractionEvidence

SYSTEM_PROMPT = """You are a clinical drug interaction assistant for use at the point of care.

You receive FDA background evidence and StatPearls clinical text chunks for one drug pair. Classify the interaction and explain it.

STRICT RULES:
1. Use ONLY the provided FDA evidence and StatPearls chunks. Never add outside knowledge.
2. Citations MUST be SOURCE_IDs from the StatPearls chunks (format: "article-XXXXX_chunk_YYYY"). The FDA evidence is background context — do NOT cite FDA section numbers (e.g. "7", "7.1") as sources.
3. If no StatPearls chunks are provided, set citation to ["FDA_LABEL"] and base your answer on the FDA evidence only.
4. If the evidence does not support a claim, do not make it.
5. Be factual and clinical. No speculation, no creativity.
6. Reference only the two drugs provided.

OUTPUT FORMAT — return ONLY a JSON array with exactly one object:
[
  {
    "drugs_involved": ["drug_a", "drug_b"],
    "severity": "RED" | "YELLOW" | "GREEN",
    "reaction_result": "clinical explanation grounded in the sources",
    "action": "STOP" | "MONITOR" | "CONSULT_DOCTOR",
    "citation": ["article-XXXXX_chunk_YYYY", "..."],
    "nurse_summary_to_doctor": "concise actionable summary",
    "confidence": 0.85
  }
]

No markdown, no text outside the JSON array."""

def build_user_prompt(evidence: InteractionEvidence, chunks: list[QueryResult])-> str:
    drug_a = evidence.drug_a.name
    drug_b = evidence.drug_b.name
    fda_evidence = evidence.evidence_text
    severity = evidence.estimated_severity
    if not chunks:
        chunk_block = "No StatPearls chunks available. Use FDA evidence only."
    else:
        lines= []
        for c in chunks:
            lines.append(f"SOURCE_ID: {c.id}\n{c.metadata.text}")
        chunk_block = "\n\n".join(lines)
    return f"""Drug pair: {drug_a} and {drug_b}

FDA background evidence (context only — do NOT cite section numbers from this):
{fda_evidence}

Preliminary severity hint (not final): {severity}

StatPearls clinical context (cite using SOURCE_ID values below):
{chunk_block}

Task: Classify the interaction severity, explain the clinical reaction grounded in the sources above, and populate the citation array with the SOURCE_IDs of StatPearls chunks you used. Return a JSON array with one DrugWarning object.
"""


# ============================================================================
# STATUS NOTE
# ============================================================================
# JSON validation, citation cross-check, enum enforcement (Severity/Action),
# citation-required, and drug/chunk hallucination checks are ALREADY satisfied
# by validators.validate_llm_response (Stage 4). Do NOT rebuild them here.
# ============================================================================


# ============================================================================
# PHASE 2 TODOs — Quality & Robustness
# ============================================================================
# [ ] Include dose + unit from DrugContext in the prompt — clinically relevant
#     for severity (warfarin 10mg vs 1mg changes risk). Fields exist on
#     evidence.drug_a.dose / .unit but are not passed to the LLM yet.
# [ ] Few-shot examples — add 1 RED and 1 GREEN worked example inside
#     SYSTEM_PROMPT so the LLM reliably learns the exact JSON shape.
# [ ] FDA-only + empty-evidence case — when chunks=[] AND evidence_text blank,
#     add explicit "insufficient data, do not speculate" instruction.
# [ ] Confidence field — define what 0.0–1.0 means in SYSTEM_PROMPT, or remove
#     it. A static "0.0" placeholder misleads downstream consumers.
# [ ] Injection hardening — escape/strip characters in chunk text + evidence
#     that could break JSON (braces, quotes, backticks).
# [ ] Prompt versioning — add PROMPT_VERSION constant, include in output so
#     every LLM response traces to the exact prompt that produced it.
# [ ] Token budget guard — count built-prompt tokens (tiktoken); drop
#     lowest-score chunks if over model context. Low urgency now (~2K vs ~1M),
#     matters if chunk count grows or a small-context model is adopted.
# [ ] Observability — log prompt_version, drug pair, chunk_count, token_count,
#     resulting severity + action; track fda_only_rate.

# ============================================================================
# PHASE 3 TODOs — Production Hardening
# ============================================================================
# [ ] Move SYSTEM_PROMPT to external file (prompts/v1/system.txt) so clinical
#     reviewers can tune wording without touching code.
# [ ] A/B prompt testing — route % of traffic to a v2 prompt in shadow mode,
#     compare validation pass rate + answer quality before promoting.
# [ ] Prompt template registry — versioned prompts with rollback, so a bad
#     prompt change can be reverted without a code deploy.
# [ ] Multi-language prompts — if nurses operate in non-English settings,
#     localize SYSTEM_PROMPT while keeping JSON output schema constant.
# [ ] Clinical review sign-off workflow — require a pharmacist to approve any
#     SYSTEM_PROMPT change before it reaches production (compliance trail).
# [ ] Citation format wiring test — prompt shows "[chunk_id: X]" but validator
#     compares bare "X"; add an end-to-end test asserting the LLM cites the
#     bare ID, not the bracket wrapper. (Verify at first integration test.)

# ============================================================================
# KNOWN LIMITATIONS (MVP)
# ============================================================================
# - Dose/unit not passed to LLM (severity reasoning ignores dose).
# - No few-shot examples (relies on schema description alone).
# - Confidence field is a static placeholder, not a real signal.
# - Chunk text not escaped before prompt insertion (low risk: StatPearls is
#   trusted source data, not user input).
# - Single prompt version, no A/B or version tracking.