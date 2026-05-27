"""
Unit tests for src/generation/prompt_template.py

Covers:
- SYSTEM_PROMPT: contains key rules, JSON schema fields, output constraints
- build_user_prompt: drug names, FDA evidence, severity, chunk block format,
  no-chunks fallback, multiple chunks joined correctly, task instruction present
"""
from __future__ import annotations

import pytest

from src.generation.prompt_template import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.pinecone_store import QueryResult, ChunkMetadata
from src.retrieval.interaction_checker import DrugContext, InteractionEvidence


# ============================================================================
# Helpers
# ============================================================================

def make_metadata(text: str = "Clinical text about drug interaction.") -> ChunkMetadata:
    return ChunkMetadata(
        text=text,
        title="Drug Interactions",
        source="StatPearls",
        article_id="NBK001",
        article_type="general",
        token_count=10,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


def make_chunk(id_: str = "NBK001_chunk_0000", text: str = "Clinical text.") -> QueryResult:
    return QueryResult(id=id_, score=0.9, metadata=make_metadata(text))


def make_drug(name: str) -> DrugContext:
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
    evidence_text: str = "Concurrent use increases bleeding risk.",
    severity: str = "RED",
) -> InteractionEvidence:
    return InteractionEvidence(
        drug_a=make_drug(drug_a),
        drug_b=make_drug(drug_b),
        evidence_text=evidence_text,
        source_drug=drug_a,
        estimated_severity=severity,
    )


# ============================================================================
# TestSystemPrompt
# ============================================================================

class TestSystemPrompt:

    def test_is_a_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 0

    def test_contains_json_array_instruction(self):
        assert "JSON array" in SYSTEM_PROMPT

    def test_contains_severity_options(self):
        assert "RED" in SYSTEM_PROMPT
        assert "YELLOW" in SYSTEM_PROMPT
        assert "GREEN" in SYSTEM_PROMPT

    def test_contains_action_options(self):
        assert "STOP" in SYSTEM_PROMPT
        assert "MONITOR" in SYSTEM_PROMPT
        assert "CONSULT_DOCTOR" in SYSTEM_PROMPT

    def test_contains_required_output_fields(self):
        for field in ["drugs_involved", "severity", "reaction_result",
                      "action", "citation", "nurse_summary_to_doctor", "confidence"]:
            assert field in SYSTEM_PROMPT, f"Missing field: {field}"

    def test_forbids_outside_knowledge(self):
        assert "outside knowledge" in SYSTEM_PROMPT.lower() or "only" in SYSTEM_PROMPT.lower()

    def test_requires_citation(self):
        assert "cite" in SYSTEM_PROMPT.lower() or "citation" in SYSTEM_PROMPT.lower()

    def test_no_markdown_instruction(self):
        assert "markdown" in SYSTEM_PROMPT.lower()

    def test_no_text_outside_json_instruction(self):
        assert "No" in SYSTEM_PROMPT and "JSON" in SYSTEM_PROMPT


# ============================================================================
# TestBuildUserPrompt — happy path
# ============================================================================

class TestBuildUserPrompt:

    def test_returns_string(self):
        result = build_user_prompt(make_evidence(), [make_chunk()])
        assert isinstance(result, str)

    def test_drug_a_name_in_prompt(self):
        result = build_user_prompt(make_evidence("warfarin", "aspirin"), [make_chunk()])
        assert "warfarin" in result

    def test_drug_b_name_in_prompt(self):
        result = build_user_prompt(make_evidence("warfarin", "aspirin"), [make_chunk()])
        assert "aspirin" in result

    def test_fda_evidence_text_in_prompt(self):
        evidence = make_evidence(evidence_text="High bleeding risk when combined.")
        result = build_user_prompt(evidence, [make_chunk()])
        assert "High bleeding risk when combined." in result

    def test_severity_hint_in_prompt(self):
        result = build_user_prompt(make_evidence(severity="RED"), [make_chunk()])
        assert "RED" in result

    def test_severity_yellow_in_prompt(self):
        result = build_user_prompt(make_evidence(severity="YELLOW"), [make_chunk()])
        assert "YELLOW" in result

    def test_task_instruction_present(self):
        result = build_user_prompt(make_evidence(), [make_chunk()])
        assert "Task" in result or "task" in result

    def test_chunk_id_appears_in_prompt(self):
        chunk = make_chunk(id_="NBK001_chunk_0000")
        result = build_user_prompt(make_evidence(), [chunk])
        assert "NBK001_chunk_0000" in result

    def test_chunk_id_formatted_with_brackets(self):
        chunk = make_chunk(id_="NBK001_chunk_0000")
        result = build_user_prompt(make_evidence(), [chunk])
        assert "[chunk_id: NBK001_chunk_0000]" in result

    def test_chunk_text_in_prompt(self):
        chunk = make_chunk(text="Warfarin prolongs bleeding time.")
        result = build_user_prompt(make_evidence(), [chunk])
        assert "Warfarin prolongs bleeding time." in result

    def test_multiple_chunks_all_appear(self):
        chunks = [
            make_chunk("chunk_0", "First chunk text."),
            make_chunk("chunk_1", "Second chunk text."),
        ]
        result = build_user_prompt(make_evidence(), chunks)
        assert "chunk_0" in result
        assert "chunk_1" in result
        assert "First chunk text." in result
        assert "Second chunk text." in result

    def test_multiple_chunks_separated_by_double_newline(self):
        chunks = [
            make_chunk("c0", "Alpha."),
            make_chunk("c1", "Beta."),
        ]
        result = build_user_prompt(make_evidence(), chunks)
        assert "\n\n" in result

    def test_return_json_instruction_in_prompt(self):
        result = build_user_prompt(make_evidence(), [make_chunk()])
        assert "JSON" in result


# ============================================================================
# TestBuildUserPrompt — no-chunks fallback
# ============================================================================

class TestBuildUserPromptNoChunks:

    def test_empty_chunks_returns_string(self):
        result = build_user_prompt(make_evidence(), [])
        assert isinstance(result, str)

    def test_fallback_message_present(self):
        result = build_user_prompt(make_evidence(), [])
        assert "FDA evidence only" in result or "No StatPearls" in result

    def test_drug_names_still_in_prompt_when_no_chunks(self):
        result = build_user_prompt(make_evidence("warfarin", "aspirin"), [])
        assert "warfarin" in result
        assert "aspirin" in result

    def test_fda_evidence_still_in_prompt_when_no_chunks(self):
        evidence = make_evidence(evidence_text="Serious bleeding risk.")
        result = build_user_prompt(evidence, [])
        assert "Serious bleeding risk." in result

    def test_no_chunk_id_format_when_no_chunks(self):
        result = build_user_prompt(make_evidence(), [])
        assert "[chunk_id:" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
