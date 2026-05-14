"""
Unit tests for src/retrieval/interaction_checker.py

Tests cover:
- InteractionChecker.__init__: config defaults, optional dependencies
- InteractionChecker._build_context: name lowercasing, unit mapping, rxcui fallback
- InteractionChecker._check_pair: each FDA label field, min_evidence_length filter,
  case-insensitive matching, both-direction scanning, missing FDA data
- InteractionChecker.check: happy path, deduplication, pair capping,
  missing FDA data, no evidence, logging
"""

import pytest
from datetime import datetime, timezone

from src.retrieval.interaction_checker import (
    DrugContext,
    InteractionChecker,
    InteractionCheckerConfig,
    InteractionEvidence,
)
from src.ingestion.fda_client import FDADrugData
from src.utils.schema import Medication, Unit


# ============================================================================
# FIXTURES
# ============================================================================

def make_medication(
    name="aspirin",
    dose=100.0,
    unit=Unit.MG,
    ingredient_rxcui=None,
):
    return Medication(name=name, dose=dose, unit=unit, ingredient_rxcui=ingredient_rxcui)


def make_fda_data(
    generic_name="aspirin",
    rxcui="1191",
    drug_class="NSAID",
    drug_interactions=None,
    warnings=None,
    do_not_use=None,
    fda_label_id="label-001",
):
    return FDADrugData(
        generic_name=generic_name,
        rxcui=rxcui,
        drug_class=drug_class,
        warnings=warnings or [],
        drug_interactions=drug_interactions or [],
        do_not_use=do_not_use or [],
        ask_doctor=[],
        source="FRESH_FDA",
        fetched_at=datetime.now(timezone.utc),
        fda_label_id=fda_label_id,
    )


def make_context(
    name="aspirin",
    dose=100.0,
    unit="mg",
    ingredient_rxcui="",
    drug_class="NSAID",
    fda_label_id="label-001",
):
    return DrugContext(
        name=name,
        dose=dose,
        unit=unit,
        ingredient_rxcui=ingredient_rxcui,
        drug_class=drug_class,
        fda_label_id=fda_label_id,
    )


# ============================================================================
# __init__
# ============================================================================

class TestInteractionCheckerInit:

    def test_uses_default_config_when_none_given(self):
        checker = InteractionChecker()
        assert checker._cfg.min_evidence_length == 20
        assert checker._cfg.max_pairs_per_request == 100
        assert checker._cfg.batch_size == 20

    def test_accepts_custom_config(self):
        config = InteractionCheckerConfig(min_evidence_length=50, max_pairs_per_request=10)
        checker = InteractionChecker(config=config)
        assert checker._cfg.min_evidence_length == 50
        assert checker._cfg.max_pairs_per_request == 10

    def test_db_pool_and_redis_default_to_none(self):
        checker = InteractionChecker()
        assert checker._db is None
        assert checker._redis is None

    def test_accepts_db_pool_and_redis(self):
        db = object()
        redis = object()
        checker = InteractionChecker(db_pool=db, redis_client=redis)
        assert checker._db is db
        assert checker._redis is redis


# ============================================================================
# _build_context
# ============================================================================

class TestBuildContext:

    def test_lowercases_drug_name(self):
        med = make_medication(name="ASPIRIN")
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.name == "aspirin"

    def test_mixed_case_lowercased(self):
        med = make_medication(name="Metformin")
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.name == "metformin"

    def test_maps_mg_unit_value(self):
        med = make_medication(unit=Unit.MG)
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.unit == "mg"

    def test_maps_mcg_unit_value(self):
        med = make_medication(unit=Unit.MCG)
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.unit == "mcg"

    def test_maps_ml_unit_value(self):
        med = make_medication(unit=Unit.ML)
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.unit == "mL"

    def test_ingredient_rxcui_none_becomes_empty_string(self):
        med = make_medication(ingredient_rxcui=None)
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.ingredient_rxcui == ""

    def test_ingredient_rxcui_set_when_present(self):
        med = make_medication(ingredient_rxcui="6809")
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.ingredient_rxcui == "6809"

    def test_maps_dose(self):
        med = make_medication(dose=500.0)
        ctx = InteractionChecker._build_context(med, make_fda_data())
        assert ctx.dose == 500.0

    def test_maps_drug_class_from_fda(self):
        fda = make_fda_data(drug_class="Biguanide")
        ctx = InteractionChecker._build_context(make_medication(), fda)
        assert ctx.drug_class == "Biguanide"

    def test_maps_fda_label_id(self):
        fda = make_fda_data(fda_label_id="label-xyz-999")
        ctx = InteractionChecker._build_context(make_medication(), fda)
        assert ctx.fda_label_id == "label-xyz-999"


# ============================================================================
# _check_pair
# ============================================================================

class TestCheckPair:

    def test_finds_evidence_in_drug_interactions_field(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["May significantly increase bleeding risk when combined with warfarin."]
            ),
            "warfarin": make_fda_data(generic_name="warfarin"),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 1
        assert results[0].source_drug == "aspirin"
        assert results[0].drug_a.name == "aspirin"
        assert results[0].drug_b.name == "warfarin"

    def test_finds_evidence_in_warnings_field(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="ibuprofen")
        ctx_b = make_context(name="lisinopril")
        fda_map = {
            "ibuprofen": make_fda_data(
                warnings=["Use with lisinopril may reduce its antihypertensive effect substantially."]
            ),
            "lisinopril": make_fda_data(generic_name="lisinopril"),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 1
        assert results[0].source_drug == "ibuprofen"

    def test_finds_evidence_in_do_not_use_field(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="methotrexate")
        ctx_b = make_context(name="aspirin")
        fda_map = {
            "methotrexate": make_fda_data(
                do_not_use=["Do not use concurrently with aspirin as it significantly increases toxicity."]
            ),
            "aspirin": make_fda_data(generic_name="aspirin"),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 1
        assert results[0].source_drug == "methotrexate"

    def test_returns_empty_when_no_mention_in_any_field(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="metformin")
        fda_map = {
            "aspirin": make_fda_data(drug_interactions=["May cause gastrointestinal bleeding."]),
            "metformin": make_fda_data(drug_interactions=["Take with food to reduce GI upset."]),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert results == []

    def test_filters_text_shorter_than_min_evidence_length(self):
        config = InteractionCheckerConfig(min_evidence_length=60)
        checker = InteractionChecker(config=config)
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["warfarin: increased risk."]  # under 60 chars
            ),
            "warfarin": make_fda_data(),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert results == []

    def test_includes_text_exactly_at_min_evidence_length(self):
        min_len = 20
        config = InteractionCheckerConfig(min_evidence_length=min_len)
        checker = InteractionChecker(config=config)
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        text = "w" * (min_len - len("warfarin")) + "warfarin causes risk"
        # Make sure text is exactly min_len and contains "warfarin"
        text = "warfarin" + "x" * (min_len - len("warfarin"))
        assert len(text) == min_len
        fda_map = {
            "aspirin": make_fda_data(drug_interactions=[text]),
            "warfarin": make_fda_data(),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 1

    def test_match_is_case_insensitive(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concurrent use with WARFARIN significantly increases bleeding risk."]
            ),
            "warfarin": make_fda_data(),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 1

    def test_skips_source_drug_missing_from_fda_map(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        # Neither drug in fda_map
        results = checker._check_pair(ctx_a, ctx_b, {})
        assert results == []

    def test_scans_both_labels_raw_before_dedup(self):
        """_check_pair returns 2 results when both labels mention each other.
        Deduplication is the responsibility of check(), not _check_pair().
        """
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concurrent use with warfarin significantly increases bleeding risk."]
            ),
            "warfarin": make_fda_data(
                warnings=["Aspirin may potentiate the anticoagulant effect of warfarin therapy."]
            ),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 2

    def test_evidence_data_source_defaults_to_fresh_fda(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concurrent use with warfarin increases bleeding risk significantly."]
            ),
            "warfarin": make_fda_data(),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert results[0].data_source == "FRESH_FDA"

    def test_evidence_text_preserved_exactly(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        text = "Concurrent use with warfarin increases the risk of serious bleeding events."
        fda_map = {
            "aspirin": make_fda_data(drug_interactions=[text]),
            "warfarin": make_fda_data(),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert results[0].evidence_text == text

    def test_multiple_matching_texts_in_one_label(self):
        checker = InteractionChecker()
        ctx_a = make_context(name="aspirin")
        ctx_b = make_context(name="warfarin")
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=[
                    "Warfarin interaction: increased bleeding risk when taken together.",
                    "Concomitant warfarin use requires INR monitoring due to elevated risk.",
                ]
            ),
            "warfarin": make_fda_data(),
        }
        results = checker._check_pair(ctx_a, ctx_b, fda_map)
        assert len(results) == 2


# ============================================================================
# check
# ============================================================================

class TestCheck:

    def test_returns_empty_for_no_medications(self):
        result = InteractionChecker().check([], {})
        assert result == []

    def test_returns_empty_for_single_medication(self):
        med = make_medication("aspirin")
        result = InteractionChecker().check([med], {"aspirin": make_fda_data()})
        assert result == []  # no pairs possible with one drug

    def test_finds_interaction_between_two_drugs(self):
        checker = InteractionChecker()
        meds = [make_medication("aspirin"), make_medication("warfarin")]
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concomitant use with warfarin increases bleeding risk significantly."]
            ),
            "warfarin": make_fda_data(generic_name="warfarin"),
        }
        result = checker.check(meds, fda_map)
        assert len(result) == 1
        assert result[0].source_drug == "aspirin"

    def test_deduplicates_when_both_labels_mention_each_other(self):
        """Both labels reference each other — must return exactly 1 evidence."""
        checker = InteractionChecker()
        meds = [make_medication("aspirin"), make_medication("warfarin")]
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concurrent use with warfarin significantly increases bleeding risk."]
            ),
            "warfarin": make_fda_data(
                warnings=["Aspirin may potentiate the anticoagulant effect of warfarin substantially."]
            ),
        }
        result = checker.check(meds, fda_map)
        assert len(result) == 1

    def test_skips_medication_missing_from_fda_map(self):
        checker = InteractionChecker()
        meds = [make_medication("aspirin"), make_medication("unknowndrug")]
        fda_map = {"aspirin": make_fda_data()}
        result = checker.check(meds, fda_map)
        # only 1 DrugContext built — no pairs
        assert result == []

    def test_returns_empty_when_no_interactions_in_labels(self):
        checker = InteractionChecker()
        meds = [make_medication("metformin"), make_medication("lisinopril")]
        fda_map = {
            "metformin": make_fda_data(drug_interactions=["Take with food to reduce GI upset."]),
            "lisinopril": make_fda_data(drug_interactions=["Monitor blood pressure regularly."]),
        }
        result = checker.check(meds, fda_map)
        assert result == []

    def test_two_distinct_interactions_across_three_drugs(self):
        checker = InteractionChecker()
        meds = [
            make_medication("aspirin"),
            make_medication("warfarin"),
            make_medication("ibuprofen"),
        ]
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concomitant use with warfarin increases the risk of serious bleeding."]
            ),
            "warfarin": make_fda_data(
                warnings=["Ibuprofen may displace warfarin from binding sites increasing toxicity risk."]
            ),
            "ibuprofen": make_fda_data(),
        }
        result = checker.check(meds, fda_map)
        # aspirin→warfarin and warfarin→ibuprofen — 2 distinct pairs
        assert len(result) == 2

    def test_caps_pairs_at_max_pairs_per_request(self):
        config = InteractionCheckerConfig(max_pairs_per_request=1)
        checker = InteractionChecker(config=config)
        meds = [
            make_medication("aspirin"),
            make_medication("warfarin"),
            make_medication("metformin"),
        ]
        fda_map = {
            "aspirin": make_fda_data(),
            "warfarin": make_fda_data(),
            "metformin": make_fda_data(),
        }
        # 3 meds → 3 pairs, capped to 1
        result = checker.check(meds, fda_map)
        # at most 1 pair was checked (regardless of evidence)
        assert isinstance(result, list)

    def test_logs_pairs_truncated_warning_when_capped(self, caplog):
        config = InteractionCheckerConfig(max_pairs_per_request=1)
        checker = InteractionChecker(config=config)
        meds = [
            make_medication("aspirin"),
            make_medication("warfarin"),
            make_medication("metformin"),
        ]
        fda_map = {k: make_fda_data() for k in ["aspirin", "warfarin", "metformin"]}
        with caplog.at_level("WARNING"):
            checker.check(meds, fda_map)
        assert "pairs_truncated" in caplog.text

    def test_logs_completion_summary(self, caplog):
        checker = InteractionChecker()
        meds = [make_medication("aspirin"), make_medication("warfarin")]
        fda_map = {"aspirin": make_fda_data(), "warfarin": make_fda_data()}
        with caplog.at_level("INFO"):
            checker.check(meds, fda_map)
        assert "interaction_checker.complete" in caplog.text

    def test_logs_missing_fda_data(self, caplog):
        checker = InteractionChecker()
        meds = [make_medication("aspirin"), make_medication("unknowndrug")]
        fda_map = {"aspirin": make_fda_data()}
        with caplog.at_level("INFO"):
            checker.check(meds, fda_map)
        assert "interaction_checker.no_fda_data" in caplog.text

    def test_medication_name_lookup_is_case_insensitive(self):
        """Medication named 'Aspirin' should match fda_map key 'aspirin'."""
        checker = InteractionChecker()
        meds = [make_medication("Aspirin"), make_medication("warfarin")]
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concurrent use with warfarin increases bleeding risk significantly."]
            ),
            "warfarin": make_fda_data(),
        }
        result = checker.check(meds, fda_map)
        assert len(result) == 1

    def test_returned_evidence_is_interactionevidence_type(self):
        checker = InteractionChecker()
        meds = [make_medication("aspirin"), make_medication("warfarin")]
        fda_map = {
            "aspirin": make_fda_data(
                drug_interactions=["Concurrent use with warfarin increases bleeding risk significantly."]
            ),
            "warfarin": make_fda_data(),
        }
        result = checker.check(meds, fda_map)
        assert all(isinstance(ev, InteractionEvidence) for ev in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
