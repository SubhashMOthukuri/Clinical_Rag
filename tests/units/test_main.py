"""Unit tests for src/main.py.

Covers:
- GET /health: status ok, component presence flags
- Correlation-ID middleware: client passthrough, UUID generation
- StageValidationError exception handler: 422 body structure
- POST /reconcile: happy path SUCCESS, no interactions, PARTIAL status,
  Stage 1 failure → 422, unverified drugs, Stage 5 not-ok (logs, returns),
  Stage 5 warnings (logs, returns), response counts
- _enrich_medications: RxcuiFound, RxcuiUnverified, FDA present/absent,
  ingredient_rxcui resolved when rxcui present, skipped when absent
- _build_response: critical_warnings counts RED only, totals, status, time
- Dependency injectors: each get_xxx returns correct app.state attribute
- Lifespan: startup sets all components on app.state, shutdown closes clients
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from src.embedding.embedder import GeminiEmbedder
from src.generation.generator import Generator
from src.ingestion.fda_client import FDAClient, FDADrugData
from src.ingestion.rxnorm_client import RxNormClient, RxcuiFound, RxcuiUnverified
from api.dependencies import (
    get_fda,
    get_generator,
    get_interaction_checker,
    get_retriever,
    get_rxnorm,
)
from api.routes.reconcile import _build_response, _enrich_medications
from src.main import app
from src.retrieval.interaction_checker import (
    DrugContext,
    InteractionChecker,
    InteractionEvidence,
)
from src.retrieval.pinecone_store import PineconeStore
from src.retrieval.retrieval import RetrievalResult, Retriever
from src.utils.schema import (
    Action,
    DataSource,
    DrugWarning,
    Medication,
    ReconciliationRequest,
    Severity,
    Status,
    Unit,
)
from src.utils.validators import StageValidationError, ValidationResult, ValidationStage

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ============================================================================
# Shared helpers
# ============================================================================

def _med(name: str = "Aspirin", dose: float = 100.0, unit: Unit = Unit.MG) -> Medication:
    return Medication(name=name, dose=dose, unit=unit)


def _fda_data(name: str = "aspirin") -> FDADrugData:
    return FDADrugData(
        generic_name=name,
        rxcui="1191",
        drug_class="Analgesic",
        warnings=["May cause bleeding"],
        drug_interactions=["Warfarin: increased bleeding risk"],
        do_not_use=[],
        ask_doctor=[],
        source="FDA",
        fetched_at=datetime.now(timezone.utc),
        fda_label_id="aspirin-label",
    )


def _drug_ctx(name: str = "aspirin") -> DrugContext:
    return DrugContext(
        name=name,
        dose=100.0,
        unit="mg",
        ingredient_rxcui="1191",
        drug_class="Analgesic",
        fda_label_id="aspirin-label",
    )


def _evidence() -> InteractionEvidence:
    return InteractionEvidence(
        drug_a=_drug_ctx("aspirin"),
        drug_b=_drug_ctx("warfarin"),
        evidence_text="Concurrent use increases bleeding risk significantly",
        source_drug="aspirin",
        estimated_severity="RED",
    )


def _warning(confidence: float = 1.0, severity: Severity = Severity.RED) -> DrugWarning:
    return DrugWarning(
        drugs_involved=["Aspirin", "Warfarin"],
        severity=severity,
        reaction_result="Increased bleeding risk",
        action=Action.STOP,
        citation=["FDA label for Aspirin"],
        nurse_summary_to_doctor="These drugs should not be combined",
        confidence=confidence,
        data_source=DataSource.FRESH_FDA,
        computed_at=datetime.now(timezone.utc),
    )


def _payload(*medications: Medication) -> dict:
    return {
        "medications": [
            {"name": m.name, "dose": m.dose, "unit": m.unit.value}
            for m in medications
        ]
    }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mocks():
    embedder = AsyncMock(spec=GeminiEmbedder)
    embedder.close = AsyncMock()

    store = MagicMock(spec=PineconeStore)
    store.close = AsyncMock()

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve_many = AsyncMock(return_value=[])

    rxnorm = MagicMock(spec=RxNormClient)
    rxnorm.aclose = AsyncMock()
    rxnorm.get_rxcui = AsyncMock(return_value=RxcuiFound("1191"))
    rxnorm.get_ingredient_rxcui = AsyncMock(return_value=None)

    fda = MagicMock(spec=FDAClient)
    fda.aclose = AsyncMock()
    fda.get_drug_data = AsyncMock(return_value=None)

    checker = MagicMock(spec=InteractionChecker)
    checker.check = MagicMock(return_value=[])

    generator = MagicMock(spec=Generator)
    generator.generate_many = AsyncMock(return_value=[])

    return dict(
        embedder=embedder,
        store=store,
        retriever=retriever,
        rxnorm=rxnorm,
        fda=fda,
        checker=checker,
        generator=generator,
    )


@pytest.fixture
def client(mocks):
    """TestClient with all lifespan components replaced by mocks."""
    with (
        patch("src.main.GeminiEmbedder", return_value=mocks["embedder"]),
        patch("src.main.PineconeStore", return_value=mocks["store"]),
        patch("src.main.Retriever", return_value=mocks["retriever"]),
        patch("src.main.RxNormClient", return_value=mocks["rxnorm"]),
        patch("src.main.FDAClient", return_value=mocks["fda"]),
        patch("src.main.InteractionChecker", return_value=mocks["checker"]),
        patch("src.main.Generator", return_value=mocks["generator"]),
    ):
        with TestClient(app) as c:
            yield c, mocks


# ============================================================================
# GET /health
# ============================================================================

class TestHealthEndpoint:
    def test_returns_status_ok(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_lists_all_components_as_present(self, client):
        c, _ = client
        components = c.get("/health").json()["components"]
        assert components["retriever"] is True
        assert components["generator"] is True
        assert components["rxnorm"] is True
        assert components["fda"] is True

    def test_component_missing_returns_false(self, client):
        c, _ = client
        del app.state.retriever
        components = c.get("/health").json()["components"]
        assert components["retriever"] is False
        # Restore so other tests see clean state
        app.state.retriever = _


# ============================================================================
# Correlation-ID middleware
# ============================================================================

class TestCorrelationIdMiddleware:
    def test_client_supplied_id_echoed_in_response(self, client):
        c, _ = client
        resp = c.get("/health", headers={"X-Correlation-ID": "clinic-42"})
        assert resp.headers["X-Correlation-ID"] == "clinic-42"

    def test_generates_uuid_when_header_absent(self, client):
        c, _ = client
        resp = c.get("/health")
        cid = resp.headers.get("X-Correlation-ID")
        assert cid is not None
        uuid.UUID(cid)  # raises ValueError if not a valid UUID4

    def test_each_request_gets_unique_id(self, client):
        c, _ = client
        cid1 = c.get("/health").headers["X-Correlation-ID"]
        cid2 = c.get("/health").headers["X-Correlation-ID"]
        assert cid1 != cid2


# ============================================================================
# StageValidationError exception handler
# ============================================================================

class TestStageValidationExceptionHandler:
    def test_returns_422(self, client):
        c, _ = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(
                ValidationStage.INPUT, "too many meds", {"count": 99}
            )
            resp = c.post("/reconcile", json=_payload(_med()))
        assert resp.status_code == 422

    def test_body_contains_error_field(self, client):
        c, _ = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(ValidationStage.INPUT, "bad input")
            resp = c.post("/reconcile", json=_payload(_med()))
        assert resp.json()["error"] == "validation_failed"

    def test_body_contains_stage_value(self, client):
        c, _ = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(ValidationStage.INPUT, "bad input")
            resp = c.post("/reconcile", json=_payload(_med()))
        assert resp.json()["stage"] == ValidationStage.INPUT.value

    def test_body_contains_message_and_details(self, client):
        c, _ = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(
                ValidationStage.INPUT, "dup med", {"name": "Aspirin"}
            )
            resp = c.post("/reconcile", json=_payload(_med()))
        body = resp.json()
        assert body["message"] == "dup med"
        assert body["details"] == {"name": "Aspirin"}

    def test_body_contains_correlation_id(self, client):
        c, _ = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(ValidationStage.INPUT, "x")
            resp = c.post(
                "/reconcile",
                json=_payload(_med()),
                headers={"X-Correlation-ID": "cid-test"},
            )
        assert resp.json()["correlation_id"] == "cid-test"


# ============================================================================
# POST /reconcile — happy path
# ============================================================================

class TestReconcileHappyPath:
    def test_success_status_with_interactions(self, client):
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        result = RetrievalResult(evidence=ev, chunks=[])
        mocks["retriever"].retrieve_many.return_value = [result]
        mocks["generator"].generate_many.return_value = [_warning(confidence=1.0)]
        mocks["rxnorm"].get_rxcui.return_value = RxcuiFound("1191")

        resp = c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0)))

        assert resp.status_code == 200
        assert resp.json()["status"] == Status.SUCCESS.value

    def test_response_counts_match_lists(self, client):
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        mocks["retriever"].retrieve_many.return_value = [RetrievalResult(evidence=ev, chunks=[])]
        mocks["generator"].generate_many.return_value = [_warning()]
        mocks["rxnorm"].get_rxcui.return_value = RxcuiFound("1191")

        body = c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0))).json()

        assert body["total_medications"] == 2
        assert body["total_warnings"] == 1
        assert body["critical_warnings"] == 1  # severity=RED

    def test_medications_present_in_response(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []

        body = c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0))).json()

        names = {m["name"] for m in body["medications"]}
        assert "Aspirin" in names
        assert "Warfarin" in names

    def test_response_time_ms_is_non_negative(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []

        body = c.post("/reconcile", json=_payload(_med())).json()

        assert body["response_time_ms"] >= 0

    def test_correlation_id_threaded_to_retriever(self, client):
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        mocks["retriever"].retrieve_many.return_value = [RetrievalResult(evidence=ev, chunks=[])]
        mocks["generator"].generate_many.return_value = [_warning()]

        c.post(
            "/reconcile",
            json=_payload(_med("Aspirin"), _med("Warfarin", 5.0)),
            headers={"X-Correlation-ID": "cid-trace"},
        )

        _, kwargs = mocks["retriever"].retrieve_many.call_args
        assert kwargs["correlation_id"] == "cid-trace"


# ============================================================================
# POST /reconcile — no interactions detected
# ============================================================================

class TestReconcileNoInteractions:
    def test_returns_success_with_empty_warnings(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []

        body = c.post("/reconcile", json=_payload(_med())).json()

        assert body["status"] == Status.SUCCESS.value
        assert body["warnings"] == []

    def test_retriever_not_called_when_no_evidence(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []

        c.post("/reconcile", json=_payload(_med()))

        mocks["retriever"].retrieve_many.assert_not_called()

    def test_generator_not_called_when_no_evidence(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []

        c.post("/reconcile", json=_payload(_med()))

        mocks["generator"].generate_many.assert_not_called()


# ============================================================================
# POST /reconcile — PARTIAL status
# ============================================================================

class TestReconcilePartialStatus:
    def test_partial_when_any_warning_confidence_below_one(self, client):
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        mocks["retriever"].retrieve_many.return_value = [RetrievalResult(evidence=ev, chunks=[])]
        mocks["generator"].generate_many.return_value = [_warning(confidence=0.5)]

        body = c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0))).json()

        assert body["status"] == Status.PARTIAL.value

    def test_success_when_all_warnings_have_full_confidence(self, client):
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        mocks["retriever"].retrieve_many.return_value = [RetrievalResult(evidence=ev, chunks=[])]
        mocks["generator"].generate_many.return_value = [_warning(confidence=1.0)]

        body = c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0))).json()

        assert body["status"] == Status.SUCCESS.value

    def test_partial_with_mixed_confidence(self, client):
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        mocks["retriever"].retrieve_many.return_value = [RetrievalResult(evidence=ev, chunks=[])]
        mocks["generator"].generate_many.return_value = [
            _warning(confidence=1.0),
            _warning(confidence=0.3, severity=Severity.YELLOW),
        ]

        body = c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0))).json()

        assert body["status"] == Status.PARTIAL.value


# ============================================================================
# POST /reconcile — Stage 1 validation failure
# ============================================================================

class TestReconcileStage1Failure:
    def test_validate_input_error_returns_422(self, client):
        c, _ = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(
                ValidationStage.INPUT, "duplicate medication", {"name": "Aspirin"}
            )
            resp = c.post("/reconcile", json=_payload(_med()))
        assert resp.status_code == 422

    def test_pipeline_stops_after_stage1_failure(self, client):
        c, mocks = client
        with patch("api.routes.reconcile.validate_input") as vi:
            vi.side_effect = StageValidationError(ValidationStage.INPUT, "bad")
            c.post("/reconcile", json=_payload(_med()))
        mocks["checker"].check.assert_not_called()
        mocks["retriever"].retrieve_many.assert_not_called()
        mocks["generator"].generate_many.assert_not_called()

    def test_invalid_pydantic_payload_returns_422(self, client):
        c, _ = client
        resp = c.post("/reconcile", json={"medications": []})
        assert resp.status_code == 422


# ============================================================================
# POST /reconcile — unverified drugs
# ============================================================================

class TestReconcileUnverifiedDrugs:
    def test_unverified_drug_appears_in_unverified_drugs(self, client):
        c, mocks = client
        mocks["rxnorm"].get_rxcui.return_value = RxcuiUnverified("Aspirin")
        mocks["checker"].check.return_value = []

        body = c.post("/reconcile", json=_payload(_med("Aspirin"))).json()

        assert "Aspirin" in body["unverified_drugs"]

    def test_verified_drug_not_in_unverified_drugs(self, client):
        c, mocks = client
        mocks["rxnorm"].get_rxcui.return_value = RxcuiFound("1191")
        mocks["checker"].check.return_value = []

        body = c.post("/reconcile", json=_payload(_med("Aspirin"))).json()

        assert "Aspirin" not in body["unverified_drugs"]

    def test_mixed_verified_unverified(self, client):
        c, mocks = client

        def side_effect(name, correlation_id=None):
            if name == "Aspirin":
                return RxcuiFound("1191")
            return RxcuiUnverified(name)

        mocks["rxnorm"].get_rxcui.side_effect = side_effect
        mocks["checker"].check.return_value = []

        body = c.post(
            "/reconcile",
            json=_payload(_med("Aspirin"), _med("Warfarin", 5.0)),
        ).json()

        assert "Warfarin" in body["unverified_drugs"]
        assert "Aspirin" not in body["unverified_drugs"]


# ============================================================================
# POST /reconcile — Stage 5 response validation
# ============================================================================

class TestReconcileStage5Validation:
    def test_stage5_not_ok_still_returns_200(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []
        bad_result = ValidationResult(ok=False, errors=["silent drops detected: []"])
        with patch("api.routes.reconcile.validate_response", return_value=bad_result):
            resp = c.post("/reconcile", json=_payload(_med()))
        assert resp.status_code == 200

    def test_stage5_warnings_still_returns_200(self, client):
        c, mocks = client
        mocks["checker"].check.return_value = []
        warn_result = ValidationResult(ok=True, warnings=["latency exceeded"])
        with patch("api.routes.reconcile.validate_response", return_value=warn_result):
            resp = c.post("/reconcile", json=_payload(_med()))
        assert resp.status_code == 200

    def test_stage5_not_ok_errors_are_logged(self, client, caplog):
        import logging
        c, mocks = client
        ev = _evidence()
        mocks["checker"].check.return_value = [ev]
        mocks["retriever"].retrieve_many.return_value = [RetrievalResult(evidence=ev, chunks=[])]
        mocks["generator"].generate_many.return_value = [_warning()]
        bad_result = ValidationResult(ok=False, errors=["count mismatch"])
        with patch("api.routes.reconcile.validate_response", return_value=bad_result):
            with caplog.at_level(logging.ERROR, logger="api.routes.reconcile"):
                c.post("/reconcile", json=_payload(_med("Aspirin"), _med("Warfarin", 5.0)))
        assert any("response_validation_failed" in r.getMessage() for r in caplog.records)


# ============================================================================
# _enrich_medications helper
# ============================================================================

class TestEnrichMedications:
    @pytest.mark.asyncio
    async def test_rxcui_found_sets_verified_and_rxcui(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiFound("1191")
        rxnorm.get_ingredient_rxcui.return_value = None
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = None

        enriched, _, _ = await _enrich_medications([_med("Aspirin")], rxnorm, fda, "cid-1")

        assert enriched[0].rxcui == "1191"
        assert enriched[0].verified is True

    @pytest.mark.asyncio
    async def test_rxcui_unverified_sets_verified_false_and_appends_unverified(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiUnverified("Aspirin")
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = None

        enriched, fda_map, unverified = await _enrich_medications(
            [_med("Aspirin")], rxnorm, fda, "cid-2"
        )

        assert enriched[0].rxcui is None
        assert enriched[0].verified is False
        assert "Aspirin" in unverified

    @pytest.mark.asyncio
    async def test_fda_data_present_added_to_fda_map(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiFound("1191")
        rxnorm.get_ingredient_rxcui.return_value = None
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = _fda_data("aspirin")

        _, fda_map, _ = await _enrich_medications([_med("Aspirin")], rxnorm, fda, "cid-3")

        assert "aspirin" in fda_map

    @pytest.mark.asyncio
    async def test_fda_data_none_not_in_fda_map(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiFound("1191")
        rxnorm.get_ingredient_rxcui.return_value = None
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = None

        _, fda_map, _ = await _enrich_medications([_med("Aspirin")], rxnorm, fda, "cid-4")

        assert "aspirin" not in fda_map

    @pytest.mark.asyncio
    async def test_ingredient_rxcui_resolved_when_rxcui_present(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiFound("1191")
        rxnorm.get_ingredient_rxcui.return_value = "5640"
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = None

        enriched, _, _ = await _enrich_medications([_med("Aspirin")], rxnorm, fda, "cid-5")

        rxnorm.get_ingredient_rxcui.assert_called_once_with("1191")
        assert enriched[0].ingredient_rxcui == "5640"

    @pytest.mark.asyncio
    async def test_ingredient_rxcui_not_called_when_rxcui_absent(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiUnverified("Aspirin")
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = None

        await _enrich_medications([_med("Aspirin")], rxnorm, fda, "cid-6")

        rxnorm.get_ingredient_rxcui.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_medications_run_in_parallel(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.side_effect = [
            RxcuiFound("1191"),
            RxcuiFound("11289"),
        ]
        rxnorm.get_ingredient_rxcui.return_value = None
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = None

        enriched, _, unverified = await _enrich_medications(
            [_med("Aspirin"), _med("Warfarin", 5.0)], rxnorm, fda, "cid-7"
        )

        assert len(enriched) == 2
        assert unverified == []

    @pytest.mark.asyncio
    async def test_fda_map_key_is_lowercase(self):
        rxnorm = AsyncMock(spec=RxNormClient)
        rxnorm.get_rxcui.return_value = RxcuiFound("1191")
        rxnorm.get_ingredient_rxcui.return_value = None
        fda = AsyncMock(spec=FDAClient)
        fda.get_drug_data.return_value = _fda_data("aspirin")

        _, fda_map, _ = await _enrich_medications([_med("Aspirin")], rxnorm, fda, "cid-8")

        assert "aspirin" in fda_map
        assert "Aspirin" not in fda_map


# ============================================================================
# _build_response helper
# ============================================================================

class TestBuildResponse:
    def _make_response(self, warnings=None, unverified=None, status=Status.SUCCESS, ms=10.0):
        medications = [_med("Aspirin"), _med("Warfarin", 5.0)]
        warnings = warnings or []
        unverified = unverified or []
        return _build_response(
            medications=medications,
            warnings=warnings,
            unverified=unverified,
            status=status,
            response_time_ms=ms,
        )

    def test_total_medications_matches_list_length(self):
        resp = self._make_response()
        assert resp.total_medications == 2

    def test_total_warnings_matches_list_length(self):
        resp = self._make_response(warnings=[_warning(), _warning()])
        assert resp.total_warnings == 2

    def test_critical_warnings_counts_only_red(self):
        warnings = [
            _warning(severity=Severity.RED),
            _warning(severity=Severity.YELLOW),
            _warning(severity=Severity.RED),
        ]
        resp = self._make_response(warnings=warnings)
        assert resp.critical_warnings == 2

    def test_critical_warnings_zero_when_no_red(self):
        warnings = [_warning(severity=Severity.YELLOW), _warning(severity=Severity.GREEN)]
        resp = self._make_response(warnings=warnings)
        assert resp.critical_warnings == 0

    def test_status_passed_through(self):
        resp = self._make_response(status=Status.PARTIAL)
        assert resp.status == Status.PARTIAL

    def test_response_time_ms_passed_through(self):
        resp = self._make_response(ms=123.45)
        assert resp.response_time_ms == pytest.approx(123.45)

    def test_computed_at_is_utc(self):
        resp = self._make_response()
        assert resp.computed_at.tzinfo is not None

    def test_unverified_drugs_passed_through(self):
        resp = self._make_response(unverified=["Aspirin"])
        assert "Aspirin" in resp.unverified_drugs


# ============================================================================
# Dependency injectors
# ============================================================================

class TestDependencyInjectors:
    def _make_request(self, **state_attrs):
        req = MagicMock()
        for k, v in state_attrs.items():
            setattr(req.app.state, k, v)
        return req

    def test_get_rxnorm_returns_state_rxnorm(self):
        sentinel = object()
        req = self._make_request(rxnorm=sentinel)
        assert get_rxnorm(req) is sentinel

    def test_get_fda_returns_state_fda(self):
        sentinel = object()
        req = self._make_request(fda=sentinel)
        assert get_fda(req) is sentinel

    def test_get_interaction_checker_returns_state_checker(self):
        sentinel = object()
        req = self._make_request(interaction_checker=sentinel)
        assert get_interaction_checker(req) is sentinel

    def test_get_retriever_returns_state_retriever(self):
        sentinel = object()
        req = self._make_request(retriever=sentinel)
        assert get_retriever(req) is sentinel

    def test_get_generator_returns_state_generator(self):
        sentinel = object()
        req = self._make_request(generator=sentinel)
        assert get_generator(req) is sentinel


# ============================================================================
# Lifespan: startup & shutdown
# ============================================================================

class TestLifespan:
    def test_startup_sets_all_components_on_app_state(self, client):
        c, mocks = client
        assert hasattr(app.state, "embedder")
        assert hasattr(app.state, "store")
        assert hasattr(app.state, "retriever")
        assert hasattr(app.state, "rxnorm")
        assert hasattr(app.state, "fda")
        assert hasattr(app.state, "interaction_checker")
        assert hasattr(app.state, "generator")

    def test_startup_stores_mock_instances_on_state(self, client):
        c, mocks = client
        assert app.state.embedder is mocks["embedder"]
        assert app.state.rxnorm is mocks["rxnorm"]
        assert app.state.fda is mocks["fda"]
        assert app.state.generator is mocks["generator"]

    def test_shutdown_closes_embedder(self, mocks):
        with (
            patch("src.main.GeminiEmbedder", return_value=mocks["embedder"]),
            patch("src.main.PineconeStore", return_value=mocks["store"]),
            patch("src.main.Retriever", return_value=mocks["retriever"]),
            patch("src.main.RxNormClient", return_value=mocks["rxnorm"]),
            patch("src.main.FDAClient", return_value=mocks["fda"]),
            patch("src.main.InteractionChecker", return_value=mocks["checker"]),
            patch("src.main.Generator", return_value=mocks["generator"]),
        ):
            with TestClient(app):
                pass  # lifespan runs startup + shutdown here
        mocks["embedder"].close.assert_called_once()

    def test_shutdown_closes_pinecone_store(self, mocks):
        with (
            patch("src.main.GeminiEmbedder", return_value=mocks["embedder"]),
            patch("src.main.PineconeStore", return_value=mocks["store"]),
            patch("src.main.Retriever", return_value=mocks["retriever"]),
            patch("src.main.RxNormClient", return_value=mocks["rxnorm"]),
            patch("src.main.FDAClient", return_value=mocks["fda"]),
            patch("src.main.InteractionChecker", return_value=mocks["checker"]),
            patch("src.main.Generator", return_value=mocks["generator"]),
        ):
            with TestClient(app):
                pass
        mocks["store"].close.assert_called_once()

    def test_shutdown_closes_rxnorm_client(self, mocks):
        with (
            patch("src.main.GeminiEmbedder", return_value=mocks["embedder"]),
            patch("src.main.PineconeStore", return_value=mocks["store"]),
            patch("src.main.Retriever", return_value=mocks["retriever"]),
            patch("src.main.RxNormClient", return_value=mocks["rxnorm"]),
            patch("src.main.FDAClient", return_value=mocks["fda"]),
            patch("src.main.InteractionChecker", return_value=mocks["checker"]),
            patch("src.main.Generator", return_value=mocks["generator"]),
        ):
            with TestClient(app):
                pass
        mocks["rxnorm"].aclose.assert_called_once()

    def test_shutdown_closes_fda_client(self, mocks):
        with (
            patch("src.main.GeminiEmbedder", return_value=mocks["embedder"]),
            patch("src.main.PineconeStore", return_value=mocks["store"]),
            patch("src.main.Retriever", return_value=mocks["retriever"]),
            patch("src.main.RxNormClient", return_value=mocks["rxnorm"]),
            patch("src.main.FDAClient", return_value=mocks["fda"]),
            patch("src.main.InteractionChecker", return_value=mocks["checker"]),
            patch("src.main.Generator", return_value=mocks["generator"]),
        ):
            with TestClient(app):
                pass
        mocks["fda"].aclose.assert_called_once()
