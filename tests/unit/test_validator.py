"""
Test suite for src/utils/validator.py

Tests the 5-stage validation pipeline for medication reconciliation.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.utils.schema import (
    Medication,
    ReconciliationRequest,
    ReconciliationResponse,
    DrugWarning,
    Unit,
    Severity,
    Action,
    Status,
    DataSource,
)


class TestMedicationValidation:
    """Test Medication schema validation."""

    def test_valid_medication(self):
        """Test creating valid medication."""
        med = Medication(
            name="aspirin",
            dose=100.0,
            unit=Unit.MG,
            frequency="once daily",
            verified=True
        )
        assert med.name == "aspirin"
        assert med.dose == 100.0
        assert med.verified is True

    def test_medication_min_dose(self):
        """Test medication dose must be > 0."""
        with pytest.raises(ValidationError) as exc:
            Medication(
                name="aspirin",
                dose=0,  # Invalid: must be > 0
                unit=Unit.MG
            )
        assert "greater than 0" in str(exc.value)

    def test_medication_max_dose(self):
        """Test medication dose must be <= 10000."""
        with pytest.raises(ValidationError) as exc:
            Medication(
                name="aspirin",
                dose=10001,  # Invalid: exceeds max
                unit=Unit.MG
            )
        assert "less than or equal to 10000" in str(exc.value)

    def test_medication_name_too_short(self):
        """Test medication name minimum length."""
        with pytest.raises(ValidationError) as exc:
            Medication(
                name="a",  # Invalid: min_length=2
                dose=100.0,
                unit=Unit.MG
            )
        assert "at least 2 characters" in str(exc.value)

    def test_medication_name_too_long(self):
        """Test medication name maximum length."""
        with pytest.raises(ValidationError) as exc:
            Medication(
                name="a" * 101,  # Invalid: max_length=100
                dose=100.0,
                unit=Unit.MG
            )
        assert "at most 100 characters" in str(exc.value)

    def test_medication_invalid_unit(self):
        """Test medication with invalid unit."""
        with pytest.raises(ValidationError):
            Medication(
                name="aspirin",
                dose=100.0,
                unit="invalid_unit"  # Invalid unit
            )

    def test_medication_optional_fields(self):
        """Test medication with optional fields."""
        med = Medication(
            name="metformin",
            dose=500.0,
            unit=Unit.MG,
            rxcui="6809",
            drug_class="biguanide"
        )
        assert med.rxcui == "6809"
        assert med.drug_class == "biguanide"

    def test_medication_rxcui_pattern(self):
        """Test RXCUI must be numeric."""
        with pytest.raises(ValidationError) as exc:
            Medication(
                name="aspirin",
                dose=100.0,
                unit=Unit.MG,
                rxcui="invalid_rxcui"
            )
        assert "String should match pattern" in str(exc.value)


class TestReconciliationRequest:
    """Test ReconciliationRequest schema validation."""

    def test_valid_request(self):
        """Test creating valid reconciliation request."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        req = ReconciliationRequest(
            medications=[med],
            patient_id="PAT_12345",
            nurse_id="NURSE_789"
        )
        assert len(req.medications) == 1
        assert req.patient_id == "PAT_12345"

    def test_request_empty_medications(self):
        """Test request must have at least 1 medication."""
        with pytest.raises(ValidationError) as exc:
            ReconciliationRequest(medications=[])
        assert "at least 1 item" in str(exc.value)

    def test_request_too_many_medications(self):
        """Test request max 50 medications."""
        meds = [
            Medication(name=f"drug_{i}", dose=100.0, unit=Unit.MG)
            for i in range(51)
        ]
        with pytest.raises(ValidationError) as exc:
            ReconciliationRequest(medications=meds)
        assert "at most 50 items" in str(exc.value)

    def test_request_patient_id_pattern(self):
        """Test patient_id must match pattern [A-Za-z0-9_-]."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        
        # Valid
        req = ReconciliationRequest(
            medications=[med],
            patient_id="PAT-123_456"
        )
        assert req.patient_id == "PAT-123_456"

        # Invalid
        with pytest.raises(ValidationError):
            ReconciliationRequest(
                medications=[med],
                patient_id="PAT@123"  # @ not allowed
            )

    def test_request_submitted_at_auto(self):
        """Test submitted_at defaults to now."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        req = ReconciliationRequest(medications=[med])
        assert req.submitted_at is not None
        assert isinstance(req.submitted_at, datetime)


class TestDrugWarning:
    """Test DrugWarning schema validation."""

    def test_valid_warning(self):
        """Test creating valid drug warning."""
        warning = DrugWarning(
            drugs_involved=["aspirin", "warfarin"],
            severity=Severity.RED,
            reaction_result="increased bleeding risk",
            action=Action.STOP,
            citation=["FDA_001"],
            nurse_summary_to_doctor="Stop aspirin due to warfarin interaction",
            confidence=0.95,
            data_source=DataSource.FRESH_FDA,
            computed_at=datetime.now(timezone.utc)
        )
        assert len(warning.drugs_involved) == 2
        assert warning.severity == Severity.RED
        assert warning.confidence == 0.95

    def test_warning_empty_drugs(self):
        """Test warning must have at least 1 drug."""
        with pytest.raises(ValidationError) as exc:
            DrugWarning(
                drugs_involved=[],
                severity=Severity.RED,
                reaction_result="test",
                action=Action.STOP,
                citation=["CITE"],
                nurse_summary_to_doctor="test",
                computed_at=datetime.now(timezone.utc)
            )
        assert "at least 1 item" in str(exc.value)

    def test_warning_confidence_bounds(self):
        """Test confidence must be between 0.0 and 1.0."""
        # Valid: 0.0
        warning = DrugWarning(
            drugs_involved=["aspirin"],
            severity=Severity.YELLOW,
            reaction_result="test",
            action=Action.MONITOR,
            citation=["CITE"],
            nurse_summary_to_doctor="test",
            confidence=0.0,
            computed_at=datetime.now(timezone.utc)
        )
        assert warning.confidence == 0.0

        # Valid: 1.0
        warning2 = DrugWarning(
            drugs_involved=["aspirin"],
            severity=Severity.YELLOW,
            reaction_result="test",
            action=Action.MONITOR,
            citation=["CITE"],
            nurse_summary_to_doctor="test",
            confidence=1.0,
            computed_at=datetime.now(timezone.utc)
        )
        assert warning2.confidence == 1.0

        # Invalid: > 1.0
        with pytest.raises(ValidationError):
            DrugWarning(
                drugs_involved=["aspirin"],
                severity=Severity.YELLOW,
                reaction_result="test",
                action=Action.MONITOR,
                citation=["CITE"],
                nurse_summary_to_doctor="test",
                confidence=1.5,
                computed_at=datetime.now(timezone.utc)
            )

        # Invalid: < 0.0
        with pytest.raises(ValidationError):
            DrugWarning(
                drugs_involved=["aspirin"],
                severity=Severity.YELLOW,
                reaction_result="test",
                action=Action.MONITOR,
                citation=["CITE"],
                nurse_summary_to_doctor="test",
                confidence=-0.1,
                computed_at=datetime.now(timezone.utc)
            )

    def test_warning_empty_reaction_result(self):
        """Test reaction_result cannot be empty."""
        with pytest.raises(ValidationError):
            DrugWarning(
                drugs_involved=["aspirin"],
                severity=Severity.RED,
                reaction_result="",  # Invalid
                action=Action.STOP,
                citation=["CITE"],
                nurse_summary_to_doctor="test",
                computed_at=datetime.now(timezone.utc)
            )

    def test_warning_no_citations(self):
        """Test warning must have at least 1 citation."""
        with pytest.raises(ValidationError):
            DrugWarning(
                drugs_involved=["aspirin"],
                severity=Severity.RED,
                reaction_result="test reaction",
                action=Action.STOP,
                citation=[],  # Invalid: must have at least 1
                nurse_summary_to_doctor="test",
                computed_at=datetime.now(timezone.utc)
            )


class TestReconciliationResponse:
    """Test ReconciliationResponse schema validation."""

    def test_valid_response(self):
        """Test creating valid response."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        warning = DrugWarning(
            drugs_involved=["aspirin"],
            severity=Severity.YELLOW,
            reaction_result="test",
            action=Action.MONITOR,
            citation=["CITE"],
            nurse_summary_to_doctor="test",
            computed_at=datetime.now(timezone.utc)
        )
        
        response = ReconciliationResponse(
            medications=[med],
            warnings=[warning],
            unverified_drugs=[],
            status=Status.SUCCESS,
            response_time_ms=150.5,
            computed_at=datetime.now(timezone.utc),
            total_medications=1,
            total_warnings=1,
            critical_warnings=0
        )
        assert response.total_medications == 1
        assert response.total_warnings == 1
        assert response.status == Status.SUCCESS

    def test_response_negative_latency(self):
        """Test response_time_ms cannot be negative."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        
        with pytest.raises(ValidationError):
            ReconciliationResponse(
                medications=[med],
                warnings=[],
                unverified_drugs=[],
                status=Status.SUCCESS,
                response_time_ms=-100.0,  # Invalid
                computed_at=datetime.now(timezone.utc),
                total_medications=1,
                total_warnings=0,
                critical_warnings=0
            )

    def test_response_negative_counts(self):
        """Test count fields cannot be negative."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        
        with pytest.raises(ValidationError):
            ReconciliationResponse(
                medications=[med],
                warnings=[],
                unverified_drugs=[],
                status=Status.SUCCESS,
                response_time_ms=100.0,
                computed_at=datetime.now(timezone.utc),
                total_medications=-1,  # Invalid
                total_warnings=0,
                critical_warnings=0
            )


class TestExtraFieldsRejected:
    """Test that extra fields are rejected (extra='forbid')."""

    def test_medication_rejects_extra_fields(self):
        """Test Medication rejects unknown fields."""
        with pytest.raises(ValidationError) as exc:
            Medication(
                name="aspirin",
                dose=100.0,
                unit=Unit.MG,
                unknown_field="should_fail"
            )
        assert "Extra inputs are not permitted" in str(exc.value)

    def test_request_rejects_extra_fields(self):
        """Test ReconciliationRequest rejects unknown fields."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        with pytest.raises(ValidationError):
            ReconciliationRequest(
                medications=[med],
                unknown_field="should_fail"
            )

    def test_response_rejects_extra_fields(self):
        """Test ReconciliationResponse rejects unknown fields."""
        med = Medication(name="aspirin", dose=100.0, unit=Unit.MG)
        with pytest.raises(ValidationError):
            ReconciliationResponse(
                medications=[med],
                warnings=[],
                unverified_drugs=[],
                status=Status.SUCCESS,
                response_time_ms=100.0,
                computed_at=datetime.now(timezone.utc),
                total_medications=1,
                total_warnings=0,
                critical_warnings=0,
                unknown_field="should_fail"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
