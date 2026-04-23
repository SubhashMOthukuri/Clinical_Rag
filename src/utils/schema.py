from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from datetime import datetime, timezone
from enum import Enum


class Severity(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class Action(str, Enum):
    STOP = "STOP"
    MONITOR = "MONITOR"
    CONSULT_DOCTOR = "CONSULT_DOCTOR"


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class DataSource(str, Enum):
    FRESH_FDA = "FRESH_FDA"
    CACHED_FDA = "CACHED_FDA"
    STATPEARLS_RAG = "STATPEARLS_RAG"
    FAERS = "FAERS"


class Unit(str, Enum):
    MG = "mg"
    MCG = "mcg"
    G = "g"
    ML = "mL"
    IU = "IU"
    UNIT = "unit"


# Applied to every model. extra="forbid" is your first line of defense
# against payload injection — unknown fields raise ValidationError.
PROD_CONFIG = ConfigDict(
    str_strip_whitespace=True,
    extra="forbid",
    validate_assignment=True,
)


class Medication(BaseModel):
    model_config = PROD_CONFIG
    name: str = Field(min_length=2, max_length=100)
    dose: float = Field(gt=0, le=10000)
    unit: Unit
    frequency: Optional[str] = Field(default=None, max_length=50)
    rxcui: Optional[str] = Field(default=None, pattern=r"^\d{1,10}$")
    drug_class: Optional[str] = Field(default=None, max_length=100)
    verified: bool = False


class DrugWarning(BaseModel):
    model_config = PROD_CONFIG
    drugs_involved: List[str] = Field(min_length=1)
    severity: Severity
    reaction_result: str = Field(min_length=1, max_length=1000)
    action: Action
    citation: List[str] = Field(min_length=1)  # every warning MUST cite
    nurse_summary_to_doctor: str = Field(min_length=1, max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_source: DataSource = DataSource.FRESH_FDA
    computed_at: datetime


class ReconciliationRequest(BaseModel):
    model_config = PROD_CONFIG
    medications: List[Medication] = Field(min_length=1, max_length=50)
    patient_id: Optional[str] = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    nurse_id: Optional[str] = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReconciliationResponse(BaseModel):
    model_config = PROD_CONFIG
    medications: List[Medication]
    warnings: List[DrugWarning]
    unverified_drugs: List[str]
    status: Status
    response_time_ms: float = Field(ge=0)
    computed_at: datetime
    total_medications: int = Field(ge=0)
    total_warnings: int = Field(ge=0)
    critical_warnings: int = Field(ge=0)