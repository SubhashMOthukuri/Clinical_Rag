from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum


class Severity(Enum):
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"

class Medication(BaseModel):
    name : str
    dose: float
    unit: str
    frequency: Optional[str]= None
    rxcui: Optional[str]=None
    drug_class: Optional[str]=None
    verified: bool = False

class Action(Enum):
    STOP = "STOP"
    MONITOR ="MONITOR"
    CONSULT_DOCTOR = "CONSULT_DOCTOR"


class Warning(BaseModel):
    drugs_involved: List[str]
    severity : Severity
    reaction_result : str
    action: Action
    citation : List[str] 
    nurse_summary_to_doctor : str
    confident: float = 0.0
    data_source :  str = "FRESH_FDA"
    compute_date : datetime

class ReconciliationRequest(BaseModel):
    medications : List[Medication]
    patient_id : Optional[str] = None
    nurse_id : Optional[str] = None
    submitted_at: datetime 

class Status(Enum):
    SUCCESS ="SUCCESS"
    PARTIAL ="PARTIAL"
    FAILED = "FAILED"

class ReconciliationResponse(BaseModel):
    medications: List[Medication]
    warnings : List[Warning]
    response_time_ms : float
    computed_at : datetime
    status : Status
    unverified_drugs: List[str]
    total_medications: int     
    total_warnings: int         
    critical_warnings: int  