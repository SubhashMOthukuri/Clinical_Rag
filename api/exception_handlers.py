"""Global FastAPI exception handlers.

Registered on the app in src/main.py via app.add_exception_handler().
Kept here so main.py stays a thin wiring file.
"""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.utils.validators import StageValidationError

logger = logging.getLogger(__name__)


async def stage_validation_handler(request: Request, exc: StageValidationError) -> JSONResponse:
    """Map StageValidationError → 422 with structured body.

    StageValidationError carries a stage label, human message, and details
    dict — surface all three so the nurse UI can show specific feedback and
    on-call engineers can grep by stage in logs.
    """
    cid = getattr(request.state, "correlation_id", "unknown")
    logger.warning(
        "validation.failed",
        extra={"cid": cid, "stage": exc.stage.value, "validation_message": exc.message},
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "stage": exc.stage.value,
            "message": exc.message,
            "details": exc.details,
            "correlation_id": cid,
        },
    )
