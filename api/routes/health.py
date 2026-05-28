"""Health-check route.

Cheap, no downstream I/O — just confirms the process is alive and all
critical app.state components were constructed by the lifespan.
Kubernetes readiness/liveness probes hit this every few seconds.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "status": "ok",
        "components": {
            "retriever": hasattr(request.app.state, "retriever"),
            "generator": hasattr(request.app.state, "generator"),
            "rxnorm": hasattr(request.app.state, "rxnorm"),
            "fda": hasattr(request.app.state, "fda"),
        },
    }
