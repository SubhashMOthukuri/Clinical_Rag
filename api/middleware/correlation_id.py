"""Correlation-ID middleware.

Every request gets a correlation_id injected into request.state and
echoed back in the X-Correlation-ID response header.  If the client
(nurse UI, upstream gateway) sends the header, its value is reused so
the ID stays consistent across the entire call chain.  Otherwise a UUID
is generated, giving on-call engineers a single token to grep across
RxNorm, FDA, Pinecone, and LLM logs.
"""
from __future__ import annotations

import uuid

from fastapi import Request


async def correlation_id_middleware(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    request.state.correlation_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response
