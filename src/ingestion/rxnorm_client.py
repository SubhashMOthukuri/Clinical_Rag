# rxnorm_client.py
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx

from src.utils.validators import StageValidationError, validate_rxnorm_response
from src.utils.circuit_breaker import _CircuitBreaker
logger = logging.getLogger(__name__)


# ---- Result type: four outcomes, not three Nones ----
@dataclass(frozen=True)
class RxcuiFound:
    rxcui: str
    from_cache: bool = False

@dataclass(frozen=True)
class RxcuiUnverified:
    drug_name: str  # in RxNorm's view, this drug isn't known — clinician review

@dataclass(frozen=True)
class RxcuiLookupFailed:
    drug_name: str
    reason: str  # timeout | network | http_4xx | http_5xx | malformed | circuit_open | bad_input

RxcuiResult = RxcuiFound | RxcuiUnverified | RxcuiLookupFailed


# ---- Observability hook: inject whatever the team uses ----
class Metrics(Protocol):
    def incr(self, name: str, tags: dict[str, str] | None = ...) -> None: ...
    def observe(self, name: str, value: float, tags: dict[str, str] | None = ...) -> None: ...

class _NoopMetrics:
    def incr(self, name, tags=None): pass
    def observe(self, name, value, tags=None): pass


# ---- Config: injectable, testable, env-overridable ----
@dataclass(frozen=True)
class RxNormConfig:
    base_url: str = "https://rxnav.nlm.nih.gov/REST"
    request_timeout_s: float = 5.0
    max_attempts: int = 2
    base_backoff_s: float = 0.2
    max_backoff_s: float = 2.0
    cache_ttl_s: float = 86_400
    cache_size: int = 10_000
    breaker_threshold: int = 5
    breaker_cooldown_s: float = 30.0


# ---- Cache with singleflight: hot keys don't thundering-herd the API ----
class _TTLCacheWithSingleflight:
    """In-process only. For multi-pod deploys, back this with Redis."""

    def __init__(self, max_size: int, ttl_s: float):
        self._data: dict[str, tuple[float, RxcuiFound | RxcuiUnverified]] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl = ttl_s

    def _get_fresh(self, key):
        entry = self._data.get(key)
        if entry is None:
            return None
        expires, value = entry
        if expires < time.monotonic():
            self._data.pop(key, None)
            return None
        return value

    async def get_or_compute(self, key, factory):
        # Fast path
        cached = self._get_fresh(key)
        if cached is not None:
            return cached, True

        async with self._lock:
            cached = self._get_fresh(key)
            if cached is not None:
                return cached, True

            if key in self._inflight:
                # Another task is already computing this — wait on it
                fut = self._inflight[key]
                wait_on_existing = True
            else:
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = fut
                wait_on_existing = False

        if wait_on_existing:
            result = await fut
            return result, False

        try:
            result = await factory()
            # Only cache definitive answers — never cache transient failures
            if isinstance(result, (RxcuiFound, RxcuiUnverified)):
                if len(self._data) >= self._max_size:
                    self._data.pop(next(iter(self._data)))  # cheap FIFO eviction
                self._data[key] = (time.monotonic() + self._ttl, result)
            fut.set_result(result)
            return result, False
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)


# ---- Client: one per process, reused ----
class RxNormClient:
    def __init__(
        self,
        config: RxNormConfig | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        metrics: Metrics | None = None,
    ):
        self._cfg = config or RxNormConfig()
        self._metrics = metrics or _NoopMetrics()
        self._cache = _TTLCacheWithSingleflight(self._cfg.cache_size, self._cfg.cache_ttl_s)
        self._breaker = _CircuitBreaker(self._cfg.breaker_threshold, self._cfg.breaker_cooldown_s)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=self._cfg.request_timeout_s,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def get_rxcui(self, drug_name: str, *, correlation_id: str | None = None) -> RxcuiResult:
        normalized = (drug_name or "").strip().lower()
        log_ctx = {"drug": drug_name, "cid": correlation_id}

        # Validate input: reject injection attempts, special characters
        if not normalized:
            self._metrics.incr("rxnorm.bad_input")
            return RxcuiLookupFailed(drug_name, "bad_input")
        
        # Ensure drug name matches safe pattern (alphanumeric, spaces, hyphens, slashes, parens)
        import re
        if not re.match(r"^[a-z0-9 \-/().]*$", normalized):
            self._metrics.incr("rxnorm.invalid_characters")
            logger.warning("rxnorm.invalid_input_chars", extra=log_ctx)
            return RxcuiLookupFailed(drug_name, "invalid_characters")

        if self._breaker.is_open():
            self._metrics.incr("rxnorm.circuit_open")
            logger.warning("rxnorm.circuit_open", extra=log_ctx)
            return RxcuiLookupFailed(drug_name, "circuit_open")

        t0 = time.perf_counter()
        try:
            result, from_cache = await self._cache.get_or_compute(
                normalized,
                lambda: self._lookup_uncached(normalized, drug_name, correlation_id),
            )
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._metrics.observe("rxnorm.latency_ms", elapsed_ms)

        self._metrics.incr("rxnorm.cache.hit" if from_cache else "rxnorm.cache.miss")
        if isinstance(result, RxcuiFound) and from_cache:
            return RxcuiFound(result.rxcui, from_cache=True)
        return result

    async def _lookup_uncached(self, normalized, original, cid) -> RxcuiResult:
        url = f"{self._cfg.base_url}/rxcui.json?name={quote(normalized)}"
        raw = await self._http_get_with_retry(url, original, cid)
        if isinstance(raw, RxcuiLookupFailed):
            return raw

        try:
            rxcui = validate_rxnorm_response(raw, original)
        except StageValidationError:
            logger.exception("rxnorm.validation_failed", extra={"drug": original, "cid": cid})
            self._metrics.incr("rxnorm.malformed")
            self._breaker.record_failure()
            return RxcuiLookupFailed(original, "malformed")

        self._breaker.record_success()
        if rxcui is None:
            self._metrics.incr("rxnorm.unverified")
            return RxcuiUnverified(original)
        self._metrics.incr("rxnorm.found")
        return RxcuiFound(rxcui)

    async def _http_get_with_retry(self, url, drug_name, cid) -> dict | RxcuiLookupFailed:
        for attempt in range(1, self._cfg.max_attempts + 1):
            try:
                resp = await self._http.get(url)
            except httpx.TimeoutException:
                self._metrics.incr("rxnorm.timeout")
                logger.warning("rxnorm.timeout", extra={"drug": drug_name, "cid": cid, "attempt": attempt})
                if attempt < self._cfg.max_attempts:
                    await self._backoff(attempt); continue
                self._breaker.record_failure()
                return RxcuiLookupFailed(drug_name, "timeout")
            except httpx.RequestError as e:
                self._metrics.incr("rxnorm.network_error")
                logger.warning("rxnorm.network", extra={"drug": drug_name, "cid": cid, "err": type(e).__name__})
                if attempt < self._cfg.max_attempts:
                    await self._backoff(attempt); continue
                self._breaker.record_failure()
                return RxcuiLookupFailed(drug_name, "network")

            status = resp.status_code
            self._metrics.incr("rxnorm.http", {"status": str(status)})

            if status == 200:
                try:
                    return resp.json()
                except ValueError:
                    self._breaker.record_failure()
                    return RxcuiLookupFailed(drug_name, "malformed")

            # Determine human-readable error message
            status_msg = self._http_status_message(status)
            if status == 429 or 500 <= status < 600:
                if attempt < self._cfg.max_attempts:
                    await self._backoff(attempt, resp); continue
                self._breaker.record_failure()
                return RxcuiLookupFailed(drug_name, status_msg)

            # 4xx other than 429 — not retryable
            return RxcuiLookupFailed(drug_name, status_msg)

        return RxcuiLookupFailed(drug_name, "exhausted")

    def _http_status_message(self, status: int) -> str:
        """Convert HTTP status code to human-readable error message."""
        status_map = {
            400: "http_400_bad_request",
            401: "http_401_unauthorized",
            403: "http_403_forbidden",
            404: "http_404_not_found",
            429: "http_429_rate_limited",
            500: "http_500_internal_error",
            502: "http_502_bad_gateway",
            503: "http_503_service_unavailable",
            504: "http_504_gateway_timeout",
        }
        return status_map.get(status, f"http_{status}_error")

    async def _backoff(self, attempt: int, resp: httpx.Response | None = None) -> None:
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                await asyncio.sleep(min(float(retry_after), self._cfg.max_backoff_s))
                return
        base = min(self._cfg.base_backoff_s * (2 ** (attempt - 1)), self._cfg.max_backoff_s)
        await asyncio.sleep(base * (0.5 + random.random()))  # jitter