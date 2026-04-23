from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Protocol
import httpx
import asyncio
import time
import json
from datetime import datetime, timezone
from urllib.parse import quote

from src.utils.validators import StageValidationError, validate_fda_response
from src.utils.circuit_breaker import _CircuitBreaker
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FDADrugData:
    generic_name: str
    rxcui: str
    drug_class: str
    warnings: list[str]
    drug_interactions: list[str]
    do_not_use: list[str]
    ask_doctor: list[str]
    source: str
    fetched_at: datetime
    fda_label_id: str = "" 

@dataclass(frozen=True)
class FDAConfig:
    base_url: str = "https://api.fda.gov/drug"
    request_timeout_s: float = 10.0
    max_attempts: int = 2
    base_backoff_s: float =1.0
    max_backoff_s: float =5.0
    cache_ttl_s: float = 604_800
    breaker_threshold: int = 5
    breaker_cooldown_s: float = 60.0


class FDAClient:
    def __init__(
        self,
        config: FDAConfig | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        redis_client=None,
    ):
        self._cfg = config or FDAConfig()
        self._redis = redis_client
        self._breaker = _CircuitBreaker(
            self._cfg.breaker_threshold,
            self._cfg.breaker_cooldown_s
        )
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=self._cfg.request_timeout_s,
        )
    async def get_drug_data(
        self,
        drug_name: str,
        *,
        correlation_id: str | None = None,
    ) -> FDADrugData | None:

        # Check Redis cache first
        if self._redis:
            cached = await self._redis.get(f"fda:{drug_name}")
            if cached:
                # Handle Redis bytes serialization
                if isinstance(cached, bytes):
                    cached = cached.decode("utf-8")
                logger.info("fda.cache_hit drug=%s cid=%s", drug_name, correlation_id)
                return self._deserialize(cached)

        # Check circuit breaker
        if self._breaker.is_open():
            logger.warning("fda.circuit_open drug=%s", drug_name)
            return None

        # Call FDA API
        result = await self._fetch_from_fda(drug_name, correlation_id)
        if result is None:
            return None

        # Store in Redis
        if self._redis:
            await self._redis.setex(
                f"fda:{drug_name}",
                int(self._cfg.cache_ttl_s),
                self._serialize(result)
            )

        return result
    
    def _serialize(self, data: FDADrugData) -> str:
        return json.dumps({
            "generic_name": data.generic_name,
            "rxcui": data.rxcui,
            "drug_class": data.drug_class,
            "warnings": data.warnings,
            "drug_interactions": data.drug_interactions,
            "do_not_use": data.do_not_use,
            "ask_doctor": data.ask_doctor,
            "source": "REDIS_CACHE",
            "fetched_at": data.fetched_at.isoformat(),
            "fda_label_id": data.fda_label_id,
        })

    def _deserialize(self, raw: str) -> FDADrugData:
        d = json.loads(raw)
        return FDADrugData(
            generic_name=d["generic_name"],
            rxcui=d["rxcui"],
            drug_class=d["drug_class"],
            warnings=d["warnings"],
            drug_interactions=d["drug_interactions"],
            do_not_use=d["do_not_use"],
            ask_doctor=d["ask_doctor"],
            source="REDIS_CACHE",
            fetched_at=datetime.fromisoformat(d["fetched_at"]),
            fda_label_id=d.get("fda_label_id", ""),
        )

    async def _fetch_from_fda(
        self,
        drug_name: str,
        cid: str | None,
    ) -> FDADrugData | None:

        url = (
            f"{self._cfg.base_url}/label.json"
            f"?search=openfda.generic_name:{quote(drug_name)}&limit=1"
        )

        for attempt in range(1, self._cfg.max_attempts + 1):
            try:
                resp = await self._http.get(url)
            except httpx.TimeoutException:
                logger.warning(
                    "fda.timeout drug=%s attempt=%d cid=%s",
                    drug_name, attempt, cid
                )
                if attempt < self._cfg.max_attempts:
                    await asyncio.sleep(self._cfg.base_backoff_s * attempt)
                    continue
                self._breaker.record_failure()
                return None
            except httpx.RequestError as e:
                logger.warning(
                    "fda.network_error drug=%s error=%s cid=%s",
                    drug_name, type(e).__name__, cid
                )
                self._breaker.record_failure()
                return None

            if resp.status_code != 200:
                logger.warning(
                    "fda.http_error drug=%s status=%d cid=%s",
                    drug_name, resp.status_code, cid
                )
                self._breaker.record_failure()
                return None

            try:
                raw = resp.json()
                results = validate_fda_response(raw, drug_name)
                self._breaker.record_success()
                return self._parse_results(results[0], drug_name)
            except StageValidationError:
                self._breaker.record_failure()
                return None

        return None

    def _parse_results(
        self,
        result: dict,
        drug_name: str,
    ) -> FDADrugData:
        openfda = result.get("openfda", {})
        return FDADrugData(
            generic_name=self._first(openfda.get("generic_name"), drug_name),
            rxcui=self._first(openfda.get("rxcui"), ""),
            drug_class=self._first(openfda.get("pharm_class_epc"), ""),
            warnings=result.get("warnings", []),
            drug_interactions=result.get("drug_interactions", []),
            do_not_use=result.get("do_not_use", []),
            ask_doctor=result.get("ask_doctor", []),
            source="FRESH_FDA",
            fetched_at=datetime.now(timezone.utc),
            fda_label_id=result.get("id", ""),
        )

    def _first(self, lst, default=""):
        if lst and isinstance(lst, list):
            return lst[0]
        return default

    async def aclose(self):
        if self._owns_http:
            await self._http.aclose()