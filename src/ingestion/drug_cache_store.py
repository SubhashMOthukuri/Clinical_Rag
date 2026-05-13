from __future__ import annotations

from datetime import datetime, timezone
import logging
import asyncio
from dataclasses import dataclass

from src.ingestion.rxnorm_client import RxcuiFound

logger = logging.getLogger(__name__)


@dataclass
class DrugRecord:
    drug_name: str
    rxcui: str           # dose-specific RXCUI (e.g. "860975" for metformin 500mg)
    ingredient_rxcui: str  # ingredient-level RXCUI (e.g. "6809" for metformin)
    normalized_name: str
    verified: bool
    lookup_count: int
    last_verified_at: datetime
    created_at: datetime


class DrugCacheStore:
    """Two-layer cache: Redis (hot, TTL-bound) in front of PostgreSQL (durable).

    Read path:  Redis → PostgreSQL → None
    Write path: PostgreSQL first, then populate Redis on the same hit
    Refresh:    background job re-verifies stale rows via RxNorm and evicts
                both Redis keys so the next read gets fresh data from PostgreSQL
    """

    def __init__(self, db_pool, redis_client=None):
        self._db = db_pool
        self._redis = redis_client  # optional — callers that omit it go straight to PostgreSQL

    async def save_drug_data(
        self,
        drug_name: str,
        rxcui: str,
        ingredient_rxcui: str,
        normalized_name: str,
    ) -> None:
        """Persist a verified drug to PostgreSQL.

        Uses INSERT … ON CONFLICT so re-ingesting the same drug is safe:
        it bumps lookup_count and refreshes the rxcui without losing history.
        drug_name is always lowercased to keep lookups case-insensitive.
        """
        try:
            async with self._db.acquire() as conn:
                await conn.execute("""
                    INSERT INTO drug_master (
                        drug_name, rxcui, ingredient_rxcui, normalized_name,
                        verified, lookup_count, last_verified_at, created_at
                    ) VALUES ($1, $2, $3, $4, TRUE, 1, $5, $5)
                    ON CONFLICT (drug_name) DO UPDATE SET
                        rxcui = EXCLUDED.rxcui,
                        ingredient_rxcui = EXCLUDED.ingredient_rxcui,
                        normalized_name = EXCLUDED.normalized_name,
                        verified = TRUE,
                        lookup_count = drug_master.lookup_count + 1,
                        last_verified_at = EXCLUDED.last_verified_at
                """, drug_name.lower(), rxcui, ingredient_rxcui, normalized_name,
                     datetime.now(timezone.utc))
                logger.info(
                    "drug_cache.saved drug=%s rxcui=%s ingredient_rxcui=%s",
                    drug_name, rxcui, ingredient_rxcui,
                )
        except Exception as e:
            logger.error("drug_cache.save_failed drug=%s error=%s", drug_name, e)
            raise

    async def get_drug_by_name(
        self,
        drug_name: str,
    ) -> DrugRecord | None:
        """Look up a drug by name using the Redis → PostgreSQL read path.

        Redis hit: returns immediately with rxcui only (ingredient_rxcui is not
        stored in Redis to keep the cached value small; callers that need it
        should use the full DrugRecord from a PostgreSQL hit).

        PostgreSQL hit: populates Redis (24h TTL) so the next call is served
        from cache without hitting the database.
        """
        # Layer 1 — Redis: fast path for hot drugs
        if self._redis:
            cached = await self._redis.get(f"rxcui:{drug_name}")
            if cached:
                logger.info("drug_cache.redis_hit drug=%s", drug_name)
                return DrugRecord(
                    drug_name=drug_name,
                    rxcui=cached.decode(),
                    ingredient_rxcui="",  # not stored in Redis; use DB hit for full record
                    normalized_name=drug_name,
                    verified=True,
                    lookup_count=0,
                    last_verified_at=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                )

        # Layer 2 — PostgreSQL: authoritative source of truth
        try:
            async with self._db.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT drug_name, rxcui, ingredient_rxcui, normalized_name,
                           verified, lookup_count,
                           last_verified_at, created_at
                    FROM drug_master
                    WHERE drug_name = $1
                """, drug_name.lower())

                if row is None:
                    logger.info("drug_cache.miss drug=%s", drug_name)
                    return None

                logger.info("drug_cache.hit drug=%s rxcui=%s", drug_name, row["rxcui"])

                # Populate Redis so the next lookup skips PostgreSQL entirely.
                # 24h TTL balances freshness against DB load; _refresh_single
                # evicts this key whenever the background job updates the rxcui.
                if self._redis:
                    await self._redis.setex(
                        f"rxcui:{drug_name}",
                        86_400,
                        row["rxcui"].encode(),
                    )

                return DrugRecord(
                    drug_name=row["drug_name"],
                    rxcui=row["rxcui"],
                    ingredient_rxcui=row["ingredient_rxcui"],
                    normalized_name=row["normalized_name"],
                    verified=row["verified"],
                    lookup_count=row["lookup_count"],
                    last_verified_at=row["last_verified_at"],
                    created_at=row["created_at"],
                )
        except Exception as e:
            # Return None instead of raising so a cache failure never crashes
            # the caller's reconciliation pipeline.
            logger.error("drug_cache.get_failed drug=%s error=%s", drug_name, e)
            return None

    async def refresh_rxnorm_cache(
        self,
        rxnorm_client,
    ) -> dict:
        """Re-verify all drugs whose last_verified_at is older than 7 days.

        Processed in batches of 10 so a large stale list doesn't open thousands
        of concurrent RxNorm connections. asyncio.gather with return_exceptions=True
        means one bad drug never cancels the rest of the batch.
        """
        refreshed = 0
        failed = 0

        # Fetch stale rows outside the batch loop so we hold the connection
        # only for the quick SELECT, not for the entire refresh duration.
        async with self._db.acquire() as conn:
            stale_rows = await conn.fetch("""
                SELECT drug_name FROM drug_master
                WHERE last_verified_at < NOW() - INTERVAL '7 days'
            """)

        batch_size = 10
        rows = list(stale_rows)

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            results = await asyncio.gather(
                *[self._refresh_single(r["drug_name"], rxnorm_client) for r in batch],
                return_exceptions=True,  # partial failure is counted, not propagated
            )
            for r in results:
                if r is True:
                    refreshed += 1
                else:
                    failed += 1

        logger.info("refresh.complete refreshed=%d failed=%d", refreshed, failed)
        return {"refreshed": refreshed, "failed": failed}

    async def _refresh_single(self, drug_name: str, rxnorm_client) -> bool:
        """Re-verify one drug against RxNorm and sync both rxcui fields.

        On success:
          1. Resolve dose-specific rxcui via get_rxcui
          2. Resolve ingredient_rxcui via get_ingredient_rxcui (allrelated endpoint)
          3. Write both to PostgreSQL
          4. Evict both Redis keys so the next read gets fresh data from DB

        Evicting both fda: and rxcui: keys is intentional — the FDA client
        also caches by drug name, so both layers must be cleared together to
        prevent a stale fda: key from serving old interaction data.
        """
        try:
            result = await rxnorm_client.get_rxcui(drug_name)
            if isinstance(result, RxcuiFound):
                # Resolve ingredient-level RXCUI from the dose-specific one.
                # Falls back to "" if the allrelated endpoint has no IN group.
                ingredient_rxcui = await rxnorm_client.get_ingredient_rxcui(result.rxcui) or ""
                async with self._db.acquire() as conn:
                    await conn.execute("""
                        UPDATE drug_master
                        SET rxcui = $1,
                            ingredient_rxcui = $2,
                            last_verified_at = $3,
                            verified = TRUE
                        WHERE drug_name = $4
                    """, result.rxcui, ingredient_rxcui, datetime.now(timezone.utc), drug_name)

                if self._redis:
                    # Evict both keys — fda: holds interaction data, rxcui: holds the
                    # rxcui string. Both are now stale after the UPDATE above.
                    await self._redis.delete(f"fda:{drug_name}")
                    await self._redis.delete(f"rxcui:{drug_name}")

                return True
            return False
        except Exception as e:
            logger.error("refresh.single_failed drug=%s error=%s", drug_name, e)
            return False