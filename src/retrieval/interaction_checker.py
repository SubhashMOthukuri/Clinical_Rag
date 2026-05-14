from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from src.ingestion.fda_client import FDADrugData
from src.utils.schema import Medication

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrugContext:
    name: str
    dose: float
    unit: str
    ingredient_rxcui: str
    drug_class: str
    fda_label_id: str


@dataclass(frozen=True)
class InteractionEvidence:
    drug_a: DrugContext
    drug_b: DrugContext
    evidence_text: str
    source_drug: str  # which drug's FDA label contained the evidence
    data_source: str = "FRESH_FDA"


@dataclass(frozen=True)
class InteractionCheckerConfig:
    min_evidence_length: int = 20
    max_pairs_per_request: int = 100
    batch_size: int = 20


class InteractionChecker:
    """Check pairwise drug interactions using FDA label text.

    For each pair of medications, scans both drugs' FDA interaction fields
    (drug_interactions, warnings, do_not_use) for mentions of the other drug.
    Returns only evidence that meets min_evidence_length to filter noise.
    """

    def __init__(
        self,
        config: InteractionCheckerConfig | None = None,
        db_pool=None,
        redis_client=None,
    ):
        self._cfg = config or InteractionCheckerConfig()
        self._db = db_pool
        self._redis = redis_client
        logger.info("interaction_checker.initialized")

    # ---- Public API ----

    def check(
        self,
        medications: Sequence[Medication],
        fda_map: dict[str, FDADrugData],
    ) -> list[InteractionEvidence]:
        """Return all pairwise interaction evidence found in FDA label text.

        fda_map must be keyed by lowercase drug name matching medication.name.
        Pairs are capped at max_pairs_per_request to prevent runaway checks on
        large medication lists.
        """
        contexts: list[DrugContext] = []
        for med in medications:
            fda = fda_map.get(med.name.lower())
            if fda is None:
                logger.info("interaction_checker.no_fda_data drug=%s", med.name)
                continue
            contexts.append(self._build_context(med, fda))

        pairs = list(combinations(contexts, 2))
        if len(pairs) > self._cfg.max_pairs_per_request:
            logger.warning(
                "interaction_checker.pairs_truncated total=%d limit=%d",
                len(pairs), self._cfg.max_pairs_per_request,
            )
            pairs = pairs[: self._cfg.max_pairs_per_request]

        evidence: list[InteractionEvidence] = []
        for ctx_a, ctx_b in pairs:
            found = self._check_pair(ctx_a, ctx_b, fda_map)
            evidence.extend(found)

        # Deduplicate — scanning both labels for the same pair produces two
        # InteractionEvidence objects for the same interaction. Keep only the
        # first one found so the LLM and nurse never see the same warning twice.
        seen: set[frozenset] = set()
        deduped: list[InteractionEvidence] = []
        for ev in evidence:
            key = frozenset([ev.drug_a.name, ev.drug_b.name])
            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        logger.info(
            "interaction_checker.complete pairs_checked=%d "
            "evidence_found=%d after_dedup=%d",
            len(pairs), len(evidence), len(deduped),
        )
        return deduped

    # ---- Private helpers ----

    @staticmethod
    def _build_context(med: Medication, fda: FDADrugData) -> DrugContext:
        return DrugContext(
            name=med.name.lower(),
            dose=med.dose,
            unit=med.unit.value,
            ingredient_rxcui=med.ingredient_rxcui or "",
            drug_class=fda.drug_class,
            fda_label_id=fda.fda_label_id,
        )

    def _check_pair(
        self,
        ctx_a: DrugContext,
        ctx_b: DrugContext,
        fda_map: dict[str, FDADrugData],
    ) -> list[InteractionEvidence]:
        """Scan both drugs' FDA label fields for mentions of the other drug."""
        results: list[InteractionEvidence] = []

        for source_ctx, target_ctx in [(ctx_a, ctx_b), (ctx_b, ctx_a)]:
            fda = fda_map.get(source_ctx.name)
            if fda is None:
                continue

            # Search across all three FDA label sections that carry interaction signals
            candidate_fields = fda.drug_interactions + fda.warnings + fda.do_not_use

            for text in candidate_fields:
                # TODO(Phase 2): also match target_ctx.drug_class (e.g. "NSAIDs")
                # so class-level warnings aren't missed when FDA text names the
                # drug class instead of the individual drug.
                if (
                    target_ctx.name in text.lower()
                    and len(text) >= self._cfg.min_evidence_length
                ):
                    results.append(InteractionEvidence(
                        drug_a=source_ctx,
                        drug_b=target_ctx,
                        evidence_text=text,
                        source_drug=source_ctx.name,
                    ))

        return results
