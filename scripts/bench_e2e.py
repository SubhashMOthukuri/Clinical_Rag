"""Self-contained end-to-end latency benchmark for MedReconcile AI.

Runs the full reconcile pipeline in-process with realistic mock delays
for every external service (OpenAI, Pinecone, Gemini, Groq, RxNorm, FDA).
No API keys or running server required.

Usage:
    python scripts/bench_e2e.py              # 100 runs, default drug pool
    python scripts/bench_e2e.py --runs 50   # fewer runs
    python scripts/bench_e2e.py --fast      # near-zero delays (pure Python overhead)
"""
from __future__ import annotations

# ── patch env before any project import touches config ───────────────────────
import os
os.environ.setdefault("OPENAI_API_KEY",   "sk-bench")
os.environ.setdefault("GEMINI_API_KEY",   "bench")
os.environ.setdefault("GROQ_API_KEY",     "bench")
os.environ.setdefault("PINECONE_API_KEY", "bench")

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

# ── realistic delay profiles (seconds) ───────────────────────────────────────
# Based on typical p50 / jitter for each service in us-east-1
DELAYS = {
    "rxnorm":  (0.15, 0.35),   # external HTTP lookup
    "fda":     (0.20, 0.50),   # external HTTP lookup, sometimes slow
    "embed":   (0.20, 0.45),   # OpenAI text-embedding-3-small
    "pinecone":(0.08, 0.20),   # managed vector DB, usually fast
    "rerank":  (0.50, 1.40),   # local cross-encoder, CPU-bound
    "gemini":  (0.60, 2.00),   # Gemini 2.0 flash
    "groq":    (0.20, 0.70),   # Groq llama-3.3-70b, usually faster
}
FAST_DELAYS = {k: (0.001, 0.002) for k in DELAYS}  # --fast mode

def jitter(key: str) -> float:
    lo, hi = _delay_profile[key]
    return random.uniform(lo, hi)

_delay_profile = DELAYS  # swapped to FAST_DELAYS by --fast


# ── 100 realistic drug pair scenarios ────────────────────────────────────────
DRUG_POOL = [
    # each tuple: (name, dose, unit, frequency)
    ("warfarin",      5.0,   "mg",  "daily"),
    ("aspirin",      81.0,   "mg",  "daily"),
    ("clopidogrel",  75.0,   "mg",  "daily"),
    ("metformin",   500.0,   "mg",  "twice daily"),
    ("lisinopril",   10.0,   "mg",  "daily"),
    ("atorvastatin", 20.0,   "mg",  "nightly"),
    ("amiodarone",  200.0,   "mg",  "daily"),
    ("digoxin",       0.125, "mg",  "daily"),
    ("sertraline",   50.0,   "mg",  "daily"),
    ("tramadol",     50.0,   "mg",  "as needed"),
    ("omeprazole",   20.0,   "mg",  "daily"),
    ("metoprolol",   25.0,   "mg",  "twice daily"),
    ("furosemide",   40.0,   "mg",  "daily"),
    ("spironolactone",25.0,  "mg",  "daily"),
    ("ciprofloxacin",500.0,  "mg",  "twice daily"),
    ("fluoxetine",   20.0,   "mg",  "daily"),
    ("simvastatin",  40.0,   "mg",  "nightly"),
    ("amlodipine",    5.0,   "mg",  "daily"),
    ("levothyroxine", 50.0,  "mcg", "daily"),
    ("prednisone",   10.0,   "mg",  "daily"),
]

def _make_payloads(n: int) -> list[dict]:
    """Generate n payloads, 2–3 medications each, cycling through drug pairs."""
    random.seed(42)
    payloads = []
    for _ in range(n):
        size = random.choice([2, 2, 3])  # mostly pairs, occasional triple
        drugs = random.sample(DRUG_POOL, size)
        payloads.append({
            "medications": [
                {"name": d[0], "dose": d[1], "unit": d[2], "frequency": d[3]}
                for d in drugs
            ]
        })
    return payloads


# ── mock external services ────────────────────────────────────────────────────
# These satisfy the same Protocol/dataclass contracts the real pipeline expects.

from src.retrieval.pinecone_store import ChunkMetadata, QueryResult
from src.ingestion.rxnorm_client import RxcuiFound, RxcuiUnverified
from src.ingestion.fda_client import FDADrugData


class MockEmbedder:
    dimensions = 768
    max_input_tokens = 8191

    async def embed(self, text: str, *, correlation_id=None) -> list[float]:
        await asyncio.sleep(jitter("embed"))
        return [0.01] * 768

    async def embed_batch(self, texts, batch_size=100, *, correlation_id=None):
        await asyncio.sleep(jitter("embed"))
        return [[0.01] * 768 for _ in texts]

    async def close(self):
        pass


class MockVectorStore:
    async def query(self, *, query_vector, top_k, namespace, correlation_id=None):
        await asyncio.sleep(jitter("pinecone"))
        return [
            QueryResult(
                id=f"article-00{i+1}_chunk_001",
                score=0.85 - i * 0.05,
                metadata=ChunkMetadata(
                    text=f"Clinical evidence chunk {i+1} about drug interactions.",
                    title="StatPearls Drug Interactions",
                    source=f"article-00{i+1}",
                    article_id=f"NBK00{i+1}",
                    article_type="review",
                    token_count=120,
                    created_at="2024-01-01",
                    updated_at="2024-01-01",
                ),
            )
            for i in range(min(top_k, 5))
        ]

    async def close(self):
        pass


FAKE_LLM_TEMPLATE = """[
  {{
    "drugs_involved": ["{drug_a}", "{drug_b}"],
    "severity": "YELLOW",
    "reaction_result": "Concurrent use may increase bleeding risk via additive antiplatelet and anticoagulant effects. Monitor INR closely.",
    "action": "MONITOR",
    "citation": ["article-001_chunk_001", "article-002_chunk_001"],
    "nurse_summary_to_doctor": "Patient on {drug_a} + {drug_b}: elevated bleeding risk, recommend INR check.",
    "confidence": 0.82
  }}
]"""


class MockGenerator:
    """Mimics Generator but sleeps instead of calling real LLMs."""

    def __init__(self):
        self._gemini_ok = True  # simulate 85% gemini success, 15% groq fallback

    async def generate_one(self, retrieval_result, *, correlation_id=None):
        from src.utils.schema import DrugWarning, Severity, Action, DataSource
        from src.utils.metrics import mvp_metrics, M

        evidence = retrieval_result.evidence
        chunks = retrieval_result.chunks

        if not chunks:
            mvp_metrics.incr(M.GENERATOR_FALLBACKS)
            return self._fda_fallback(evidence, "no_chunks")

        drug_a = evidence.drug_a.name
        drug_b = evidence.drug_b.name

        # simulate primary provider (gemini) with 85% success
        use_gemini = random.random() < 0.85
        provider = "gemini" if use_gemini else "groq"

        try:
            with mvp_metrics.time(M.GENERATOR_LLM_LATENCY, provider=provider):
                await asyncio.sleep(jitter(provider))

            if use_gemini:
                mvp_metrics.incr(M.GENERATOR_GEMINI_OK)
            else:
                mvp_metrics.incr(M.GENERATOR_GROQ_OK)

            allowed = {c.id for c in chunks}
            from src.utils.validators import validate_llm_response
            raw = FAKE_LLM_TEMPLATE.format(drug_a=drug_a, drug_b=drug_b)
            warnings = validate_llm_response(
                raw_output=raw,
                allowed_drug_names={drug_a, drug_b},
                allowed_citation_sources=allowed,
            )
            return warnings[0]
        except Exception as e:
            mvp_metrics.incr(M.GENERATOR_FALLBACKS)
            return self._fda_fallback(evidence, str(e))

    async def generate_many(self, retrieval_results, *, correlation_id=None):
        tasks = [self.generate_one(r, correlation_id=correlation_id) for r in retrieval_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        from src.utils.schema import DrugWarning
        return [
            r if isinstance(r, DrugWarning)
            else self._fda_fallback(retrieval_results[i].evidence, type(r).__name__)
            for i, r in enumerate(results)
        ]

    def _fda_fallback(self, evidence, reason):
        from src.utils.schema import DrugWarning, Severity, Action, DataSource
        return DrugWarning(
            drugs_involved=[evidence.drug_a.name, evidence.drug_b.name],
            severity=Severity.YELLOW,
            reaction_result="FDA evidence only — LLM unavailable.",
            action=Action.CONSULT_DOCTOR,
            citation=["FDA_LABEL"],
            nurse_summary_to_doctor=f"Degraded mode: {reason[:80]}",
            confidence=0.5,
            data_source=DataSource.FRESH_FDA,
            computed_at=datetime.now(timezone.utc),
        )


class MockRxNormClient:
    _rxcui_map = {
        "warfarin": "11289", "aspirin": "1191", "clopidogrel": "174742",
        "metformin": "6809",  "lisinopril": "29046", "atorvastatin": "83367",
        "amiodarone": "703",  "digoxin": "3407", "sertraline": "36437",
        "tramadol": "41493",  "omeprazole": "7646", "metoprolol": "6918",
        "furosemide": "4603", "spironolactone": "9997", "ciprofloxacin": "2551",
        "fluoxetine": "4493", "simvastatin": "36567", "amlodipine": "17767",
        "levothyroxine": "10582", "prednisone": "8638",
    }

    async def get_rxcui(self, drug_name: str, *, correlation_id=None):
        await asyncio.sleep(jitter("rxnorm"))
        rxcui = self._rxcui_map.get(drug_name.lower())
        return RxcuiFound(rxcui=rxcui) if rxcui else RxcuiUnverified(drug_name=drug_name)

    async def get_ingredient_rxcui(self, rxcui: str, *, correlation_id=None):
        await asyncio.sleep(jitter("rxnorm") * 0.3)
        return rxcui

    async def aclose(self):
        pass


class MockFDAClient:
    async def get_drug_data(self, drug_name: str, *, correlation_id=None):
        await asyncio.sleep(jitter("fda"))
        return FDADrugData(
            generic_name=drug_name,
            rxcui="00000",
            drug_class="mock-class",
            warnings=[f"Monitor closely when used with other drugs."],
            drug_interactions=[f"{drug_name} may interact with warfarin-class agents."],
            do_not_use=[],
            ask_doctor=["If you take blood thinners."],
            source="FDA_LABEL_MOCK",
            fetched_at=datetime.now(timezone.utc),
            fda_label_id="mock-label-id",
        )

    async def aclose(self):
        pass


class MockInteractionChecker:
    """Returns one YELLOW interaction per drug pair that both have FDA data."""

    def check(self, medications, fda_map):
        from src.retrieval.interaction_checker import InteractionEvidence, DrugContext

        pairs = []
        for i in range(len(medications)):
            for j in range(i + 1, len(medications)):
                a, b = medications[i], medications[j]
                fda_a = fda_map.get(a.name.lower())
                fda_b = fda_map.get(b.name.lower())
                if not fda_a or not fda_b:
                    continue
                ctx_a = DrugContext(
                    name=a.name, dose=a.dose, unit=a.unit.value,
                    ingredient_rxcui=a.ingredient_rxcui or "00000",
                    drug_class=fda_a.drug_class,
                    fda_label_id=fda_a.fda_label_id,
                )
                ctx_b = DrugContext(
                    name=b.name, dose=b.dose, unit=b.unit.value,
                    ingredient_rxcui=b.ingredient_rxcui or "00000",
                    drug_class=fda_b.drug_class,
                    fda_label_id=fda_b.fda_label_id,
                )
                pairs.append(InteractionEvidence(
                    drug_a=ctx_a, drug_b=ctx_b,
                    evidence_text=f"Monitor closely when {a.name} and {b.name} are used together.",
                    source_drug=a.name,
                    estimated_severity="YELLOW",
                ))
        return pairs


# ── mock reranker to avoid loading the 86 MB cross-encoder ───────────────────
import unittest.mock as _mock

_rerank_sleep_patch = None


# ── pipeline runner ────────────────────────────────────────────────────────────

async def run_pipeline(payload: dict, components: dict) -> dict:
    """Call the full reconcile pipeline with mock components, return timing info."""
    from src.utils.schema import (
        Medication, ReconciliationRequest, Status, Severity,
        ReconciliationResponse,
    )
    from src.utils.validators import validate_input, validate_response
    from src.utils.metrics import mvp_metrics, M

    request_obj = ReconciliationRequest(**payload)
    validate_input(request_obj)

    mvp_metrics.incr(M.RECONCILE_REQUESTS)
    mvp_metrics.observe(M.MEDS_PER_REQUEST, len(request_obj.medications))

    t0 = time.perf_counter()

    rxnorm  = components["rxnorm"]
    fda     = components["fda"]
    checker = components["checker"]
    retriever = components["retriever"]
    generator = components["generator"]

    # Stage 2: enrich meds in parallel
    async def _one(med):
        rx_task  = rxnorm.get_rxcui(med.name)
        fda_task = fda.get_drug_data(med.name)
        rx_result, fda_data = await asyncio.gather(rx_task, fda_task)
        from src.ingestion.rxnorm_client import RxcuiFound, RxcuiUnverified
        is_unverified = isinstance(rx_result, RxcuiUnverified)
        rxcui = rx_result.rxcui if isinstance(rx_result, RxcuiFound) else None
        ingredient_rxcui = None
        if rxcui:
            ingredient_rxcui = await rxnorm.get_ingredient_rxcui(rxcui)
        enriched = med.model_copy(update={"rxcui": rxcui, "ingredient_rxcui": ingredient_rxcui, "verified": rxcui is not None})
        return enriched, fda_data, is_unverified

    results = await asyncio.gather(*[_one(m) for m in request_obj.medications])
    enriched_meds, fda_map_list, unverified_list = [], {}, []
    for enriched, fda_data, is_unverified in results:
        enriched_meds.append(enriched)
        if fda_data:
            fda_map_list[enriched.name.lower()] = fda_data
        if is_unverified:
            unverified_list.append(enriched.name)

    # Stage 3: interaction check
    evidences = checker.check(enriched_meds, fda_map_list)
    mvp_metrics.observe(M.DRUG_PAIRS_PER_REQUEST, len(evidences))

    # Stage 4+5: retrieve + generate
    if evidences:
        retrieval_results = await retriever.retrieve_many(evidences)
        warnings = await generator.generate_many(retrieval_results)
    else:
        warnings = []

    elapsed_ms = (time.perf_counter() - t0) * 1000
    mvp_metrics.observe(M.RECONCILE_LATENCY, elapsed_ms)

    has_fda_fallback = any("FDA_LABEL" in w.citation for w in warnings)
    status = Status.PARTIAL if has_fda_fallback else Status.SUCCESS
    mvp_metrics.incr(M.RECONCILE_STATUS_PARTIAL if has_fda_fallback else M.RECONCILE_STATUS_SUCCESS)

    for w in warnings:
        if w.severity == Severity.RED:    mvp_metrics.incr(M.WARNINGS_SEVERITY_RED)
        elif w.severity == Severity.YELLOW: mvp_metrics.incr(M.WARNINGS_SEVERITY_YELLOW)
        else:                               mvp_metrics.incr(M.WARNINGS_SEVERITY_GREEN)

    return {"elapsed_ms": elapsed_ms, "status": status.value, "warnings": len(warnings)}


# ── main ──────────────────────────────────────────────────────────────────────

async def main(n_runs: int, fast: bool) -> None:
    global _delay_profile
    if fast:
        _delay_profile = FAST_DELAYS

    # patch out CrossEncoder so we don't load the 86 MB model
    with _mock.patch("src.retrieval.retrieval.CrossEncoder") as mock_ce:
        # make rerank() just sleep and return candidates as-is
        async def _fake_rerank(query, candidates):
            await asyncio.sleep(jitter("rerank"))
            return candidates[:3]

        from src.retrieval.retrieval import Retriever

        components = {
            "rxnorm":    MockRxNormClient(),
            "fda":       MockFDAClient(),
            "checker":   MockInteractionChecker(),
            "retriever": None,   # built after patch
            "generator": MockGenerator(),
        }

        mock_embedder = MockEmbedder()
        mock_store    = MockVectorStore()

        # Build retriever with mock components, skip real CrossEncoder
        retriever = Retriever.__new__(Retriever)
        retriever._embedder        = mock_embedder
        retriever._store           = mock_store
        retriever._namespace       = "full_v1"
        retriever._retrieve_k      = 10
        retriever._rerank_n        = 3
        retriever._score_threshold = 0.5
        retriever._reranker        = None
        # monkey-patch _rerank to use async sleep
        retriever._rerank = _fake_rerank  # type: ignore
        components["retriever"] = retriever

        payloads = _make_payloads(n_runs)

        print(f"\nMedReconcile E2E Latency Benchmark")
        print(f"Mode   : {'FAST (near-zero delays)' if fast else 'realistic mock delays'}")
        print(f"Runs   : {n_runs}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")

        wall_times: list[float] = []
        statuses: dict[str, int] = {}
        errors = 0

        for i, payload in enumerate(payloads):
            try:
                result = await run_pipeline(payload, components)
                wall_times.append(result["elapsed_ms"])
                s = result["status"]
                statuses[s] = statuses.get(s, 0) + 1
                meds = len(payload["medications"])
                print(
                    f"  [{i+1:03d}/{n_runs}] {result['elapsed_ms']:7.1f}ms  "
                    f"status={s:<8}  meds={meds}  warnings={result['warnings']}"
                )
            except Exception as e:
                errors += 1
                print(f"  [{i+1:03d}/{n_runs}] ERROR: {e}")

    # ── fetch metrics snapshot ─────────────────────────────────────────────
    from src.utils.metrics import mvp_metrics

    snap = mvp_metrics.snapshot()
    h = snap["histograms"]
    c = snap["counters"]

    def pct(vals, p):
        if not vals: return 0.0
        s = sorted(vals)
        k = (len(s) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    def stat_row(label, key):
        d = h.get(key, {})
        if not d or d.get("count", 0) == 0:
            return f"  {label:<36} no data"
        return (
            f"  {label:<36} "
            f"mean={d['mean']:7.1f}ms  "
            f"p50={d['p50']:7.1f}ms  "
            f"p95={d['p95']:7.1f}ms  "
            f"max={d['max']:7.1f}ms  (n={d['count']})"
        )

    def ctr(key):
        return c.get(key, 0)

    SEP = "=" * 80
    sep = "-" * 78

    print(f"\n{SEP}")
    print(f"  END-TO-END LATENCY REPORT  ({n_runs} runs)")
    print(SEP)

    print(f"\n  Outcome summary:")
    print(f"  {sep}")
    total_ok = sum(statuses.values())
    for s, cnt in sorted(statuses.items()):
        bar = "█" * int(cnt / n_runs * 40)
        print(f"  {s:<10} {cnt:4d}  {bar}")
    if errors:
        print(f"  ERRORS     {errors:4d}")

    print(f"\n  Per-stage latency breakdown:")
    print(f"  {sep}")
    print(stat_row("reconcile end-to-end",        "reconcile.latency_ms"))
    print(stat_row("  ├─ embed (OpenAI mock)",     "retriever.embed_latency_ms"))
    print(stat_row("  ├─ pinecone query (mock)",   "retriever.pinecone_latency_ms"))
    print(stat_row("  ├─ cross-encoder rerank",    "retriever.rerank_latency_ms"))
    print(stat_row("  ├─ LLM gemini (mock)",       "generator.llm_latency_ms{provider=gemini}"))
    print(stat_row("  └─ LLM groq (mock)",         "generator.llm_latency_ms{provider=groq}"))

    print(f"\n  Retriever quality:")
    print(f"  {sep}")
    chunks_d = h.get("retriever.chunks_returned", {})
    if chunks_d and chunks_d.get("count", 0) > 0:
        print(f"  {'avg chunks / pair':<36} {chunks_d['mean']:.1f}  "
              f"(p50={chunks_d['p50']:.0f}, max={chunks_d['max']:.0f})")
    print(f"  {'empty retrieval results':<36} {ctr('retriever.empty_results.total')}")
    print(f"  {'rerank failures':<36} {ctr('retriever.rerank_failures.total')}")
    print(f"  {'embed errors':<36} {ctr('retriever.embed_errors.total')}")
    print(f"  {'pinecone errors':<36} {ctr('retriever.pinecone_errors.total')}")

    print(f"\n  Generator:")
    print(f"  {sep}")
    total_llm = ctr("generator.gemini_ok.total") + ctr("generator.groq_ok.total")
    total_fb  = ctr("generator.fda_fallbacks.total")
    fallback_pct = total_fb / max(total_llm + total_fb, 1) * 100
    print(f"  {'gemini ok':<36} {ctr('generator.gemini_ok.total')}")
    print(f"  {'groq ok':<36} {ctr('generator.groq_ok.total')}")
    print(f"  {'fda fallbacks':<36} {total_fb}  ({fallback_pct:.1f}%)")

    print(f"\n  Warning severity distribution:")
    print(f"  {sep}")
    total_warn = ctr("warnings.severity.red") + ctr("warnings.severity.yellow") + ctr("warnings.severity.green")
    print(f"  {'RED   (critical)':<36} {ctr('warnings.severity.red')}")
    print(f"  {'YELLOW (caution)':<36} {ctr('warnings.severity.yellow')}")
    print(f"  {'GREEN  (safe)':<36} {ctr('warnings.severity.green')}")
    print(f"  {'total warnings generated':<36} {total_warn}")

    print(f"\n{SEP}")
    if wall_times:
        print(f"  SUMMARY:")
        print(f"    total runs        : {n_runs}")
        print(f"    mean latency      : {statistics.mean(wall_times):.0f}ms")
        print(f"    median (p50)      : {pct(wall_times, 50):.0f}ms")
        print(f"    p95 latency       : {pct(wall_times, 95):.0f}ms")
        print(f"    p99 latency       : {pct(wall_times, 99):.0f}ms")
        print(f"    max latency       : {max(wall_times):.0f}ms")
        print(f"    min latency       : {min(wall_times):.0f}ms")
        partial_pct = statuses.get("PARTIAL", 0) / n_runs * 100
        print(f"    PARTIAL rate      : {partial_pct:.0f}%  (FDA fallback instead of LLM)")
    print(SEP + "\n")

    # ── save baseline.json ────────────────────────────────────────────────────
    baseline = {
        "meta": {
            "runs": n_runs,
            "mode": "fast" if fast else "realistic_mock",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "mean_ms":   round(statistics.mean(wall_times), 1) if wall_times else 0,
            "p50_ms":    round(pct(wall_times, 50), 1)         if wall_times else 0,
            "p95_ms":    round(pct(wall_times, 95), 1)         if wall_times else 0,
            "p99_ms":    round(pct(wall_times, 99), 1)         if wall_times else 0,
            "max_ms":    round(max(wall_times), 1)             if wall_times else 0,
            "min_ms":    round(min(wall_times), 1)             if wall_times else 0,
            "partial_rate_pct": round(statuses.get("PARTIAL", 0) / n_runs * 100, 1),
            "error_count": errors,
        },
        "per_stage_ms": {
            k: v for k, v in h.items()
        },
        "counters": c,
        "status_counts": statuses,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "baseline.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  Baseline saved → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedReconcile E2E latency benchmark")
    parser.add_argument("--runs", type=int, default=100, help="Number of requests (default: 100)")
    parser.add_argument("--fast", action="store_true", help="Near-zero delays — measure pure Python overhead")
    args = parser.parse_args()

    asyncio.run(main(n_runs=args.runs, fast=args.fast))
