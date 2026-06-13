"""End-to-end latency tester for MedReconcile AI.

Sends N reconcile requests to the running server, then fetches /metrics
and prints a per-stage latency breakdown table.

Usage:
    # default: 5 requests against localhost:8000
    python scripts/latency_test.py

    # custom runs / base url
    python scripts/latency_test.py --runs 10 --url http://localhost:8000

Requires the server to be running:
    uvicorn src.main:app --reload
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

try:
    import httpx
except ImportError:
    raise SystemExit("httpx not installed — run: pip install httpx")


# ---------------------------------------------------------------------------
# Test payloads — realistic drug pairs that exercise the full pipeline
# ---------------------------------------------------------------------------
PAYLOADS = [
    {
        "medications": [
            {"name": "warfarin", "dose": 5.0, "unit": "mg", "frequency": "daily"},
            {"name": "aspirin",  "dose": 81.0, "unit": "mg", "frequency": "daily"},
        ]
    },
    {
        "medications": [
            {"name": "metformin", "dose": 500.0, "unit": "mg", "frequency": "twice daily"},
            {"name": "lisinopril", "dose": 10.0, "unit": "mg", "frequency": "daily"},
            {"name": "atorvastatin", "dose": 20.0, "unit": "mg", "frequency": "nightly"},
        ]
    },
    {
        "medications": [
            {"name": "clopidogrel", "dose": 75.0, "unit": "mg", "frequency": "daily"},
            {"name": "omeprazole",  "dose": 20.0, "unit": "mg", "frequency": "daily"},
        ]
    },
    {
        "medications": [
            {"name": "amiodarone", "dose": 200.0, "unit": "mg", "frequency": "daily"},
            {"name": "digoxin",    "dose": 0.125, "unit": "mg", "frequency": "daily"},
        ]
    },
    {
        "medications": [
            {"name": "sertraline",  "dose": 50.0, "unit": "mg", "frequency": "daily"},
            {"name": "tramadol",    "dose": 50.0, "unit": "mg", "frequency": "as needed"},
            {"name": "ondansetron", "dose": 4.0,  "unit": "mg", "frequency": "as needed"},
        ]
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _row(label: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"  {label:<35} no data"
    mean = statistics.mean(values)
    p50  = _pct(values, 50)
    p95  = _pct(values, 95)
    mx   = max(values)
    return (
        f"  {label:<35} "
        f"mean={mean:7.1f}{unit}  "
        f"p50={p50:7.1f}{unit}  "
        f"p95={p95:7.1f}{unit}  "
        f"max={mx:7.1f}{unit}"
    )


def _counter_row(label: str, value: int) -> str:
    return f"  {label:<35} {value}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(base_url: str, n_runs: int) -> None:
    print(f"\nMedReconcile Latency Test — {n_runs} runs against {base_url}")
    print(f"Started at {datetime.now().strftime('%H:%M:%S')}\n")

    wall_times: list[float] = []
    statuses: dict[str, int] = {"SUCCESS": 0, "PARTIAL": 0, "FAILED": 0, "HTTP_ERROR": 0}

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        # health check first
        try:
            r = client.get("/health")
            r.raise_for_status()
            print(f"  /health → {r.json()['status']}")
        except Exception as e:
            raise SystemExit(f"Server not reachable at {base_url}: {e}")

        print(f"  Sending {n_runs} POST /reconcile requests...\n")
        for i in range(n_runs):
            payload = PAYLOADS[i % len(PAYLOADS)]
            n_meds = len(payload["medications"])
            t0 = time.perf_counter()
            try:
                r = client.post("/reconcile", json=payload)
                elapsed = (time.perf_counter() - t0) * 1000
                wall_times.append(elapsed)

                if r.status_code == 200:
                    body = r.json()
                    status = body.get("status", "?")
                    n_warn = body.get("total_warnings", 0)
                    n_crit = body.get("critical_warnings", 0)
                    statuses[status] = statuses.get(status, 0) + 1
                    print(
                        f"  [{i+1:02d}] {elapsed:7.1f}ms  "
                        f"status={status:<8} meds={n_meds}  "
                        f"warnings={n_warn}  critical={n_crit}"
                    )
                else:
                    statuses["HTTP_ERROR"] += 1
                    print(f"  [{i+1:02d}] {elapsed:7.1f}ms  HTTP {r.status_code}: {r.text[:80]}")
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                wall_times.append(elapsed)
                statuses["HTTP_ERROR"] += 1
                print(f"  [{i+1:02d}] {elapsed:7.1f}ms  ERROR: {e}")

        # fetch metrics snapshot
        print("\n  Fetching /metrics snapshot...")
        try:
            snap = client.get("/metrics").json()
        except Exception as e:
            print(f"  Could not fetch /metrics: {e}")
            snap = None

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  END-TO-END LATENCY REPORT")
    print("=" * 75)

    print(f"\n  Requests sent : {n_runs}")
    for s, cnt in statuses.items():
        if cnt:
            print(f"  {s:<15}: {cnt}")

    print(f"\n  Wall-clock (client-observed, includes network RTT):")
    print(_row("total request time", wall_times))

    if snap:
        h = snap.get("histograms", {})
        c = snap.get("counters", {})

        def hist(key: str) -> list[float]:
            """Reconstruct flat value list from histogram stats for display."""
            # /metrics returns pre-aggregated stats, not raw values.
            # We display the stats dict directly.
            return []

        print("\n  Per-stage latency (from server-side /metrics):")
        print("  " + "-" * 71)

        def stat_row(label: str, key: str) -> str:
            d = h.get(key)
            if not d or d["count"] == 0:
                return f"  {label:<35} no data"
            return (
                f"  {label:<35} "
                f"mean={d['mean']:7.1f}ms  "
                f"p50={d['p50']:7.1f}ms  "
                f"p95={d['p95']:7.1f}ms  "
                f"max={d['max']:7.1f}ms  "
                f"(n={d['count']})"
            )

        print(stat_row("reconcile end-to-end",    "reconcile.latency_ms"))
        print(stat_row("  embed (OpenAI)",         "retriever.embed_latency_ms"))
        print(stat_row("  pinecone query",         "retriever.pinecone_latency_ms"))
        print(stat_row("  cross-encoder rerank",   "retriever.rerank_latency_ms"))
        print(stat_row("  LLM gemini",             "generator.llm_latency_ms{provider=gemini}"))
        print(stat_row("  LLM groq (fallback)",    "generator.llm_latency_ms{provider=groq}"))

        print("\n  Retriever quality counters:")
        print("  " + "-" * 71)
        def ctr(key: str) -> int:
            return c.get(key, 0)

        print(_counter_row("chunks returned (total obs)", h.get("retriever.chunks_returned", {}).get("count", 0)))
        chunks_d = h.get("retriever.chunks_returned", {})
        if chunks_d and chunks_d.get("count", 0) > 0:
            print(f"  {'  mean chunks / pair':<35} {chunks_d['mean']:.1f}  (p50={chunks_d['p50']:.1f})")
        print(_counter_row("empty retrieval results",  ctr("retriever.empty_results.total")))
        print(_counter_row("rerank failures",           ctr("retriever.rerank_failures.total")))
        print(_counter_row("embed errors",              ctr("retriever.embed_errors.total")))
        print(_counter_row("pinecone errors",           ctr("retriever.pinecone_errors.total")))

        print("\n  Generator counters:")
        print("  " + "-" * 71)
        print(_counter_row("gemini ok",             ctr("generator.gemini_ok.total")))
        print(_counter_row("gemini errors",         ctr("generator.gemini_errors.total")))
        print(_counter_row("groq ok",               ctr("generator.groq_ok.total")))
        print(_counter_row("groq errors",           ctr("generator.groq_errors.total")))
        print(_counter_row("fda fallbacks",         ctr("generator.fda_fallbacks.total")))
        print(_counter_row("circuit breaker opens", ctr("breaker.gemini.open.total") + ctr("breaker.groq.open.total")))

        print("\n  Warning severity distribution:")
        print("  " + "-" * 71)
        print(_counter_row("RED (critical)",  ctr("warnings.severity.red")))
        print(_counter_row("YELLOW (caution)", ctr("warnings.severity.yellow")))
        print(_counter_row("GREEN (safe)",    ctr("warnings.severity.green")))

    print("\n" + "=" * 75)
    if wall_times:
        fda_fallback_rate = statuses.get("PARTIAL", 0) / n_runs * 100
        print(f"  Median wall latency : {_pct(wall_times, 50):.0f}ms")
        print(f"  p95 wall latency    : {_pct(wall_times, 95):.0f}ms")
        print(f"  PARTIAL rate        : {fda_fallback_rate:.0f}%  (LLM fallback to FDA)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedReconcile end-to-end latency test")
    parser.add_argument("--runs", type=int, default=5, help="Number of requests to send (default: 5)")
    parser.add_argument("--url",  type=str, default="http://localhost:8000", help="Server base URL")
    args = parser.parse_args()
    run(base_url=args.url, n_runs=args.runs)
