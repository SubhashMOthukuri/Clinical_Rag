#!/usr/bin/env python3
"""Pass 4: Score retrieval quality across chunk config namespaces.

Loads data/eval/retrieval_eval.json, embeds each query, retrieves the top-K
chunks from each named Pinecone namespace, and scores how many queries have
at least one expected keyword in the top-K results.

Usage:
    # Compare all three configs:
    python scripts/04_eval_retrieval.py \\
        --namespaces chunks_v1_512_80 chunks_v2_256_32 chunks_v3_1024_128

    # Quick check on one namespace:
    python scripts/04_eval_retrieval.py --namespaces chunks_v1_512_80

    # More results per query:
    python scripts/04_eval_retrieval.py --namespaces chunks_v1_512_80 --top-k 5

Scoring:
    A query passes if at least 1 of the top-K chunks contains ALL expected
    keywords (case-insensitive substring match).  The namespace with the
    highest pass rate wins.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.embedding.embedder import OpenAIEmbedder
from src.retrieval.pinecone_store import PineconeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _keywords_found(chunks: list, keywords: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, found_keywords) where found_keywords are those matched."""
    found = set()
    for chunk in chunks:
        text = chunk.metadata.text.lower()
        for kw in keywords:
            if kw.lower() in text:
                found.add(kw)
    return len(found) == len(keywords), sorted(found)


async def eval_namespace(
    namespace: str,
    queries: list[dict],
    embedder: GeminiEmbedder,
    store: PineconeStore,
    top_k: int,
) -> dict:
    results = []

    for q in queries:
        query_text = q["query"]
        keywords = q["expected_keywords"]

        try:
            vec = await embedder.embed(query_text)
            chunks = await store.query(vec, top_k=top_k, namespace=namespace)
        except Exception as e:
            logger.warning("query_failed namespace=%s query=%r error=%s", namespace, query_text[:50], e)
            results.append({"query": query_text, "passed": False, "found": [], "error": str(e)})
            continue

        passed, found = _keywords_found(chunks, keywords)

        results.append({
            "query": query_text,
            "passed": passed,
            "expected_keywords": keywords,
            "found_keywords": found,
            "missing_keywords": [k for k in keywords if k not in found],
            "top_chunk_preview": chunks[0].metadata.text[:120] if chunks else "",
            "top_chunk_article": chunks[0].metadata.article_id if chunks else "",
        })

        status = "✓" if passed else "✗"
        logger.info("[%s] %s  %s", namespace, status, query_text[:60])

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "namespace": namespace,
        "score": passed_count / len(queries) if queries else 0.0,
        "passed": passed_count,
        "total": len(queries),
        "results": results,
    }


async def main_async(args: argparse.Namespace) -> None:
    eval_path = ROOT / args.eval
    if not eval_path.exists():
        logger.error("Eval file not found: %s", eval_path)
        sys.exit(1)

    queries = json.loads(eval_path.read_text())
    logger.info("Loaded %d eval queries from %s", len(queries), eval_path)

    embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)
    store = PineconeStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        dimensions=768,
    )

    namespace_scores: list[dict] = []

    try:
        for ns in args.namespaces:
            logger.info("--- Evaluating namespace: %s ---", ns)
            score_data = await eval_namespace(ns, queries, embedder, store, args.top_k)
            namespace_scores.append(score_data)
    finally:
        if hasattr(embedder, "close"):
            await embedder.close()
        await store.close()

    # ── print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  RETRIEVAL EVAL  (top-{args.top_k}, {len(queries)} queries)")
    print("=" * 62)
    print(f"  {'Namespace':<30} {'Score':>6}  {'Pass/Total'}")
    print("  " + "-" * 58)
    best = max(namespace_scores, key=lambda x: x["score"])
    for s in sorted(namespace_scores, key=lambda x: -x["score"]):
        marker = " ◀ best" if s["namespace"] == best["namespace"] else ""
        print(
            f"  {s['namespace']:<30} {s['score']:>5.0%}   "
            f"{s['passed']}/{s['total']}{marker}"
        )
    print("=" * 62)

    # ── per-query breakdown for each namespace ────────────────────────────────
    if args.verbose:
        for s in namespace_scores:
            print(f"\n  [{s['namespace']}]")
            for r in s["results"]:
                status = "✓" if r["passed"] else "✗"
                print(f"    {status} {r['query'][:55]}")
                if not r["passed"]:
                    missing = r.get("missing_keywords", [])
                    print(f"      missing: {missing}")
                    if r.get("top_chunk_preview"):
                        print(f"      top chunk: {r['top_chunk_preview']}...")

    # ── write full results JSON ───────────────────────────────────────────────
    if args.output:
        out_path = ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(namespace_scores, indent=2))
        print(f"\nFull results written to {out_path}")

    winner = best["namespace"]
    print(f"\nWinner: {winner}  ({best['score']:.0%} pass rate)")
    print(f"Next: python scripts/03_ingest_statpearls.py with namespace={winner} for full re-ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespaces", nargs="+", required=True,
        help="One or more Pinecone namespaces to evaluate and compare",
    )
    parser.add_argument(
        "--eval", default="data/eval/retrieval_eval.json",
        help="Eval queries JSON file",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of chunks to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query breakdown",
    )
    parser.add_argument(
        "--output", default="",
        help="Optional path to write full JSON results (e.g. data/eval/results.json)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
