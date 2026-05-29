#!/usr/bin/env python3
"""Pass 3: Embed chunks and upsert to a Pinecone namespace.

Reads the chunks JSONL produced by 02_chunk_statpearls.py, embeds each
chunk with OpenAIEmbedder (text-embedding-3-small, 768d), and upserts the
vectors to Pinecone under the given namespace.

Usage:
    # Typical flow — run once per config variant:
    python scripts/03_ingest_statpearls.py \\
        --chunks data/processed/chunks_v1.jsonl \\
        --namespace chunks_v1_512_80

    python scripts/03_ingest_statpearls.py \\
        --chunks data/processed/chunks_v2.jsonl \\
        --namespace chunks_v2_256_32

Checkpoint support: chunk IDs already upserted are tracked and skipped on
re-run so an interruption doesn't waste API quota.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.embedding.embedder import OpenAIEmbedder
from src.retrieval.pinecone_store import ChunkMetadata, PineconeStore, VectorRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_BATCH = 100   # texts per OpenAI embeddings.create call (up to 2048)
UPSERT_BATCH = 100  # vectors per Pinecone upsert
BATCH_DELAY_S = 0.0  # no delay needed for OpenAI; override with --batch-delay if throttled


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def _save_checkpoint(path: Path, ids: set[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(ids)))
    os.replace(tmp, path)


def _read_chunks(path: Path, done_ids: set[str]) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk["chunk_id"] not in done_ids:
                chunks.append(chunk)
    return chunks


async def ingest(
    chunks_path: Path,
    namespace: str,
    checkpoint_path: Path,
    embedder: GeminiEmbedder,
    store: PineconeStore,
    batch_delay: float = 0.0,
) -> dict:
    done_ids = _load_checkpoint(checkpoint_path)
    if done_ids:
        logger.info("Resuming — %d chunks already ingested", len(done_ids))

    pending = _read_chunks(chunks_path, done_ids)
    total = len(pending)
    logger.info("%d chunks to embed and upsert → namespace=%s", total, namespace)

    upserted = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for batch_start in range(0, total, EMBED_BATCH):
        batch = pending[batch_start : batch_start + EMBED_BATCH]
        texts = [c["content"] for c in batch]

        logger.info(
            "embedding batch %d-%d / %d",
            batch_start + 1, batch_start + len(batch), total,
        )

        vectors = None
        for attempt in range(1, 4):
            try:
                vectors = await embedder.embed_batch(texts)
                break
            except Exception as e:
                err_str = str(e)
                # Rate limit: honour the API's window (~60s) rather than retrying too fast.
                # Circuit breaker also opens on 429; the 65s wait lets it reset.
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                wait = 65 if is_rate_limit else 2 ** attempt  # 65s | 2/4/8s
                logger.warning(
                    "embed_failed attempt=%d/%d wait=%ds rate_limit=%s error=%s",
                    attempt, 3, wait, is_rate_limit, err_str[:120],
                )
                if attempt == 3:
                    logger.error("embed_giving_up batch_start=%d — skipping batch", batch_start)
                    break
                await asyncio.sleep(wait)
        if vectors is None:
            continue

        if batch_delay > 0:
            await asyncio.sleep(batch_delay)

        records = [
            VectorRecord(
                id=chunk["chunk_id"],
                values=vec,
                metadata=ChunkMetadata(
                    text=chunk["content"],
                    title=chunk["title"],
                    source=chunk.get("source", "StatPearls"),
                    article_id=chunk["article_id"],
                    article_type=chunk.get("article_type", "general"),
                    token_count=chunk.get("char_count", len(chunk["content"])) // 4,
                    created_at=now_iso,
                    updated_at=now_iso,
                ),
            )
            for chunk, vec in zip(batch, vectors)
        ]

        # Upsert in sub-batches if embed batch > upsert limit
        for i in range(0, len(records), UPSERT_BATCH):
            sub = records[i : i + UPSERT_BATCH]
            try:
                await store.upsert_batch(sub, namespace=namespace)
                for r in sub:
                    done_ids.add(r.id)
                upserted += len(sub)
            except Exception as e:
                logger.error("upsert_failed error=%s", e)
                continue
            finally:
                _save_checkpoint(checkpoint_path, done_ids)
            logger.info("checkpoint saved  upserted=%d / %d", upserted, total)

    _save_checkpoint(checkpoint_path, done_ids)
    return dict(total=total, upserted=upserted, skipped=total - upserted)


async def main_async(args: argparse.Namespace) -> None:
    chunks_path = ROOT / args.chunks
    checkpoint_path = ROOT / args.checkpoint

    if not chunks_path.exists():
        logger.error("Chunks file not found: %s  — run 02_chunk_statpearls.py first", chunks_path)
        sys.exit(1)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)
    store = PineconeStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME,
        dimensions=768,
    )

    try:
        stats = await ingest(
            chunks_path, args.namespace, checkpoint_path,
            embedder, store, batch_delay=args.batch_delay,
        )
    finally:
        # GeminiEmbedder has no close(); guard for other embedder implementations
        if hasattr(embedder, "close"):
            await embedder.close()
        await store.close()

    print(f"\nNamespace : {args.namespace}")
    print(f"Upserted  : {stats['upserted']} / {stats['total']}")
    print(f"Skipped   : {stats['skipped']}")
    print(f"\nNext: python scripts/04_eval_retrieval.py --namespaces {args.namespace}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", required=True,
        help="Chunks JSONL file from 02_chunk_statpearls.py",
    )
    parser.add_argument(
        "--namespace", required=True,
        help="Pinecone namespace to write to (e.g. chunks_v1_512_80)",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint file path (default: derived from --chunks path)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run",
    )
    parser.add_argument(
        "--batch-delay", type=float, default=BATCH_DELAY_S,
        help="Seconds to sleep between embed batches. Set to 65 on Gemini free tier (100 embeddings/min limit)",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        stem = Path(args.chunks).stem
        args.checkpoint = f"data/processed/{stem}_ingest_checkpoint.json"

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
