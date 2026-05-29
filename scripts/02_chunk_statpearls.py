#!/usr/bin/env python3
"""Pass 2: Chunk filtered articles with a named ChunkConfig variant.

Reads the articles_filtered.jsonl produced by 01_extract_statpearls.py,
applies the requested chunk config, and writes chunks to
data/processed/chunks_{config_name}.jsonl.

Run once per config variant to produce the A/B candidates:

    python scripts/02_chunk_statpearls.py --config v1   # 512 chars, 80 overlap  (default)
    python scripts/02_chunk_statpearls.py --config v2   # 256 chars, 32 overlap  (small)
    python scripts/02_chunk_statpearls.py --config v3   # 1024 chars, 128 overlap (large)

Output file is named after the config so you can't accidentally mix them.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "statpearls_processor",
    ROOT / "src" / "ingestion" / "statpearls-processor.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["statpearls_processor"] = _mod
_spec.loader.exec_module(_mod)
StatPearlsProcessor = _mod.StatPearlsProcessor

from src.chunking import ChunkConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Named configs — add more variants here as needed
CONFIGS: dict[str, ChunkConfig] = {
    "v1": ChunkConfig(max_chunk_size=512,  min_chunk_size=252,  overlap_size=80),    # ~128 tokens, balanced
    "v2": ChunkConfig(max_chunk_size=256,  min_chunk_size=128,  overlap_size=32),    # ~64 tokens, precise
    "v3": ChunkConfig(max_chunk_size=1024, min_chunk_size=512,  overlap_size=128),   # ~256 tokens, contextual
    "v4": ChunkConfig(max_chunk_size=2048, min_chunk_size=1024, overlap_size=256),   # ~512 tokens, large context
}

# Pinecone namespace each config writes to (used in 03_ingest and 04_eval)
NAMESPACES: dict[str, str] = {
    "v1": "chunks_v1_512_80",
    "v2": "chunks_v2_256_32",
    "v3": "chunks_v3_1024_128",
    "v4": "chunks_v4_2048_256",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", choices=list(CONFIGS), required=True,
        help="Chunk config variant to apply",
    )
    parser.add_argument(
        "--input", default="data/processed/statpearls/articles_filtered.jsonl",
        help="Filtered articles JSONL from 01b_filter_articles.py",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/statpearls",
        help="Directory for the output chunks JSONL",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    namespace = NAMESPACES[args.config]

    input_path = ROOT / args.input
    output_path = ROOT / args.output_dir / f"chunks_{args.config}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input not found: %s  — run 01_extract_statpearls.py first", input_path)
        sys.exit(1)

    logger.info(
        "Chunking with config=%s  max=%d min=%d overlap=%d  → %s",
        args.config, cfg.max_chunk_size, cfg.min_chunk_size, cfg.overlap_size, output_path.name,
    )

    processor = StatPearlsProcessor(config=cfg)
    stats = processor.chunk_articles(input_path, output_path, config=cfg)

    print(f"\nConfig   : {args.config}  (namespace → {namespace})")
    print(f"Articles : {stats['total_articles']}")
    print(f"Chunks   : {stats['total_chunks']}")
    print(f"Failed   : {stats['failed_articles']}")
    if stats['total_articles']:
        ratio = stats['total_chunks'] / stats['total_articles']
        print(f"Avg chunks/article: {ratio:.1f}")
    print(f"Output   : {output_path}")
    print(f"\nNext: python scripts/03_ingest_statpearls.py --chunks {output_path} --namespace {namespace}")


if __name__ == "__main__":
    main()
