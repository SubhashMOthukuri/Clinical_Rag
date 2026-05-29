#!/usr/bin/env python3
"""Pass 1: Extract ALL StatPearls articles to clean JSONL.

Parses every .nxml file in the collection folder and writes one JSON line
per article to data/processed/statpearls/articles_all.jsonl.

Run this ONCE.  It is drug-agnostic — no filtering here.  Use
01b_filter_articles.py to filter for any drug list without re-parsing XML.

Features:
  - Checkpoint resume: safe to interrupt and re-run with --resume
  - Per-file timeout: a corrupt/huge XML never hangs the run
  - Atomic output: output file is never half-written

Usage:
    python scripts/01_extract_statpearls.py
    python scripts/01_extract_statpearls.py --resume   # continue interrupted run
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--articles-dir",
        default="data/raw/statpearls/statpearls_NBK430685",
        help="Folder containing .nxml files (one per article)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/statpearls/articles_all.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/processed/statpearls/articles_all_checkpoint.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    articles_dir = ROOT / args.articles_dir
    output_path = ROOT / args.output
    checkpoint_path = ROOT / args.checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not articles_dir.exists():
        logger.error("Articles directory not found: %s", articles_dir)
        sys.exit(1)

    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    processor = StatPearlsProcessor()
    stats = processor.extract_articles(articles_dir, output_path, checkpoint_path)

    print(f"\nExtracted : {stats['extracted']} / {stats['total_files']} articles")
    print(f"Failed    : {stats['failed']}")
    print(f"Empty     : {stats['empty']}")
    print(f"Output    : {output_path}")
    print(f"\nNext: python scripts/01b_filter_articles.py")


if __name__ == "__main__":
    main()
