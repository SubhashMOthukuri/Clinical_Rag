#!/usr/bin/env python3
"""Pass 1b: Filter extracted articles to those mentioning target drugs.

Reads articles_all.jsonl (from 01_extract_statpearls.py) and keeps only
articles whose title or section text contains at least one target drug name.
Fast — pure string search, no XML parsing.

Re-run anytime with a different --drugs list without re-parsing XML.

Usage:
    python scripts/01b_filter_articles.py
    python scripts/01b_filter_articles.py --drugs warfarin aspirin metformin
    python scripts/01b_filter_articles.py --dry-run   # count matches, write nothing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DRUGS = [
    "warfarin", "ibuprofen", "aspirin", "metformin",
    "lisinopril", "atorvastatin", "metoprolol",
    "omeprazole", "amoxicillin", "sertraline",
]


def _matches(article: dict, drugs: list[str]) -> bool:
    text = (
        article.get("title", "") + " "
        + " ".join(s["text"] for s in article.get("sections", []))
    ).lower()
    return any(drug in text for drug in drugs)


def filter_articles(
    input_path: Path,
    output_path: Path,
    drugs: list[str],
    dry_run: bool = False,
) -> dict:
    total = kept = 0

    tmp_path = output_path.with_suffix(".tmp")
    out = open(os.devnull, "w") if dry_run else open(tmp_path, "w")

    with open(input_path) as src, out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            article = json.loads(line)
            if _matches(article, drugs):
                kept += 1
                if not dry_run:
                    out.write(line + "\n")

            if total % 1000 == 0:
                logger.info("scanned %d  kept %d", total, kept)

    if not dry_run:
        os.replace(tmp_path, output_path)

    logger.info("done  scanned=%d kept=%d filtered_out=%d", total, kept, total - kept)
    return dict(total=total, kept=kept, filtered_out=total - kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/statpearls/articles_all.jsonl",
        help="All-articles JSONL from 01_extract_statpearls.py",
    )
    parser.add_argument(
        "--output",
        default="data/processed/statpearls/articles_filtered.jsonl",
    )
    parser.add_argument(
        "--drugs", nargs="+", default=DEFAULT_DRUGS,
        help="Drug names to filter for (case-insensitive substring match)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count matches without writing output",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input not found: %s  — run 01_extract_statpearls.py first", input_path)
        sys.exit(1)

    logger.info("Filtering for drugs: %s", args.drugs)
    stats = filter_articles(input_path, output_path, args.drugs, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run — would keep {stats['kept']} / {stats['total']} articles")
    else:
        print(f"\nKept     : {stats['kept']} / {stats['total']} articles")
        print(f"Filtered : {stats['filtered_out']}")
        print(f"Output   : {output_path}")
        print(f"\nNext: python scripts/02_chunk_statpearls.py --config v1")


if __name__ == "__main__":
    main()