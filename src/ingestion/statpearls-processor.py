from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

FILE_TIMEOUT_S = 30  # max seconds allowed per XML file before it is skipped


@dataclass
class ChunkConfig:
    max_chunk_size: int = 1000
    min_chunk_size: int = 200
    overlap_size: int = 100
    include_title: bool = True  # TODO(prod): field defined but never read — wire into _assign_ids prefix logic or remove

    def __post_init__(self) -> None:
        if self.overlap_size >= self.max_chunk_size:
            raise ValueError("overlap_size must be less than max_chunk_size")


@dataclass(frozen=True)
class ArticleChunk:
    chunk_id: str
    article_id: str
    title: str
    section: str
    chunk_index: int
    total_chunks: int
    content: str
    char_count: int
    article_type: str = "general"
    source: str = "StatPearls"


@dataclass
class Article:
    """Intermediate representation produced by Pass 1 (extract_articles).

    Stores cleaned text only — no chunking decisions yet — so Pass 2
    (chunk_articles) can re-chunk with any ChunkConfig without re-parsing XML.
    """
    article_id: str
    title: str
    article_type: str
    sections: list[dict]  # each: {"title": str, "text": str}
    source: str = "StatPearls"


class StatPearlsProcessor:
    def __init__(self, config: ChunkConfig | None = None):
        self._cfg = config or ChunkConfig()

    def process_file(
        self,
        filepath: Path,
    ) -> list[ArticleChunk]:
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error("xml.parse_failed file=%s error=%s", filepath.name, e)
            return []

        article_id = filepath.stem
        title = self._extract_title(root)
        sections = self._extract_sections(root)
        article_type = self._detect_article_type(title, sections)

        # Carry section title alongside each chunk
        tagged: list[tuple[str, str]] = []
        for section_title, section_text in sections:
            for chunk_text in self._chunk_text(section_text):
                tagged.append((section_title, chunk_text))

        return self._assign_ids(tagged, article_id, title, article_type)

    def _extract_title(self, root: ET.Element) -> str:
        for tag in ["title", "article-title"]:
            el = root.find(f".//{tag}")
            if el is not None and el.text:
                return self._clean_text(el.text, add_period=False)
        return "Unknown"

    def _extract_sections(
        self,
        root: ET.Element,
        min_length: int | None = None,
    ) -> list[tuple[str, str]]:
        # Pass 1 (extract_articles) passes min_length=0 to keep all sections;
        # process_file leaves it None to fall back to config's min_chunk_size.
        threshold = min_length if min_length is not None else self._cfg.min_chunk_size
        sections = []
        for sec in root.findall(".//sec"):
            sec_title_el = sec.find("title")
            sec_title = self._clean_text(
                sec_title_el.text if sec_title_el is not None and sec_title_el.text
                else "General",
                add_period=False,
            )
            paragraphs = []
            # TODO(prod): only <p> tags scanned — <table>, <list>, <boxed-text> content silently lost
            for p in sec.findall(".//p"):
                text = self._get_element_text(p)
                if text:
                    paragraphs.append(text)

            section_text = " ".join(paragraphs)
            if len(section_text) >= threshold:
                sections.append((sec_title, section_text))

        return sections

    def _clean_text(self, text: str, add_period: bool = True) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\{[^}]+\}", "", text)
        text = re.sub(r"\s+", " ", text)
        # TODO(prod): strips Greek letters (α β μ), em-dashes, degree signs — audit against real medical text before prod
        text = re.sub(r"[^\w\s.,;:!?()\-\/]", "", text)
        text = text.strip()
        if add_period and text and text[-1] not in ".!?":
            text += "."
        return text

    def _get_element_text(self, element: ET.Element) -> str:
        text = "".join(element.itertext())
        return self._clean_text(text)

    def _chunk_text(self, text: str, cfg: ChunkConfig | None = None) -> list[str]:
        cfg = cfg or self._cfg
        if len(text) <= cfg.max_chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + cfg.max_chunk_size

            if end < len(text):
                boundary = text.rfind(".", start, end)
                if boundary > start + cfg.min_chunk_size:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if len(chunk) >= cfg.min_chunk_size:
                chunks.append(chunk)

            next_start = end - cfg.overlap_size
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def _assign_ids(
        self,
        tagged_chunks: list[tuple[str, str]],
        article_id: str,
        title: str,
        article_type: str,
    ) -> list[ArticleChunk]:
        total = len(tagged_chunks)
        result = []
        for idx, (section, content) in enumerate(tagged_chunks):
            result.append(ArticleChunk(
                chunk_id=f"{article_id}_chunk_{idx:04d}",
                article_id=article_id,
                title=title,
                section=section,
                chunk_index=idx,
                total_chunks=total,
                content=content,
                char_count=len(content),
                article_type=article_type,
                source="StatPearls",
            ))
        return result

    # TODO(prod): keyword match misclassifies — "drug abuse" → drug_interaction, anatomy article with "drug" in one sentence → mislabelled
    def _detect_article_type(
        self,
        title: str,
        sections: list[tuple[str, str]],
    ) -> str:
        drug_keywords = [
            "drug", "medication", "pharmacology",
            "interaction", "pharmacokinetics", "dosage",
        ]
        title_lower = title.lower()
        if any(kw in title_lower for kw in drug_keywords):
            return "drug_interaction"

        section_text = " ".join(
            f"{sec_title} {sec_body[:200]}"
            for sec_title, sec_body in sections
        ).lower()
        if any(kw in section_text for kw in drug_keywords):
            return "drug_interaction"

        return "general"

    # ---- Pass 1 helpers ----

    def _extract_article(self, filepath: Path) -> Article | None:
        """Parse one XML file into a cleaned Article. Runs inside ThreadPoolExecutor."""
        # TODO(prod): ET.parse loads full XML tree into memory — a multi-MB file can OOM before timeout fires
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error("xml.parse_failed file=%s error=%s", filepath.name, e)
            return None

        article_id = filepath.stem
        title = self._extract_title(root)
        # min_length=0: keep every section regardless of length; chunking
        # decisions belong to Pass 2, not Pass 1.
        sections = self._extract_sections(root, min_length=0)
        article_type = self._detect_article_type(title, sections)

        return Article(
            article_id=article_id,
            title=title,
            article_type=article_type,
            sections=[{"title": t, "text": text} for t, text in sections],
        )

    def _load_checkpoint(self, path: Path) -> set[str]:
        if not path.exists():
            return set()
        try:
            with open(path) as f:
                return set(json.load(f))
        except Exception:
            return set()

    def _save_checkpoint(self, path: Path, processed_ids: set[str]) -> None:
        """Atomically persist the processed-id set so a crash never corrupts it."""
        # TODO(prod): serialises full set every 100 files — write cost grows linearly; switch to append-only log at scale
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(list(processed_ids), f)
        os.replace(tmp, path)

    # ---- Pass 2 helpers ----

    def _chunk_article(self, article_dict: dict, cfg: ChunkConfig) -> list[dict]:
        """Turn one Article dict (from Pass 1 JSONL) into a list of chunk dicts."""
        article_id = article_dict["article_id"]
        title = article_dict["title"]
        article_type = article_dict["article_type"]

        tagged: list[tuple[str, str]] = []
        for section in article_dict["sections"]:
            for chunk_text in self._chunk_text(section["text"], cfg):
                tagged.append((section["title"], chunk_text))

        return [
            {
                "chunk_id": c.chunk_id,
                "article_id": c.article_id,
                "title": c.title,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "content": c.content,
                "char_count": c.char_count,
                "article_type": c.article_type,
                "source": c.source,
            }
            for c in self._assign_ids(tagged, article_id, title, article_type)
        ]

    # ---- Public pipeline methods ----

    def extract_articles(
        self,
        directory: Path,
        output_path: Path,
        checkpoint_path: Path | None = None,
    ) -> dict:
        """Pass 1: XML → cleaned text JSONL. Safe to interrupt and resume.

        Three production strategies:
        - Checkpoint resume: skips article_ids already in checkpoint; saves
          progress to checkpoint_path every 100 extracted articles.
        - Timeout per file: each XML parse is capped at FILE_TIMEOUT_S seconds
          via a ThreadPoolExecutor so a corrupt/huge file never hangs the run.
        - Atomic writes: output is written to a .tmp file throughout; only
          os.replace'd to output_path on clean completion.  The checkpoint is
          also written atomically (tmp+replace) so neither file is ever corrupt.
        """
        files = list(directory.glob("*.nxml"))
        total_files = len(files)
        processed_ids = self._load_checkpoint(checkpoint_path) if checkpoint_path else set()

        extracted = 0
        failed = 0
        empty = 0
        skipped = len(processed_ids)

        tmp_path = output_path.with_suffix(".tmp")
        # Resume: append to existing .tmp so prior extracted articles survive.
        # Fresh start: open new .tmp for writing.
        is_resume = bool(processed_ids) and tmp_path.exists()
        mode = "a" if is_resume else "w"

        try:
            # TODO(prod): max_workers=1 — executor used only for timeout, not speed; switch to ProcessPoolExecutor for real parallelism
            # TODO(prod): no .lock file — concurrent runs will both write to the same .tmp and corrupt output
            with (
                open(tmp_path, mode) as f,
                concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
            ):
                for i, filepath in enumerate(files, 1):
                    article_id = filepath.stem
                    if article_id in processed_ids:
                        continue

                    if i % 100 == 0 or i == total_files:
                        logger.info(
                            "extract.progress %d/%d extracted=%d failed=%d skipped=%d",
                            i, total_files, extracted, failed, skipped,
                        )

                    # Strategy 2: enforce per-file timeout
                    future = executor.submit(self._extract_article, filepath)
                    try:
                        article = future.result(timeout=FILE_TIMEOUT_S)
                    except concurrent.futures.TimeoutError:
                        logger.warning("extract.timeout file=%s", filepath.name)
                        failed += 1
                        continue
                    except Exception as e:
                        logger.error("extract.failed file=%s error=%s", filepath.name, e)
                        failed += 1
                        continue

                    if article is None:
                        failed += 1
                        continue

                    if not article.sections:
                        logger.info("extract.empty file=%s", filepath.name)
                        empty += 1
                        continue

                    f.write(json.dumps({
                        "article_id": article.article_id,
                        "title": article.title,
                        "article_type": article.article_type,
                        "sections": article.sections,
                        "source": article.source,
                    }) + "\n")
                    extracted += 1
                    processed_ids.add(article_id)

                    # Strategy 1: checkpoint every 100 newly extracted articles
                    if checkpoint_path and extracted % 100 == 0:
                        self._save_checkpoint(checkpoint_path, processed_ids)

            # Strategy 3: atomic promotion of .tmp → output_path
            if extracted > 0 or is_resume:
                os.replace(tmp_path, output_path)
            else:
                tmp_path.unlink(missing_ok=True)

            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, processed_ids)

        except Exception:
            # TODO(prod): disk-full mid-run deletes .tmp and loses all progress for this run; checkpoint survives but output is gone
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "extract.complete total=%d extracted=%d failed=%d empty=%d skipped=%d",
            total_files, extracted, failed, empty, skipped,
        )
        return {
            "total_files": total_files,
            "extracted": extracted,
            "failed": failed,
            "empty": empty,
            "skipped": skipped,
        }

    def chunk_articles(
        self,
        processed_path: Path,
        output_path: Path,
        config: ChunkConfig | None = None,
    ) -> dict:
        """Pass 2: cleaned text JSONL → chunks JSONL. Re-run anytime to re-chunk.

        Reads the Article JSONL written by extract_articles and applies chunking
        with the given (or default) ChunkConfig.  The output is written to a
        .tmp file and atomically replaced so output_path is never half-written.
        """
        cfg = config or self._cfg
        total_articles = 0
        total_chunks = 0
        failed_articles = 0

        tmp_path = output_path.with_suffix(".tmp")
        try:
            with open(processed_path) as src, open(tmp_path, "w") as dst:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    total_articles += 1
                    article_id = "unknown"
                    try:
                        article_dict = json.loads(line)
                        article_id = article_dict.get("article_id", "unknown")
                        chunks = self._chunk_article(article_dict, cfg)
                        for chunk in chunks:
                            dst.write(json.dumps(chunk) + "\n")
                            total_chunks += 1
                    except Exception as e:
                        logger.error("chunk.article_failed article=%s error=%s", article_id, e)
                        failed_articles += 1

            os.replace(tmp_path, output_path)

        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "chunk.complete articles=%d chunks=%d failed=%d",
            total_articles, total_chunks, failed_articles,
        )
        return {
            "total_articles": total_articles,
            "total_chunks": total_chunks,
            "failed_articles": failed_articles,
        }

    def process_directory(
        self,
        directory: Path,
        output_path: Path,
    ) -> dict:
        warnings.warn(
            "process_directory has no checkpoint, timeout, or resilience. "
            "Use extract_articles() + chunk_articles() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        files = list(directory.glob("*.nxml"))
        total_files = len(files)
        total_chunks = 0
        failed_files = 0

        tmp_path = output_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                for i, filepath in enumerate(files, 1):
                    if i % 100 == 0 or i == total_files:
                        logger.info(
                            "processor.progress %d/%d files failed=%d",
                            i, total_files, failed_files,
                        )
                    chunks = self.process_file(filepath)
                    if not chunks:
                        failed_files += 1
                        continue
                    for chunk in chunks:
                        f.write(json.dumps({
                            "chunk_id": chunk.chunk_id,
                            "article_id": chunk.article_id,
                            "title": chunk.title,
                            "section": chunk.section,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "content": chunk.content,
                            "char_count": chunk.char_count,
                            "article_type": chunk.article_type,
                            "source": chunk.source,
                        }) + "\n")
                        total_chunks += 1

            os.replace(tmp_path, output_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info(
            "processor.complete total_chunks=%d failed=%d",
            total_chunks, failed_files,
        )
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "failed_files": failed_files,
        }