from __future__ import annotations

import json
import logging
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    max_chunk_size: int = 1000
    min_chunk_size: int = 200
    overlap_size: int = 100
    include_title: bool = True

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
    ) -> list[tuple[str, str]]:
        sections = []
        for sec in root.findall(".//sec"):
            sec_title_el = sec.find("title")
            sec_title = self._clean_text(
                sec_title_el.text if sec_title_el is not None and sec_title_el.text
                else "General",
                add_period=False,
            )
            paragraphs = []
            for p in sec.findall(".//p"):
                text = self._get_element_text(p)
                if text:
                    paragraphs.append(text)

            section_text = " ".join(paragraphs)
            if len(section_text) >= self._cfg.min_chunk_size:
                sections.append((sec_title, section_text))

        return sections

    def _clean_text(self, text: str, add_period: bool = True) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\{[^}]+\}", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,;:!?()\-\/]", "", text)
        text = text.strip()
        if add_period and text and text[-1] not in ".!?":
            text += "."
        return text

    def _get_element_text(self, element: ET.Element) -> str:
        text = "".join(element.itertext())
        return self._clean_text(text)

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self._cfg.max_chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._cfg.max_chunk_size

            if end < len(text):
                boundary = text.rfind(".", start, end)
                if boundary > start + self._cfg.min_chunk_size:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if len(chunk) >= self._cfg.min_chunk_size:
                chunks.append(chunk)

            next_start = end - self._cfg.overlap_size
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

    def process_directory(
        self,
        directory: Path,
        output_path: Path,
    ) -> dict:
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
