from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    max_chunk_size: int = 512
    min_chunk_size: int = 252
    overlap_size: int = 80
    include_title: bool = True  # TODO(prod): wire into _assign_ids prefix logic or remove

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


class TextChunker:
    """Splits cleaned article text into overlapping chunks with metadata."""

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._cfg = config or ChunkConfig()

    def chunk_text(self, text: str, cfg: ChunkConfig | None = None) -> list[str]:
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

    def assign_ids(
        self,
        tagged_chunks: list[tuple[str, str]],
        article_id: str,
        title: str,
        article_type: str,
        source: str = "StatPearls",
    ) -> list[ArticleChunk]:
        total = len(tagged_chunks)
        return [
            ArticleChunk(
                chunk_id=f"{article_id}_chunk_{idx:04d}",
                article_id=article_id,
                title=title,
                section=section,
                chunk_index=idx,
                total_chunks=total,
                content=content,
                char_count=len(content),
                article_type=article_type,
                source=source,
            )
            for idx, (section, content) in enumerate(tagged_chunks)
        ]

    def chunk_article(self, article_dict: dict, cfg: ChunkConfig | None = None) -> list[dict]:
        """Turn one Article dict (from Pass 1 JSONL) into a list of chunk dicts."""
        cfg = cfg or self._cfg
        article_id = article_dict["article_id"]
        title = article_dict["title"]
        article_type = article_dict["article_type"]
        source = article_dict.get("source", "StatPearls")

        tagged: list[tuple[str, str]] = []
        for section in article_dict["sections"]:
            for chunk_text in self.chunk_text(section["text"], cfg):
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
            for c in self.assign_ids(tagged, article_id, title, article_type, source)
        ]
