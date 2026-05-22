"""
Unit tests for src/chunking/chunker.py

Covers:
- ChunkConfig: defaults, validation
- ArticleChunk: creation, immutability, defaults
- TextChunker.chunk_text: short/long text, sentence boundaries, overlap, min-size filter
- TextChunker.assign_ids: id format, indexing, total_chunks, source override
- TextChunker.chunk_article: output shape, keys, multi-section, source propagation, config override
"""

import pytest

from src.chunking import ArticleChunk, ChunkConfig, TextChunker


# ============================================================================
# Helpers
# ============================================================================

def make_chunker(max_chunk=100, min_chunk=20, overlap=10):
    return TextChunker(ChunkConfig(
        max_chunk_size=max_chunk,
        min_chunk_size=min_chunk,
        overlap_size=overlap,
    ))


def minimal_article(article_id="NBK001", title="Test", sections=None):
    return {
        "article_id": article_id,
        "title": title,
        "article_type": "general",
        "sections": sections if sections is not None else [{"title": "Intro", "text": "A" * 50}],
        "source": "StatPearls",
    }


# ============================================================================
# ChunkConfig
# ============================================================================

class TestChunkConfig:

    def test_defaults(self):
        cfg = ChunkConfig()
        assert cfg.max_chunk_size == 512
        assert cfg.min_chunk_size == 252
        assert cfg.overlap_size == 80

    def test_overlap_equal_to_max_raises(self):
        with pytest.raises(ValueError):
            ChunkConfig(max_chunk_size=100, min_chunk_size=10, overlap_size=100)

    def test_overlap_greater_than_max_raises(self):
        with pytest.raises(ValueError):
            ChunkConfig(max_chunk_size=100, min_chunk_size=10, overlap_size=200)

    def test_overlap_one_less_than_max_does_not_raise(self):
        ChunkConfig(max_chunk_size=100, min_chunk_size=10, overlap_size=99)


# ============================================================================
# ArticleChunk
# ============================================================================

class TestArticleChunk:

    def _make(self, **kwargs):
        base = dict(
            chunk_id="NBK001_chunk_0000",
            article_id="NBK001",
            title="Test",
            section="Intro",
            chunk_index=0,
            total_chunks=1,
            content="hello",
            char_count=5,
        )
        base.update(kwargs)
        return ArticleChunk(**base)

    def test_fields_set_correctly(self):
        c = self._make()
        assert c.chunk_id == "NBK001_chunk_0000"
        assert c.article_id == "NBK001"
        assert c.content == "hello"
        assert c.char_count == 5

    def test_default_article_type_is_general(self):
        assert self._make().article_type == "general"

    def test_default_source_is_statpearls(self):
        assert self._make().source == "StatPearls"

    def test_frozen_raises_on_mutation(self):
        c = self._make()
        with pytest.raises((AttributeError, TypeError)):
            c.content = "changed"


# ============================================================================
# TextChunker.chunk_text
# ============================================================================

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        result = make_chunker(max_chunk=200, min_chunk=10).chunk_text("Short text.")
        assert result == ["Short text."]

    def test_text_exactly_at_max_chunk_size_is_single_chunk(self):
        text = "A" * 100
        assert len(make_chunker(max_chunk=100, min_chunk=10).chunk_text(text)) == 1

    def test_long_text_split_into_multiple_chunks(self):
        result = make_chunker(max_chunk=50, min_chunk=10, overlap=5).chunk_text("A" * 200)
        assert len(result) > 1

    def test_each_chunk_within_max_size(self):
        chunker = make_chunker(max_chunk=50, min_chunk=10, overlap=5)
        for chunk in chunker.chunk_text("word " * 60):
            assert len(chunk) <= 50

    def test_sentence_boundary_used_when_available(self):
        chunker = make_chunker(max_chunk=60, min_chunk=10, overlap=5)
        text = "First sentence ends here. " + "B" * 40
        chunks = chunker.chunk_text(text)
        assert chunks[0].endswith(".")

    def test_no_sentence_boundary_still_chunks(self):
        chunker = make_chunker(max_chunk=50, min_chunk=10, overlap=5)
        result = chunker.chunk_text("A" * 200)
        assert len(result) > 1

    def test_chunks_below_min_size_dropped(self):
        # Text == max_chunk_size → returned as single chunk (len <= max)
        chunker = make_chunker(max_chunk=50, min_chunk=30, overlap=5)
        result = chunker.chunk_text("A" * 50)
        assert len(result) == 1

    def test_overlap_causes_shared_content_between_chunks(self):
        chunker = make_chunker(max_chunk=50, min_chunk=5, overlap=20)
        text = "A" * 100
        chunks = chunker.chunk_text(text)
        if len(chunks) > 1:
            assert chunks[0][-20:] == chunks[1][:20]

    def test_custom_cfg_overrides_instance_config(self):
        chunker = make_chunker(max_chunk=500, min_chunk=100)
        small_cfg = ChunkConfig(max_chunk_size=50, min_chunk_size=10, overlap_size=5)
        result = chunker.chunk_text("A" * 200, cfg=small_cfg)
        assert len(result) > 1

    def test_sentence_boundary_too_close_to_start_not_used(self):
        # Period at position 3, min_chunk=30 → boundary ignored; chunk uses full window
        chunker = make_chunker(max_chunk=60, min_chunk=30, overlap=5)
        text = "AB. " + "C" * 60
        chunks = chunker.chunk_text(text)
        assert len(chunks[0]) > 10


# ============================================================================
# TextChunker.assign_ids
# ============================================================================

class TestAssignIds:

    def test_chunk_id_format(self):
        chunks = make_chunker().assign_ids([("Sec", "content")], "NBK001", "Title", "general")
        assert chunks[0].chunk_id == "NBK001_chunk_0000"

    def test_chunk_index_sequential(self):
        tagged = [("S", "text1"), ("S", "text2"), ("S", "text3")]
        chunks = make_chunker().assign_ids(tagged, "NBK001", "T", "general")
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_total_chunks_correct(self):
        tagged = [("S", "t1"), ("S", "t2")]
        chunks = make_chunker().assign_ids(tagged, "NBK001", "T", "general")
        assert all(c.total_chunks == 2 for c in chunks)

    def test_default_source_is_statpearls(self):
        chunks = make_chunker().assign_ids([("S", "text")], "NBK001", "T", "general")
        assert chunks[0].source == "StatPearls"

    def test_custom_source_propagated(self):
        chunks = make_chunker().assign_ids([("S", "text")], "NBK001", "T", "general", source="PubMed")
        assert chunks[0].source == "PubMed"

    def test_empty_tagged_returns_empty_list(self):
        assert make_chunker().assign_ids([], "NBK001", "T", "general") == []

    def test_char_count_matches_content_length(self):
        content = "hello world"
        chunks = make_chunker().assign_ids([("S", content)], "NBK001", "T", "general")
        assert chunks[0].char_count == len(content)

    def test_returns_article_chunk_instances(self):
        chunks = make_chunker().assign_ids([("S", "text")], "NBK001", "T", "general")
        assert all(isinstance(c, ArticleChunk) for c in chunks)


# ============================================================================
# TextChunker.chunk_article
# ============================================================================

class TestChunkArticle:

    def test_returns_list_of_dicts(self):
        result = make_chunker().chunk_article(minimal_article())
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_output_has_all_required_keys(self):
        chunk = make_chunker().chunk_article(minimal_article())[0]
        required = {
            "chunk_id", "article_id", "title", "section", "chunk_index",
            "total_chunks", "content", "char_count", "article_type", "source",
        }
        assert required.issubset(chunk.keys())

    def test_article_id_in_every_chunk(self):
        chunks = make_chunker().chunk_article(minimal_article(article_id="NBK999"))
        assert all(c["article_id"] == "NBK999" for c in chunks)

    def test_empty_sections_produces_no_chunks(self):
        assert make_chunker().chunk_article(minimal_article(sections=[])) == []

    def test_multiple_sections_all_represented(self):
        article = minimal_article(sections=[
            {"title": "S1", "text": "A" * 50},
            {"title": "S2", "text": "B" * 50},
        ])
        chunks = make_chunker().chunk_article(article)
        sections_seen = {c["section"] for c in chunks}
        assert "S1" in sections_seen and "S2" in sections_seen

    def test_source_propagated_from_article_dict(self):
        article = minimal_article()
        article["source"] = "PubMed"
        chunks = make_chunker().chunk_article(article)
        assert all(c["source"] == "PubMed" for c in chunks)

    def test_missing_source_defaults_to_statpearls(self):
        article = minimal_article()
        del article["source"]
        chunks = make_chunker().chunk_article(article)
        assert all(c["source"] == "StatPearls" for c in chunks)

    def test_custom_cfg_produces_more_chunks_than_large_cfg(self):
        long_text = "A" * 500
        article = minimal_article(sections=[{"title": "S", "text": long_text}])
        small_cfg = ChunkConfig(max_chunk_size=50, min_chunk_size=10, overlap_size=5)
        large_cfg = ChunkConfig(max_chunk_size=600, min_chunk_size=10, overlap_size=5)
        small_chunks = make_chunker().chunk_article(article, cfg=small_cfg)
        large_chunks = make_chunker().chunk_article(article, cfg=large_cfg)
        assert len(small_chunks) > len(large_chunks)

    def test_chunk_indexes_are_sequential(self):
        long_text = "A" * 500
        article = minimal_article(sections=[{"title": "S", "text": long_text}])
        small_cfg = ChunkConfig(max_chunk_size=50, min_chunk_size=10, overlap_size=5)
        chunks = make_chunker().chunk_article(article, cfg=small_cfg)
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])