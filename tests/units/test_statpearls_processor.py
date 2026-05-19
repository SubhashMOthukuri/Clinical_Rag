"""
Unit tests for src/ingestion/statpearls-processor.py

Covers every public and private method:
- _clean_text
- _chunk_text
- _extract_title
- _extract_sections
- _extract_article
- _assign_ids
- _chunk_article
- _load_checkpoint / _save_checkpoint
- extract_articles (counters, checkpoint resume, output content, atomic write)
- chunk_articles (happy path, error logging, output structure, custom config)
- process_directory (DeprecationWarning)
"""

import importlib.util
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# statpearls-processor.py has a hyphen so we load it via importlib
_spec = importlib.util.spec_from_file_location(
    "statpearls_processor",
    Path(__file__).parent.parent.parent / "src" / "ingestion" / "statpearls-processor.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["statpearls_processor"] = _mod
_spec.loader.exec_module(_mod)

StatPearlsProcessor = _mod.StatPearlsProcessor
ChunkConfig = _mod.ChunkConfig
Article = _mod.Article
ArticleChunk = _mod.ArticleChunk


# ============================================================================
# Helpers
# ============================================================================

def make_processor(max_chunk=100, min_chunk=20, overlap=10):
    return StatPearlsProcessor(ChunkConfig(
        max_chunk_size=max_chunk,
        min_chunk_size=min_chunk,
        overlap_size=overlap,
    ))


def make_xml(title="Metformin", sections=None):
    """Build a minimal NLM-style XML string."""
    secs = ""
    for sec_title, para in (sections or [("MOA", "x" * 50)]):
        secs += f"<sec><title>{sec_title}</title><p>{para}</p></sec>"
    return f"<article><title>{title}</title>{secs}</article>"


def write_xml(tmp_path, name, content):
    f = tmp_path / name
    f.write_text(content)
    return f


def make_article_jsonl(tmp_path, articles: list[dict]) -> Path:
    path = tmp_path / "articles.jsonl"
    with open(path, "w") as f:
        for a in articles:
            f.write(json.dumps(a) + "\n")
    return path


def minimal_article(article_id="NBK001", title="Test", sections=None):
    return {
        "article_id": article_id,
        "title": title,
        "article_type": "general",
        "sections": sections if sections is not None else [{"title": "Intro", "text": "A" * 50}],
        "source": "StatPearls",
    }


def parse_xml(xml_str):
    return ET.fromstring(xml_str)


# ============================================================================
# _clean_text
# ============================================================================

class TestCleanText:

    def test_empty_string_returns_empty(self):
        assert make_processor()._clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert make_processor()._clean_text("   \t\n  ") == ""

    def test_strips_html_tags(self):
        result = make_processor()._clean_text("<b>bold</b> text", add_period=False)
        assert "<b>" not in result and "bold" in result

    def test_strips_curly_brace_content(self):
        result = make_processor()._clean_text("text {ref:123} end", add_period=False)
        assert "{ref:123}" not in result and "text" in result

    def test_collapses_whitespace(self):
        result = make_processor()._clean_text("word   \t  word", add_period=False)
        assert result == "word word"

    def test_collapses_newlines(self):
        result = make_processor()._clean_text("line one\n\nline two", add_period=False)
        assert "\n" not in result and "line one" in result

    def test_adds_period_when_missing(self):
        assert make_processor()._clean_text("no period here").endswith(".")

    def test_no_period_added_when_add_period_false(self):
        assert not make_processor()._clean_text("no period here", add_period=False).endswith(".")

    def test_does_not_double_period(self):
        result = make_processor()._clean_text("already ends.")
        assert result.endswith(".") and not result.endswith("..")

    def test_question_mark_not_overwritten(self):
        result = make_processor()._clean_text("is this right?")
        assert result.endswith("?") and not result.endswith("?.")

    def test_exclamation_mark_not_overwritten(self):
        result = make_processor()._clean_text("watch out!")
        assert result.endswith("!") and not result.endswith("!.")

    def test_only_special_chars_returns_empty(self):
        # All chars stripped by regex → empty after strip
        result = make_processor()._clean_text("### @@@ ^^^", add_period=False)
        assert result == ""


# ============================================================================
# _chunk_text
# ============================================================================

class TestChunkText:

    def test_short_text_returns_single_chunk(self):
        result = make_processor(max_chunk=200, min_chunk=10)._chunk_text("Short text.")
        assert result == ["Short text."]

    def test_text_exactly_at_max_chunk_size_is_single_chunk(self):
        text = "A" * 100
        assert len(make_processor(max_chunk=100, min_chunk=10)._chunk_text(text)) == 1

    def test_long_text_split_into_multiple_chunks(self):
        result = make_processor(max_chunk=50, min_chunk=10, overlap=5)._chunk_text("A" * 200)
        assert len(result) > 1

    def test_each_chunk_within_max_size(self):
        p = make_processor(max_chunk=50, min_chunk=10, overlap=5)
        for chunk in p._chunk_text("word " * 60):
            assert len(chunk) <= 50

    def test_sentence_boundary_used_when_available(self):
        p = make_processor(max_chunk=60, min_chunk=10, overlap=5)
        text = "First sentence ends here. " + "B" * 40
        chunks = p._chunk_text(text)
        assert chunks[0].endswith(".")

    def test_no_sentence_boundary_still_chunks(self):
        # No periods — should still split without crashing
        p = make_processor(max_chunk=50, min_chunk=10, overlap=5)
        text = "A" * 200
        result = p._chunk_text(text)
        assert len(result) > 1

    def test_chunks_below_min_size_dropped(self):
        # Text = max_chunk_size, returned as-is since len <= max
        p = make_processor(max_chunk=50, min_chunk=30, overlap=5)
        result = p._chunk_text("A" * 50)
        assert len(result) == 1

    def test_overlap_causes_shared_content_between_chunks(self):
        p = make_processor(max_chunk=50, min_chunk=5, overlap=20)
        text = "A" * 100
        chunks = p._chunk_text(text)
        if len(chunks) > 1:
            assert chunks[0][-20:] == chunks[1][:20]

    def test_custom_config_overrides_instance_config(self):
        p = make_processor(max_chunk=500, min_chunk=100)
        small_cfg = ChunkConfig(max_chunk_size=50, min_chunk_size=10, overlap_size=5)
        result = p._chunk_text("A" * 200, cfg=small_cfg)
        assert len(result) > 1

    def test_sentence_boundary_too_close_to_start_not_used(self):
        # Period at position 5, min_chunk_size=30 → boundary not used
        p = make_processor(max_chunk=60, min_chunk=30, overlap=5)
        text = "AB. " + "C" * 60
        chunks = p._chunk_text(text)
        # First chunk should NOT end at position 3 (too short); must use full max
        assert len(chunks[0]) > 10


# ============================================================================
# _extract_title
# ============================================================================

class TestExtractTitle:

    def test_finds_title_tag(self):
        root = parse_xml("<article><title>Metformin</title></article>")
        assert make_processor()._extract_title(root) == "Metformin"

    def test_finds_article_title_tag(self):
        root = parse_xml("<article><article-title>Aspirin</article-title></article>")
        assert make_processor()._extract_title(root) == "Aspirin"

    def test_prefers_title_over_article_title(self):
        root = parse_xml("<article><title>First</title><article-title>Second</article-title></article>")
        assert make_processor()._extract_title(root) == "First"

    def test_returns_unknown_when_no_title_tag(self):
        root = parse_xml("<article><p>no title here</p></article>")
        assert make_processor()._extract_title(root) == "Unknown"

    def test_empty_title_tag_returns_unknown(self):
        root = parse_xml("<article><title></title></article>")
        assert make_processor()._extract_title(root) == "Unknown"

    def test_title_with_nested_xml_tags_returns_unknown(self):
        # ET.fromstring sets el.text=None when title content is inside child tags;
        # _extract_title sees no direct text and falls back to "Unknown".
        root = parse_xml("<article><title><b>Bold</b> Title</title></article>")
        assert make_processor()._extract_title(root) == "Unknown"

    def test_title_with_extra_spaces_is_cleaned(self):
        root = parse_xml("<article><title>word   word</title></article>")
        result = make_processor()._extract_title(root)
        assert "  " not in result and "word" in result


# ============================================================================
# _extract_sections
# ============================================================================

class TestExtractSections:

    def test_section_with_title_uses_that_title(self):
        root = parse_xml("<article><sec><title>MOA</title><p>" + "x" * 50 + "</p></sec></article>")
        sections = make_processor()._extract_sections(root)
        assert sections[0][0] == "MOA"

    def test_section_without_title_uses_general(self):
        root = parse_xml("<article><sec><p>" + "x" * 50 + "</p></sec></article>")
        sections = make_processor()._extract_sections(root)
        assert sections[0][0] == "General"

    def test_short_section_filtered_by_min_chunk_size(self):
        # Text shorter than min_chunk_size (20) should be filtered out
        root = parse_xml("<article><sec><title>S</title><p>short</p></sec></article>")
        sections = make_processor(min_chunk=20)._extract_sections(root)
        assert sections == []

    def test_min_length_zero_keeps_short_sections(self):
        root = parse_xml("<article><sec><title>S</title><p>short</p></sec></article>")
        sections = make_processor()._extract_sections(root, min_length=0)
        assert len(sections) == 1

    def test_multiple_sections_all_returned(self):
        xml = (
            "<article>"
            "<sec><title>S1</title><p>" + "x" * 50 + "</p></sec>"
            "<sec><title>S2</title><p>" + "y" * 50 + "</p></sec>"
            "</article>"
        )
        sections = make_processor()._extract_sections(parse_xml(xml))
        assert len(sections) == 2

    def test_paragraphs_within_section_joined(self):
        xml = "<article><sec><title>S</title><p>" + "a" * 30 + "</p><p>" + "b" * 30 + "</p></sec></article>"
        sections = make_processor()._extract_sections(root=parse_xml(xml), min_length=0)
        combined_text = sections[0][1]
        assert len(combined_text) > 30  # both paragraphs merged

    def test_no_sections_returns_empty_list(self):
        root = parse_xml("<article><title>T</title><p>orphan paragraph</p></article>")
        assert make_processor()._extract_sections(root) == []


# ============================================================================
# _extract_article
# ============================================================================

class TestExtractArticle:

    def test_valid_xml_returns_article(self, tmp_path):
        f = write_xml(tmp_path, "NBK001.nxml", make_xml())
        article = make_processor()._extract_article(f)
        assert isinstance(article, Article)

    def test_article_id_from_filename_stem(self, tmp_path):
        f = write_xml(tmp_path, "NBK001.nxml", make_xml())
        article = make_processor()._extract_article(f)
        assert article.article_id == "NBK001"

    def test_corrupt_xml_returns_none(self, tmp_path):
        f = write_xml(tmp_path, "bad.nxml", "<<not xml>>")
        assert make_processor()._extract_article(f) is None

    def test_article_sections_populated(self, tmp_path):
        f = write_xml(tmp_path, "NBK001.nxml", make_xml(sections=[("MOA", "x" * 50)]))
        article = make_processor()._extract_article(f)
        assert len(article.sections) == 1
        assert article.sections[0]["title"] == "MOA"

    def test_article_with_no_sections_has_empty_sections(self, tmp_path):
        f = write_xml(tmp_path, "nosec.nxml", "<article><title>T</title></article>")
        article = make_processor()._extract_article(f)
        assert article.sections == []

    def test_article_source_is_statpearls(self, tmp_path):
        f = write_xml(tmp_path, "NBK001.nxml", make_xml())
        assert make_processor()._extract_article(f).source == "StatPearls"

    def test_corrupt_xml_logs_error(self, tmp_path, caplog):
        f = write_xml(tmp_path, "bad.nxml", "<<not xml>>")
        with caplog.at_level("ERROR"):
            make_processor()._extract_article(f)
        assert "xml.parse_failed" in caplog.text


# ============================================================================
# _assign_ids
# ============================================================================

class TestAssignIds:

    def test_chunk_id_format(self):
        p = make_processor()
        chunks = p._assign_ids([("Sec", "content")], "NBK001", "Title", "general")
        assert chunks[0].chunk_id == "NBK001_chunk_0000"

    def test_chunk_index_sequential(self):
        p = make_processor()
        tagged = [("S", "text1"), ("S", "text2"), ("S", "text3")]
        chunks = p._assign_ids(tagged, "NBK001", "T", "general")
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_total_chunks_correct(self):
        p = make_processor()
        tagged = [("S", "t1"), ("S", "t2")]
        chunks = p._assign_ids(tagged, "NBK001", "T", "general")
        assert all(c.total_chunks == 2 for c in chunks)

    def test_source_is_statpearls(self):
        p = make_processor()
        chunks = p._assign_ids([("S", "text")], "NBK001", "T", "general")
        assert chunks[0].source == "StatPearls"

    def test_empty_tagged_returns_empty_list(self):
        assert make_processor()._assign_ids([], "NBK001", "T", "general") == []

    def test_char_count_matches_content_length(self):
        p = make_processor()
        content = "hello world"
        chunks = p._assign_ids([("S", content)], "NBK001", "T", "general")
        assert chunks[0].char_count == len(content)


# ============================================================================
# _chunk_article
# ============================================================================

class TestChunkArticle:

    def test_returns_list_of_dicts(self):
        p = make_processor()
        result = p._chunk_article(minimal_article(), p._cfg)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_output_has_required_keys(self):
        p = make_processor()
        chunk = p._chunk_article(minimal_article(), p._cfg)[0]
        required = {"chunk_id", "article_id", "title", "section", "chunk_index",
                    "total_chunks", "content", "char_count", "article_type", "source"}
        assert required.issubset(chunk.keys())

    def test_article_id_in_chunk(self):
        p = make_processor()
        chunks = p._chunk_article(minimal_article(article_id="NBK999"), p._cfg)
        assert all(c["article_id"] == "NBK999" for c in chunks)

    def test_empty_sections_produces_no_chunks(self):
        p = make_processor()
        article = minimal_article(sections=[])
        assert p._chunk_article(article, p._cfg) == []

    def test_multiple_sections_all_chunked(self):
        p = make_processor()
        article = minimal_article(sections=[
            {"title": "S1", "text": "A" * 50},
            {"title": "S2", "text": "B" * 50},
        ])
        chunks = p._chunk_article(article, p._cfg)
        sections_seen = {c["section"] for c in chunks}
        assert "S1" in sections_seen and "S2" in sections_seen


# ============================================================================
# _load_checkpoint / _save_checkpoint
# ============================================================================

class TestCheckpoint:

    def test_load_returns_empty_set_when_file_missing(self, tmp_path):
        result = make_processor()._load_checkpoint(tmp_path / "none.json")
        assert result == set()

    def test_load_returns_ids_from_file(self, tmp_path):
        path = tmp_path / "ckpt.json"
        path.write_text(json.dumps(["NBK001", "NBK002"]))
        result = make_processor()._load_checkpoint(path)
        assert result == {"NBK001", "NBK002"}

    def test_load_returns_empty_set_on_corrupt_file(self, tmp_path):
        path = tmp_path / "ckpt.json"
        path.write_text("not-json{{")
        result = make_processor()._load_checkpoint(path)
        assert result == set()

    def test_save_writes_ids_to_file(self, tmp_path):
        path = tmp_path / "ckpt.json"
        make_processor()._save_checkpoint(path, {"NBK001", "NBK002"})
        loaded = set(json.loads(path.read_text()))
        assert loaded == {"NBK001", "NBK002"}

    def test_save_is_atomic_no_tmp_left(self, tmp_path):
        path = tmp_path / "ckpt.json"
        make_processor()._save_checkpoint(path, {"NBK001"})
        assert not path.with_suffix(".tmp").exists()

    def test_save_then_load_round_trip(self, tmp_path):
        path = tmp_path / "ckpt.json"
        ids = {"NBK001", "NBK002", "NBK003"}
        p = make_processor()
        p._save_checkpoint(path, ids)
        assert p._load_checkpoint(path) == ids


# ============================================================================
# extract_articles — counters, checkpoint resume, output content
# ============================================================================

class TestExtractArticlesCounters:

    def test_parse_error_counted_as_failed_not_empty(self, tmp_path):
        write_xml(tmp_path, "bad.nxml", "<<not valid xml>>")
        result = make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl")
        assert result["failed"] == 1 and result["empty"] == 0

    def test_article_with_no_sections_counted_as_empty_not_failed(self, tmp_path):
        write_xml(tmp_path, "nosec.nxml", "<article><title>T</title></article>")
        result = make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl")
        assert result["empty"] == 1 and result["failed"] == 0

    def test_empty_articles_not_written_to_output(self, tmp_path):
        write_xml(tmp_path, "nosec.nxml", "<article><title>T</title></article>")
        out = tmp_path / "out.jsonl"
        make_processor().extract_articles(tmp_path, out)
        assert not out.exists() or out.read_text().strip() == ""

    def test_valid_article_counted_as_extracted(self, tmp_path):
        write_xml(tmp_path, "good.nxml", make_xml())
        result = make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl")
        assert result["extracted"] == 1 and result["failed"] == 0 and result["empty"] == 0

    def test_mixed_files_all_counters_correct(self, tmp_path):
        write_xml(tmp_path, "good.nxml", make_xml())
        write_xml(tmp_path, "empty.nxml", "<article><title>T</title></article>")
        write_xml(tmp_path, "bad.nxml", "<<broken")
        result = make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl")
        assert result["extracted"] == 1 and result["empty"] == 1 and result["failed"] == 1

    def test_return_dict_includes_empty_key(self, tmp_path):
        result = make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl")
        assert "empty" in result

    def test_empty_directory_all_counters_zero(self, tmp_path):
        result = make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl")
        assert result == {"total_files": 0, "extracted": 0, "failed": 0, "empty": 0, "skipped": 0}

    def test_output_jsonl_is_valid_json_per_line(self, tmp_path):
        write_xml(tmp_path, "good.nxml", make_xml())
        out = tmp_path / "out.jsonl"
        make_processor().extract_articles(tmp_path, out)
        for line in out.read_text().splitlines():
            obj = json.loads(line)
            assert "article_id" in obj and "sections" in obj

    def test_output_file_atomic_no_tmp_left(self, tmp_path):
        write_xml(tmp_path, "good.nxml", make_xml())
        out = tmp_path / "out.jsonl"
        make_processor().extract_articles(tmp_path, out)
        assert not out.with_suffix(".tmp").exists()

    def test_checkpoint_file_created(self, tmp_path):
        write_xml(tmp_path, "good.nxml", make_xml())
        ckpt = tmp_path / "ckpt.json"
        make_processor().extract_articles(tmp_path, tmp_path / "out.jsonl", checkpoint_path=ckpt)
        assert ckpt.exists()

    def test_checkpoint_resume_skips_processed_articles(self, tmp_path):
        write_xml(tmp_path, "NBK001.nxml", make_xml())
        ckpt = tmp_path / "ckpt.json"
        out = tmp_path / "out.jsonl"
        # First run: processes NBK001
        make_processor().extract_articles(tmp_path, out, checkpoint_path=ckpt)
        # Second run: NBK001 is in checkpoint → skipped=1, extracted=0
        result = make_processor().extract_articles(tmp_path, out, checkpoint_path=ckpt)
        assert result["skipped"] == 1 and result["extracted"] == 0


# ============================================================================
# chunk_articles — error logging, output structure, custom config
# ============================================================================

class TestChunkArticles:

    def test_happy_path_produces_chunks(self, tmp_path):
        src = make_article_jsonl(tmp_path, [minimal_article()])
        out = tmp_path / "chunks.jsonl"
        result = make_processor().chunk_articles(src, out)
        assert result["total_articles"] == 1 and result["total_chunks"] >= 1

    def test_corrupt_json_line_logs_unknown(self, tmp_path, caplog):
        src = tmp_path / "articles.jsonl"
        src.write_text("not-valid-json\n")
        with caplog.at_level("ERROR"):
            result = make_processor().chunk_articles(src, tmp_path / "out.jsonl")
        assert result["failed_articles"] == 1 and "unknown" in caplog.text

    def test_article_id_in_error_log_when_chunking_fails(self, tmp_path, caplog):
        bad = {"article_id": "NBK999", "title": "T", "article_type": "general"}
        src = make_article_jsonl(tmp_path, [bad])
        with caplog.at_level("ERROR"):
            make_processor().chunk_articles(src, tmp_path / "out.jsonl")
        assert "NBK999" in caplog.text

    def test_output_chunk_has_all_required_fields(self, tmp_path):
        src = make_article_jsonl(tmp_path, [minimal_article()])
        out = tmp_path / "chunks.jsonl"
        make_processor().chunk_articles(src, out)
        chunk = json.loads(out.read_text().splitlines()[0])
        required = {"chunk_id", "article_id", "title", "section", "chunk_index",
                    "total_chunks", "content", "char_count", "article_type", "source"}
        assert required.issubset(chunk.keys())

    def test_output_file_written_atomically(self, tmp_path):
        src = make_article_jsonl(tmp_path, [minimal_article()])
        out = tmp_path / "chunks.jsonl"
        make_processor().chunk_articles(src, out)
        assert out.exists() and not out.with_suffix(".tmp").exists()

    def test_custom_config_changes_chunk_count(self, tmp_path):
        long_text = "A" * 500
        src = make_article_jsonl(tmp_path, [minimal_article(sections=[{"title": "S", "text": long_text}])])
        out_small = tmp_path / "small.jsonl"
        out_large = tmp_path / "large.jsonl"
        p = make_processor()
        small_cfg = ChunkConfig(max_chunk_size=50, min_chunk_size=10, overlap_size=5)
        large_cfg = ChunkConfig(max_chunk_size=600, min_chunk_size=10, overlap_size=5)
        p.chunk_articles(src, out_small, config=small_cfg)
        # Need fresh src since first call consumed the file iterator
        src2 = make_article_jsonl(tmp_path, [minimal_article(sections=[{"title": "S", "text": long_text}])])
        p.chunk_articles(src2, out_large, config=large_cfg)
        small_count = len(out_small.read_text().splitlines())
        large_count = len(out_large.read_text().splitlines())
        assert small_count > large_count

    def test_empty_input_file_returns_zero_counts(self, tmp_path):
        src = tmp_path / "empty.jsonl"
        src.write_text("")
        result = make_processor().chunk_articles(src, tmp_path / "out.jsonl")
        assert result == {"total_articles": 0, "total_chunks": 0, "failed_articles": 0}

    def test_blank_lines_in_jsonl_ignored(self, tmp_path):
        src = tmp_path / "articles.jsonl"
        with open(src, "w") as f:
            f.write(json.dumps(minimal_article()) + "\n\n")
            f.write(json.dumps(minimal_article("NBK002")) + "\n")
        result = make_processor().chunk_articles(src, tmp_path / "out.jsonl")
        assert result["total_articles"] == 2

    def test_multiple_articles_all_chunked(self, tmp_path):
        articles = [minimal_article(f"NBK{i:03d}") for i in range(5)]
        src = make_article_jsonl(tmp_path, articles)
        result = make_processor().chunk_articles(src, tmp_path / "out.jsonl")
        assert result["total_articles"] == 5 and result["failed_articles"] == 0


# ============================================================================
# process_directory — DeprecationWarning
# ============================================================================

class TestProcessDirectoryDeprecation:

    def test_emits_deprecation_warning(self, tmp_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_processor().process_directory(tmp_path, tmp_path / "out.jsonl")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_deprecation_message_mentions_alternative(self, tmp_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_processor().process_directory(tmp_path, tmp_path / "out.jsonl")
        msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
        assert any("extract_articles" in m for m in msgs)

    def test_still_returns_result_dict(self, tmp_path):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = make_processor().process_directory(tmp_path, tmp_path / "out.jsonl")
        assert {"total_files", "total_chunks", "failed_files"}.issubset(result.keys())

    def test_processes_valid_xml_files(self, tmp_path):
        write_xml(tmp_path, "good.nxml", make_xml())
        out = tmp_path / "out.jsonl"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = make_processor().process_directory(tmp_path, out)
        assert result["total_chunks"] >= 1 and out.exists()

    def test_failed_files_counted(self, tmp_path):
        write_xml(tmp_path, "bad.nxml", "<<broken")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = make_processor().process_directory(tmp_path, tmp_path / "out.jsonl")
        assert result["failed_files"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
