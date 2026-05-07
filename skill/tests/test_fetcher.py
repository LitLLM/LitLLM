"""Unit tests for the fetcher module — no network calls."""

from __future__ import annotations

import pytest

from litllm import PaperFetcher
from litllm.fetcher import _dedupe_papers, _safe_filename, sanitize_title


def test_init_creates_cache_dir(tmp_path):
    cache = tmp_path / "cache"
    fetcher = PaperFetcher(cache_dir=str(cache))
    assert (cache / "pdfs").is_dir()
    assert fetcher.pdf_cache_path == str(cache / "pdfs")


def test_search_papers_rejects_unknown_api(tmp_path):
    fetcher = PaperFetcher(cache_dir=str(tmp_path / "c"))
    with pytest.raises(ValueError, match="Unsupported search API"):
        fetcher.search_papers(["foo"], api="serper")


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Hello, World!", "Hello World"),
        ('"Quoted" (parens)', "Quoted parens"),
        (None, None),
        ("", None),
    ],
)
def test_sanitize_title(title, expected):
    assert sanitize_title(title) == expected


@pytest.mark.parametrize(
    "pid,expected",
    [
        ("2403.05530", "2403.05530"),
        ("hello/world", "helloworld"),
        ("a b!c@d#e", "abcde"),
        ("", None),
        (None, None),
    ],
)
def test_safe_filename(pid, expected):
    assert _safe_filename(pid) == expected


def test_dedupe_versioned_arxiv_ids():
    papers = [
        {"arxiv_id": "1706.03762v1", "title": "Attention v1"},
        {"arxiv_id": "1706.03762v2", "title": "Attention v2"},
        {"arxiv_id": "2403.05530", "title": "Other"},
    ]
    result = _dedupe_papers(papers)
    assert len(result) == 2
    assert {p["title"] for p in result} == {"Attention v1", "Other"}


def test_dedupe_falls_back_through_id_priority():
    papers = [
        {"arxiv_id": None, "doi": "10.1/x", "title": "By DOI"},
        {"arxiv_id": None, "doi": "10.1/x", "title": "Also DOI x"},
        {"arxiv_id": None, "doi": None, "openalex_id": "W123", "title": "By OA"},
        {"arxiv_id": None, "doi": None, "openalex_id": None, "title": "Just Title"},
        {"arxiv_id": None, "doi": None, "openalex_id": None, "title": "JUST title"},
    ]
    result = _dedupe_papers(papers)
    titles = [p["title"] for p in result]
    assert "By DOI" in titles
    assert "Also DOI x" not in titles
    assert "By OA" in titles
    # Title dedup is case-insensitive — only the first wins
    assert sum(1 for t in titles if t.lower() == "just title") == 1
