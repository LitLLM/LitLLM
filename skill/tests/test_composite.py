"""Composite pipeline tests — mocked client + fetcher; no network."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litllm.composite import (
    CompositeLitLLM,
    _build_reference_block,
    _normalize_title,
    _paper_key,
    _papers_to_bibtex,
    _parse_ranking_scores,
    _safify,
)


# ---- pure-function tests --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello world", "hello_world"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("a/b\\c.d", "a_b_c.d"),
        ("3.5_bibfile", "3.5_bibfile"),
    ],
)
def test_safify(raw, expected):
    assert _safify(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("Attention\n\nIs   All", "attention is all"), ("", ""), (None, "")],
)
def test_normalize_title(raw, expected):
    assert _normalize_title(raw) == expected


def test_paper_key_priority():
    assert _paper_key({"arxiv_id": "1", "doi": "x", "openalex_id": "y"}) == "1"
    assert _paper_key({"doi": "x", "openalex_id": "y"}) == "x"
    assert _paper_key({"openalex_id": "y"}) == "y"
    assert _paper_key({}) is None


def test_parse_ranking_scores_explicit():
    text = """
    <probability>
    paper_id: 1234.5678
    score: 73/100
    </probability>
    """
    assert _parse_ranking_scores(text) == {"1234.5678": 73}


def test_parse_ranking_scores_inline():
    text = "<probability>[1234.5678]: 42</probability>"
    assert _parse_ranking_scores(text) == {"1234.5678": 42}


def test_parse_ranking_scores_keeps_max_per_id():
    text = """
    <probability>paper_id: 1234.5678 score: 30/100</probability>
    <probability>paper_id: 1234.5678 score: 80/100</probability>
    """
    assert _parse_ranking_scores(text)["1234.5678"] == 80


def test_papers_to_bibtex_skips_no_abstract():
    out = _papers_to_bibtex([
        {"title": "A", "abstract": "", "authors": []},
        {"title": "B", "abstract": "real", "authors": [{"name": "Smith"}], "publication_date": "2024-01-01"},
    ])
    assert "title = {A}" not in out
    assert "title = {B}" in out
    assert "year = {2024}" in out
    assert "author = {Smith}" in out


def test_build_reference_block_format():
    block = _build_reference_block(
        [{"arxiv_id": "1", "title": "T1", "abstract": "A1"}, {"doi": "x", "title": "T2", "abstract": "A2"}]
    )
    assert "arxiv id: 1" in block
    assert "Title: T1" in block
    assert "arxiv id: x" in block  # doi falls through via _paper_key


# ---- pipeline tests with mocks --------------------------------------------------


def _make_pipeline(tmp_path, **overrides) -> CompositeLitLLM:
    paper_path = tmp_path / "paper.md"
    paper_path.write_text("# Title\n\nAbstract goes here.")

    client = AsyncMock()
    fetcher = MagicMock()
    fetcher.search_papers = MagicMock(return_value=[])
    fetcher.fetch_pdfs = MagicMock(return_value={})

    pipe = CompositeLitLLM(
        paper_path=str(paper_path),
        output_dir=str(tmp_path / "out"),
        client=client,
        fetcher=fetcher,
        **overrides,
    )
    return pipe


@pytest.mark.asyncio
async def test_pipeline_aborts_when_no_queries(tmp_path):
    pipe = _make_pipeline(tmp_path)
    for agent in pipe.agents.values():
        agent.execute = AsyncMock(return_value={"response": "{}", "json_content": {"queries": []}})
    out = await pipe.run()
    assert "No search queries" in out


@pytest.mark.asyncio
async def test_pipeline_aborts_when_no_papers(tmp_path):
    pipe = _make_pipeline(tmp_path)
    pipe.agents["keyword_extraction"].execute = AsyncMock(
        return_value={"response": "{}", "json_content": {"queries": ["q1"]}}
    )
    pipe.fetcher.search_papers.return_value = []
    out = await pipe.run()
    assert "No related papers" in out


@pytest.mark.asyncio
async def test_pipeline_filters_by_threshold(tmp_path):
    pipe = _make_pipeline(tmp_path, ranking_score_threshold=70)
    pipe.agents["keyword_extraction"].execute = AsyncMock(
        return_value={"response": "{}", "json_content": {"queries": ["q1"]}}
    )
    pipe.fetcher.search_papers.return_value = [
        {"arxiv_id": "1", "title": "Hi", "abstract": "a"},
        {"arxiv_id": "2", "title": "Lo", "abstract": "a"},
    ]
    pipe.fetcher.fetch_pdfs.return_value = {}
    pipe.agents["debate_ranking"].execute = AsyncMock(
        return_value={
            "response": (
                "<probability>paper_id: 1 score: 90/100</probability>\n"
                "<probability>paper_id: 2 score: 30/100</probability>"
            )
        }
    )
    pipe.agents["summary"].execute = AsyncMock(return_value={"response": "summary"})
    out = await pipe.run()
    # Only paper 1 survives the filter; we now pass the dict, not just the ID.
    pipe.fetcher.fetch_pdfs.assert_called_once()
    passed = pipe.fetcher.fetch_pdfs.call_args.args[0]
    assert [p["arxiv_id"] for p in passed] == ["1"]
    assert "No PDFs" in out


@pytest.mark.asyncio
async def test_pipeline_caches_step_outputs(tmp_path):
    pipe = _make_pipeline(tmp_path)
    pipe.agents["keyword_extraction"].execute = AsyncMock(
        return_value={"response": "{}", "json_content": {"queries": ["q1"]}}
    )
    pipe.fetcher.search_papers.return_value = []
    await pipe.run()
    cached = pipe._load_cached("1_generated_queries")
    assert cached == ["q1"]


@pytest.mark.asyncio
async def test_pipeline_loads_cached_queries(tmp_path):
    pipe = _make_pipeline(tmp_path)
    # Pre-write the cache
    cache_path = pipe._path("1_generated_queries")
    import os
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(["cached query"], f)
    pipe.agents["keyword_extraction"].execute = AsyncMock(
        return_value={"response": "{}", "json_content": {"queries": ["fresh"]}}
    )
    pipe.fetcher.search_papers.return_value = []
    await pipe.run()
    pipe.agents["keyword_extraction"].execute.assert_not_called()
    pipe.fetcher.search_papers.assert_called_once_with(
        queries=["cached query"], api="semanticscholar", limit_per_query=10
    )
