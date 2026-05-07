"""Deep research tests — selection modes (no network, no torch)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litllm.deep_research import (
    _filter_by_abstract,
    _filter_by_full_text,
    _key,
    run_deep_research,
)


def _make_agents():
    """Map of agent-name → AsyncMock."""
    return {
        "debate_ranking": AsyncMock(),
        "full_text_selection": AsyncMock(),
        "summary": AsyncMock(),
    }


@pytest.mark.asyncio
async def test_filter_by_abstract_threshold():
    agent = AsyncMock()
    agent.execute = AsyncMock(
        return_value={
            "response": (
                "<probability>[1234]: 90</probability>\n"
                "<probability>[5678]: 30</probability>"
            )
        }
    )
    candidates = [
        {"arxiv_id": "1234", "title": "A", "abstract": "x"},
        {"arxiv_id": "5678", "title": "B", "abstract": "y"},
    ]
    out = await _filter_by_abstract(candidates, "main abstract", agent, AsyncMock(), 70, 20)
    keys = [_key(p) for p in out]
    assert keys == ["1234"]


@pytest.mark.asyncio
async def test_filter_by_full_text(tmp_path):
    pdf = tmp_path / "1234.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")
    fetcher = MagicMock()
    fetcher.fetch_pdfs = MagicMock(return_value={"1234": str(pdf)})

    agent = AsyncMock()
    agent.execute = AsyncMock(return_value={"response": "<probability>85</probability>"})

    # extract_text_from_pdf will fail on our minimal stub — patch it.
    import litllm.deep_research as dr

    dr.extract_text_from_pdf = lambda p: "candidate full text"

    out = await _filter_by_full_text(
        [{"arxiv_id": "1234", "title": "A"}],
        "main full text",
        fetcher,
        agent,
        AsyncMock(),
        set(),
        70,
    )
    assert len(out) == 1


@pytest.mark.asyncio
async def test_run_deep_research_aborts_when_no_refs():
    fetcher = MagicMock()
    fetcher.get_referenced_works_from_openalex = MagicMock(
        return_value={"referenced_works": [], "related_works": []}
    )
    fetcher.search_papers = MagicMock(return_value=[])

    out = await run_deep_research(
        main_paper={"arxiv_id": "main", "title": "Main", "abstract": "m"},
        seed_papers=[{"arxiv_id": "1", "title": "Seed1", "abstract": "s1"}],
        fetcher=fetcher,
        client=AsyncMock(),
        agents=_make_agents(),
    )
    # Only seed papers survive; nothing expanded.
    assert {p["arxiv_id"] for p in out} == {"1"}


@pytest.mark.asyncio
async def test_run_deep_research_rejects_unknown_selection_mode():
    fetcher = MagicMock()
    fetcher.get_referenced_works_from_openalex = MagicMock(
        return_value={"referenced_works": ["W123"], "related_works": []}
    )
    fetcher.search_papers = MagicMock(
        return_value=[{"openalex_id": "W123", "title": "Other", "abstract": "x"}]
    )
    with pytest.raises(ValueError, match="Unknown selection_mode"):
        await run_deep_research(
            main_paper={"arxiv_id": "main", "title": "M", "abstract": "m"},
            seed_papers=[{"arxiv_id": "1", "title": "S", "abstract": "s"}],
            fetcher=fetcher,
            client=AsyncMock(),
            agents=_make_agents(),
            selection_mode="bogus",
        )


@pytest.mark.asyncio
async def test_run_deep_research_embedding_mode_requires_embedder():
    fetcher = MagicMock()
    fetcher.get_referenced_works_from_openalex = MagicMock(
        return_value={"referenced_works": ["W123"], "related_works": []}
    )
    fetcher.search_papers = MagicMock(
        return_value=[{"openalex_id": "W123", "title": "C", "abstract": "x"}]
    )
    with pytest.raises(ValueError, match="requires an Embedder"):
        await run_deep_research(
            main_paper={"arxiv_id": "main", "title": "M", "abstract": "m"},
            seed_papers=[{"arxiv_id": "1", "title": "S", "abstract": "s"}],
            fetcher=fetcher,
            client=AsyncMock(),
            agents=_make_agents(),
            selection_mode="embedding",
            embedder=None,
        )


@pytest.mark.asyncio
async def test_run_deep_research_abstract_mode_expands_one_level():
    fetcher = MagicMock()
    # Level 0: seed paper "1" has reference "W999"
    # Level 1: paper "W999" has no further references (BFS stops)
    fetcher.get_referenced_works_from_openalex = MagicMock(
        side_effect=[
            {"referenced_works": ["W999"], "related_works": []},
            {"referenced_works": [], "related_works": []},
        ]
    )
    fetcher.search_papers = MagicMock(
        return_value=[{"arxiv_id": "999", "title": "Expanded", "abstract": "text"}]
    )

    agents = _make_agents()
    agents["debate_ranking"].execute = AsyncMock(
        return_value={"response": "<probability>[999]: 90</probability>"}
    )

    out = await run_deep_research(
        main_paper={"arxiv_id": "main", "title": "M", "abstract": "m"},
        seed_papers=[{"arxiv_id": "1", "title": "S", "abstract": "s"}],
        fetcher=fetcher,
        client=AsyncMock(),
        agents=agents,
        selection_mode="abstract",
        max_depth=2,
    )
    keys = sorted(_key(p) for p in out)
    assert keys == ["1", "999"]
