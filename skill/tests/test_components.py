"""Component tests — Agent dispatch, brace escaping, JSON retry, factory."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from litllm import Agent, build_agents
from litllm.prompts import LITLLM_PROMPTS


def test_build_agents_includes_required():
    agents = build_agents()
    required = {
        "keyword_extraction",
        "debate_ranking",
        "bibliography_locator",
        "bibliography_extraction",
        "full_text_selection",
        "semantic_relevance",
        "title_validator",
        "summary",
    }
    assert required.issubset(agents.keys())
    for name in required:
        assert agents[name].name == name
        assert agents[name].system  # non-empty
        assert agents[name].user


def test_build_agents_includes_optional_when_present():
    agents = build_agents()
    assert "keyword_extraction_v2" in agents
    assert "keyword_extraction_v3" in agents
    assert "query_translator" in agents


def _mock_client(response: str = "ok", json_content: Any = None) -> AsyncMock:
    client = AsyncMock()
    client.complete = AsyncMock(return_value={"response": response, "json_content": None})
    client.complete_json = AsyncMock(return_value={"response": response, "json_content": json_content})
    return client


@pytest.mark.asyncio
async def test_agent_executes_text():
    client = _mock_client(response="hello")
    agent = Agent(name="test", system="sys", user="user {x}")
    out = await agent.execute(client, {"x": "world"})
    assert out["response"] == "hello"
    client.complete.assert_awaited_once()
    client.complete_json.assert_not_awaited()
    # Argument check: brace-escaping leaves plain strings intact
    call_kwargs = client.complete.await_args.kwargs
    assert call_kwargs["prompt_format_kwargs"] == {"x": "world"}


@pytest.mark.asyncio
async def test_agent_brace_escapes_user_strings():
    client = _mock_client()
    agent = Agent(name="t", system="s", user="user {x}")
    await agent.execute(client, {"x": "literal {brace}"})
    kwargs = client.complete.await_args.kwargs["prompt_format_kwargs"]
    assert kwargs["x"] == "literal {{brace}}"


@pytest.mark.asyncio
async def test_agent_keeps_non_string_values():
    client = _mock_client()
    agent = Agent(name="t", system="s", user="u")
    await agent.execute(client, {"n": 42, "lst": [1, 2]})
    kwargs = client.complete.await_args.kwargs["prompt_format_kwargs"]
    assert kwargs == {"n": 42, "lst": [1, 2]}


@pytest.mark.asyncio
async def test_agent_json_path_returns_parsed():
    client = _mock_client(json_content={"foo": 1})
    agent = Agent(name="t", system="s", user="u")
    out = await agent.execute(client, expect_json=True)
    assert out["json_content"] == {"foo": 1}
    client.complete_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_json_retries_then_gives_up():
    client = AsyncMock()
    client.complete_json = AsyncMock(
        return_value={"response": "garbage", "json_content": None}
    )
    agent = Agent(name="t", system="s", user="u")
    out = await agent.execute(client, expect_json=True, max_retries=2)
    # Returns soft error after exhausting retries (initial + 2 retries = 3 calls)
    assert client.complete_json.await_count == 3
    assert "error" in out


@pytest.mark.asyncio
async def test_agent_uses_prompt_text_correctly():
    client = _mock_client()
    agent = Agent(name="t", system="my-system", user="my-user-template")
    await agent.execute(client)
    kwargs = client.complete.await_args.kwargs
    assert kwargs["system_prompt"] == "my-system"
    assert kwargs["user_prompt_template"] == "my-user-template"
