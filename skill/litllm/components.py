"""Pipeline components — one per (system, user) prompt pair.

The original reviewertoo had 7 component classes (KeywordExtractionAgent,
DebateRankingAgent, ...) all inheriting from a shared BaseCheckAgent. They
were typed marker classes; the real logic lived in the parent. Here we
collapse to a single `Agent` dataclass plus a `build_agents()` factory.

Each Agent carries its own prompts and exposes an `execute()` coroutine
that calls an LLMClient, with optional JSON parsing and retry-on-failure
behavior identical to the original BaseCheckAgent.execute().
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient
from .prompts import LITLLM_PROMPTS, LitLLMPrompts

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """One LLM-driven pipeline step identified by a name and prompt pair."""

    name: str
    system: str
    user: str

    async def execute(
        self,
        client: LLMClient,
        prompt_format_kwargs: dict[str, Any] | None = None,
        files: list[str] | None = None,
        expect_json: bool = False,
        max_retries: int = 3,
        **complete_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the agent. Returns the LLMClient.complete() dict, with json_content
        populated when expect_json=True. Retries up to max_retries on JSON-parse
        failure or transient errors."""
        # Brace-escape string values so user-provided text (LaTeX, dict examples,
        # etc.) doesn't break the .format() pass inside LLMClient.
        escaped: dict[str, Any] = {}
        for key, value in (prompt_format_kwargs or {}).items():
            escaped[key] = value.replace("{", "{{").replace("}", "}}") if isinstance(value, str) else value

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if expect_json:
                    result = await client.complete_json(
                        system_prompt=self.system,
                        user_prompt_template=self.user,
                        files=files,
                        prompt_format_kwargs=escaped,
                        **complete_kwargs,
                    )
                    if result.get("json_content") is None and result.get("response"):
                        raise ValueError("Could not parse JSON from LLM response.")
                else:
                    result = await client.complete(
                        system_prompt=self.system,
                        user_prompt_template=self.user,
                        files=files,
                        prompt_format_kwargs=escaped,
                        **complete_kwargs,
                    )
                return result
            except (ValueError, TypeError, AttributeError) as e:
                last_exc = e
                logger.warning(
                    "[%s] attempt %d/%d failed: %s", self.name, attempt + 1, max_retries + 1, e
                )
                if attempt < max_retries:
                    await asyncio.sleep(1)
        return {"response": "", "json_content": None, "error": str(last_exc) if last_exc else "unknown"}


_REQUIRED_AGENTS = (
    "keyword_extraction",
    "debate_ranking",
    "bibliography_locator",
    "bibliography_extraction",
    "full_text_selection",
    "semantic_relevance",
    "title_validator",
    "summary",
)
_OPTIONAL_AGENTS = ("keyword_extraction_v2", "keyword_extraction_v3", "query_translator")


def build_agents(prompts: LitLLMPrompts = LITLLM_PROMPTS) -> dict[str, Agent]:
    """Build the full dict of litllm pipeline agents from a prompt set."""
    out: dict[str, Agent] = {}
    for name in _REQUIRED_AGENTS:
        pair = getattr(prompts, name)
        out[name] = Agent(name=name, system=pair.system, user=pair.user)
    for name in _OPTIONAL_AGENTS:
        pair = getattr(prompts, name, None)
        if pair is not None:
            out[name] = Agent(name=name, system=pair.system, user=pair.user)
    return out
