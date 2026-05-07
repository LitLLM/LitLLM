"""OpenAI-compatible LLM client.

Speaks any endpoint exposing the OpenAI chat-completions and embeddings APIs:
OpenAI proper, vLLM, Ollama, OpenRouter, Together, Groq, LM Studio, etc.

Configured via environment:
    LITLLM_API_KEY    — required
    LITLLM_BASE_URL   — default https://api.openai.com/v1
    LITLLM_MODEL      — default gpt-4o-mini
    LITLLM_EMBED_MODEL — default text-embedding-3-small
"""

from __future__ import annotations

import os
import re
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from .utils import clean_markdown, extract_text_from_pdf, parse_llm_json_response

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class LLMConfigError(ValueError):
    """Raised when required configuration (API key, base URL) is missing."""


class LLMClient:
    """Minimal OpenAI-compatible async client used by all litllm components."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("LITLLM_API_KEY")
        if not self.api_key:
            raise LLMConfigError(
                "LITLLM_API_KEY is not set. Export it or pass api_key= explicitly. "
                "Any OpenAI-compatible provider works (OpenAI, vLLM, Ollama, OpenRouter, ...)."
            )
        self.base_url = base_url or os.getenv("LITLLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("LITLLM_MODEL", DEFAULT_MODEL)
        self.embed_model = embed_model or os.getenv("LITLLM_EMBED_MODEL", DEFAULT_EMBED_MODEL)

        self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def complete(
        self,
        system_prompt: str,
        user_prompt_template: str,
        files: list[str] | None = None,
        prompt_format_kwargs: dict[str, Any] | None = None,
        use_reasoning: bool = False,
        reasoning_effort: str = "medium",
        logprobs: bool = False,
        top_logprobs: int = 5,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 20000,
    ) -> dict[str, Any]:
        """Run a chat completion. Returns {"response": str, "usage": dict, "finish_reason": str, ...}."""
        prompt_format_kwargs = dict(prompt_format_kwargs or {})
        system_prompt = re.sub(r"<think>.*?</think>", "", system_prompt, flags=re.DOTALL)
        user_text = user_prompt_template

        # Inline file content under the {paper_text} placeholder, or append a labeled section.
        for path in files or []:
            if not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                content = extract_text_from_pdf(path)
            elif ext in (".md", ".markdown"):
                with open(path, "r", encoding="utf-8") as f:
                    content = clean_markdown(f.read())
            elif ext == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                continue

            if "paper_text" not in prompt_format_kwargs:
                prompt_format_kwargs["paper_text"] = content
            else:
                # Escape braces — appended content goes through .format() and may contain LaTeX.
                safe = content.replace("{", "{{").replace("}", "}}")
                fname = os.path.basename(path)
                user_text += f"\n\n--- Content of {fname} ---\n{safe}\n--- End of {fname} ---"

        if "{paper_text}" in user_text and "paper_text" not in prompt_format_kwargs:
            prompt_format_kwargs["paper_text"] = "[Paper text not available.]"

        async def _call() -> Any:
            final_user = user_text.format(**prompt_format_kwargs)
            final_user = re.sub(r"<think>.*?</think>", "", final_user, flags=re.DOTALL)
            params: dict[str, Any] = {
                "model": model or self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_user},
                ],
            }
            if use_reasoning:
                params["reasoning_effort"] = reasoning_effort
            else:
                params.update({"temperature": temperature, "max_tokens": max_tokens, "top_p": 1.0})
            if logprobs:
                params.update({"logprobs": True, "top_logprobs": top_logprobs})
            return await self._client.chat.completions.create(**params)

        try:
            response = await _call()
        except BadRequestError as e:
            if "maximum context length" not in str(e).lower():
                raise
            paper_text = prompt_format_kwargs.get("paper_text", "")
            if not paper_text or len(paper_text) < 100:
                raise
            prompt_format_kwargs["paper_text"] = paper_text[: len(paper_text) // 2]
            response = await _call()

        choice = response.choices[0]
        out: dict[str, Any] = {
            "response": choice.message.content,
            "model_used": model or self.model,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        if use_reasoning:
            out["reasoning"] = getattr(choice.message, "reasoning_content", None)
        if logprobs:
            out["logprobs"] = choice.logprobs
        return out

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt_template: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat completion + robust JSON parse. Returns the same dict as complete()
        plus json_content (None on parse failure)."""
        result = await self.complete(system_prompt, user_prompt_template, **kwargs)
        result["json_content"] = parse_llm_json_response(result["response"] or "")
        return result

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Get embeddings via the OpenAI-compatible /embeddings endpoint."""
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=model or self.embed_model,
            input=texts,
        )
        return [d.embedding for d in response.data]
