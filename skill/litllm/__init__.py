"""litllm — literature search and related-work generation."""

from .components import Agent, build_agents
from .fetcher import PaperFetcher
from .llm_client import LLMClient, LLMConfigError
from .prompts import LITLLM_PROMPTS, LitLLMPrompts, PromptPair
from .utils import clean_markdown, extract_text_from_pdf, parse_llm_json_response, read_paper

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "LITLLM_PROMPTS",
    "LLMClient",
    "LLMConfigError",
    "LitLLMPrompts",
    "PaperFetcher",
    "PromptPair",
    "__version__",
    "build_agents",
    "clean_markdown",
    "extract_text_from_pdf",
    "parse_llm_json_response",
    "read_paper",
]
