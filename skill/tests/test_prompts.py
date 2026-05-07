"""Verify the prompts module loads and exposes the expected templates."""

from litllm.prompts import LITLLM_PROMPTS, PromptPair


def test_prompts_loaded():
    assert isinstance(LITLLM_PROMPTS.keyword_extraction, PromptPair)
    assert "{paper_text}" in LITLLM_PROMPTS.keyword_extraction.user


def test_all_required_prompts_present():
    required = [
        "keyword_extraction",
        "debate_ranking",
        "bibliography_locator",
        "bibliography_extraction",
        "full_text_selection",
        "semantic_relevance",
        "title_validator",
        "summary",
    ]
    for name in required:
        pair = getattr(LITLLM_PROMPTS, name)
        assert isinstance(pair, PromptPair)
        assert pair.system and pair.user, f"{name} has empty system or user prompt"


def test_optional_query_strategies():
    assert LITLLM_PROMPTS.keyword_extraction_v2 is not None
    assert LITLLM_PROMPTS.keyword_extraction_v3 is not None
    assert LITLLM_PROMPTS.query_translator is not None


def test_debate_ranking_has_required_placeholders():
    user = LITLLM_PROMPTS.debate_ranking.user
    assert "{query_paper}" in user
    assert "{reference_papers}" in user


def test_title_validator_format_is_valid():
    """Confirm the title_validator template can be .format()-ed without crashing."""
    user = LITLLM_PROMPTS.title_validator.user
    formatted = user.format(title_a="A", title_b="B")
    assert "Title A" in formatted and "Title B" in formatted
