"""Unit tests for LLMClient configuration and utilities (no network calls)."""

from __future__ import annotations

import pytest

from litllm import LLMConfigError, parse_llm_json_response
from litllm.llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LITLLM_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="LITLLM_API_KEY"):
        LLMClient()


def test_env_driven_config(monkeypatch):
    monkeypatch.setenv("LITLLM_API_KEY", "test-key")
    monkeypatch.delenv("LITLLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITLLM_MODEL", raising=False)
    client = LLMClient()
    assert client.api_key == "test-key"
    assert client.base_url == DEFAULT_BASE_URL
    assert client.model == DEFAULT_MODEL


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("LITLLM_API_KEY", "env-key")
    monkeypatch.setenv("LITLLM_MODEL", "env-model")
    client = LLMClient(api_key="explicit", base_url="https://custom/v1", model="explicit-model")
    assert client.api_key == "explicit"
    assert client.base_url == "https://custom/v1"
    assert client.model == "explicit-model"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('```json\n{"x": 1}\n```', {"x": 1}),
        ("plain prose then {\"x\": 2}", {"x": 2}),
        ("<think>noise</think>\n```\n[1, 2, 3]\n```", [1, 2, 3]),
        ("not json at all", None),
        ("", None),
    ],
)
def test_parse_llm_json_response(raw, expected):
    assert parse_llm_json_response(raw) == expected
