"""Shared utilities: PDF/markdown extraction and JSON parsing from LLM output."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pdfminer.high_level import extract_text as _pdfminer_extract_text

logging.getLogger("pdfminer").setLevel(logging.ERROR)


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    return _pdfminer_extract_text(pdf_path)


_IMAGE_DATA_URI = re.compile(r"^\s*!\[.*?\]\(data:image/.*?\)\s*$")


def clean_markdown(markdown_text: str) -> str:
    """Strip page-number-only lines and inlined base64 image data URIs."""
    cleaned = []
    for line in markdown_text.split("\n"):
        if line.strip().isdigit():
            continue
        if _IMAGE_DATA_URI.match(line):
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned).strip())


def parse_llm_json_response(raw_response: str) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON object/list from an LLM response that may contain
    `<think>` blocks, markdown fences, or trailing prose."""
    if not raw_response:
        return None

    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else cleaned

    first_brace = candidate.find("{")
    first_bracket = candidate.find("[")
    if first_brace == -1 and first_bracket == -1:
        return None
    start = first_brace if (first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket)) else first_bracket
    candidate = candidate[start:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def read_paper(path: str) -> str:
    """Load a paper as text. Accepts .pdf or .md/.txt."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".md", ".markdown", ".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return clean_markdown(f.read()) if ext != ".txt" else f.read()
    raise ValueError(f"Unsupported paper format: {ext}. Use .pdf, .md, or .txt.")
