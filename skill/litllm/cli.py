"""Command-line entry point for litllm."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from . import __version__
from .components import build_agents
from .composite import CompositeLitLLM
from .fetcher import PaperFetcher
from .llm_client import LLMClient, LLMConfigError
from .utils import read_paper

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    # Silence per-request HTTP logs from openai's httpx unless --verbose.
    if not verbose:
        for noisy in ("httpx", "httpcore", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litllm",
        description="Literature search and related-work generation for academic papers.",
    )
    parser.add_argument("--version", action="version", version=f"litllm {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    rw = sub.add_parser("related-work", help="Generate a related-work section for a paper.")
    rw.add_argument("paper", help="Path to the paper (PDF or markdown).")
    rw.add_argument("--out", default="litllm-output", help="Output directory.")
    rw.add_argument("--deep-research", action="store_true", help="Expand citation graph (BFS, max depth 2).")
    rw.add_argument("--limit-per-query", type=int, default=10)
    rw.add_argument("--ranking-threshold", type=int, default=60)
    rw.add_argument("--main-paper-title", default="", help="Title of the input paper (used for dedup).")
    rw.add_argument(
        "--selection-mode",
        choices=["abstract", "full-text", "embedding"],
        default="abstract",
        help="Deep-research candidate selection strategy.",
    )
    rw.add_argument(
        "--api",
        choices=["semanticscholar", "arxiv", "openalex"],
        default="semanticscholar",
        help="Search backend.",
    )

    kw = sub.add_parser("keywords", help="Extract search queries from a paper.")
    kw.add_argument("paper")

    rk = sub.add_parser("rank", help="Debate-rank candidate papers against a target paper.")
    rk.add_argument("paper")
    rk.add_argument("--candidates", required=True, help="JSON file with candidate papers (list of dicts).")

    bib = sub.add_parser("bib", help="Extract bibliography titles from a paper PDF.")
    bib.add_argument("paper")

    return parser


async def _cmd_related_work(args: argparse.Namespace) -> int:
    pipeline = CompositeLitLLM(
        paper_path=args.paper,
        output_dir=args.out,
        deep_research=args.deep_research,
        selection_mode=args.selection_mode,
        main_paper_title=args.main_paper_title,
        ranking_score_threshold=args.ranking_threshold,
        limit_per_query=args.limit_per_query,
        search_api=args.api,
    )
    summary = await pipeline.run()
    print(summary)
    return 0


async def _cmd_keywords(args: argparse.Namespace) -> int:
    client = LLMClient()
    agents = build_agents()
    paper_text = read_paper(args.paper)
    result = await agents["keyword_extraction"].execute(
        client, {"paper_text": paper_text}, expect_json=True
    )
    queries = (result.get("json_content") or {}).get("queries", [])
    print(json.dumps(queries, indent=2))
    return 0


async def _cmd_rank(args: argparse.Namespace) -> int:
    client = LLMClient()
    agents = build_agents()
    paper_text = read_paper(args.paper)
    with open(args.candidates, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    block = "\n\n".join(
        f"arxiv id: {p.get('arxiv_id') or p.get('doi') or p.get('openalex_id')}\nTitle: {p.get('title')}\nAbstract: {p.get('abstract', 'N/A')}"
        for p in candidates
    )
    result = await agents["debate_ranking"].execute(
        client, {"query_paper": paper_text, "reference_papers": block}
    )
    print(result.get("response") or "")
    return 0


async def _cmd_bib(args: argparse.Namespace) -> int:
    client = LLMClient()
    agents = build_agents()
    paper_text = read_paper(args.paper)
    locator = await agents["bibliography_locator"].execute(client, {"paper_text": paper_text})
    bib_text = locator.get("response") or ""
    extractor = await agents["bibliography_extraction"].execute(
        client, {"bibliography_text": bib_text}, expect_json=True
    )
    titles = [r.get("title") for r in (extractor.get("json_content") or {}).get("references", []) if r.get("title")]
    print(json.dumps(titles, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    handler = {
        "related-work": _cmd_related_work,
        "keywords": _cmd_keywords,
        "rank": _cmd_rank,
        "bib": _cmd_bib,
    }[args.command]

    try:
        return asyncio.run(handler(args))
    except LLMConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
