"""The 4-step litllm pipeline: keywords → fetch → rank+filter → summarize.

Ported from src/agents/litllm/types/composite.py. Differences vs the original:

- Drops the BaseReviewer parent (replaced by inlined safify/save_output).
- Drops the per-experiment 4-level output nesting (model/strategy/mode/file);
  the standalone CLI just writes to <output_dir>/<name>.md.
- Deep-research citation-graph traversal lives in a separate module
  (litllm/deep_research.py); composite calls into it when --deep-research is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from typing import Any

import tqdm

from .components import Agent, build_agents
from .fetcher import PaperFetcher
from .llm_client import LLMClient
from .prompts import LITLLM_PROMPTS, LitLLMPrompts
from .utils import clean_markdown, extract_text_from_pdf, parse_llm_json_response

logger = logging.getLogger(__name__)


def _safify(name: str) -> str:
    """Filesystem-safe slug — keeps alphanumerics, dots, and hyphens."""
    return re.sub(r"[^\w.\-]+", "_", name).strip("_")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").lower()).strip()


def _paper_key(p: dict[str, Any]) -> str | None:
    return p.get("arxiv_id") or p.get("doi") or p.get("openalex_id")


def _parse_ranking_scores(output_text: str) -> dict[str, int]:
    """Extract `{paper_id: score}` from a ranking output string.
    Tolerates two formats: explicit `paper_id: X / score: Y/100` and
    inline `<probability>[ID]: Y</probability>`."""
    scores: dict[str, int] = {}
    for pid, score in re.findall(
        r"paper_id:\s*([^\s\n]+)[\s\S]*?score:\s*(\d+)\s*/100",
        output_text,
        flags=re.IGNORECASE,
    ):
        pid = pid.strip()
        try:
            scores[pid] = max(int(score), scores.get(pid, 0))
        except ValueError:
            pass
    for pid, score in re.findall(
        r"<probability>\s*\[?(.*?)\]?:\s*(\d+)\s*</probability>",
        output_text,
        flags=re.IGNORECASE,
    ):
        pid = pid.strip()
        if not pid:
            continue
        try:
            scores[pid] = max(int(score), scores.get(pid, 0))
        except ValueError:
            pass
    return scores


def _papers_to_bibtex(papers: list[dict[str, Any]]) -> str:
    """Render search results as @article BibTeX entries (skips entries without abstract)."""
    entries: list[str] = []
    for item in papers:
        if not item.get("abstract"):
            continue
        bibid = (item.get("bibid") or _normalize_title(item.get("title", "")) or "unnamed_entry").replace(" ", "_").replace(":", "_").replace(",", "_")
        title = item.get("title", "")
        authors_field = item.get("authors") or []
        authors = " and ".join(a.get("name", a) if isinstance(a, dict) else str(a) for a in authors_field)
        year = (item.get("publication_date") or "")[:4]
        doi = item.get("doi") or ""
        arxiv_id = item.get("arxiv_id") or ""
        abstract = (item.get("abstract") or "").replace("\n", " ")

        entry = f"@article{{{bibid},\n"
        entry += f"  title = {{{title}}},\n"
        entry += f"  author = {{{authors}}},\n"
        if year:
            entry += f"  year = {{{year}}},\n"
        if doi:
            entry += f"  doi = {{{doi}}},\n"
        if arxiv_id:
            entry += f"  eprint = {{{arxiv_id}}},\n"
            entry += "  archivePrefix = {arXiv},\n"
        entry += f"  abstract = {{{abstract}}}\n"
        entry += "}\n"
        entries.append(entry)
    return "\n".join(entries)


def _build_reference_block(papers: list[dict[str, Any]]) -> str:
    """Format candidate papers as the `reference_papers` argument for debate ranking."""
    return "\n\n".join(
        f"arxiv id: {_paper_key(p)}\nTitle: {p.get('title')}\nAbstract: {p.get('abstract', 'N/A')}"
        for p in papers
    )


class CompositeLitLLM:
    """End-to-end literature-search pipeline for a single paper."""

    def __init__(
        self,
        paper_path: str,
        output_dir: str = "litllm-output",
        *,
        client: LLMClient | None = None,
        fetcher: PaperFetcher | None = None,
        prompts: LitLLMPrompts = LITLLM_PROMPTS,
        agents: dict[str, Agent] | None = None,
        deep_research: bool = False,
        selection_mode: str = "abstract",
        main_paper_title: str = "",
        ranking_score_threshold: int = 60,
        ranking_batch_size: int = 10,
        limit_per_query: int = 10,
        search_api: str = "semanticscholar",
    ):
        self.paper_path = paper_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.client = client or LLMClient()
        self.fetcher = fetcher or PaperFetcher()
        self.prompts = prompts
        self.agents = agents or build_agents(prompts)

        self.deep_research = deep_research
        self.selection_mode = selection_mode
        self.main_paper_title = main_paper_title
        self.ranking_score_threshold = ranking_score_threshold
        self.ranking_batch_size = ranking_batch_size
        self.limit_per_query = limit_per_query
        self.search_api = search_api

    # ---- helpers ----

    def _path(self, name: str) -> str:
        return os.path.join(self.output_dir, f"{_safify(name)}.md")

    def _load_cached(self, name: str) -> Any:
        path = self._path(name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    def _save(self, name: str, content: Any) -> str:
        path = self._path(name)
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _read_paper_text(self) -> str:
        if self.paper_path.endswith(".md"):
            with open(self.paper_path, "r", encoding="utf-8") as f:
                return clean_markdown(f.read())
        if self.paper_path.endswith(".pdf"):
            return extract_text_from_pdf(self.paper_path)
        with open(self.paper_path, "r", encoding="utf-8") as f:
            return f.read()

    # ---- pipeline steps ----

    async def _step_keywords(self, paper_text: str) -> list[str]:
        cached = self._load_cached("1_generated_queries")
        if isinstance(cached, list) and cached:
            logger.info("Loaded cached queries from %s", self._path("1_generated_queries"))
            return cached

        result = await self.agents["keyword_extraction"].execute(
            self.client,
            {"paper_text": paper_text},
            expect_json=True,
        )
        json_content = result.get("json_content") or {}
        queries = json_content.get("queries", []) if isinstance(json_content, dict) else []
        self._save("1_generated_queries", queries)
        return queries

    async def _step_fetch(self, queries: list[str]) -> list[dict[str, Any]]:
        cached = self._load_cached("2_fetched_papers")
        if isinstance(cached, list) and cached:
            logger.info("Loaded cached fetched papers")
            return cached

        papers = self.fetcher.search_papers(
            queries=queries, api=self.search_api, limit_per_query=self.limit_per_query
        )

        # Drop the main paper if it shows up as a result
        if self.main_paper_title:
            target = _normalize_title(self.main_paper_title)
            papers = [p for p in papers if _normalize_title(p.get("title", "")) != target]

        if self.deep_research:
            from .deep_research import run_deep_research  # local import to avoid cycle

            main_paper_data_list = self.fetcher.search_papers(
                queries=[self.main_paper_title or queries[0]], limit_per_query=1
            )
            if main_paper_data_list:
                papers = await run_deep_research(
                    main_paper=main_paper_data_list[0],
                    seed_papers=papers,
                    fetcher=self.fetcher,
                    client=self.client,
                    agents=self.agents,
                    selection_mode=self.selection_mode,
                    main_paper_full_text=self._read_paper_text(),
                )

        self._save("2_fetched_papers", papers)
        return papers

    async def _step_rank(self, paper_text: str, papers: list[dict[str, Any]]) -> str:
        results: list[str] = []
        num_batches = math.ceil(len(papers) / self.ranking_batch_size)
        for i in tqdm.tqdm(range(num_batches), desc="Ranking papers"):
            batch = papers[i * self.ranking_batch_size : (i + 1) * self.ranking_batch_size]
            ranked = await self.agents["debate_ranking"].execute(
                self.client,
                {"query_paper": paper_text, "reference_papers": _build_reference_block(batch)},
            )
            response = ranked.get("response") or ""
            if response:
                results.append(response)
        ranking_output = "\n\n".join(results)
        self._save("3_ranked_papers", ranking_output)
        return ranking_output

    def _step_filter(self, papers: list[dict[str, Any]], ranking_output: str) -> list[dict[str, Any]]:
        scores = _parse_ranking_scores(ranking_output)
        if not scores:
            logger.warning("No scores parsed from ranking output. Skipping filter step.")
            return papers
        filtered = [
            p for p in papers
            if (key := _paper_key(p)) and key in scores and scores[key] >= self.ranking_score_threshold
        ]
        logger.info("Filtered %d/%d papers above score >= %d", len(filtered), len(papers), self.ranking_score_threshold)
        self._save("3.5_filtered_papers", filtered)
        self._save("3.5_bibfile", _papers_to_bibtex(filtered))
        return filtered

    async def _step_summarize(self, papers: list[dict[str, Any]]) -> str:
        with_ids = [p for p in papers if p.get("arxiv_id") or p.get("doi") or p.get("openalex_id")]
        pdf_path_map = self.fetcher.fetch_pdfs(with_ids)
        pdf_paths = list(pdf_path_map.values())
        if not pdf_paths:
            self._save("4_related_papers_summary", "")
            return "No PDFs available for summarization."

        tasks = [self.agents["summary"].execute(self.client, files=[path]) for path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        summaries = [
            (r.get("response", "") if not isinstance(r, Exception) else f"Error: {r}")
            for r in results
        ]
        final = "\n\n---\n\n".join(s for s in summaries if s)
        self._save("4_related_papers_summary", final)
        return final

    # ---- entrypoint ----

    async def run(self) -> str:
        """Run all four steps. Returns the final summary string."""
        logger.info("Starting litllm pipeline for %s", self.paper_path)
        paper_text = self._read_paper_text()

        queries = await self._step_keywords(paper_text)
        if not queries:
            return "No search queries could be generated."

        papers = await self._step_fetch(queries)
        if not papers:
            return "No related papers were found."

        ranking_output = await self._step_rank(paper_text, papers)
        filtered = self._step_filter(papers, ranking_output)
        if not filtered:
            self._save("4_related_papers_summary", "")
            return "No related papers passed the ranking threshold."

        return await self._step_summarize(filtered)
