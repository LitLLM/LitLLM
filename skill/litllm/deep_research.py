"""Citation-graph BFS expansion ("deep research") — ported from
CompositeLitLLMAgent._run_deep_research().

Starts from a set of seed papers (the initial search results), traverses up to
max_depth levels of references via OpenAlex (with a related-works fallback),
and at each level filters candidates with one of three selection modes:

- "embedding": SPECTER similarity ≥ similarity_threshold (lazy-loads
  sentence-transformers; install with `pip install 'litllm[embeddings]'`)
- "abstract": batched debate-ranking on abstracts, score ≥ probability_threshold
- "full-text": per-candidate full-text comparison, score ≥ probability_threshold
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import deque
from typing import Any

from .components import Agent
from .fetcher import PaperFetcher
from .llm_client import LLMClient
from .utils import clean_markdown, extract_text_from_pdf

logger = logging.getLogger(__name__)


def _key(p: dict[str, Any]) -> str | None:
    return p.get("arxiv_id") or p.get("doi") or p.get("openalex_id")


class Embedder:
    """SPECTER-based paper embeddings.

    Heavy deps (torch, sentence-transformers, scikit-learn) are lazy-loaded so
    `litllm` works without them when deep_research isn't using embedding mode.
    """

    def __init__(self, model_name: str = "sentence-transformers/allenai-specter"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Embedding selection requires sentence-transformers. "
                "Install with: pip install 'litllm[embeddings]'"
            ) from e
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._cache: dict[str, Any] = {}

    def embed(self, paper_id: str, title: str | None, abstract: str | None) -> Any:
        if not paper_id:
            return None
        if paper_id in self._cache:
            return self._cache[paper_id]
        text = (title or "") + self._model.tokenizer.sep_token + (abstract or "")
        try:
            vector = self._model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        except Exception as e:
            logger.warning("Embedding failed for %s: %s", paper_id, e)
            return None
        self._cache[paper_id] = vector
        return vector

    @staticmethod
    def similarities(main_vec: Any, candidates: dict[str, Any]) -> dict[str, float]:
        if main_vec is None or not candidates:
            return {}
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        ids = list(candidates.keys())
        matrix = np.array([candidates[cid] for cid in ids])
        sims = cosine_similarity([main_vec], matrix)[0]
        return {cid: float(sims[i]) for i, cid in enumerate(ids)}


async def _filter_by_embedding(
    candidates: list[dict[str, Any]],
    main_paper: dict[str, Any],
    embedder: Embedder,
    visited_ids: set[str],
    similarity_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    main_vec = embedder.embed(_key(main_paper) or "main", main_paper.get("title"), main_paper.get("abstract"))
    if main_vec is None:
        return []
    candidate_vecs: dict[str, Any] = {}
    for paper in candidates:
        pid = _key(paper)
        if not pid or pid in visited_ids:
            continue
        vec = embedder.embed(pid, paper.get("title"), paper.get("abstract"))
        if vec is not None:
            candidate_vecs[pid] = vec

    sims = embedder.similarities(main_vec, candidate_vecs)
    above = sorted(
        ((pid, score) for pid, score in sims.items() if score >= similarity_threshold),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_k]

    selected: list[dict[str, Any]] = []
    for pid, score in above:
        match = next((p for p in candidates if _key(p) == pid), None)
        if match:
            logger.info("  + EMBED: %s (sim=%.2f) %s", pid, score, match.get("title"))
            selected.append(match)
    return selected


async def _filter_by_abstract(
    candidates: list[dict[str, Any]],
    main_paper_abstract: str,
    ranking_agent: Agent,
    client: LLMClient,
    probability_threshold: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    num_batches = math.ceil(len(candidates) / batch_size)
    for i in range(num_batches):
        batch = candidates[i * batch_size : (i + 1) * batch_size]
        block = "\n\n".join(
            f"ID: {_key(p)}\nTitle: {p.get('title')}\nAbstract: {p.get('abstract', 'N/A')}"
            for p in batch if _key(p)
        )
        result = await ranking_agent.execute(
            client,
            {"query_paper": main_paper_abstract, "reference_papers": block},
        )
        text = result.get("response") or ""
        for pid, prob in re.findall(r"<probability>\s*\[?(.*?)\]?:\s*(\d+)\s*</probability>", text):
            try:
                if int(prob) >= probability_threshold:
                    pid = pid.strip()
                    match = next((p for p in batch if _key(p) == pid), None)
                    if match:
                        logger.info("  + ABS: %s (%s%%) %s", pid, prob, match.get("title"))
                        selected.append(match)
            except ValueError:
                continue
    return selected


async def _filter_by_full_text(
    candidates: list[dict[str, Any]],
    main_paper_full_text: str,
    fetcher: PaperFetcher,
    full_text_agent: Agent,
    client: LLMClient,
    visited_ids: set[str],
    probability_threshold: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for paper in candidates:
        pid = _key(paper)
        if not pid or pid in visited_ids:
            continue
        pdf_map = fetcher.fetch_pdfs([paper])
        pdf_path = pdf_map.get(pid)
        if not pdf_path or not os.path.exists(pdf_path):
            continue
        candidate_text = extract_text_from_pdf(pdf_path)
        result = await full_text_agent.execute(
            client,
            {"query_paper": main_paper_full_text, "candidate_paper": candidate_text},
        )
        text = result.get("response") or ""
        match = re.search(r"<probability>.*?(\d+)\s*</probability>", text, re.DOTALL)
        if match and int(match.group(1)) >= probability_threshold:
            logger.info("  + FT: %s (%s%%) %s", pid, match.group(1), paper.get("title"))
            selected.append(paper)
    return selected


async def run_deep_research(
    *,
    main_paper: dict[str, Any],
    seed_papers: list[dict[str, Any]],
    fetcher: PaperFetcher,
    client: LLMClient,
    agents: dict[str, Agent],
    selection_mode: str = "abstract",
    main_paper_full_text: str = "",
    embedder: Embedder | None = None,
    max_depth: int = 2,
    max_total_papers: int = 1000,
    probability_threshold: int = 70,
    similarity_threshold: float = 0.80,
    top_k_after_threshold: int = 10,
    selection_batch_size: int = 20,
) -> list[dict[str, Any]]:
    """Expand the citation graph BFS-style and return the accumulated paper set."""
    logger.info("Starting deep research (mode=%s, max_depth=%d)", selection_mode, max_depth)

    queue: deque[tuple[dict[str, Any], int]] = deque()
    final_papers: dict[str, dict[str, Any]] = {}
    visited: set[str] = set()
    for p in seed_papers:
        pid = _key(p)
        if pid:
            queue.append((p, 0))
            final_papers[pid] = p
            visited.add(pid)

    main_abstract = main_paper.get("abstract", "")

    while queue and len(final_papers) < max_total_papers:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        # Try OpenAlex referenced_works first; fall back to related_works.
        api_data = fetcher.get_referenced_works_from_openalex(current)
        candidate_ids_or_dicts: list[Any] = []
        if api_data.get("referenced_works"):
            candidate_ids_or_dicts.extend(api_data["referenced_works"])
        elif api_data.get("related_works"):
            candidate_ids_or_dicts.extend(api_data["related_works"])
        if not candidate_ids_or_dicts:
            continue

        # Resolve string OpenAlex IDs into full paper records.
        ids_to_fetch = [item.split("/")[-1] for item in candidate_ids_or_dicts if isinstance(item, str)]
        new_papers = [item for item in candidate_ids_or_dicts if isinstance(item, dict)]
        if ids_to_fetch:
            new_papers.extend(fetcher.search_papers(queries=[], openalex_ids=ids_to_fetch, limit_per_query=1))

        if not new_papers:
            continue

        if selection_mode == "embedding":
            if embedder is None:
                raise ValueError("selection_mode='embedding' requires an Embedder. Pass embedder= or use 'abstract'/'full-text'.")
            selected = await _filter_by_embedding(
                new_papers, main_paper, embedder, visited, similarity_threshold, top_k_after_threshold
            )
        elif selection_mode == "abstract":
            selected = await _filter_by_abstract(
                new_papers, main_abstract, agents["debate_ranking"], client, probability_threshold, selection_batch_size
            )
        elif selection_mode == "full-text":
            selected = await _filter_by_full_text(
                new_papers, main_paper_full_text, fetcher, agents["full_text_selection"], client, visited, probability_threshold
            )
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode!r}. Use embedding/abstract/full-text.")

        for paper in selected:
            pid = _key(paper)
            if pid and pid not in visited:
                visited.add(pid)
                final_papers[pid] = paper
                queue.append((paper, depth + 1))
                if len(final_papers) >= max_total_papers:
                    break

    logger.info("Deep research complete. Found %d papers.", len(final_papers))
    return list(final_papers.values())
