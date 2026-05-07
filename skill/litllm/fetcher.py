"""Paper search and PDF fetching across arXiv, OpenAlex, and Semantic Scholar.

Ported from src/services/paper_fetcher_service.py in the original reviewertoo
codebase, with these intentional differences:

- Drops the Serper (paid Google Scholar) backend.
- Drops the `extract_bibliography_with_llm` method — bibliography parsing
  moves into the composite orchestrator if needed.
- Configuration via env vars: LITLLM_CONTACT_EMAIL, LITLLM_S2_API_KEY.
- Cache lives at .litllm-cache/pdfs/ in the cwd by default.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from typing import Any, Callable
from urllib.parse import quote
from xml.etree import ElementTree

import arxiv
import numpy as np
import requests
import semanticscholar
import tqdm

logger = logging.getLogger(__name__)


# ---- Semantic Scholar pacing ---------------------------------------------------
# Free-tier S2 caps at ~1 RPS. Process-wide lock + monotonic timestamp serializes
# all S2 calls so concurrent pipelines don't trigger TCP RST (seen as
# ConnectionRefusedError in tenacity RetryError).

_S2_PACE_LOCK = threading.Lock()
_S2_LAST_CALL_TS = 0.0
_S2_MIN_INTERVAL_SEC = float(os.getenv("LITLLM_S2_MIN_INTERVAL_SEC", "1.05"))


def _s2_throttle() -> None:
    global _S2_LAST_CALL_TS
    with _S2_PACE_LOCK:
        wait = _S2_LAST_CALL_TS + _S2_MIN_INTERVAL_SEC - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _S2_LAST_CALL_TS = time.monotonic()


# ---- Helpers ------------------------------------------------------------------


def sanitize_title(title: str | None) -> str | None:
    if not title:
        return None
    return title.translate(str.maketrans("", "", ".:'\",()!?"))


def _safe_filename(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    return "".join(c for c in paper_id if c.isalnum() or c in "-_.").strip()


def _make_request_with_backoff(
    request_func: Callable[[], requests.Response],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 0.1,
) -> requests.Response:
    delay = initial_delay
    last_exception: Exception | None = None
    for retry in range(max_retries + 1):
        try:
            response = request_func()
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_exception = e
            if retry >= max_retries:
                raise
            if getattr(e, "response", None) is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except (ValueError, TypeError):
                        pass
            jitter_amount = random.uniform(-jitter, jitter) * delay
            actual_delay = min(delay + jitter_amount, max_delay)
            logger.warning(f"Request failed ({retry + 1}/{max_retries + 1}): {e}. Retrying in {actual_delay:.2f}s.")
            time.sleep(actual_delay)
            delay = min(delay * backoff_factor, max_delay)
    if last_exception:
        raise last_exception
    raise RuntimeError("unreachable")


def _process_abstract_inverted_index(idx: dict | None) -> str:
    """Turn OpenAlex's word→positions inverted index back into a readable abstract."""
    if not idx or not isinstance(idx, dict):
        return "Abstract not available"
    word_positions: list[tuple[int, str]] = []
    for word, positions in idx.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)


def _download_pdf(url: str, save_path: str) -> bool:
    """Download a PDF from a URL with content-type and magic-bytes validation."""
    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("PDF download failed for %s: %s", url, e)
        return False

    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and "octet-stream" not in content_type:
        logger.warning("Non-PDF response from %s (content-type=%r); skipping.", url, content_type)
        return False

    try:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with open(save_path, "rb") as f:
            magic = f.read(5)
        if magic != b"%PDF-":
            logger.warning("Downloaded file from %s is not a valid PDF (magic=%r); deleting.", url, magic)
            os.remove(save_path)
            return False
        return True
    except OSError as e:
        logger.warning("Failed to write PDF from %s: %s", url, e)
        return False


def _resolve_arxiv_by_title(title: str) -> str | None:
    """Search arXiv for a paper by title; return its short_id if found.

    Used as a fallback when a paper has only a DOI/OpenAlex ID and no PDF URL.
    Mirrors the title-fallback in src/services/paper_fetcher_service.py:fetch_paper_from_arxiv.
    """
    sanitized = sanitize_title(title)
    if not sanitized:
        return None
    try:
        search = arxiv.Search(
            query=f'ti:"{sanitized}"',
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in search.results():
            return result.get_short_id()
    except Exception as e:
        logger.warning("arXiv title lookup failed for %r: %s", title, e)
    return None


# ---- Backend: arXiv -----------------------------------------------------------


def _search_arxiv(queries: list[str], limit_per_query: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for query in tqdm.tqdm(queries, desc="Searching arXiv"):
        try:
            search = arxiv.Search(
                query=query,
                max_results=limit_per_query,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for r in search.results():
                results.append(
                    {
                        "title": r.title,
                        "arxiv_id": r.get_short_id(),
                        "doi": r.doi,
                        "abstract": r.summary.replace("\n", " ").strip(),
                        "openalex_id": None,
                        "publication_date": str(r.published),
                        "authors": [a.name for a in r.authors],
                        "pdf_url": r.pdf_url,
                    }
                )
        except Exception as e:
            logger.error(f"arXiv search for '{query}' failed: {e}")
    return results


# ---- Backend: OpenAlex --------------------------------------------------------


def _fetch_from_openalex(
    query: str | dict | None,
    email: str | None = None,
    limit: int = 1,
    openalex_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search or fetch a single paper from OpenAlex."""
    base_url = "https://api.openalex.org/works"
    headers = {"User-Agent": f"mailto:{email}"} if email else {}
    params: dict[str, Any]

    if openalex_id:
        base_url += f"/{quote(openalex_id)}"
        params = {}
    elif isinstance(query, str):
        sanitized = sanitize_title(query)
        if not sanitized:
            return []
        params = {"search": sanitized, "sort": "relevance_score:desc", "per_page": limit}
    elif isinstance(query, dict):
        params = {"search": query.get("search") or "", "sort": "relevance_score:desc", "per_page": limit}
    else:
        return []

    try:
        response = _make_request_with_backoff(lambda: requests.get(base_url, params=params, headers=headers))
    except requests.RequestException as e:
        logger.error(f"OpenAlex request failed: {e}")
        return []

    data = response.json()
    if "meta" in data:
        if data["meta"]["count"] == 0:
            return []
        results = data.get("results", [])
    elif data.get("title"):
        return [_normalize_openalex_paper(data)]
    else:
        return []

    return [_normalize_openalex_paper(p) for p in results]


def _normalize_openalex_paper(paper: dict[str, Any]) -> dict[str, Any]:
    arxiv_id: str | None = None
    oa_url = paper.get("open_access", {}).get("oa_url") or ""
    if "arxiv" in paper.get("indexed_in", []):
        if paper.get("ids", {}).get("arxiv"):
            arxiv_id = paper["ids"]["arxiv"].split("/")[-1]
        elif "arxiv.org" in oa_url:
            arxiv_id = oa_url.split("/")[-1]
    oa_id = paper.get("id", "")
    return {
        "title": paper.get("title"),
        "arxiv_id": arxiv_id,
        "publication_date": paper.get("publication_date"),
        "openalex_id": oa_id.split("/")[-1] if oa_id else None,
        "doi": paper.get("doi"),
        "cited_by_count": paper.get("cited_by_count"),
        "relevance_score": paper.get("relevance_score"),
        "abstract": _process_abstract_inverted_index(paper.get("abstract_inverted_index")),
        "referenced_works": paper.get("referenced_works", []),
        "related_works": paper.get("related_works", []),
        "pdf_url": oa_url or None,
    }


# ---- Backend: Semantic Scholar -----------------------------------------------


def _search_semantic_scholar(
    queries: list[str], limit_per_query: int, api_key: str | None = None
) -> list[dict[str, Any]]:
    s2 = semanticscholar.SemanticScholar(api_key=api_key, timeout=30, retry=False)
    results: list[dict[str, Any]] = []
    for query in tqdm.tqdm(queries, desc="Searching Semantic Scholar"):
        for _ in range(5):
            try:
                _s2_throttle()
                items = s2.search_paper(query, limit=limit_per_query)
                for ix, item in enumerate(items):
                    if ix > limit_per_query:
                        break
                    item = dict(item)
                    ext = item.get("externalIds") or {}
                    open_access_pdf = item.get("openAccessPdf") or {}
                    results.append(
                        {
                            "title": item.get("title"),
                            "arxiv_id": ext.get("ArXiv"),
                            "doi": ext.get("DOI"),
                            "abstract": item.get("abstract"),
                            "openalex_id": None,
                            "publication_date": str(item.get("publicationDate")),
                            "authors": item.get("authors") or [],
                            "bibid": (item.get("title") or "").lower().replace(" ", "_"),
                            "pdf_url": open_access_pdf.get("url") or None,
                        }
                    )
                break
            except Exception as e:
                logger.warning(f"S2 search for '{query}' failed: {e}. Retrying.")
                time.sleep(5)
    return results


# ---- Public API ---------------------------------------------------------------


class PaperFetcher:
    """Search and download academic papers, with on-disk PDF caching."""

    def __init__(
        self,
        email: str | None = None,
        s2_api_key: str | None = None,
        cache_dir: str = ".litllm-cache",
    ):
        self.email = email or os.getenv("LITLLM_CONTACT_EMAIL")
        self.s2_api_key = s2_api_key or os.getenv("LITLLM_S2_API_KEY") or os.getenv("S2_API_KEY")
        self.pdf_cache_path = os.path.join(cache_dir, "pdfs")
        os.makedirs(self.pdf_cache_path, exist_ok=True)

    def search_papers(
        self,
        queries: list[str],
        api: str = "semanticscholar",
        limit_per_query: int = 10,
        openalex_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run `queries` against `api` and return de-duplicated paper records."""
        if openalex_ids:
            all_papers: list[dict[str, Any]] = []
            for oid in tqdm.tqdm(openalex_ids, desc="Fetching OpenAlex papers"):
                all_papers.extend(_fetch_from_openalex(query=None, email=self.email, openalex_id=oid))
        elif api == "openalex":
            all_papers = []
            for q in tqdm.tqdm(queries, desc="Searching OpenAlex"):
                all_papers.extend(_fetch_from_openalex(q, email=self.email, limit=limit_per_query))
        elif api == "arxiv":
            all_papers = _search_arxiv(queries, limit_per_query)
        elif api == "semanticscholar":
            all_papers = _search_semantic_scholar(queries, limit_per_query, api_key=self.s2_api_key)
        else:
            raise ValueError(
                f"Unsupported search API: '{api}'. Use 'arxiv', 'openalex', or 'semanticscholar'."
            )

        return _dedupe_papers(all_papers)

    def fetch_pdfs(
        self,
        papers: list[dict[str, Any]],
        resolve_title_fallback: bool = True,
    ) -> dict[str, str]:
        """Download PDFs for the given papers (cache-aware).

        For each paper, tries URLs in priority order:
          1. paper["pdf_url"] — set by S2 openAccessPdf or OpenAlex oa_url
          2. https://arxiv.org/pdf/{arxiv_id} when arxiv_id is present
          3. arXiv title lookup (when resolve_title_fallback=True) — last-ditch
             attempt for DOI/OpenAlex-only papers that may have an arXiv version

        Returns {paper_id: local_path} where paper_id = arxiv_id | doi | openalex_id."""
        local_paths: dict[str, str] = {}
        for paper in tqdm.tqdm(papers, desc="Fetching PDFs"):
            paper_id = paper.get("arxiv_id") or paper.get("doi") or paper.get("openalex_id")
            safe = _safe_filename(paper_id)
            if not paper_id or not safe:
                continue
            pdf_path = os.path.join(self.pdf_cache_path, f"{safe}.pdf")
            if os.path.exists(pdf_path):
                local_paths[paper_id] = pdf_path
                continue

            urls: list[str] = []
            if paper.get("pdf_url"):
                urls.append(paper["pdf_url"])
            if paper.get("arxiv_id"):
                urls.append(f"https://arxiv.org/pdf/{paper['arxiv_id']}")

            # Last-ditch: resolve by title to find an arXiv version of the paper.
            if not urls and resolve_title_fallback and paper.get("title"):
                resolved_id = _resolve_arxiv_by_title(paper["title"])
                if resolved_id:
                    logger.info("Resolved %r to arXiv:%s by title lookup", paper["title"], resolved_id)
                    urls.append(f"https://arxiv.org/pdf/{resolved_id}")

            for url in urls:
                if _download_pdf(url, pdf_path):
                    local_paths[paper_id] = pdf_path
                    break
        return local_paths

    def get_referenced_works_from_semantic_scholar(
        self, paper_info: list[dict[str, Any]]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """For each paper with an arxiv_id, fetch its reference list from S2.
        Returns {"references": {arxiv_id: [{arxiv_id, title, abstract}, ...]}}."""
        api_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        fields = "references.title,references.externalIds,referenceCount,references.abstract"
        params = {"fields": fields}
        headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else {}

        arxiv_ids = [o["arxiv_id"] for o in paper_info]
        s2_paper_ids = [f"ARXIV:{aid}" for aid in arxiv_ids]
        s2_results: list[dict[str, Any]] = []

        batch_size = 50
        for i in tqdm.tqdm(
            range(0, len(s2_paper_ids), batch_size),
            desc="Fetching S2 references",
            total=int(np.ceil(len(s2_paper_ids) / batch_size)),
        ):
            batch = s2_paper_ids[i : i + batch_size]
            try:
                _s2_throttle()
                resp = requests.post(api_url, params=params, json={"ids": batch}, headers=headers, timeout=30)
                resp.raise_for_status()
                s2_results.extend(resp.json())
            except requests.RequestException as e:
                logger.warning(f"S2 batch references request failed: {e}")
                time.sleep(2)

        if not s2_results:
            return {"references": {}}

        references_dict: dict[str, list[dict[str, Any]]] = {}
        for i, result in enumerate(s2_results):
            if not result:
                continue
            refs = result.get("references") or []
            query_id = arxiv_ids[i]
            kept: list[dict[str, Any]] = []
            for ref in refs:
                ext = ref.get("externalIds") or {}
                if not ext.get("ArXiv"):
                    continue
                kept.append({"arxiv_id": ext["ArXiv"], "title": ref.get("title"), "abstract": ref.get("abstract")})
            if kept:
                references_dict[query_id] = kept
        return {"references": references_dict}

    def get_referenced_works_from_openalex(
        self, paper_info: dict[str, Any]
    ) -> dict[str, list[Any]]:
        """DOI-first, title-fallback lookup of referenced/related works."""
        doi = paper_info.get("doi")
        title = paper_info.get("title")

        if doi:
            try:
                response = _make_request_with_backoff(
                    lambda: requests.get("https://api.openalex.org/works", params={"filter": f"doi:{doi}"})
                )
                data = response.json()
                if data.get("results"):
                    work = data["results"][0]
                    return {
                        "referenced_works": work.get("referenced_works", []),
                        "related_works": work.get("related_works", []),
                    }
            except requests.RequestException as e:
                logger.warning(f"OpenAlex DOI lookup for {doi} failed: {e}")

        if title:
            try:
                metadata = _fetch_from_openalex(title, email=self.email, limit=1)
                if metadata:
                    work = metadata[0]
                    return {
                        "referenced_works": work.get("referenced_works", []),
                        "related_works": work.get("related_works", []),
                    }
            except requests.RequestException as e:
                logger.warning(f"OpenAlex title search for '{title}' failed: {e}")

        return {"referenced_works": [], "related_works": []}


def _dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-dupe by versioned-arxiv-id → doi → openalex_id → lowercase title."""
    seen: dict[str, dict[str, Any]] = {}
    for paper in papers:
        unique_id: str | None = None
        if paper.get("arxiv_id"):
            unique_id = re.sub(r"v\d+$", "", paper["arxiv_id"])
        if not unique_id:
            unique_id = paper.get("doi") or paper.get("openalex_id")
        if not unique_id and paper.get("title"):
            unique_id = paper["title"].lower()
        if unique_id and unique_id not in seen:
            seen[unique_id] = paper
    return list(seen.values())
