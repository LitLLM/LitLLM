"""Tests for PDF download URL resolution and pdf_url extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from litllm.fetcher import (
    PaperFetcher,
    _download_pdf,
    _normalize_openalex_paper,
    _resolve_arxiv_by_title,
)


def _fake_response(status: int, body: bytes, content_type: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    resp.iter_content = MagicMock(return_value=[body])
    if status >= 400:
        from requests import HTTPError
        resp.raise_for_status.side_effect = HTTPError(f"{status} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_download_pdf_writes_valid_pdf(tmp_path):
    target = tmp_path / "x.pdf"
    body = b"%PDF-1.4\nfake content\n%%EOF\n"
    with patch("litllm.fetcher.requests.get", return_value=_fake_response(200, body, "application/pdf")):
        assert _download_pdf("http://example/x.pdf", str(target))
    assert target.read_bytes() == body


def test_download_pdf_rejects_html_response(tmp_path):
    target = tmp_path / "x.pdf"
    body = b"<html>oops</html>"
    with patch("litllm.fetcher.requests.get", return_value=_fake_response(200, body, "text/html")):
        assert not _download_pdf("http://example/x", str(target))
    assert not target.exists()


def test_download_pdf_rejects_invalid_magic(tmp_path):
    """Some endpoints return content-type:pdf but actually deliver garbage."""
    target = tmp_path / "x.pdf"
    body = b"<html>fake-pdf-claim</html>"
    with patch("litllm.fetcher.requests.get", return_value=_fake_response(200, body, "application/pdf")):
        assert not _download_pdf("http://example/x", str(target))
    assert not target.exists()  # invalid file is cleaned up


def test_download_pdf_handles_4xx(tmp_path):
    target = tmp_path / "x.pdf"
    with patch("litllm.fetcher.requests.get", return_value=_fake_response(404, b"", "text/html")):
        assert not _download_pdf("http://example/x", str(target))


def test_normalize_openalex_extracts_oa_url():
    paper = {
        "id": "https://openalex.org/W123",
        "title": "Sample",
        "publication_date": "2024",
        "open_access": {"oa_url": "https://example.org/sample.pdf"},
    }
    out = _normalize_openalex_paper(paper)
    assert out["pdf_url"] == "https://example.org/sample.pdf"


def test_normalize_openalex_no_oa_url():
    paper = {"id": "https://openalex.org/W123", "title": "T", "publication_date": "2024"}
    out = _normalize_openalex_paper(paper)
    assert out["pdf_url"] is None


def test_fetch_pdfs_uses_pdf_url_first(tmp_path):
    fetcher = PaperFetcher(cache_dir=str(tmp_path / "c"))
    body = b"%PDF-1.4 stub\n%%EOF\n"
    with patch("litllm.fetcher.requests.get", return_value=_fake_response(200, body, "application/pdf")) as g:
        out = fetcher.fetch_pdfs(
            [{"doi": "10.x/abc", "pdf_url": "https://oa.example/abc.pdf", "title": "X"}],
            resolve_title_fallback=False,
        )
    # _safe_filename strips '/' rather than replacing it.
    assert out == {"10.x/abc": str(tmp_path / "c" / "pdfs" / "10.xabc.pdf")}
    g.assert_called_once()
    assert g.call_args.args[0] == "https://oa.example/abc.pdf"


def test_fetch_pdfs_falls_back_to_arxiv_when_pdf_url_fails(tmp_path):
    fetcher = PaperFetcher(cache_dir=str(tmp_path / "c"))
    body = b"%PDF-1.4 stub\n%%EOF\n"
    bad = _fake_response(404, b"", "text/html")
    good = _fake_response(200, body, "application/pdf")
    with patch("litllm.fetcher.requests.get", side_effect=[bad, good]):
        out = fetcher.fetch_pdfs(
            [{"arxiv_id": "1234.5678", "pdf_url": "https://broken/x.pdf", "title": "X"}],
            resolve_title_fallback=False,
        )
    assert "1234.5678" in out


def test_fetch_pdfs_caches_existing_files(tmp_path):
    cache = tmp_path / "c" / "pdfs"
    cache.mkdir(parents=True)
    (cache / "1234.5678.pdf").write_bytes(b"%PDF-1.4 cached\n%%EOF\n")
    fetcher = PaperFetcher(cache_dir=str(tmp_path / "c"))
    with patch("litllm.fetcher.requests.get") as g:
        out = fetcher.fetch_pdfs([{"arxiv_id": "1234.5678", "title": "X"}])
    assert "1234.5678" in out
    g.assert_not_called()


def test_fetch_pdfs_skips_papers_without_id(tmp_path):
    fetcher = PaperFetcher(cache_dir=str(tmp_path / "c"))
    out = fetcher.fetch_pdfs([{"title": "no ids"}])
    assert out == {}


def test_resolve_arxiv_by_title_uses_arxiv_search():
    fake_result = MagicMock()
    fake_result.get_short_id.return_value = "2024.0001"
    fake_search = MagicMock()
    fake_search.results.return_value = iter([fake_result])
    with patch("litllm.fetcher.arxiv.Search", return_value=fake_search):
        assert _resolve_arxiv_by_title("Some title") == "2024.0001"


def test_resolve_arxiv_by_title_handles_empty_results():
    fake_search = MagicMock()
    fake_search.results.return_value = iter([])
    with patch("litllm.fetcher.arxiv.Search", return_value=fake_search):
        assert _resolve_arxiv_by_title("nonexistent") is None
