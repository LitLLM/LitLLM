---
name: litllm
description: Generate a related-work section, find prior literature, or rank candidate citations for an academic paper. Uses LLM-driven debate ranking over Semantic Scholar / arXiv / OpenAlex search, with optional citation-graph BFS expansion (deep research). Use when the user provides a paper (PDF or markdown) and asks for related work, citations, prior art, or a literature review.
argument-hint: [paper-path]
---

# litllm — Literature Search & Related-Work Generation

Standalone Claude Code skill backed by a pip-installable Python CLI. Speaks any
OpenAI-compatible LLM endpoint (OpenAI, vLLM, Ollama, OpenRouter, Together,
Groq, LM Studio, ...).

## When to invoke

Activate this skill when the user:
- shares a PDF or markdown paper and asks for "related work", "prior art",
  "citations", "literature review", "what's been done on this"
- asks to "rank these candidate citations against my paper"
- asks for "deep research" / "expand the citation graph" / "find references of
  references"
- asks to "extract the bibliography from this PDF"

Do **not** invoke for:
- single-paper summarization (use a generic summarization skill)
- writing the related-work *prose* (use a writing skill — this skill produces
  ranked candidates + summaries, not the polished section)

## Setup (one-time, per machine)

```bash
curl -sSL https://raw.githubusercontent.com/LitLLM/LitLLM/main/skill/install.sh | bash
```

That single line drops `SKILL.md` into `~/.claude/skills/litllm/` and installs
the `litllm` CLI on PATH. The package is named `litllm-skill` on PyPI to
disambiguate from the popular `litellm` package.

Set env vars (any OpenAI-compatible endpoint works):

```bash
export LITLLM_API_KEY="sk-..."
export LITLLM_BASE_URL="https://api.openai.com/v1"   # default
export LITLLM_MODEL="gpt-4o-mini"                    # default
export LITLLM_CONTACT_EMAIL="you@example.com"        # OpenAlex/S2 politeness
export LITLLM_S2_API_KEY="..."                       # optional, higher S2 rate
```

For embedding-based deep research:

```bash
pip install 'litllm-skill[embeddings]'
```

## Commands

```bash
# Full 4-step pipeline: keywords → fetch → rank+filter → summarize
litllm related-work paper.pdf --out ./out

# Citation-graph expansion (BFS, depth 2, ≤1000 papers)
litllm related-work paper.pdf --deep-research --selection-mode abstract

# Pick the search backend
litllm related-work paper.pdf --api arxiv          # arxiv | openalex | semanticscholar

# Tune ranking
litllm related-work paper.pdf --ranking-threshold 70 --limit-per-query 15

# Single-step utilities
litllm keywords paper.pdf                # → JSON list of search queries
litllm rank paper.pdf --candidates c.json   # → debate-ranking output with scores
litllm bib paper.pdf                     # → JSON list of cited paper titles
```

## Output

`related-work` writes a phased tree to `--out`. Each step is cached so re-runs
skip completed work.

```
out/
├── 1_generated_queries.md      # JSON list of search queries
├── 2_fetched_papers.md         # JSON of search results (post-dedup)
├── 3_ranked_papers.md          # Debate-ranking arguments + scores per paper
├── 3.5_filtered_papers.md      # JSON of papers above ranking threshold
├── 3.5_bibfile.md              # Filtered candidates as @article BibTeX
└── 4_related_papers_summary.md # Concatenated per-paper summaries
```

## How it works (so you can explain the output)

1. **Keyword extraction** — LLM reads the paper and emits 8 diverse search queries.
2. **Fetch** — runs queries against Semantic Scholar (default), arXiv, or OpenAlex.
   With `--deep-research`, BFS-walks the citation graph (depth 2, ≤1000 papers),
   picking expansions per `--selection-mode`:
   - `abstract` (default): batched LLM debate ranking on abstracts, ≥70/100 keeps
   - `full-text`: per-candidate LLM full-text comparison, ≥70/100 keeps
   - `embedding`: SPECTER cosine similarity ≥0.80, top-10 keeps
3. **Rank** — batched debate ranking of fetched papers vs. the query paper.
   Each candidate gets arguments-for, arguments-against, and a 0-100 score.
4. **Filter & summarize** — drops candidates below `--ranking-threshold`,
   downloads the survivors' PDFs, summarizes each in parallel.

## Selection-mode trade-offs

| Mode | Cost | Quality | Notes |
|------|------|---------|-------|
| `abstract` | Low | Good | Default. LLM sees only title+abstract. |
| `full-text` | High | Best | Downloads candidate PDFs. Slow, accurate. |
| `embedding` | One-time GPU | Fast | Local SPECTER model; no per-candidate API cost. |

## Related skills

- `huggingface-papers` — fetch individual paper metadata; pairs well as a
  pre-step ("look up this arXiv ID, then run litllm on its PDF")
- `agent-research-skills/literature-search` — alternative with heuristic
  ranking (citations + recency + venue), no LLM debate
