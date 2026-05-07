"""Prompt templates for litllm components.

Ported from src/prompts/structures.py and src/prompts/litllm/default.py in
the original reviewertoo codebase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPair:
    """A system+user prompt pair."""

    system: str
    user: str


@dataclass(frozen=True)
class LitLLMPrompts:
    """All prompts needed by the litllm pipeline."""

    keyword_extraction: PromptPair
    debate_ranking: PromptPair
    bibliography_locator: PromptPair
    bibliography_extraction: PromptPair
    full_text_selection: PromptPair
    semantic_relevance: PromptPair
    title_validator: PromptPair
    summary: PromptPair
    keyword_extraction_v2: PromptPair | None = None
    keyword_extraction_v3: PromptPair | None = None
    query_translator: PromptPair | None = None


# A short paper-summary prompt used by the final step of the composite pipeline.
# (Original reviewertoo COMPOSITE_PROMPTS.summary, distilled.)
_SUMMARY = PromptPair(
    system="You are a careful research assistant who summarizes academic papers concisely and faithfully.",
    user="""Summarize the following paper in 4-6 bullet points covering:
- The problem it addresses
- The key contribution / method
- The main experimental setup or evidence
- The headline result
- Any important limitations
- One-sentence relevance for someone writing a related-work section

{paper_text}
""",
)


LITLLM_PROMPTS = LitLLMPrompts(
    keyword_extraction=PromptPair(
        system="You are a helpful research assistant who is helping with literature review of a research idea.",
        user="""You are a helpful research assistant helping with writing a literature review for a research idea. You will be given a text of a scientific paper or a research idea. Your goal is to generate a diverse set of mutually exclusive search queries to help find directly relevant and citable papers using academic search engines.

Here is the text:
{paper_text}

## Instructions:
*   Generate 8 search queries, none of which should be more than 5 keywords. Please write the query in a similar fashion as a human would use search engine.
*   The queries should capture the main focus of the text.
*   Please make sure to generate different search queries using a variety of key words so as to get maximum papers that could be cited.
*   Use a variety of keywords and phrasings to maximize diversity and avoid redundancy.
*   Focus on maximizing recall (retrieving a broad but relevant set of papers).
*   In addition to the queries, also provide a reasoning for the generated queries.
*   The reasoning should explain how each query targets a different facet of the research (e.g., methodology, application, problem domain) and why it is distinct from the other queries.
*   Extract the relevant sentences from the paper(s) that justify your reasoning.
*   Put the extracted sentences in quotes and put them at the end of each of your reasonings.
*   Please return a JSON with a key "queries" that has the list of queries and a "reasoning" key that has the reasoning for the queries.
*   Do not generate anything else apart from the JSON.

## Example Output:

```json
{{
  "queries": [
    "query 1",
    "query 2",
    "query 3",
    "query 4",
    "query 5",
    "query 6",
    "query 7",
    "query 8"
  ],
  "reasoning": "The queries are designed to capture the main focus of the text. The first query focuses on the main method discussed, the second query captures the application of the method, and the third query is more general to capture any related work."
}}
```

## Response:""",
    ),
    keyword_extraction_v2=PromptPair(
        system="You are a helpful research assistant who is helping with literature review of a research idea.",
        user="""You are a helpful research assistant assisting with a literature review for a research idea. You will be given the abstract of a scientific paper. Your goal is to generate a diverse set of mutually exclusive search queries to help find relevant and citable academic papers using scholarly search engines.

Here is the abstract:
{paper_text}

## Instructions:
* Generate 10 search queries written in the natural, concise style typically used by researchers using academic search engines (e.g., OpenAlex, Semantic Scholar, or Google Scholar).
* Each query should reflect a **different angle** of the abstract (e.g., method, task, dataset, domain, application, or novelty).
* Use a **variety of keywords and phrasings** to maximize diversity and avoid redundancy.
* Focus on **maximizing recall** (retrieving a broad but relevant set of papers), not just precision.
* Avoid stopwords or overly verbose phrasing; use terms that researchers would actually search for.
* Return a JSON object with the following structure:

```json
{{
  "queries": [
    "first query here",
    "second query here"
  ]
}}
```""",
    ),
    keyword_extraction_v3=PromptPair(
        system="You are a helpful research assistant who is helping with literature review of a research idea.",
        user="""You will be provided with an abstract of a scientific document. Your task is to extract the 4-word core research concept. Then, generate 3 related search queries based on that concept for the Semantic Scholar API. Your response should be structured in a JSON format, with a 'core-concept' key and a 'queries' key. Do not generate anything other than the JSON in your response.

Abstract: {paper_text}

### Response:
""",
    ),
    query_translator=PromptPair(
        system="""You are an expert research assistant who specializes in translating high-level research topics into keyword queries for the OpenAlex API.

Your goal is to **maximize recall** — generate simple, unquoted, and loosely structured keyword-based queries that return as many relevant results as possible.

Do NOT over-refine. Do NOT assume OpenAlex understands Boolean logic, semantic meaning, or full-text content.""",
        user="""You will be given a search query. Your task is to split it into multiple simplified versions to reduce complexity:

1. If the original query contains any boolean operators, consider splitting them into multiple queries
2. Avoid any Boolean logic (like `OR`, `AND`, parentheses).
3. Keep the query **broad and loose** — 3 to 5 terms is ideal.
4. Do NOT use quotes or special characters.

### Final Output Format:
Return a single, top-level JSON object with a single key: `simplified_queries`.

```json
{{
  "simplified_queries": [
    "your first query",
    "your second query",
    "your third query"
  ]
}}
```

### Your Response:
""",
    ),
    debate_ranking=PromptPair(
        system="You are a helpful research assistant who is helping with literature review of a research idea.",
        user="""You are a helpful research assistant. Your task is to rank some papers based on their relevance to a query paper.

Given the query paper:
<query_paper>
{query_paper}
</query_paper>

And the following candidate reference paper abstracts:
<candidate_paper_abstracts>
{reference_papers}
</candidate_paper_abstracts>

## Instructions:
* For EVERY candidate paper, provide a relevance score between 0 and 100 representing the probability that the query paper would cite it.
* The relevance score MUST be an integer between 0 and 100 (inclusive).
* The score MUST be written using digits only (e.g., 0, 17, 42, 50, 100).
* Do NOT write numbers in words (e.g., "thirty-seven" or "fifty").
* If a candidate paper happens to be a duplicate of the query paper, it should receive a score of 0.
* Provide arguments for and against citing the candidate paper, extracting supporting sentences from the candidate's abstract.
* Format your response for EACH paper inside using the specified tags below.

### Response Format for EACH paper:
<arguments_for>
[paper's id]: [Arguments for including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_for>
<arguments_against>
[paper's id]: [Arguments for not including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_against>
<probability>
paper_id: [paper's id]
score: [Final Probability Score Based on the Arguments]/100
</probability>

Note: your response MUST contain arguments and probabilities for ALL the candidate paper abstracts.

## Your Response:
""",
    ),
    bibliography_locator=PromptPair(
        system="You are a text-processing utility. Your only function is to extract and locate text from the bibliography section of a scientific paper.",
        user="""The following is some text from a scientific paper. Find the section header for the bibliography (e.g., "References", "Bibliography", "Works Cited") and return the text of the bibliography section. Do not add any other text, explanation, or formatting.

**Paper Text:**
<paper_text>
{paper_text}
</paper_text>
""",
    ),
    bibliography_extraction=PromptPair(
        system="You are an expert assistant specializing in parsing academic bibliographies. Your task is to extract all references from the provided bibliography text and format them as a JSON list. You must handle various citation formats gracefully and only output a valid JSON object.",
        user="""Please parse the following bibliography text and return a JSON list where each object has the key 'title'. Ignore entries that do not appear to be valid academic papers or books.

**Bibliography Text:**
```
{bibliography_text}
```

**Instructions:**
- Extract the title of each publication.
- Do not include authors, year, or publication venue in the title field.
- If you cannot parse an entry, skip it.
- Your entire response MUST be a single, valid JSON object containing a list under the key "references".

**Example Output:**

```json
{{
  "references": [
    {{"title": "Bert: Pre-training of deep bidirectional transformers for language understanding"}},
    {{"title": "Attention is all you need"}}
  ]
}}
```

**Your JSON Response:**
""",
    ),
    title_validator=PromptPair(
        system="""You are a meticulous validation assistant. Your task is to determine if two paper titles refer to the same publication. One title may be a revised version, truncated, or have minor formatting differences.

Respond ONLY with a valid JSON object in the format: {{"match": true}} or {{"match": false}}.""",
        user="""Please determine if these two titles refer to the same paper.

Title A: "{title_a}"
Title B: "{title_b}"

**Instructions:**
- Respond with a valid JSON object indicating whether it was a match or not.
- ONLY include a JSON in your response. Do NOT include anything else.

**Example response:**

```json
{{"match": true}}
```

**Your response:**
""",
    ),
    semantic_relevance=PromptPair(
        system="""You are an expert academic research assistant. You will be shown a query paper and a candidate paper and your task is to analyze the semantic relevance of the query paper and the candidate paper. Assess how relevant the candidate paper is to the subject matter, research scope, and focus of the query paper. Consider topical overlap, methodological similarity, shared objectives, and whether the candidate contributes meaningfully to the themes of the query paper.""",
        user="""You are an expert academic research assistant.

**Input:**
Query Paper Details:
Title: {query_title}
Abstract: {query_abstract}
Full Paper:
{query_full}

Candidate Paper:
Title: {candidate_title}
Abstract: {candidate_abstract}

**Instructions:**
Analyze the semantic relevance of the query paper and the candidate paper. Assess how relevant the candidate paper is to the subject matter, research scope, and focus of the query paper.

Task Rubric:
* 5 (Direct Correspondence): Candidate directly addresses the same research problem as the query paper.
* 4 (Primary Topical Focus): Candidate's central theme is closely related to the query paper.
* 3 (Substantial Topical Coverage): Candidate covers significant aspects of the query paper's domain.
* 2 (Peripheral Topical Treatment): Candidate addresses the query paper's subject as a secondary element.
* 1 (Tangential Relevance): Minimal substantive overlap.
* 0 (No Substantive Relevance): Candidate is from a different domain or research area.

**Output Format:**
```json
{{
 "paper_to_paper_relevance": {{
   "relevanceScore": 0,
   "confidenceLevel": 0,
   "summaryStatement": "..."
 }}
}}
```
""",
    ),
    full_text_selection=PromptPair(
        system="You are an expert research assistant. Your task is to perform a deep, full-text comparison between two papers and assess relevance for citation.",
        user="""You are deciding whether a candidate paper is relevant enough to the main paper to warrant further exploration of its own references.

**Main Paper (Full Text):**
<query_paper>
{query_paper}
</query_paper>

**Candidate Paper (Full Text):**
<candidate_paper>
{candidate_paper}
</candidate_paper>

## Instructions:
1.  Analyze both full texts, paying close attention to methodology, datasets, and key results.
2.  Provide a relevance score between 0 and 100 representing the probability that the main paper would cite the candidate paper.
3.  Provide arguments for and against citing the paper, extracting supporting sentences from the candidate's full text.
4.  Format your response using the specified tags.

### Response Format:
<arguments_for>
[Reason for including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_for>
<arguments_against>
[Reason for not including the paper]
Extracted Sentences: "Sentence 1", "Sentence 2", ...
</arguments_against>
<probability>
[Final Probability Score Based on the Arguments]
</probability>""",
    ),
    summary=_SUMMARY,
)
