"""
arxiv_retriever.py

Robust arXiv search that:
- accepts a single query string or a list of queries
- cleans bulleted / numbered LLM output before sending
- sanitizes queries (arXiv 500s on certain punctuation / very long inputs)
- retries with exponential backoff on transient errors
- merges & de-duplicates results across sub-queries
"""

import re
import time
import logging
from typing import List, Dict, Union

import arxiv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_queries(query: Union[str, List[str], None]) -> List[str]:
    """Split a string/list/blob into clean, individual query strings."""
    if query is None:
        return []

    if isinstance(query, (list, tuple)):
        raw_lines = []
        for item in query:
            if item is None:
                continue
            raw_lines.extend(str(item).splitlines())
    else:
        raw_lines = str(query).splitlines()

    cleaned = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading bullets ("* ", "- ", "• ")
        line = re.sub(r"^[\*\-\u2022]+\s*", "", line)
        # Strip leading numbering ("1. ", "1) ")
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Collapse internal whitespace
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line)

    # Dedupe (case-insensitive) preserving order
    seen, unique = set(), []
    for q in cleaned:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            unique.append(q)
    return unique


# arXiv treats most punctuation as query syntax. Strip anything that isn't
# a word char, whitespace, or quotes. Also cap length to avoid HTTP 500s on
# huge LLM-generated blobs.
_ARXIV_BAD_CHARS = re.compile(r"[^\w\s\"']+", re.UNICODE)
_MAX_ARXIV_QUERY_LEN = 200


def _sanitize_for_arxiv(q: str) -> str:
    q = _ARXIV_BAD_CHARS.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > _MAX_ARXIV_QUERY_LEN:
        q = q[:_MAX_ARXIV_QUERY_LEN].rsplit(" ", 1)[0]
    return q


def _search_one(client: arxiv.Client, query: str, max_results: int) -> List[Dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for paper in client.results(search):
        papers.append({
            "title": paper.title,
            "content": paper.summary,
            "url": paper.entry_id,
        })
    return papers


def _search_with_backoff(
    client: arxiv.Client, query: str, max_results: int, max_retries: int = 3
) -> List[Dict]:
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return _search_one(client, query, max_results)
        except Exception as e:
            msg = str(e)
            transient = any(
                code in msg
                for code in ("500", "502", "503", "504", "timed out", "Connection")
            )
            if not transient or attempt == max_retries:
                raise
            logger.warning(
                f"⏳ arXiv transient error (attempt {attempt}/{max_retries}) — "
                f"sleeping {delay:.1f}s: {e}"
            )
            time.sleep(delay)
            delay *= 2
    return []


def search_arxiv(query: Union[str, List[str]], max_results: int = 5) -> List[Dict]:
    """
    Search arXiv for one or many queries. Results are merged and de-duplicated
    by URL. Failures on individual sub-queries are logged but do not abort.
    """
    queries = _normalize_queries(query)
    if not queries:
        logger.warning("⚠️ Arxiv: no valid queries after normalization.")
        return []

    client = arxiv.Client()
    all_papers, seen_urls = [], set()

    for idx, raw_q in enumerate(queries):
        q = _sanitize_for_arxiv(raw_q)
        if not q:
            logger.info(f"↪️ Arxiv [{idx+1}/{len(queries)}] skipped (empty after sanitize)")
            continue
        try:
            papers = _search_with_backoff(client, q, max_results)
            for p in papers:
                url = p.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_papers.append(p)
            logger.info(f"✅ Arxiv [{idx+1}/{len(queries)}] '{q[:60]}' → {len(papers)} papers")
            # Be polite to the arXiv API between sub-queries
            if idx < len(queries) - 1:
                time.sleep(1.0)
        except Exception as e:
            logger.error(f"⚠️ Arxiv failed for query {idx+1} ('{q[:60]}'): {e}")
            continue

    logger.info(f"📚 Arxiv total: {len(all_papers)} unique papers from {len(queries)} query(ies).")
    return all_papers
