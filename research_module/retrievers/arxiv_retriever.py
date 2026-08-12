import logging
import re
import time
from typing import Dict, List, Union

import arxiv


logger = logging.getLogger(__name__)

_MAX_QUERY_LENGTH = 200
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0

_BAD_CHARS = re.compile(r"""[^\w\s"']+""", re.UNICODE)


def _normalize_queries(
    query: Union[str, List[str], None],
) -> List[str]:
    if query is None:
        return []

    if isinstance(query, (list, tuple)):
        lines = []

        for item in query:
            if item is not None:
                lines.extend(str(item).splitlines())
    else:
        lines = str(query).splitlines()

    queries = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^[*\-\u2022]+\s*",
            "",
            line,
        )

        line = re.sub(
            r"^\d+[\.)]\s*",
            "",
            line,
        )

        line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        if not line:
            continue

        key = line.lower()

        if key in seen:
            continue

        seen.add(key)
        queries.append(line)

    return queries


def _sanitize_query(query: str) -> str:
    query = _BAD_CHARS.sub(" ", query)
    query = re.sub(r"\s+", " ", query).strip()

    if len(query) > _MAX_QUERY_LENGTH:
        query = query[:_MAX_QUERY_LENGTH].rsplit(
            " ",
            1,
        )[0]

    return query


def _search_one(
    client: arxiv.Client,
    query: str,
    max_results: int,
) -> List[Dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for paper in client.results(search):
        papers.append(
            {
                "title": paper.title,
                "content": paper.summary,
                "url": paper.entry_id,
            }
        )

    return papers


def _search_with_retry(
    client: arxiv.Client,
    query: str,
    max_results: int,
) -> List[Dict]:
    delay = _RETRY_DELAY

    for attempt in range(_MAX_RETRIES):
        try:
            return _search_one(
                client,
                query,
                max_results,
            )

        except Exception as exc:
            message = str(exc)

            transient = any(
                error in message
                for error in (
                    "500",
                    "502",
                    "503",
                    "504",
                    "timed out",
                    "Connection",
                )
            )

            if not transient or attempt == _MAX_RETRIES - 1:
                raise

            logger.warning(
                "arXiv request failed. "
                "Retrying in %.1fs: %s",
                delay,
                exc,
            )

            time.sleep(delay)
            delay *= 2

    return []


def search_arxiv(
    query: Union[str, List[str]],
    max_results: int = 5,
) -> List[Dict]:
    queries = _normalize_queries(query)

    if not queries:
        logger.warning(
            "No valid arXiv queries were provided."
        )
        return []

    client = arxiv.Client()

    papers = []
    seen_urls = set()

    for index, raw_query in enumerate(queries):
        query = _sanitize_query(raw_query)

        if not query:
            continue

        try:
            results = _search_with_retry(
                client,
                query,
                max_results,
            )

            for paper in results:
                url = paper.get("url")

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)
                papers.append(paper)

            logger.info(
                "arXiv query %d/%d returned %d papers.",
                index + 1,
                len(queries),
                len(results),
            )

            if index < len(queries) - 1:
                time.sleep(1)

        except Exception as exc:
            logger.error(
                "arXiv search failed for query '%s': %s",
                query[:60],
                exc,
            )

    logger.info(
        "arXiv returned %d unique papers.",
        len(papers),
    )

    return papers