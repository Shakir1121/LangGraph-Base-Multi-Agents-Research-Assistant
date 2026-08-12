import logging
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from typing import Any, Dict, List

from research_module.retrievers.arxiv_retriever import search_arxiv
from research_module.retrievers.openalex_retriever import OpenAlexRetriever
from research_module.tools.tavily_tool import tavily_search


logger = logging.getLogger(__name__)

SEARCH_TIMEOUT = 12.0


class HybridRetriever:

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.openalex = OpenAlexRetriever()

    def search_papers(
        self,
        query: str,
        arxiv_limit: int = 20,
        openalex_limit: int = 20,
        tavily_limit: int = 10,
    ) -> Dict[str, Any]:

        results = {
            "arxiv_papers": [],
            "openalex_papers": [],
            "tavily_papers": [],
            "total_papers": 0,
            "sources_used": [],
        }

        start = time.monotonic()

        executor = ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        futures = {
            executor.submit(
                self._fetch_arxiv,
                query,
                arxiv_limit,
            ): "arxiv",

            executor.submit(
                self._fetch_openalex,
                query,
                openalex_limit,
            ): "openalex",

            executor.submit(
                self._fetch_tavily,
                query,
                tavily_limit,
            ): "tavily",
        }

        pending = set(futures)

        try:
            while pending:

                remaining = SEARCH_TIMEOUT - (
                    time.monotonic() - start
                )

                if remaining <= 0:
                    logger.warning(
                        "Hybrid retrieval timed out."
                    )
                    break

                done, pending = wait(
                    pending,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    source = futures[future]

                    try:
                        papers = future.result()

                    except Exception as exc:
                        logger.warning(
                            "%s retrieval failed: %s",
                            source,
                            exc,
                        )
                        continue

                    if not papers:
                        continue

                    results[
                        f"{source}_papers"
                    ] = papers

                    results["sources_used"].append(
                        source
                    )

                    logger.info(
                        "%s returned %d papers.",
                        source,
                        len(papers),
                    )

        finally:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

        results["total_papers"] = sum(
            len(results[key])
            for key in (
                "arxiv_papers",
                "openalex_papers",
                "tavily_papers",
            )
        )

        elapsed = time.monotonic() - start

        logger.info(
            "Hybrid retrieval finished in %.2fs: %d papers.",
            elapsed,
            results["total_papers"],
        )

        return results

    def _fetch_arxiv(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:

        try:
            papers = search_arxiv(
                query,
                max_results=limit,
            )

            return [
                self._normalize_paper(
                    paper,
                    "arxiv",
                )
                for paper in papers
            ]

        except Exception as exc:
            logger.warning(
                "arXiv retrieval failed: %s",
                exc,
            )
            return []

    def _fetch_openalex(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:

        try:
            papers = self.openalex.search_papers(
                query,
                limit=limit,
            )

            return [
                self._normalize_paper(
                    paper,
                    "openalex",
                )
                for paper in papers
            ]

        except Exception as exc:
            logger.warning(
                "OpenAlex retrieval failed: %s",
                exc,
            )
            return []

    def _fetch_tavily(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:

        try:
            response = tavily_search(query)

            if isinstance(response, dict):
                response = response.get(
                    "results",
                    [],
                )

            if not isinstance(response, list):
                return []

            papers = []

            for item in response[:limit]:

                if not isinstance(item, dict):
                    continue

                papers.append(
                    {
                        "title": item.get(
                            "title",
                            "",
                        ),
                        "abstract": item.get(
                            "content",
                            "",
                        ),
                        "url": item.get(
                            "url",
                            "",
                        ),
                        "source": "tavily",
                        "authors": [],
                        "publication_date": None,
                        "citation_count": 0,
                    }
                )

            return papers

        except Exception as exc:
            logger.warning(
                "Tavily retrieval failed: %s",
                exc,
            )
            return []

    @staticmethod
    def _normalize_paper(
        paper: Any,
        source: str,
    ) -> Dict[str, Any]:

        if source == "arxiv":
            return {
                "title": getattr(
                    paper,
                    "title",
                    "",
                ),
                "abstract": getattr(
                    paper,
                    "summary",
                    "",
                ),
                "url": getattr(
                    paper,
                    "entry_id",
                    "",
                ),
                "source": "arxiv",
                "authors": [
                    author.name
                    for author in getattr(
                        paper,
                        "authors",
                        [],
                    )
                ],
                "publication_date": str(
                    getattr(
                        paper,
                        "published",
                        "",
                    )
                ),
                "citation_count": 0,
            }

        if source == "openalex":
            return {
                "title": getattr(
                    paper,
                    "title",
                    "",
                ),
                "abstract": getattr(
                    paper,
                    "abstract",
                    "",
                ),
                "url": getattr(
                    paper,
                    "url",
                    "",
                ),
                "source": "openalex",
                "authors": [
                    author.display_name
                    for author in getattr(
                        paper,
                        "authors",
                        [],
                    )
                ],
                "publication_date": str(
                    getattr(
                        paper,
                        "publication_date",
                        "",
                    )
                ),
                "citation_count": getattr(
                    paper,
                    "citation_count",
                    0,
                ),
            }

        return paper


def hybrid_search(
    query: str,
    arxiv_limit: int = 20,
    openalex_limit: int = 20,
    tavily_limit: int = 10,
) -> Dict[str, Any]:

    retriever = HybridRetriever()

    return retriever.search_papers(
        query,
        arxiv_limit,
        openalex_limit,
        tavily_limit,
    )