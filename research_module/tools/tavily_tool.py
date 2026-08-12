import logging
import os

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()

logger = logging.getLogger(__name__)


def tavily_search(query: str, max_results: int = 5):
    """Search Tavily and return normalized results."""

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        logger.warning("TAVILY_API_KEY is not configured.")
        return []

    try:
        logger.info(
            f"Tavily search: '{query[:50]}...' "
            f"(max_results={max_results})"
        )

        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
        )

        if isinstance(response, dict):
            results = response.get("results", [])
        elif isinstance(response, list):
            results = response
        else:
            logger.warning(
                f"Unexpected Tavily response type: {type(response)}"
            )
            return []

        processed = [
            {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
            }
            for result in results[:max_results]
            if isinstance(result, dict)
        ]

        logger.info(
            f"Tavily returned {len(processed)} result(s)"
        )

        return processed

    except Exception as exc:
        logger.error(
            f"Tavily search failed: {exc}",
            exc_info=True,
        )
        return []