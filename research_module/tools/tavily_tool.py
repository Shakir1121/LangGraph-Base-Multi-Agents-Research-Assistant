import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
logger = logging.getLogger(__name__)


def tavily_search(query, max_results=5):
    """
    Search using Tavily API
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default 5)
    
    Returns:
        List of search results with title, content, and url
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning(" No Tavily API key found in environment")
            return []

        logger.info(f"🔍 Tavily: Searching '{query[:50]}...' (max_results={max_results})")
        client = TavilyClient(api_key=api_key)

        # Make the API call
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )
        
        logger.debug(f"📨 Tavily raw response type: {type(response)}")
        
        # Handle different response formats
        if isinstance(response, dict):
            results = response.get("results", [])
            logger.info(f" Tavily: Got response with {len(results)} results")
            
            processed = [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", "")
                }
                for r in results
            ]
            return processed
        
        elif isinstance(response, list):
            logger.info(f" Tavily: Got list response with {len(response)} items")
            processed = [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", "")
                }
                for r in response
            ]
            return processed
        
        else:
            logger.warning(f" Tavily: Unexpected response type: {type(response)}")
            return []

    except Exception as e:
        logger.error(f" Tavily failed: {e}", exc_info=True)
        return []