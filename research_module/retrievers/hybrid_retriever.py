"""
Hybrid retriever combining multiple sources: ArXiv, OpenAlex, and Tavily
Runs all retrievers in parallel for comprehensive paper collection
"""

import logging
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time

from research_module.retrievers.arxiv_retriever import search_arxiv
from research_module.retrievers.openalex_retriever import OpenAlexRetriever
from research_module.tools.tavily_tool import tavily_search

logger = logging.getLogger(__name__)

# Hard cap on how long a single hybrid retrieval may take. When sources are
# slow (OpenAlex retries, ArXiv transient errors, Tavily timeouts) the whole
# search used to block for many minutes. We abort remaining sources past this
# time so the LLM sections start streaming quickly.
DEFAULT_SEARCH_TIMEOUT = 12.0


class HybridRetriever:
    """Combines ArXiv, OpenAlex, and Tavily for comprehensive paper retrieval"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.openalex = OpenAlexRetriever()
        self.lock = threading.Lock()
    
    def search_papers(
        self, 
        query: str, 
        arxiv_limit: int = 20,
        openalex_limit: int = 20,
        tavily_limit: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all sources in parallel
        
        Args:
            query: Research topic/query
            arxiv_limit: Number of ArXiv papers to retrieve
            openalex_limit: Number of OpenAlex papers to retrieve
            tavily_limit: Number of Tavily web results to retrieve
        
        Returns:
            Dict with keys: arxiv_papers, openalex_papers, tavily_papers
        """
        results = {
            "arxiv_papers": [],
            "openalex_papers": [],
            "tavily_papers": [],
            "total_papers": 0,
            "sources_used": []
}
        
        logger.info(f"🔄 HybridRetriever: Starting parallel search for '{query}'")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._fetch_arxiv, query, arxiv_limit): "arxiv",
                executor.submit(self._fetch_openalex, query, openalex_limit): "openalex",
                executor.submit(self._fetch_tavily, query, tavily_limit): "tavily"
            }
            
            # Collect results as they complete, but never block past the hard
            # timeout. As soon as DEFAULT_SEARCH_TIMEOUT elapses, stop waiting
            # and return whatever has completed so far (remaining sources are
            # abandoned in the background). This guarantees the caller never
            # hangs for minutes when a source is slow/unresponsive.
            remaining = set(futures)
            while remaining:
                elapsed_total = time.time() - start_time
                if elapsed_total >= DEFAULT_SEARCH_TIMEOUT:
                    logger.warning(
                        f"⏰ Hybrid retrieval reached {DEFAULT_SEARCH_TIMEOUT:.0f}s "
                        f"timeout with {len(remaining)} source(s) still running — "
                        f"returning partial results."
                    )
                    break

                done, _ = wait(remaining, timeout=0.5, return_when=FIRST_COMPLETED)
                if not done:
                    continue
                for future in done:
                    remaining.discard(future)
                    source = futures[future]
                    try:
                        papers = future.result()
                        if papers:
                            results[f"{source}_papers"] = papers
                            results["sources_used"].append(source)
                            logger.info(f"✅ {source.upper()}: {len(papers)} paper(s)")
                    except Exception as e:
                        logger.warning(f"⚠️ {source.upper()} retrieval failed: {str(e)}")
        
        results["total_papers"] = (
            len(results["arxiv_papers"]) + 
            len(results["openalex_papers"]) + 
            len(results["tavily_papers"])
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"📊 HybridRetriever: {results['total_papers']} papers from "
            f"{len(results['sources_used'])} source(s) in {elapsed:.2f}s"
        )
        
        return results
    
    def _fetch_arxiv(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch from ArXiv"""
        try:
            papers = search_arxiv(query, max_results=limit)
            return [self._normalize_paper(p, "arxiv") for p in papers]
        except Exception as e:
            logger.error(f"ArXiv retrieval error: {str(e)}")
            return []
    
    def _fetch_openalex(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch from OpenAlex"""
        try:
            papers = self.openalex.search_papers(query, limit=limit)
            return [self._normalize_paper(p, "openalex") for p in papers]
        except Exception as e:
            logger.error(f"OpenAlex retrieval error: {str(e)}")
            return []
    
    def _fetch_tavily(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch from Tavily web search"""
        try:
            results = tavily_search(query)
            logger.debug(f"🔍 Tavily raw response type: {type(results)}, content: {results}")
            
            papers = []
            
            # Handle list response
            if isinstance(results, list):
                logger.info(f"✅ TAVILY: Processing list with {len(results)} items")
                for i, result in enumerate(results[:limit]):
                    if isinstance(result, dict):
                        papers.append({
                            "title": result.get("title", ""),
                            "abstract": result.get("content", ""),
                            "url": result.get("url", ""),
                            "source": "tavily",
                            "authors": [],
                            "publication_date": None,
                            "citation_count": 0
                        })
            
            # Handle dict response with nested results
            elif isinstance(results, dict):
                if "results" in results:
                    logger.info(f"✅ TAVILY: Processing dict with 'results' key, {len(results['results'])} items")
                    for i, result in enumerate(results["results"][:limit]):
                        papers.append({
                            "title": result.get("title", ""),
                            "abstract": result.get("content", ""),
                            "url": result.get("url", ""),
                            "source": "tavily",
                            "authors": [],
                            "publication_date": None,
                            "citation_count": 0
                        })
                else:
                    logger.warning(f"⚠️ TAVILY: Dict response but no 'results' key. Keys: {results.keys()}")
            else:
                logger.warning(f"⚠️ TAVILY: Unexpected response type: {type(results)}")
            
            if papers:
                logger.info(f"✅ TAVILY: {len(papers)} paper(s) extracted")
            else:
                logger.warning(f"⚠️ TAVILY: No papers extracted from response")
                
            return papers
        except Exception as e:
            logger.error(f" Tavily retrieval error: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def _normalize_paper(paper: Any, source: str) -> Dict[str, Any]:
        """Normalize paper format across sources"""
        try:
            if source == "arxiv":
                return {
                    "title": getattr(paper, "title", ""),
                    "abstract": getattr(paper, "summary", ""),
                    "url": getattr(paper, "entry_id", ""),
                    "source": "arxiv",
                    "authors": [a.name for a in getattr(paper, "authors", [])],
                    "publication_date": str(getattr(paper, "published", "")),
                    "citation_count": 0
                }
            elif source == "openalex":
                return {
                    "title": getattr(paper, "title", ""),
                    "abstract": getattr(paper, "abstract", ""),
                    "url": getattr(paper, "url", ""),
                    "source": "openalex",
                    "authors": [a.display_name for a in getattr(paper, "authors", [])],
                    "publication_date": str(getattr(paper, "publication_date", "")),
                    "citation_count": getattr(paper, "citation_count", 0)
                }
            else:
                return paper
        except Exception as e:
            logger.warning(f"Paper normalization error: {str(e)}")
            return paper


def hybrid_search(
    query: str,
    arxiv_limit: int = 20,
    openalex_limit: int = 20,
    tavily_limit: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function for hybrid search"""
    retriever = HybridRetriever()
    return retriever.search_papers(query, arxiv_limit, openalex_limit, tavily_limit)
