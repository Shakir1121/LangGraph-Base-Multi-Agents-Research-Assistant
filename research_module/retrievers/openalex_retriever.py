"""
OpenAlex Retriever - Access to comprehensive scholarly metadata
https://docs.openalex.org/

OpenAlex provides free access to 240M+ scholarly works including:
- Journal articles, preprints, book chapters
- Author and institution information
- Citation counts and research concepts
- Open access status and URLs

No API key required for basic usage. Polite pool available with email.
"""

import logging
import time
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import urllib.parse

logger = logging.getLogger(__name__)

# Default API settings
DEFAULT_OPENALEX_API_KEY = "qGLXmkpXcmeJ4OjXagW4n5"
DEFAULT_OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_RATE_LIMIT = 10  # requests per second


@dataclass
class OpenAlexAuthor:
    """Author data structure from OpenAlex"""
    id: str
    display_name: str
    orcid: Optional[str] = None
    works_count: int = 0
    cited_by_count: int = 0


@dataclass
class OpenAlexInstitution:
    """Institution data structure from OpenAlex"""
    id: str
    display_name: str
    ror: Optional[str] = None
    country_code: Optional[str] = None
    works_count: int = 0


@dataclass
class OpenAlexConcept:
    """Research concept/topic from OpenAlex"""
    id: str
    display_name: str
    level: int = 0
    works_count: int = 0


@dataclass
class OpenAlexPaper:
    """Paper data structure from OpenAlex"""
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[OpenAlexAuthor] = None
    publication_date: Optional[str] = None
    citation_count: int = 0
    open_access: bool = False
    concepts: List[OpenAlexConcept] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.concepts is None:
            self.concepts = []


class OpenAlexRetriever:
    """
    OpenAlex API client for scholarly paper retrieval
    
    Features:
    - Search papers by title/keywords
    - Search authors and institutions
    - Retrieve papers by ID
    - Rate limiting and retry logic
    - Error handling with exponential backoff
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        base_url: str = DEFAULT_OPENALEX_BASE_URL
    ):
        """
        Initialize OpenAlex retriever
        
        Args:
            api_key: Optional API key (for polite pool)
            rate_limit: Requests per second (default 10)
            base_url: API base URL
        """
        # Set API key with priority: parameter > environment > default
        import os
        self.api_key = (
            api_key or 
            os.getenv("OPENALEX_API_KEY") or 
            DEFAULT_OPENALEX_API_KEY
        )
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.min_request_interval = 1.0 / rate_limit
        
        logger.info(f"✅ OpenAlexRetriever initialized (rate limit: {rate_limit} req/s)")

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _get_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> Optional[Dict[str, Any]]:
        """
        GET request with exponential backoff retry logic
        
        Args:
            url: Request URL
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff factor
        
        Returns:
            JSON response or None on failure
        """
        self._enforce_rate_limit()
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "User-Agent": f"ResearchCopilot/1.0 ({self.api_key})"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f" Request failed after {max_retries} attempts: {str(e)}")
                    return None
        
        return None

    def search_papers(self, query: str, limit: int = 10) -> List[OpenAlexPaper]:
        """
        Search for papers by title/keywords
        
        Args:
            query: Search query
            limit: Maximum results to return
        
        Returns:
            List of OpenAlexPaper objects
        """
        try:
            # Encode query
            encoded_query = urllib.parse.quote(query)
            
            # Build URL
            url = (
                f"{self.base_url}/works?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=relevance_score:desc"
            )
            
            logger.info(f"🔍 OpenAlex: Searching '{query[:40]}...'")
            
            # Get response
            data = self._get_with_retry(url)
            if not data:
                return []
            
            # Parse results
            papers = []
            for result in data.get("results", [])[:limit]:
                paper = self._parse_paper(result)
                if paper:
                    papers.append(paper)
            
            logger.info(f"✅ OpenAlex: Found {len(papers)} paper(s)")
            return papers
        
        except Exception as e:
            logger.error(f" OpenAlex search error: {str(e)}")
            return []

    def search_authors(self, query: str, limit: int = 10) -> List[OpenAlexAuthor]:
        """
        Search for authors
        
        Args:
            query: Author name query
            limit: Maximum results
        
        Returns:
            List of OpenAlexAuthor objects
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"{self.base_url}/authors?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=cited_by_count:desc"
            )
            
            logger.info(f"👤 OpenAlex: Searching authors '{query}'")
            data = self._get_with_retry(url)
            if not data:
                return []
            
            authors = []
            for result in data.get("results", [])[:limit]:
                author = self._parse_author(result)
                if author:
                    authors.append(author)
            
            logger.info(f"✅ OpenAlex: Found {len(authors)} author(s)")
            return authors
        
        except Exception as e:
            logger.error(f" OpenAlex author search error: {str(e)}")
            return []

    def search_institutions(self, query: str, limit: int = 10) -> List[OpenAlexInstitution]:
        """
        Search for institutions
        
        Args:
            query: Institution name query
            limit: Maximum results
        
        Returns:
            List of OpenAlexInstitution objects
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"{self.base_url}/institutions?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=works_count:desc"
            )
            
            logger.info(f"🏢 OpenAlex: Searching institutions '{query}'")
            data = self._get_with_retry(url)
            if not data:
                return []
            
            institutions = []
            for result in data.get("results", [])[:limit]:
                inst = self._parse_institution(result)
                if inst:
                    institutions.append(inst)
            
            logger.info(f"✅ OpenAlex: Found {len(institutions)} institution(s)")
            return institutions
        
        except Exception as e:
            logger.error(f" OpenAlex institution search error: {str(e)}")
            return []

    def search_concepts(self, query: str, limit: int = 10) -> List[OpenAlexConcept]:
        """
        Search for research concepts/topics
        
        Args:
            query: Concept query
            limit: Maximum results
        
        Returns:
            List of OpenAlexConcept objects
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = (
                f"{self.base_url}/concepts?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=works_count:desc"
            )
            
            logger.info(f"💡 OpenAlex: Searching concepts '{query}'")
            data = self._get_with_retry(url)
            if not data:
                return []
            
            concepts = []
            for result in data.get("results", [])[:limit]:
                concept = self._parse_concept(result)
                if concept:
                    concepts.append(concept)
            
            logger.info(f" OpenAlex: Found {len(concepts)} concept(s)")
            return concepts
        
        except Exception as e:
            logger.error(f" OpenAlex concept search error: {str(e)}")
            return []

    def get_paper_by_id(self, paper_id: str) -> Optional[OpenAlexPaper]:
        """
        Retrieve a specific paper by ID
        
        Args:
            paper_id: OpenAlex paper ID (e.g., 'W2741809807')
        
        Returns:
            OpenAlexPaper object or None
        """
        try:
            url = f"{self.base_url}/works/{paper_id}"
            logger.info(f"📄 OpenAlex: Fetching paper {paper_id}")
            data = self._get_with_retry(url)
            if not data:
                return None
            
            paper = self._parse_paper(data)
            if paper:
                logger.info(f" OpenAlex: Retrieved {paper.title[:50]}...")
            return paper
        
        except Exception as e:
            logger.error(f" OpenAlex get paper error: {str(e)}")
            return None

    def _parse_paper(self, result: Dict[str, Any]) -> Optional[OpenAlexPaper]:
        """Parse OpenAlex work JSON to OpenAlexPaper"""
        try:
            # Extract authors
            authors = []
            for author_data in result.get("authorships", []):
                author_info = author_data.get("author", {})
                author = OpenAlexAuthor(
                    id=author_info.get("id", ""),
                    display_name=author_info.get("display_name", "Unknown"),
                    orcid=author_info.get("orcid"),
                    works_count=author_info.get("works_count", 0),
                    cited_by_count=author_info.get("cited_by_count", 0)
                )
                authors.append(author)
            
            # Extract concepts
            concepts = []
            for concept_data in result.get("concepts", []):
                concept = OpenAlexConcept(
                    id=concept_data.get("id", ""),
                    display_name=concept_data.get("display_name", ""),
                    level=concept_data.get("level", 0),
                    works_count=concept_data.get("works_count", 0)
                )
                concepts.append(concept)
            
            # Create paper object
            paper = OpenAlexPaper(
                id=result.get("id", ""),
                title=result.get("title", ""),
                abstract=result.get("abstract", ""),
                authors=authors,
                publication_date=result.get("publication_date"),
                citation_count=result.get("cited_by_count", 0),
                open_access=result.get("open_access", {}).get("is_oa", False),
                concepts=concepts,
                doi=result.get("doi"),
                url=result.get("url")
            )
            
            return paper
        
        except Exception as e:
            logger.warning(f"  Error parsing paper: {str(e)}")
            return None

    def _parse_author(self, result: Dict[str, Any]) -> Optional[OpenAlexAuthor]:
        """Parse OpenAlex author JSON"""
        try:
            return OpenAlexAuthor(
                id=result.get("id", ""),
                display_name=result.get("display_name", ""),
                orcid=result.get("orcid"),
                works_count=result.get("works_count", 0),
                cited_by_count=result.get("cited_by_count", 0)
            )
        except Exception as e:
            logger.warning(f"  Error parsing author: {str(e)}")
            return None

    def _parse_institution(self, result: Dict[str, Any]) -> Optional[OpenAlexInstitution]:
        """Parse OpenAlex institution JSON"""
        try:
            return OpenAlexInstitution(
                id=result.get("id", ""),
                display_name=result.get("display_name", ""),
                ror=result.get("ror"),
                country_code=result.get("country_code"),
                works_count=result.get("works_count", 0)
            )
        except Exception as e:
            logger.warning(f"  Error parsing institution: {str(e)}")
            return None

    def _parse_concept(self, result: Dict[str, Any]) -> Optional[OpenAlexConcept]:
        """Parse OpenAlex concept JSON"""
        try:
            return OpenAlexConcept(
                id=result.get("id", ""),
                display_name=result.get("display_name", ""),
                level=result.get("level", 0),
                works_count=result.get("works_count", 0)
            )
        except Exception as e:
            logger.warning(f"  Error parsing concept: {str(e)}")
            return None


# Convenience functions
def search_papers(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None
) -> List[OpenAlexPaper]:
    """Quick function to search papers"""
    retriever = OpenAlexRetriever(api_key=api_key)
    return retriever.search_papers(query, limit)


def search_authors(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None
) -> List[OpenAlexAuthor]:
    """Quick function to search authors"""
    retriever = OpenAlexRetriever(api_key=api_key)
    return retriever.search_authors(query, limit)


def search_institutions(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None
) -> List[OpenAlexInstitution]:
    """Quick function to search institutions"""
    retriever = OpenAlexRetriever(api_key=api_key)
    return retriever.search_institutions(query, limit)
