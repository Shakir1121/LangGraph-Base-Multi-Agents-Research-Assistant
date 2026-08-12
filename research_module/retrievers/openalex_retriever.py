import logging
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_RATE_LIMIT = 10


@dataclass
class OpenAlexAuthor:
    id: str
    display_name: str
    orcid: Optional[str] = None
    works_count: int = 0
    cited_by_count: int = 0


@dataclass
class OpenAlexInstitution:
    id: str
    display_name: str
    ror: Optional[str] = None
    country_code: Optional[str] = None
    works_count: int = 0


@dataclass
class OpenAlexConcept:
    id: str
    display_name: str
    level: int = 0
    works_count: int = 0


@dataclass
class OpenAlexPaper:
    id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[List[OpenAlexAuthor]] = None
    publication_date: Optional[str] = None
    citation_count: int = 0
    open_access: bool = False
    concepts: Optional[List[OpenAlexConcept]] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []

        if self.concepts is None:
            self.concepts = []


class OpenAlexRetriever:

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = DEFAULT_RATE_LIMIT,
        base_url: str = DEFAULT_OPENALEX_BASE_URL,
    ):
        self.api_key = (
            api_key
            or os.getenv("OPENALEX_API_KEY")
        )

        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.min_request_interval = 1.0 / rate_limit

        logger.info(
            "OpenAlexRetriever initialized"
        )

    def _enforce_rate_limit(self):
        elapsed = time.time() - self.last_request_time

        if elapsed < self.min_request_interval:
            time.sleep(
                self.min_request_interval - elapsed
            )

        self.last_request_time = time.time()

    def _get_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> Optional[Dict[str, Any]]:

        for attempt in range(max_retries):

            self._enforce_rate_limit()

            try:
                headers = {
                    "User-Agent": "ResearchCopilot/1.0"
                }

                if self.api_key:
                    headers["Authorization"] = (
                        f"Bearer {self.api_key}"
                    )

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=10,
                )

                response.raise_for_status()

                return response.json()

            except requests.exceptions.RequestException as exc:

                if attempt < max_retries - 1:
                    wait_time = (
                        backoff_factor ** attempt
                    )

                    logger.warning(
                        f"OpenAlex request failed. "
                        f"Retrying in {wait_time}s: {exc}"
                    )

                    time.sleep(wait_time)

                else:
                    logger.error(
                        f"OpenAlex request failed "
                        f"after {max_retries} attempts: {exc}"
                    )

        return None

    def search_papers(
        self,
        query: str,
        limit: int = 10,
    ) -> List[OpenAlexPaper]:

        try:
            encoded_query = urllib.parse.quote(
                query
            )

            url = (
                f"{self.base_url}/works?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=relevance_score:desc"
            )

            logger.info(
                f"Searching OpenAlex: {query[:40]}..."
            )

            data = self._get_with_retry(url)

            if not data:
                return []

            papers = []

            for result in data.get("results", [])[:limit]:
                paper = self._parse_paper(result)

                if paper:
                    papers.append(paper)

            logger.info(
                f"OpenAlex returned {len(papers)} papers"
            )

            return papers

        except Exception as exc:
            logger.error(
                f"OpenAlex paper search failed: {exc}"
            )
            return []

    def search_authors(
        self,
        query: str,
        limit: int = 10,
    ) -> List[OpenAlexAuthor]:

        try:
            encoded_query = urllib.parse.quote(
                query
            )

            url = (
                f"{self.base_url}/authors?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=cited_by_count:desc"
            )

            data = self._get_with_retry(url)

            if not data:
                return []

            authors = []

            for result in data.get("results", [])[:limit]:
                author = self._parse_author(result)

                if author:
                    authors.append(author)

            return authors

        except Exception as exc:
            logger.error(
                f"OpenAlex author search failed: {exc}"
            )
            return []

    def search_institutions(
        self,
        query: str,
        limit: int = 10,
    ) -> List[OpenAlexInstitution]:

        try:
            encoded_query = urllib.parse.quote(
                query
            )

            url = (
                f"{self.base_url}/institutions?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=works_count:desc"
            )

            data = self._get_with_retry(url)

            if not data:
                return []

            institutions = []

            for result in data.get("results", [])[:limit]:
                institution = (
                    self._parse_institution(result)
                )

                if institution:
                    institutions.append(institution)

            return institutions

        except Exception as exc:
            logger.error(
                f"OpenAlex institution search failed: {exc}"
            )
            return []

    def search_concepts(
        self,
        query: str,
        limit: int = 10,
    ) -> List[OpenAlexConcept]:

        try:
            encoded_query = urllib.parse.quote(
                query
            )

            url = (
                f"{self.base_url}/concepts?"
                f"search={encoded_query}&"
                f"per_page={min(limit, 50)}&"
                f"sort=works_count:desc"
            )

            data = self._get_with_retry(url)

            if not data:
                return []

            concepts = []

            for result in data.get("results", [])[:limit]:
                concept = self._parse_concept(result)

                if concept:
                    concepts.append(concept)

            return concepts

        except Exception as exc:
            logger.error(
                f"OpenAlex concept search failed: {exc}"
            )
            return []

    def get_paper_by_id(
        self,
        paper_id: str,
    ) -> Optional[OpenAlexPaper]:

        try:
            url = (
                f"{self.base_url}/works/{paper_id}"
            )

            data = self._get_with_retry(url)

            if not data:
                return None

            return self._parse_paper(data)

        except Exception as exc:
            logger.error(
                f"OpenAlex paper retrieval failed: {exc}"
            )
            return None

    def _parse_paper(
        self,
        result: Dict[str, Any],
    ) -> Optional[OpenAlexPaper]:

        try:
            authors = []

            for author_data in result.get(
                "authorships",
                [],
            ):
                author_info = author_data.get(
                    "author",
                    {},
                )

                authors.append(
                    OpenAlexAuthor(
                        id=author_info.get(
                            "id",
                            "",
                        ),
                        display_name=author_info.get(
                            "display_name",
                            "Unknown",
                        ),
                        orcid=author_info.get(
                            "orcid"
                        ),
                        works_count=author_info.get(
                            "works_count",
                            0,
                        ),
                        cited_by_count=author_info.get(
                            "cited_by_count",
                            0,
                        ),
                    )
                )

            concepts = []

            for concept_data in result.get(
                "concepts",
                [],
            ):
                concepts.append(
                    OpenAlexConcept(
                        id=concept_data.get(
                            "id",
                            "",
                        ),
                        display_name=concept_data.get(
                            "display_name",
                            "",
                        ),
                        level=concept_data.get(
                            "level",
                            0,
                        ),
                        works_count=concept_data.get(
                            "works_count",
                            0,
                        ),
                    )
                )

            return OpenAlexPaper(
                id=result.get("id", ""),
                title=result.get("title", ""),
                abstract=result.get("abstract"),
                authors=authors,
                publication_date=result.get(
                    "publication_date"
                ),
                citation_count=result.get(
                    "cited_by_count",
                    0,
                ),
                open_access=result.get(
                    "open_access",
                    {},
                ).get(
                    "is_oa",
                    False,
                ),
                concepts=concepts,
                doi=result.get("doi"),
                url=result.get("url"),
            )

        except Exception as exc:
            logger.warning(
                f"Failed to parse OpenAlex paper: {exc}"
            )
            return None

    def _parse_author(
        self,
        result: Dict[str, Any],
    ) -> Optional[OpenAlexAuthor]:

        try:
            return OpenAlexAuthor(
                id=result.get("id", ""),
                display_name=result.get(
                    "display_name",
                    "",
                ),
                orcid=result.get("orcid"),
                works_count=result.get(
                    "works_count",
                    0,
                ),
                cited_by_count=result.get(
                    "cited_by_count",
                    0,
                ),
            )

        except Exception as exc:
            logger.warning(
                f"Failed to parse author: {exc}"
            )
            return None

    def _parse_institution(
        self,
        result: Dict[str, Any],
    ) -> Optional[OpenAlexInstitution]:

        try:
            return OpenAlexInstitution(
                id=result.get("id", ""),
                display_name=result.get(
                    "display_name",
                    "",
                ),
                ror=result.get("ror"),
                country_code=result.get(
                    "country_code"
                ),
                works_count=result.get(
                    "works_count",
                    0,
                ),
            )

        except Exception as exc:
            logger.warning(
                f"Failed to parse institution: {exc}"
            )
            return None

    def _parse_concept(
        self,
        result: Dict[str, Any],
    ) -> Optional[OpenAlexConcept]:

        try:
            return OpenAlexConcept(
                id=result.get("id", ""),
                display_name=result.get(
                    "display_name",
                    "",
                ),
                level=result.get(
                    "level",
                    0,
                ),
                works_count=result.get(
                    "works_count",
                    0,
                ),
            )

        except Exception as exc:
            logger.warning(
                f"Failed to parse concept: {exc}"
            )
            return None


def search_papers(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> List[OpenAlexPaper]:

    retriever = OpenAlexRetriever(
        api_key=api_key
    )

    return retriever.search_papers(
        query,
        limit,
    )


def search_authors(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> List[OpenAlexAuthor]:

    retriever = OpenAlexRetriever(
        api_key=api_key
    )

    return retriever.search_authors(
        query,
        limit,
    )


def search_institutions(
    query: str,
    limit: int = 10,
    api_key: Optional[str] = None,
) -> List[OpenAlexInstitution]:

    retriever = OpenAlexRetriever(
        api_key=api_key
    )

    return retriever.search_institutions(
        query,
        limit,
    )