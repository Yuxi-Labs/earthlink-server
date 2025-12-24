"""Search providers - web search for knowledge acquisition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class SearchResult:
    """Web search result."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Search for query and return results."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass


class SerperSearchProvider(SearchProvider):
    """
    Search using Serper API (Google search).
    
    Requires SERPER_API_KEY environment variable.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        client = await self._get_client()

        payload = {
            "q": query,
            "num": num_results,
        }

        response = await client.post(self.base_url, json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="serper",
            ))

        return results


class TavilySearchProvider(SearchProvider):
    """
    Search using Tavily API (AI-focused search).
    
    Requires TAVILY_API_KEY environment variable.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_depth: str = "basic",  # "basic" or "advanced"
        include_answer: bool = False,
    ) -> list[SearchResult]:
        client = await self._get_client()

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }

        response = await client.post(self.base_url, json=payload)
        response.raise_for_status()
        data = response.json()

        results = []

        # Include AI-generated answer if available
        if include_answer and "answer" in data:
            results.append(SearchResult(
                title="AI Summary",
                url="",
                snippet=data["answer"],
                source="tavily_answer",
            ))

        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source="tavily",
                metadata={
                    "score": item.get("score", 0),
                    "published_date": item.get("published_date"),
                },
            ))

        return results


class DuckDuckGoSearchProvider(SearchProvider):
    """
    Search using DuckDuckGo (no API key required).
    
    Uses the instant answer API, which has limitations.
    """

    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": "Earthlink/1.0"},
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        client = await self._get_client()

        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        response = await client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []

        # Abstract (main result)
        if data.get("Abstract"):
            results.append(SearchResult(
                title=data.get("Heading", query),
                url=data.get("AbstractURL", ""),
                snippet=data.get("Abstract", ""),
                source="duckduckgo",
            ))

        # Related topics
        for item in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in item:
                results.append(SearchResult(
                    title=item.get("Text", "")[:100],
                    url=item.get("FirstURL", ""),
                    snippet=item.get("Text", ""),
                    source="duckduckgo",
                ))

        return results[:num_results]


class CompositeSearchProvider(SearchProvider):
    """
    Combines multiple search providers.
    
    Queries multiple providers and merges results.
    """

    def __init__(self, providers: list[SearchProvider]):
        self.providers = providers

    async def close(self) -> None:
        for provider in self.providers:
            await provider.close()

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        import asyncio

        # Query all providers concurrently
        tasks = [
            provider.search(query, num_results)
            for provider in self.providers
        ]

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        all_results = []
        seen_urls = set()

        for results in results_lists:
            if isinstance(results, Exception):
                continue

            for result in results:
                # Deduplicate by URL
                if result.url and result.url in seen_urls:
                    continue
                seen_urls.add(result.url)
                all_results.append(result)

        return all_results[:num_results]
