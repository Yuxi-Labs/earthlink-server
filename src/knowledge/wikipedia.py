"""Wikipedia knowledge source - access to Wikipedia via API."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class WikipediaArticle:
    """Wikipedia article data."""

    title: str
    page_id: int
    summary: str = ""
    content: str = ""
    url: str = ""
    categories: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


class WikipediaSource:
    """
    Wikipedia knowledge source.
    
    Provides access to Wikipedia content for agents to learn from.
    Uses the Wikipedia API (no scraping).
    """

    def __init__(
        self,
        language: str = "en",
        user_agent: str = "EarthlinkBot/1.0 (https://github.com/Yuxi-Labs/earthlink; research@yuxilabs.com) python-httpx/0.28",
    ):
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search Wikipedia for articles matching query.
        
        Returns list of search results with title and snippet.
        """
        client = await self._get_client()

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }

        response = await client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item["title"],
                "page_id": item["pageid"],
                "snippet": item.get("snippet", ""),
                "word_count": item.get("wordcount", 0),
            })

        return results

    async def get_summary(self, title: str) -> WikipediaArticle | None:
        """
        Get article summary (extract) by title.
        
        Uses the REST API for cleaner summaries.
        """
        client = await self._get_client()

        # Use REST API for summary
        rest_url = f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary/{title}"

        try:
            response = await client.get(rest_url)
            response.raise_for_status()
            data = response.json()

            return WikipediaArticle(
                title=data.get("title", title),
                page_id=data.get("pageid", 0),
                summary=data.get("extract", ""),
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            )
        except httpx.HTTPStatusError:
            return None

    async def get_article(self, title: str) -> WikipediaArticle | None:
        """
        Get full article content by title.
        """
        client = await self._get_client()

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|categories|links",
            "explaintext": True,  # Plain text, not HTML
            "exsectionformat": "plain",
            "cllimit": 20,  # Categories limit
            "pllimit": 50,  # Links limit
            "format": "json",
        }

        response = await client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        page = list(pages.values())[0]
        if "missing" in page:
            return None

        categories = [
            cat["title"].replace("Category:", "")
            for cat in page.get("categories", [])
        ]

        links = [link["title"] for link in page.get("links", [])]

        return WikipediaArticle(
            title=page.get("title", title),
            page_id=page.get("pageid", 0),
            content=page.get("extract", ""),
            categories=categories,
            links=links,
            url=f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}",
        )

    async def get_random(self, count: int = 1) -> list[str]:
        """Get random article titles."""
        client = await self._get_client()

        params = {
            "action": "query",
            "list": "random",
            "rnlimit": count,
            "rnnamespace": 0,  # Main namespace only
            "format": "json",
        }

        response = await client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        return [
            item["title"]
            for item in data.get("query", {}).get("random", [])
        ]

    async def get_links(self, title: str) -> list[str]:
        """Get all links from an article."""
        client = await self._get_client()

        params = {
            "action": "query",
            "titles": title,
            "prop": "links",
            "pllimit": "max",
            "format": "json",
        }

        all_links = []
        continue_token = None

        while True:
            if continue_token:
                params["plcontinue"] = continue_token

            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                links = page.get("links", [])
                all_links.extend([link["title"] for link in links])

            if "continue" in data:
                continue_token = data["continue"].get("plcontinue")
            else:
                break

        return all_links

    async def get_categories(self, title: str) -> list[str]:
        """Get categories for an article."""
        client = await self._get_client()

        params = {
            "action": "query",
            "titles": title,
            "prop": "categories",
            "cllimit": "max",
            "format": "json",
        }

        response = await client.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            categories = page.get("categories", [])
            return [
                cat["title"].replace("Category:", "")
                for cat in categories
            ]

        return []
