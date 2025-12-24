"""Knowledge module - external knowledge sources for agents."""

from .wikipedia import WikipediaSource, WikipediaArticle
from .ollama import OllamaGateway, OllamaResponse
from .search import (
    SearchProvider,
    SearchResult,
    SerperSearchProvider,
    TavilySearchProvider,
    DuckDuckGoSearchProvider,
    CompositeSearchProvider,
)
from .reddit import RedditSource, RedditPost, RedditComment
from .twitter import TwitterSource, Tweet, TwitterUser
from .geo import GeoSource, get_geo_source

__all__ = [
    "WikipediaSource",
    "WikipediaArticle",
    "OllamaGateway",
    "OllamaResponse",
    "SearchProvider",
    "SearchResult",
    "SerperSearchProvider",
    "TavilySearchProvider",
    "DuckDuckGoSearchProvider",
    "CompositeSearchProvider",
    "RedditSource",
    "RedditPost",
    "RedditComment",
    "TwitterSource",
    "Tweet",
    "TwitterUser",
    "GeoSource",
    "get_geo_source",
]
    "Tweet",
    "TwitterUser",
]
