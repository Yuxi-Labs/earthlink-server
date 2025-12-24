"""Twitter/X knowledge source - access to Twitter via API."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx


@dataclass
class Tweet:
    """Twitter/X tweet data."""

    id: str
    text: str
    author_id: str
    author_username: str = ""
    author_name: str = ""
    created_at: str = ""
    retweet_count: int = 0
    reply_count: int = 0
    like_count: int = 0
    quote_count: int = 0
    impression_count: int = 0
    lang: str = ""
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    is_retweet: bool = False
    referenced_tweet_id: str = ""


@dataclass
class TwitterUser:
    """Twitter/X user data."""

    id: str
    username: str
    name: str
    description: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    created_at: str = ""
    verified: bool = False
    location: str = ""
    url: str = ""


class TwitterSource:
    """
    Twitter/X knowledge source.
    
    Provides access to Twitter/X content for agents to learn from.
    Uses Twitter API v2 (requires authentication).
    
    Note: Requires TWITTER_BEARER_TOKEN environment variable.
    """

    def __init__(
        self,
        bearer_token: str | None = None,
    ):
        import os
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        if not self.bearer_token:
            raise ValueError(
                "Twitter Bearer Token required. "
                "Set TWITTER_BEARER_TOKEN environment variable or pass bearer_token parameter."
            )
        
        self.base_url = "https://api.twitter.com/2"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "User-Agent": "Earthlink/1.0",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make request to Twitter API."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = await client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Twitter API request failed: {e}") from e

    def _parse_tweet(self, tweet_data: dict[str, Any], includes: dict[str, Any] | None = None) -> Tweet:
        """Parse tweet data into Tweet object."""
        # Extract entities
        entities = tweet_data.get("entities", {})
        hashtags = [tag["tag"] for tag in entities.get("hashtags", [])]
        mentions = [mention["username"] for mention in entities.get("mentions", [])]
        urls = [url["expanded_url"] for url in entities.get("urls", []) if url.get("expanded_url")]
        
        # Check if retweet
        referenced_tweets = tweet_data.get("referenced_tweets", [])
        is_retweet = any(ref.get("type") == "retweeted" for ref in referenced_tweets)
        referenced_id = ""
        if referenced_tweets:
            referenced_id = referenced_tweets[0].get("id", "")
        
        # Get author info from includes if available
        author_username = ""
        author_name = ""
        if includes and "users" in includes:
            author_id = tweet_data.get("author_id", "")
            for user in includes["users"]:
                if user.get("id") == author_id:
                    author_username = user.get("username", "")
                    author_name = user.get("name", "")
                    break
        
        # Get metrics
        public_metrics = tweet_data.get("public_metrics", {})
        
        return Tweet(
            id=tweet_data.get("id", ""),
            text=tweet_data.get("text", ""),
            author_id=tweet_data.get("author_id", ""),
            author_username=author_username,
            author_name=author_name,
            created_at=tweet_data.get("created_at", ""),
            retweet_count=public_metrics.get("retweet_count", 0),
            reply_count=public_metrics.get("reply_count", 0),
            like_count=public_metrics.get("like_count", 0),
            quote_count=public_metrics.get("quote_count", 0),
            impression_count=public_metrics.get("impression_count", 0),
            lang=tweet_data.get("lang", ""),
            hashtags=hashtags,
            mentions=mentions,
            urls=urls,
            is_retweet=is_retweet,
            referenced_tweet_id=referenced_id,
        )

    async def search_recent_tweets(
        self,
        query: str,
        max_results: int = 10,
        language: str | None = None,
    ) -> list[Tweet]:
        """
        Search recent tweets (last 7 days).
        
        Args:
            query: Search query (Twitter search syntax)
            max_results: Number of results (10-100)
            language: Language code (e.g., "en")
            
        Returns:
            List of Tweet objects
        """
        params = {
            "query": query,
            "max_results": min(max(max_results, 10), 100),
            "tweet.fields": "author_id,created_at,public_metrics,entities,referenced_tweets,lang",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        
        if language:
            params["query"] += f" lang:{language}"
        
        data = await self._make_request("/tweets/search/recent", params)
        
        tweets = []
        tweet_list = data.get("data", [])
        includes = data.get("includes", {})
        
        for tweet_data in tweet_list:
            tweets.append(self._parse_tweet(tweet_data, includes))
        
        return tweets

    async def get_user_tweets(
        self,
        user_id: str,
        max_results: int = 10,
        exclude_retweets: bool = False,
        exclude_replies: bool = False,
    ) -> list[Tweet]:
        """
        Get tweets from a user's timeline.
        
        Args:
            user_id: Twitter user ID
            max_results: Number of results (5-100)
            exclude_retweets: Exclude retweets
            exclude_replies: Exclude replies
            
        Returns:
            List of Tweet objects
        """
        params = {
            "max_results": min(max(max_results, 5), 100),
            "tweet.fields": "author_id,created_at,public_metrics,entities,referenced_tweets,lang",
        }
        
        excludes = []
        if exclude_retweets:
            excludes.append("retweets")
        if exclude_replies:
            excludes.append("replies")
        
        if excludes:
            params["exclude"] = ",".join(excludes)
        
        endpoint = f"/users/{user_id}/tweets"
        data = await self._make_request(endpoint, params)
        
        tweets = []
        tweet_list = data.get("data", [])
        
        for tweet_data in tweet_list:
            tweets.append(self._parse_tweet(tweet_data))
        
        return tweets

    async def get_user_by_username(self, username: str) -> TwitterUser:
        """
        Get user information by username.
        
        Args:
            username: Twitter username (without @)
            
        Returns:
            TwitterUser object
        """
        params = {
            "user.fields": "description,public_metrics,created_at,verified,location,url",
        }
        
        endpoint = f"/users/by/username/{username}"
        data = await self._make_request(endpoint, params)
        
        user_data = data.get("data", {})
        metrics = user_data.get("public_metrics", {})
        
        return TwitterUser(
            id=user_data.get("id", ""),
            username=user_data.get("username", ""),
            name=user_data.get("name", ""),
            description=user_data.get("description", ""),
            followers_count=metrics.get("followers_count", 0),
            following_count=metrics.get("following_count", 0),
            tweet_count=metrics.get("tweet_count", 0),
            created_at=user_data.get("created_at", ""),
            verified=user_data.get("verified", False),
            location=user_data.get("location", ""),
            url=user_data.get("url", ""),
        )

    async def get_trending_topics(self, woeid: int = 1) -> list[dict[str, Any]]:
        """
        Get trending topics (requires higher API tier).
        
        Args:
            woeid: Where On Earth ID (1 = worldwide)
            
        Returns:
            List of trending topics
            
        Note: This endpoint is only available with elevated or academic research access.
        """
        # Note: Twitter API v2 doesn't have a direct trends endpoint
        # This is a placeholder that would need the v1.1 API
        raise NotImplementedError(
            "Trending topics requires Twitter API v1.1 or elevated access. "
            "Use search_recent_tweets with popular hashtags instead."
        )

    async def search_hashtag(
        self,
        hashtag: str,
        max_results: int = 10,
    ) -> list[Tweet]:
        """
        Search tweets by hashtag.
        
        Args:
            hashtag: Hashtag (without #)
            max_results: Number of results
            
        Returns:
            List of Tweet objects
        """
        query = f"#{hashtag} -is:retweet"  # Exclude retweets for cleaner results
        return await self.search_recent_tweets(query, max_results)

    async def search_by_topic(
        self,
        topic: str,
        max_results: int = 10,
        language: str = "en",
    ) -> list[Tweet]:
        """
        Search tweets by topic with quality filters.
        
        Args:
            topic: Topic to search for
            max_results: Number of results
            language: Language code
            
        Returns:
            List of Tweet objects
        """
        # Quality filters: minimum engagement, exclude retweets
        query = f"{topic} -is:retweet min_faves:5"
        return await self.search_recent_tweets(query, max_results, language)


# Example usage
async def main():
    """Example usage of Twitter source."""
    import os
    
    # Check if bearer token is available
    if not os.getenv("TWITTER_BEARER_TOKEN"):
        print("TWITTER_BEARER_TOKEN not set. Skipping example.")
        return
    
    twitter = TwitterSource()
    
    try:
        # Search recent tweets
        tweets = await twitter.search_by_topic("artificial intelligence", max_results=5)
        print(f"Found {len(tweets)} tweets about AI")
        
        for tweet in tweets:
            print(f"\n@{tweet.author_username}: {tweet.text[:100]}...")
            print(f"  ❤️ {tweet.like_count} | 🔁 {tweet.retweet_count} | 💬 {tweet.reply_count}")
        
        # Search by hashtag
        hashtag_tweets = await twitter.search_hashtag("MachineLearning", max_results=5)
        print(f"\nFound {len(hashtag_tweets)} tweets with #MachineLearning")
        
        # Get user info
        user = await twitter.get_user_by_username("openai")
        print(f"\n{user.name} (@{user.username})")
        print(f"  Followers: {user.followers_count:,}")
        print(f"  Tweets: {user.tweet_count:,}")
        
        # Get user's recent tweets
        user_tweets = await twitter.get_user_tweets(
            user.id,
            max_results=5,
            exclude_retweets=True,
        )
        print(f"\nRecent tweets from @{user.username}:")
        for tweet in user_tweets[:3]:
            print(f"  - {tweet.text[:80]}...")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await twitter.close()


if __name__ == "__main__":
    asyncio.run(main())
