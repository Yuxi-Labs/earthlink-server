"""Reddit knowledge source - access to Reddit via API."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx


@dataclass
class RedditPost:
    """Reddit post data."""

    id: str
    title: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    selftext: str = ""
    url: str = ""
    permalink: str = ""
    is_self: bool = True
    link_flair_text: str = ""
    upvote_ratio: float = 0.0


@dataclass
class RedditComment:
    """Reddit comment data."""

    id: str
    post_id: str
    author: str
    body: str
    score: int
    created_utc: float
    permalink: str = ""
    parent_id: str = ""


class RedditSource:
    """
    Reddit knowledge source.
    
    Provides access to Reddit content for agents to learn from.
    Uses Reddit's public JSON API (no authentication required for read-only).
    """

    def __init__(
        self,
        user_agent: str = "Earthlink/1.0 (research project)",
    ):
        self.base_url = "https://www.reddit.com"
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

    async def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make request to Reddit API."""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}.json"
        
        try:
            response = await client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Reddit API request failed: {e}") from e

    async def get_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",  # hot, new, top, rising
        limit: int = 25,
        time_filter: str = "day",  # hour, day, week, month, year, all
    ) -> list[RedditPost]:
        """
        Get posts from a subreddit.
        
        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort method (hot, new, top, rising)
            limit: Number of posts to return (max 100)
            time_filter: Time filter for 'top' sort
            
        Returns:
            List of RedditPost objects
        """
        endpoint = f"/r/{subreddit}/{sort}"
        params = {"limit": min(limit, 100)}
        
        if sort == "top":
            params["t"] = time_filter
        
        data = await self._make_request(endpoint, params)
        posts = []
        
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            posts.append(RedditPost(
                id=post_data.get("id", ""),
                title=post_data.get("title", ""),
                subreddit=post_data.get("subreddit", ""),
                author=post_data.get("author", ""),
                score=post_data.get("score", 0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=post_data.get("created_utc", 0.0),
                selftext=post_data.get("selftext", ""),
                url=post_data.get("url", ""),
                permalink=post_data.get("permalink", ""),
                is_self=post_data.get("is_self", True),
                link_flair_text=post_data.get("link_flair_text", ""),
                upvote_ratio=post_data.get("upvote_ratio", 0.0),
            ))
        
        return posts

    async def get_post_comments(
        self,
        subreddit: str,
        post_id: str,
        limit: int = 100,
        sort: str = "best",  # best, top, new, controversial, old
    ) -> list[RedditComment]:
        """
        Get comments from a Reddit post.
        
        Args:
            subreddit: Subreddit name
            post_id: Post ID
            limit: Number of comments to return
            sort: Sort method
            
        Returns:
            List of RedditComment objects
        """
        endpoint = f"/r/{subreddit}/comments/{post_id}"
        params = {
            "limit": limit,
            "sort": sort,
        }
        
        data = await self._make_request(endpoint, params)
        
        # Reddit returns [post_data, comments_data]
        if len(data) < 2:
            return []
        
        comments = []
        self._extract_comments(
            data[1].get("data", {}).get("children", []),
            comments,
            post_id,
        )
        
        return comments

    def _extract_comments(
        self,
        children: list[dict[str, Any]],
        comments: list[RedditComment],
        post_id: str,
    ) -> None:
        """Recursively extract comments from nested structure."""
        for child in children:
            if child.get("kind") != "t1":  # t1 = comment
                continue
            
            comment_data = child.get("data", {})
            
            comments.append(RedditComment(
                id=comment_data.get("id", ""),
                post_id=post_id,
                author=comment_data.get("author", ""),
                body=comment_data.get("body", ""),
                score=comment_data.get("score", 0),
                created_utc=comment_data.get("created_utc", 0.0),
                permalink=comment_data.get("permalink", ""),
                parent_id=comment_data.get("parent_id", ""),
            ))
            
            # Recursively extract replies
            replies = comment_data.get("replies")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                self._extract_comments(reply_children, comments, post_id)

    async def search_posts(
        self,
        query: str,
        subreddit: str | None = None,
        sort: str = "relevance",  # relevance, hot, top, new, comments
        time_filter: str = "all",
        limit: int = 25,
    ) -> list[RedditPost]:
        """
        Search Reddit posts.
        
        Args:
            query: Search query
            subreddit: Limit to specific subreddit (optional)
            sort: Sort method
            time_filter: Time filter
            limit: Number of results
            
        Returns:
            List of RedditPost objects
        """
        if subreddit:
            endpoint = f"/r/{subreddit}/search"
        else:
            endpoint = "/search"
        
        params = {
            "q": query,
            "sort": sort,
            "t": time_filter,
            "limit": min(limit, 100),
            "restrict_sr": "true" if subreddit else "false",
        }
        
        data = await self._make_request(endpoint, params)
        posts = []
        
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            posts.append(RedditPost(
                id=post_data.get("id", ""),
                title=post_data.get("title", ""),
                subreddit=post_data.get("subreddit", ""),
                author=post_data.get("author", ""),
                score=post_data.get("score", 0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=post_data.get("created_utc", 0.0),
                selftext=post_data.get("selftext", ""),
                url=post_data.get("url", ""),
                permalink=post_data.get("permalink", ""),
                is_self=post_data.get("is_self", True),
                link_flair_text=post_data.get("link_flair_text", ""),
                upvote_ratio=post_data.get("upvote_ratio", 0.0),
            ))
        
        return posts

    async def get_subreddit_info(self, subreddit: str) -> dict[str, Any]:
        """
        Get information about a subreddit.
        
        Args:
            subreddit: Subreddit name
            
        Returns:
            Subreddit metadata
        """
        endpoint = f"/r/{subreddit}/about"
        data = await self._make_request(endpoint)
        
        subreddit_data = data.get("data", {})
        return {
            "name": subreddit_data.get("display_name", ""),
            "title": subreddit_data.get("title", ""),
            "description": subreddit_data.get("public_description", ""),
            "subscribers": subreddit_data.get("subscribers", 0),
            "active_users": subreddit_data.get("accounts_active", 0),
            "created_utc": subreddit_data.get("created_utc", 0.0),
            "is_nsfw": subreddit_data.get("over18", False),
            "url": subreddit_data.get("url", ""),
        }

    async def get_trending_subreddits(self) -> list[str]:
        """
        Get trending subreddits.
        
        Returns:
            List of trending subreddit names
        """
        endpoint = "/api/trending_subreddits"
        data = await self._make_request(endpoint)
        
        # Parse trending subreddits from response
        # The API returns markdown-formatted text
        trending = []
        for name in data.get("subreddit_names", []):
            trending.append(name)
        
        return trending


# Example usage
async def main():
    """Example usage of Reddit source."""
    reddit = RedditSource()
    
    try:
        # Get posts from a subreddit
        posts = await reddit.get_subreddit_posts("artificial", sort="hot", limit=10)
        print(f"Found {len(posts)} posts from r/artificial")
        
        for post in posts[:3]:
            print(f"\n{post.title}")
            print(f"  Score: {post.score} | Comments: {post.num_comments}")
            print(f"  {post.permalink}")
        
        # Get comments from first post
        if posts:
            first_post = posts[0]
            comments = await reddit.get_post_comments(
                first_post.subreddit,
                first_post.id,
                limit=10,
            )
            print(f"\nFound {len(comments)} comments")
        
        # Search for posts
        search_results = await reddit.search_posts(
            "machine learning",
            subreddit="artificial",
            limit=5,
        )
        print(f"\nSearch found {len(search_results)} results")
        
        # Get subreddit info
        info = await reddit.get_subreddit_info("artificial")
        print(f"\nr/artificial has {info['subscribers']:,} subscribers")
        
    finally:
        await reddit.close()


if __name__ == "__main__":
    asyncio.run(main())
