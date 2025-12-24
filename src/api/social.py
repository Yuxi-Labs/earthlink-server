"""Social media API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..knowledge.reddit import RedditSource
from ..knowledge.twitter import TwitterSource

router = APIRouter(prefix="/social", tags=["social"])

# Initialize sources
reddit = RedditSource()
twitter = None  # Will be initialized if TWITTER_BEARER_TOKEN is set

try:
    twitter = TwitterSource()
except ValueError:
    # Twitter token not available
    pass


@router.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await reddit.close()
    if twitter:
        await twitter.close()


# Reddit endpoints
@router.get("/reddit/subreddit/{subreddit}")
async def get_subreddit_posts(
    subreddit: str,
    sort: str = Query("hot", regex="^(hot|new|top|rising)$"),
    limit: int = Query(25, ge=1, le=100),
    time_filter: str = Query("day", regex="^(hour|day|week|month|year|all)$"),
) -> dict[str, Any]:
    """
    Get posts from a subreddit.
    
    - **subreddit**: Subreddit name (without r/)
    - **sort**: Sort method (hot, new, top, rising)
    - **limit**: Number of posts (1-100)
    - **time_filter**: Time filter for 'top' sort
    """
    try:
        posts = await reddit.get_subreddit_posts(
            subreddit=subreddit,
            sort=sort,
            limit=limit,
            time_filter=time_filter,
        )
        return {
            "subreddit": subreddit,
            "count": len(posts),
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "author": p.author,
                    "score": p.score,
                    "num_comments": p.num_comments,
                    "created_utc": p.created_utc,
                    "selftext": p.selftext,
                    "url": p.url,
                    "permalink": f"https://reddit.com{p.permalink}",
                    "upvote_ratio": p.upvote_ratio,
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reddit/post/{subreddit}/{post_id}/comments")
async def get_post_comments(
    subreddit: str,
    post_id: str,
    limit: int = Query(100, ge=1, le=500),
    sort: str = Query("best", regex="^(best|top|new|controversial|old)$"),
) -> dict[str, Any]:
    """
    Get comments from a Reddit post.
    
    - **subreddit**: Subreddit name
    - **post_id**: Post ID
    - **limit**: Number of comments
    - **sort**: Sort method
    """
    try:
        comments = await reddit.get_post_comments(
            subreddit=subreddit,
            post_id=post_id,
            limit=limit,
            sort=sort,
        )
        return {
            "post_id": post_id,
            "count": len(comments),
            "comments": [
                {
                    "id": c.id,
                    "author": c.author,
                    "body": c.body,
                    "score": c.score,
                    "created_utc": c.created_utc,
                    "parent_id": c.parent_id,
                }
                for c in comments
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reddit/search")
async def search_reddit(
    query: str = Query(..., min_length=1),
    subreddit: str | None = None,
    sort: str = Query("relevance", regex="^(relevance|hot|top|new|comments)$"),
    time_filter: str = Query("all", regex="^(hour|day|week|month|year|all)$"),
    limit: int = Query(25, ge=1, le=100),
) -> dict[str, Any]:
    """
    Search Reddit posts.
    
    - **query**: Search query
    - **subreddit**: Limit to specific subreddit (optional)
    - **sort**: Sort method
    - **time_filter**: Time filter
    - **limit**: Number of results
    """
    try:
        posts = await reddit.search_posts(
            query=query,
            subreddit=subreddit,
            sort=sort,
            time_filter=time_filter,
            limit=limit,
        )
        return {
            "query": query,
            "subreddit": subreddit,
            "count": len(posts),
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "subreddit": p.subreddit,
                    "author": p.author,
                    "score": p.score,
                    "num_comments": p.num_comments,
                    "permalink": f"https://reddit.com{p.permalink}",
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reddit/subreddit/{subreddit}/info")
async def get_subreddit_info(subreddit: str) -> dict[str, Any]:
    """
    Get information about a subreddit.
    
    - **subreddit**: Subreddit name
    """
    try:
        info = await reddit.get_subreddit_info(subreddit)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Twitter endpoints
@router.get("/twitter/search")
async def search_twitter(
    query: str = Query(..., min_length=1),
    max_results: int = Query(10, ge=10, le=100),
    language: str | None = None,
) -> dict[str, Any]:
    """
    Search recent tweets (last 7 days).
    
    - **query**: Search query (Twitter search syntax)
    - **max_results**: Number of results (10-100)
    - **language**: Language code (e.g., "en")
    
    Requires TWITTER_BEARER_TOKEN environment variable.
    """
    if not twitter:
        raise HTTPException(
            status_code=503,
            detail="Twitter API not configured. Set TWITTER_BEARER_TOKEN environment variable.",
        )
    
    try:
        tweets = await twitter.search_recent_tweets(
            query=query,
            max_results=max_results,
            language=language,
        )
        return {
            "query": query,
            "count": len(tweets),
            "tweets": [
                {
                    "id": t.id,
                    "text": t.text,
                    "author_username": t.author_username,
                    "author_name": t.author_name,
                    "created_at": t.created_at,
                    "like_count": t.like_count,
                    "retweet_count": t.retweet_count,
                    "reply_count": t.reply_count,
                    "hashtags": t.hashtags,
                    "mentions": t.mentions,
                }
                for t in tweets
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twitter/hashtag/{hashtag}")
async def search_twitter_hashtag(
    hashtag: str,
    max_results: int = Query(10, ge=10, le=100),
) -> dict[str, Any]:
    """
    Search tweets by hashtag.
    
    - **hashtag**: Hashtag (without #)
    - **max_results**: Number of results
    
    Requires TWITTER_BEARER_TOKEN environment variable.
    """
    if not twitter:
        raise HTTPException(
            status_code=503,
            detail="Twitter API not configured. Set TWITTER_BEARER_TOKEN environment variable.",
        )
    
    try:
        tweets = await twitter.search_hashtag(hashtag, max_results)
        return {
            "hashtag": hashtag,
            "count": len(tweets),
            "tweets": [
                {
                    "id": t.id,
                    "text": t.text,
                    "author_username": t.author_username,
                    "like_count": t.like_count,
                    "retweet_count": t.retweet_count,
                }
                for t in tweets
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twitter/topic/{topic}")
async def search_twitter_topic(
    topic: str,
    max_results: int = Query(10, ge=10, le=100),
    language: str = Query("en"),
) -> dict[str, Any]:
    """
    Search tweets by topic with quality filters.
    
    - **topic**: Topic to search for
    - **max_results**: Number of results
    - **language**: Language code
    
    Requires TWITTER_BEARER_TOKEN environment variable.
    """
    if not twitter:
        raise HTTPException(
            status_code=503,
            detail="Twitter API not configured. Set TWITTER_BEARER_TOKEN environment variable.",
        )
    
    try:
        tweets = await twitter.search_by_topic(topic, max_results, language)
        return {
            "topic": topic,
            "count": len(tweets),
            "tweets": [
                {
                    "id": t.id,
                    "text": t.text,
                    "author_username": t.author_username,
                    "like_count": t.like_count,
                    "retweet_count": t.retweet_count,
                }
                for t in tweets
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twitter/user/{username}")
async def get_twitter_user(username: str) -> dict[str, Any]:
    """
    Get user information by username.
    
    - **username**: Twitter username (without @)
    
    Requires TWITTER_BEARER_TOKEN environment variable.
    """
    if not twitter:
        raise HTTPException(
            status_code=503,
            detail="Twitter API not configured. Set TWITTER_BEARER_TOKEN environment variable.",
        )
    
    try:
        user = await twitter.get_user_by_username(username)
        return {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "description": user.description,
            "followers_count": user.followers_count,
            "following_count": user.following_count,
            "tweet_count": user.tweet_count,
            "verified": user.verified,
            "location": user.location,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twitter/user/{user_id}/tweets")
async def get_user_tweets(
    user_id: str,
    max_results: int = Query(10, ge=5, le=100),
    exclude_retweets: bool = False,
    exclude_replies: bool = False,
) -> dict[str, Any]:
    """
    Get tweets from a user's timeline.
    
    - **user_id**: Twitter user ID
    - **max_results**: Number of results (5-100)
    - **exclude_retweets**: Exclude retweets
    - **exclude_replies**: Exclude replies
    
    Requires TWITTER_BEARER_TOKEN environment variable.
    """
    if not twitter:
        raise HTTPException(
            status_code=503,
            detail="Twitter API not configured. Set TWITTER_BEARER_TOKEN environment variable.",
        )
    
    try:
        tweets = await twitter.get_user_tweets(
            user_id=user_id,
            max_results=max_results,
            exclude_retweets=exclude_retweets,
            exclude_replies=exclude_replies,
        )
        return {
            "user_id": user_id,
            "count": len(tweets),
            "tweets": [
                {
                    "id": t.id,
                    "text": t.text,
                    "created_at": t.created_at,
                    "like_count": t.like_count,
                    "retweet_count": t.retweet_count,
                }
                for t in tweets
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
