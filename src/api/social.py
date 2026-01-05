"""Social media API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.data.sources import RedditSource, XSource

router = APIRouter(prefix="/social", tags=["social"])

# Initialize sources
reddit = RedditSource()
x = None  # Will be initialized if X_BEARER_TOKEN is set

try:
    x = XSource()
except ValueError:
    # X token not available
    pass


@router.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    await reddit.close()
    if x:
        await x.close()


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


# X endpoints
@router.get("/x/search")
async def search_x(
    query: str = Query(..., min_length=1),
    max_results: int = Query(10, ge=10, le=100),
    language: str | None = None,
) -> dict[str, Any]:
    """
    Search recent posts (last 7 days).
    
    - **query**: Search query (X search syntax)
    - **max_results**: Number of results (10-100)
    - **language**: Language code (e.g., "en")
    
    Requires X_BEARER_TOKEN environment variable.
    """
    if not x:
        raise HTTPException(
            status_code=503,
            detail="X API not configured. Set X_BEARER_TOKEN environment variable.",
        )
    
    try:
        posts = await x.search_recent_posts(
            query=query,
            max_results=max_results,
            language=language,
        )
        return {
            "query": query,
            "count": len(posts),
            "posts": [
                {
                    "id": p.id,
                    "text": p.text,
                    "author_username": p.author_username,
                    "author_name": p.author_name,
                    "created_at": p.created_at,
                    "like_count": p.like_count,
                    "repost_count": p.repost_count,
                    "reply_count": p.reply_count,
                    "hashtags": p.hashtags,
                    "mentions": p.mentions,
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/x/hashtag/{hashtag}")
async def search_x_hashtag(
    hashtag: str,
    max_results: int = Query(10, ge=10, le=100),
) -> dict[str, Any]:
    """
    Search posts by hashtag.
    
    - **hashtag**: Hashtag (without #)
    - **max_results**: Number of results
    
    Requires X_BEARER_TOKEN environment variable.
    """
    if not x:
        raise HTTPException(
            status_code=503,
            detail="X API not configured. Set X_BEARER_TOKEN environment variable.",
        )
    
    try:
        posts = await x.search_hashtag(hashtag, max_results)
        return {
            "hashtag": hashtag,
            "count": len(posts),
            "posts": [
                {
                    "id": p.id,
                    "text": p.text,
                    "author_username": p.author_username,
                    "like_count": p.like_count,
                    "repost_count": p.repost_count,
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/x/topic/{topic}")
async def search_x_topic(
    topic: str,
    max_results: int = Query(10, ge=10, le=100),
    language: str = Query("en"),
) -> dict[str, Any]:
    """
    Search posts by topic with quality filters.
    
    - **topic**: Topic to search for
    - **max_results**: Number of results
    - **language**: Language code
    
    Requires X_BEARER_TOKEN environment variable.
    """
    if not x:
        raise HTTPException(
            status_code=503,
            detail="X API not configured. Set X_BEARER_TOKEN environment variable.",
        )
    
    try:
        posts = await x.search_by_topic(topic, max_results, language)
        return {
            "topic": topic,
            "count": len(posts),
            "posts": [
                {
                    "id": p.id,
                    "text": p.text,
                    "author_username": p.author_username,
                    "like_count": p.like_count,
                    "repost_count": p.repost_count,
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/x/user/{username}")
async def get_x_user(username: str) -> dict[str, Any]:
    """
    Get user information by username.
    
    - **username**: X username (without @)
    
    Requires X_BEARER_TOKEN environment variable.
    """
    if not x:
        raise HTTPException(
            status_code=503,
            detail="X API not configured. Set X_BEARER_TOKEN environment variable.",
        )
    
    try:
        user = await x.get_user_by_username(username)
        return {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "description": user.description,
            "followers_count": user.followers_count,
            "following_count": user.following_count,
            "post_count": user.post_count,
            "verified": user.verified,
            "location": user.location,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/x/user/{user_id}/posts")
async def get_user_posts(
    user_id: str,
    max_results: int = Query(10, ge=5, le=100),
    exclude_reposts: bool = False,
    exclude_replies: bool = False,
) -> dict[str, Any]:
    """
    Get posts from a user's timeline.
    
    - **user_id**: X user ID
    - **max_results**: Number of results (5-100)
    - **exclude_reposts**: Exclude reposts
    - **exclude_replies**: Exclude replies
    
    Requires X_BEARER_TOKEN environment variable.
    """
    if not x:
        raise HTTPException(
            status_code=503,
            detail="X API not configured. Set X_BEARER_TOKEN environment variable.",
        )
    
    try:
        posts = await x.get_user_posts(
            user_id=user_id,
            max_results=max_results,
            exclude_reposts=exclude_reposts,
            exclude_replies=exclude_replies,
        )
        return {
            "user_id": user_id,
            "count": len(posts),
            "posts": [
                {
                    "id": p.id,
                    "text": p.text,
                    "created_at": p.created_at,
                    "like_count": p.like_count,
                    "repost_count": p.repost_count,
                }
                for p in posts
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
