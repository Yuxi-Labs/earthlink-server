"""
Test Wikipedia API integration - LIVE test with real API calls.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Direct import to avoid circular dependencies
sys.path.insert(0, str(src_path / "knowledge"))

import wikipedia


async def test_wikipedia_live():
    """Test Wikipedia with REAL API call."""
    print("Testing Wikipedia API (LIVE)...")
    
    wiki = wikipedia.WikipediaSource()
    
    try:
        # Search for articles
        print("  Searching for 'Python programming language'...")
        results = await wiki.search("Python programming language", limit=3)
        
        if not results:
            print("  ❌ FAILED: No results returned")
            return False
        
        print(f"  ✓ Found {len(results)} results")
        print(f"    Top result: {results[0]['title']}")
        
        # Get full article
        print(f"  Fetching article '{results[0]['title']}'...")
        article = await wiki.get_article(results[0]['title'])
        
        if not article:
            print("  ❌ FAILED: Could not fetch article")
            return False
        
        print(f"  ✓ Got article: {len(article.content)} chars")
        print(f"    URL: {article.url}")
        print(f"    Summary: {article.summary[:100]}...")
        
        await wiki.close()
        
        print("✅ Wikipedia API WORKING")
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        await wiki.close()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_wikipedia_live())
    sys.exit(0 if success else 1)
