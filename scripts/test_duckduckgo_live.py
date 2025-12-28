"""
Test DuckDuckGo search API integration - LIVE test with real API calls.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "knowledge"))

import search


async def test_duckduckgo_live():
    """Test DuckDuckGo with REAL API call."""
    print("Testing DuckDuckGo Search API (LIVE)...")
    
    ddg = search.DuckDuckGoSearchProvider()
    
    try:
        # Search
        print("  Searching for 'artificial intelligence'...")
        results = await ddg.search("artificial intelligence", num_results=5)
        
        if not results:
            print("  ❌ FAILED: No results returned")
            return False
        
        print(f"  ✓ Found {len(results)} results")
        for i, result in enumerate(results[:3], 1):
            print(f"    {i}. {result.title}")
            print(f"       {result.url}")
            print(f"       {result.snippet[:80]}...")
        
        print("✅ DuckDuckGo Search API WORKING")
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_duckduckgo_live())
    sys.exit(0 if success else 1)
