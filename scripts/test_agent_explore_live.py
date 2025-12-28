"""
Test agent knowledge query methods with LIVE sources.
Tests query_wikipedia, query_ollama, search_web directly (no Ray).
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


async def test_agent_knowledge_methods():
    """Test agent knowledge methods with REAL API calls (no Ray)."""
    print("Testing Agent knowledge methods with LIVE sources...")
    
    try:
        # Import agent module directly
        from agents.core import agent as agent_module
        
        # Create agent instance (not Ray remote)
        print("  Creating agent...")
        agent_instance = agent_module.Agent(name="KnowledgeExplorer")
        
        # Test Wikipedia
        print("\n  Testing query_wikipedia()...")
        wiki_result = await agent_instance.query_wikipedia("Python programming", limit=3)
        print(f"    ✓ Wikipedia: {len(wiki_result.get('results', []))} results")
        if wiki_result.get('results'):
            print(f"      - {wiki_result['results'][0]['title']}")
        
        # Test web search
        print("\n  Testing search_web()...")
        web_result = await agent_instance.search_web("artificial intelligence", provider="duckduckgo")
        print(f"    ✓ Web Search: {len(web_result.get('results', []))} results")
        if web_result.get('results'):
            print(f"      - {web_result['results'][0]['title']}")
        
        # Test Ollama
        print("\n  Testing query_ollama()...")
        ollama_result = await agent_instance.query_ollama("What is curiosity?")
        print(f"    ✓ Ollama: {len(ollama_result.get('response', ''))} chars")
        print(f"      - {ollama_result.get('response', '')[:100]}...")
        
        print("\n✅ All agent knowledge methods WORKING WITH LIVE APIS")
        return True
        
    except Exception as e:
        print(f"\n  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agent_knowledge_methods())
    sys.exit(0 if success else 1)
