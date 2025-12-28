"""
Test Ollama API integration - LIVE test with real API calls.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(src_path / "knowledge"))

import ollama


async def test_ollama_live():
    """Test Ollama with REAL API call."""
    print("Testing Ollama API (LIVE)...")
    
    gateway = ollama.OllamaGateway(base_url="http://localhost:11434")
    
    try:
        # Check if available
        print("  Checking if Ollama is available...")
        available = await gateway.is_available()
        
        if not available:
            print("  ❌ Ollama is NOT RUNNING")
            print("     Start with: docker compose up ollama -d")
            print("     Or: ollama serve")
            return False
        
        print("  ✓ Ollama is running")
        
        # List models
        print("  Listing available models...")
        models = await gateway.list_models()
        
        if not models:
            print("  ❌ No models available")
            print("     Pull a model with: ollama pull llama3.2")
            return False
        
        print(f"  ✓ Found {len(models)} models:")
        for model in models:
            print(f"    - {model['name']}")
        
        # Generate response
        model_name = models[0]["name"]
        print(f"  Generating response with {model_name}...")
        response = await gateway.generate(
            prompt="What is curiosity in one sentence?",
            model=model_name
        )
        
        if not response or not response.response:
            print("  ❌ No response from model")
            return False
        
        print(f"  ✓ Got response ({len(response.response)} chars):")
        print(f"    {response.response[:200]}...")
        
        await gateway.close()
        
        print("✅ Ollama API WORKING")
        return True
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        await gateway.close()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_live())
    sys.exit(0 if success else 1)
