"""Ollama gateway - access to LLMs via Ollama."""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


@dataclass
class OllamaResponse:
    """Response from Ollama."""

    model: str
    response: str
    done: bool = True
    context: list[int] = field(default_factory=list)
    total_duration: int = 0
    load_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


class OllamaGateway:
    """
    Gateway to LLM models via Ollama.
    
    This is NOT the agent's cognition - agents think for themselves.
    Ollama is an external knowledge source that agents can query,
    like looking something up in a library.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,  # LLM responses can be slow
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        client = await self._get_client()

        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()

        return data.get("models", [])

    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        client = await self._get_client()

        try:
            response = await client.post(
                "/api/pull",
                json={"name": model},
                timeout=600.0,  # Model downloads can be slow
            )
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        context: list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> OllamaResponse:
        """
        Generate text completion.
        
        Args:
            prompt: The prompt to generate from
            model: Model to use (defaults to default_model)
            system: System prompt
            context: Previous context for continuation
            options: Model options (temperature, etc.)
            
        Returns:
            OllamaResponse with generated text
        """
        client = await self._get_client()

        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        if options:
            payload["options"] = options

        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        return OllamaResponse(
            model=data.get("model", ""),
            response=data.get("response", ""),
            done=data.get("done", True),
            context=data.get("context", []),
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0),
        )

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Generate text completion with streaming.
        
        Yields chunks of generated text.
        """
        client = await self._get_client()

        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": True,
        }

        if system:
            payload["system"] = system

        async with client.stream("POST", "/api/generate", json=payload) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> OllamaResponse:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}
            model: Model to use
            options: Model options
            
        Returns:
            OllamaResponse with assistant reply
        """
        client = await self._get_client()

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
        }

        if options:
            payload["options"] = options

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        return OllamaResponse(
            model=data.get("model", ""),
            response=message.get("content", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0),
        )

    async def embeddings(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Get embeddings for text.
        
        Note: Requires an embedding model like nomic-embed-text.
        """
        client = await self._get_client()

        payload = {
            "model": model or "nomic-embed-text",
            "prompt": text,
        }

        response = await client.post("/api/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        return data.get("embedding", [])

    # -------------------------------------------------------------------------
    # Agent-specific query methods
    # -------------------------------------------------------------------------

    async def query_knowledge(
        self,
        question: str,
        context: str = "",
        model: str | None = None,
    ) -> str:
        """
        Query LLM for factual knowledge.
        
        This is how agents access external knowledge - NOT for decision making.
        """
        system = """You are a knowledge source providing factual information.
Answer the question concisely and accurately.
If you're unsure, say so. Do not make up information."""

        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"
        else:
            prompt = question

        response = await self.generate(
            prompt=prompt,
            model=model,
            system=system,
            options={
                "temperature": 0.1,  # Low temperature for factual responses
            },
        )

        return response.response

    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        model: str | None = None,
    ) -> str:
        """Summarize text using LLM."""
        prompt = f"""Summarize the following text in {max_length} words or less:

{text}

Summary:"""

        response = await self.generate(
            prompt=prompt,
            model=model,
            options={"temperature": 0.3},
        )

        return response.response

    async def extract_entities(
        self,
        text: str,
        model: str | None = None,
    ) -> list[dict[str, str]]:
        """Extract named entities from text."""
        prompt = f"""Extract named entities from the following text.
Return as JSON list with "name" and "type" fields.
Types: PERSON, ORGANIZATION, LOCATION, DATE, CONCEPT

Text: {text}

JSON:"""

        response = await self.generate(
            prompt=prompt,
            model=model,
            options={"temperature": 0.1},
        )

        # Parse response
        try:
            import json
            # Try to extract JSON from response
            text = response.response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception:
            return []

    async def extract_relations(
        self,
        text: str,
        model: str | None = None,
    ) -> list[dict[str, str]]:
        """Extract relations between entities."""
        prompt = f"""Extract relationships between entities from the following text.
Return as JSON list with "subject", "predicate", "object" fields.

Text: {text}

JSON:"""

        response = await self.generate(
            prompt=prompt,
            model=model,
            options={"temperature": 0.1},
        )

        try:
            import json
            text = response.response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception:
            return []
