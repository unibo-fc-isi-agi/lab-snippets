"""
Embedding model of this lecture, via any OpenAI-compatible API (by default, a local Ollama server).
`embed` uses OpenAI's client directly, `langchain_embeddings` wraps the same model for LangChain.

Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL
(defaults: Ollama's nomic-embed-text, install it with: ollama pull nomic-embed-text).
"""
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

base_url = os.environ.get("EMBEDDINGS_BASE_URL", "http://localhost:11434/v1")
api_key = os.environ.get("EMBEDDINGS_API_KEY", "ollama")  # Ollama ignores it, but the client requires one
model = os.environ.get("EMBEDDINGS_MODEL", "nomic-embed-text")

client = OpenAI(base_url=base_url, api_key=api_key)


def embed(texts: list[str]) -> list[list[float]]:
    """One vector per text, computed in a single (batch) request."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]  # same order as the input texts


def langchain_embeddings() -> OpenAIEmbeddings:
    """The same model, as a LangChain Embeddings object (e.g. for LangChain's vector stores)."""
    return OpenAIEmbeddings(
        base_url=base_url, api_key=api_key, model=model,
        check_embedding_ctx_length=False,  # otherwise texts are sent as OpenAI's token IDs, which other servers reject
    )
