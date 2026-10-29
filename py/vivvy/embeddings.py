"""Built-in embedding adapters for vivvy.

Provides an `Embedder` base interface and concrete adapters for:
- OpenAI (`OpenAIEmbedder`)
- Sentence-Transformers (`SentenceTransformersEmbedder`)
- Local Ollama (`OllamaEmbedder`)
- Zero-dependency deterministic testing (`DeterministicEmbedder`)
"""

from abc import ABC, abstractmethod
import hashlib
import json
import math
from typing import Any, List, Optional, Sequence
import urllib.request
import urllib.error


class Embedder(ABC):
    """Abstract base class for embedding adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical name/identifier of the embedding model (e.g. 'text-embedding-3-small')."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The dimensionality of embeddings produced by this model (e.g. 1536)."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector of floats."""
        pass

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple text strings into vectors of floats.

        Subclasses may override this with an optimized batched implementation.
        """
        return [self.embed(t) for t in texts]


class OpenAIEmbedder(Embedder):
    """Embedding adapter for the OpenAI Embeddings API (and OpenAI-compatible endpoints)."""

    # Known dimensions for common OpenAI models
    KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize the OpenAI embedding adapter.

        Args:
            model: Model identifier (e.g. 'text-embedding-3-small').
            dimensions: Optional vector dimensionality override (supported by text-embedding-3-*).
            api_key: OpenAI API key. If omitted, uses OPENAI_API_KEY environment variable.
            client: Pre-instantiated `openai.OpenAI` client.
            **kwargs: Extra parameters passed to `openai.OpenAI(...)` if client is not provided.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required to use OpenAIEmbedder. "
                "Install it with: pip install 'vivvy[openai]' or pip install openai"
            ) from e

        self._model = model
        self._dimensions = dimensions or self.KNOWN_DIMENSIONS.get(model)
        self._custom_dims_supported = model.startswith("text-embedding-3")

        if client is not None:
            self._client = client
        else:
            self._client = openai.OpenAI(api_key=api_key, **kwargs)

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Probe dimension on first access if unknown
            sample = self.embed("dim_probe")
            self._dimensions = len(sample)
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        params: dict[str, Any] = {
            "model": self._model,
            "input": text,
        }
        if self._custom_dims_supported and self._dimensions is not None:
            params["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**params)
        embedding = response.data[0].embedding
        if self._dimensions is None:
            self._dimensions = len(embedding)
        return embedding

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        params: dict[str, Any] = {
            "model": self._model,
            "input": list(texts),
        }
        if self._custom_dims_supported and self._dimensions is not None:
            params["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**params)
        # Ensure ordered by index
        sorted_data = sorted(response.data, key=lambda x: getattr(x, "index", 0))
        results = [item.embedding for item in sorted_data]
        if self._dimensions is None and results:
            self._dimensions = len(results[0])
        return results


class SentenceTransformersEmbedder(Embedder):
    """Embedding adapter for local sentence-transformers models."""

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        model: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize local sentence-transformers adapter.

        Args:
            model_name_or_path: Hugging Face hub model id or local directory path.
            device: Computation device ('cpu', 'cuda', 'mps', etc.).
            model: Pre-instantiated `SentenceTransformer` object.
            **kwargs: Extra parameters passed to `SentenceTransformer(...)`.
        """
        try:
            import sentence_transformers
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required to use SentenceTransformersEmbedder. "
                "Install it with: pip install 'vivvy[sentence-transformers]' or pip install sentence-transformers"
            ) from e

        self._model_name = model_name_or_path
        if model is not None:
            self._model = model
        else:
            self._model = sentence_transformers.SentenceTransformer(
                model_name_or_path, device=device, **kwargs
            )
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(list(texts), convert_to_numpy=True)
        return vecs.tolist()


class OllamaEmbedder(Embedder):
    """Embedding adapter for local Ollama instances (uses standard library HTTP)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
    ):
        """Initialize the Ollama embedding adapter.

        Args:
            model: Name of the Ollama embedding model (e.g. 'nomic-embed-text').
            base_url: Base URL where Ollama is listening.
            dimensions: Expected embedding dimensions. If None, probed dynamically.
            timeout: HTTP request timeout in seconds.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            sample = self.embed("dim_probe")
            self._dimensions = len(sample)
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        url = f"{self._base_url}/api/embeddings"
        payload = json.dumps({"model": self._model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embedding = data["embedding"]
                if self._dimensions is None:
                    self._dimensions = len(embedding)
                return embedding
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {e}") from e

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class DeterministicEmbedder(Embedder):
    """Deterministic, zero-dependency embedding generator for testing and offline development.

    Generates reproducible unit-norm vector projections from text strings.
    """

    def __init__(self, name: str = "test-model", dimensions: int = 1536):
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._name = name
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> List[float]:
        dims = self._dimensions
        vec = [0.0] * dims

        # Use SHA-256 blocks to derive deterministic pseudo-random components
        text_bytes = text.encode("utf-8")
        h = hashlib.sha256(text_bytes).digest()

        for i in range(dims):
            # Seed each dimension from hash + index
            idx_bytes = i.to_bytes(4, byteorder="little")
            component_hash = hashlib.sha256(h + idx_bytes).digest()
            val = int.from_bytes(component_hash[:4], byteorder="little", signed=True)
            vec[i] = float(val) / (2**31)

        # Normalize to unit vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]
