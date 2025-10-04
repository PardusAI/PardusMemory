from abc import ABC, abstractmethod
from typing import List, Optional
import openai
from openai import OpenAI


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a given text."""
        pass


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI-based embedding service with configurable endpoints."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "text-embedding-3-small",
                 dimensions: Optional[int] = None):
        """
        Initialize OpenAI embedding service.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            model: Embedding model to use.
            dimensions: Embedding dimensions. If None, uses model default.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.dimensions = dimensions
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a given text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {e}")


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing purposes."""
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._cache = {}
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding based on text hash."""
        if text in self._cache:
            return self._cache[text]
        
        # Generate deterministic but pseudo-random embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to float values
        embedding = []
        for i in range(0, min(len(hash_hex), self.dimensions * 2), 2):
            hex_pair = hash_hex[i:i+2]
            val = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(val)
        
        # Pad or truncate to desired dimensions
        while len(embedding) < self.dimensions:
            embedding.append(0.0)
        
        embedding = embedding[:self.dimensions]
        self._cache[text] = embedding
        return embedding