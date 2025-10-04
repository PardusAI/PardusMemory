from .core import MemoryGraph, MemoryEntry, SimilarityFunction, CosineSimilarity, CustomSimilarity
from .embedding_service import EmbeddingService, OpenAIEmbeddingService, MockEmbeddingService
from .database import JSONDatabase
from .compression import LLMCompressor, OpenAICompressor, MockCompressor

__version__ = "0.1.0"
__all__ = [
    "MemoryGraph", "MemoryEntry", "SimilarityFunction", "CosineSimilarity", "CustomSimilarity",
    "EmbeddingService", "OpenAIEmbeddingService", "MockEmbeddingService",
    "JSONDatabase", "LLMCompressor", "OpenAICompressor", "MockCompressor"
]