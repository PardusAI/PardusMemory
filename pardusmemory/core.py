from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pydantic import BaseModel


@dataclass
class MemoryEntry:
    """A single memory entry in the graph system."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    compressed: bool = False


class SimilarityFunction:
    """Base class for similarity functions."""
    
    def __call__(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        raise NotImplementedError


class CosineSimilarity(SimilarityFunction):
    """Standard cosine similarity between embeddings."""
    
    def __call__(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        if entry1.embedding is None or entry2.embedding is None:
            return 0.0
        
        dot_product = np.dot(entry1.embedding, entry2.embedding)
        norm1 = np.linalg.norm(entry1.embedding)
        norm2 = np.linalg.norm(entry2.embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class CustomSimilarity(SimilarityFunction):
    """Custom similarity function that combines multiple factors."""
    
    def __init__(self, 
                 similarity_weight: float = 0.1,
                 time_weight: float = 0.9,
                 base_similarity: Optional[SimilarityFunction] = None):
        self.similarity_weight = similarity_weight
        self.time_weight = time_weight
        self.base_similarity = base_similarity or CosineSimilarity()
    
    def __call__(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        similarity_score = self.base_similarity(entry1, entry2)
        time_diff = (entry1.timestamp - entry2.timestamp).total_seconds()
        time_diff_normalized = np.tanh(time_diff / (24 * 3600))  # Normalize by day
        
        return (self.similarity_weight * similarity_score + 
                self.time_weight * time_diff_normalized)


class MemoryGraph:
    """Core memory graph system with custom similarity rules."""
    
    def __init__(self, 
                 similarity_function: Optional[SimilarityFunction] = None,
                 embedding_service=None,
                 database=None):
        self.similarity_function = similarity_function or CosineSimilarity()
        self.embedding_service = embedding_service
        self.database = database
        self.entries: Dict[str, MemoryEntry] = {}
    
    def add_entry(self, 
                  content: str, 
                  entry_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new memory entry."""
        if entry_id is None:
            entry_id = f"entry_{len(self.entries)}_{datetime.now().timestamp()}"
        
        embedding = None
        if self.embedding_service:
            embedding_result = self.embedding_service.get_embedding(content)
            # Ensure embedding is numpy array
            if isinstance(embedding_result, list):
                embedding = np.array(embedding_result)
            else:
                embedding = embedding_result
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.entries[entry_id] = entry
        
        if self.database:
            self.database.store_entry(entry)
        
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry."""
        return self.entries.get(entry_id)
    
    def find_similar(self, 
                    query: str, 
                    top_k: int = 5,
                    threshold: float = 0.0) -> List[tuple[MemoryEntry, float]]:
        """Find similar entries based on the similarity function."""
        if not self.entries:
            return []
        
        # Create query entry
        query_embedding = None
        if self.embedding_service:
            query_embedding = self.embedding_service.get_embedding(query)
        
        query_entry = MemoryEntry(
            id="query",
            content=query,
            embedding=query_embedding
        )
        
        # Calculate similarities
        similarities = []
        for entry in self.entries.values():
            similarity = self.similarity_function(query_entry, entry)
            if similarity >= threshold:
                similarities.append((entry, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def update_similarity_function(self, similarity_function: SimilarityFunction):
        """Update the similarity function used for retrieval."""
        self.similarity_function = similarity_function
    
    def compress_entries(self, entry_ids: List[str], compressor) -> str:
        """Compress multiple entries into a single summary."""
        entries_to_compress = [self.entries[eid] for eid in entry_ids if eid in self.entries]
        
        if not entries_to_compress:
            return ""
        
        # Get content for compression - check if entries have conversation format
        content_to_compress = []
        for entry in entries_to_compress:
            if entry.metadata.get("role") and "conversation" in entry.metadata.get("type", ""):
                # If it's a conversation entry, format as message
                content_to_compress.append({
                    "role": entry.metadata.get("role", "unknown"),
                    "content": entry.content
                })
            else:
                content_to_compress.append(entry.content)
        
        summary = compressor.compress(content_to_compress)
        
        # Create new compressed entry
        summary_id = f"compressed_{datetime.now().timestamp()}"
        self.add_entry(summary, summary_id, {"compressed": True, "source_entries": entry_ids})
        
        # Mark original entries as compressed and remove them
        for entry_id in entry_ids:
            if entry_id in self.entries:
                self.entries[entry_id].compressed = True
                del self.entries[entry_id]
        
        return summary_id
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all non-compressed entries."""
        return [entry for entry in self.entries.values() if not entry.compressed]
    
    def compress_conversation(self, conversation_messages: List[Dict[str, Any]], compressor) -> str:
        """
        Compress a conversation in OpenAI format and store it as a memory entry.
        
        Args:
            conversation_messages: List of conversation messages in OpenAI format
            compressor: Compressor instance to use
            
        Returns:
            ID of the compressed memory entry
        """
        if not conversation_messages:
            return ""
        
        # Compress the conversation
        summary = compressor.compress(conversation_messages)
        
        # Create memory entry for the compressed conversation
        conversation_id = f"conversation_{datetime.now().timestamp()}"
        entry_id = self.add_entry(
            summary,
            conversation_id,
            {
                "type": "compressed_conversation",
                "original_messages": len(conversation_messages),
                "compressed": True
            }
        )
        
        return entry_id