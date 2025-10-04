"""
Basic usage example of PardusMemory library.

This example demonstrates:
1. Setting up a memory graph with custom similarity
2. Adding and retrieving memory entries
3. Using different embedding services
4. Knowledge compression
5. Database operations
"""

from pardusmemory import (
    MemoryGraph, 
    MockEmbeddingService, 
    JSONDatabase, 
    MockCompressor,
    CustomSimilarity,
    CosineSimilarity,
    MemoryEntry
)


def basic_example():
    """Basic memory graph usage example."""
    print("=== Basic Memory Graph Example ===\n")
    
    # Initialize components
    embedding_service = MockEmbeddingService(dimensions=384)
    database = JSONDatabase("basic_example_memory.json")
    compressor = MockCompressor()
    
    # Create memory graph with standard cosine similarity
    memory_graph = MemoryGraph(
        similarity_function=CosineSimilarity(),
        embedding_service=embedding_service,
        database=database
    )
    
    # Add memory entries
    print("1. Adding memory entries...")
    entries = [
        "Python is a high-level programming language with dynamic typing.",
        "Machine learning algorithms can be implemented using Python libraries like scikit-learn.",
        "Neural networks are a subset of machine learning inspired by biological neurons.",
        "Data preprocessing is a crucial step in any machine learning pipeline.",
        "Python decorators allow modification of function behavior."
    ]
    
    entry_ids = []
    for i, content in enumerate(entries):
        entry_id = memory_graph.add_entry(
            content,
            metadata={"category": "programming" if "Python" in content else "ml"}
        )
        entry_ids.append(entry_id)
        print(f"  Added entry {i+1}: {content[:50]}...")
    
    # Find similar entries
    print("\n2. Finding entries similar to 'Python programming'...")
    similar = memory_graph.find_similar("Python programming", top_k=3)
    for entry, similarity in similar:
        print(f"  Similarity {similarity:.3f}: {entry.content}")
    
    # Compress related entries
    print("\n3. Compressing Python-related entries...")
    python_entries = [eid for i, eid in enumerate(entry_ids) if "Python" in entries[i]]
    if len(python_entries) > 1:
        compressed_id = memory_graph.compress_entries(python_entries, compressor)
        print(f"  Compressed {len(python_entries)} entries into: {compressed_id}")
    
    print(f"\n4. Total entries after compression: {len(memory_graph.get_all_entries())}")
    
    # Clean up
    database.clear_database()
    print("\nExample completed!")


def custom_similarity_example():
    """Example using custom similarity function."""
    print("\n=== Custom Similarity Example ===\n")
    
    # Create custom similarity that weights time more heavily
    custom_sim = CustomSimilarity(
        similarity_weight=0.2,  # 20% weight to content similarity
        time_weight=0.8,        # 80% weight to temporal proximity
        base_similarity=CosineSimilarity()
    )
    
    embedding_service = MockEmbeddingService()
    memory_graph = MemoryGraph(
        similarity_function=custom_sim,
        embedding_service=embedding_service
    )
    
    # Add entries with different timestamps
    import time
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    
    # Add entries at different times
    entry1_id = memory_graph.add_entry("First entry about AI")
    time.sleep(0.1)  # Small delay
    
    entry2_id = memory_graph.add_entry("Second entry about machine learning")
    time.sleep(0.1)
    
    entry3_id = memory_graph.add_entry("Third entry about neural networks")
    
    # Find similar entries - should favor more recent entries
    print("Finding entries similar to 'artificial intelligence'...")
    similar = memory_graph.find_similar("artificial intelligence", top_k=3)
    
    for entry, similarity in similar:
        time_diff = datetime.now() - entry.timestamp
        print(f"  Similarity {similarity:.3f} (age: {time_diff.total_seconds():.1f}s): {entry.content}")
    
    print("\nCustom similarity example completed!")


if __name__ == "__main__":
    basic_example()
    custom_similarity_example()