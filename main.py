def main():
    """Example usage of the PardusMemory library."""
    from pardusmemory import (
        MemoryGraph, 
        MockEmbeddingService, 
        JSONDatabase, 
        MockCompressor,
        CustomSimilarity,
        CosineSimilarity
    )
    
    print("PardusMemory - Memory Graph System Demo")
    print("=" * 50)
    
    # Initialize components
    embedding_service = MockEmbeddingService(dimensions=384)
    database = JSONDatabase("demo_memory.json")
    compressor = MockCompressor()
    
    # Create memory graph with custom similarity
    custom_similarity = CustomSimilarity(
        similarity_weight=0.3,
        time_weight=0.7,
        base_similarity=CosineSimilarity()
    )
    
    memory_graph = MemoryGraph(
        similarity_function=custom_similarity,
        embedding_service=embedding_service,
        database=database
    )
    
    # Add some memory entries
    print("\n1. Adding memory entries...")
    entry1_id = memory_graph.add_entry(
        "I learned about Python decorators today. They allow functions to be modified dynamically.",
        metadata={"topic": "programming", "difficulty": "intermediate"}
    )
    
    entry2_id = memory_graph.add_entry(
        "Decorators use the @ symbol and can be stacked for multiple modifications.",
        metadata={"topic": "programming", "difficulty": "intermediate"}
    )
    
    entry3_id = memory_graph.add_entry(
        "Yesterday I studied machine learning algorithms, specifically neural networks.",
        metadata={"topic": "ml", "difficulty": "advanced"}
    )
    
    print(f"Added entries: {entry1_id}, {entry2_id}, {entry3_id}")
    
    # Find similar entries
    print("\n2. Finding similar entries...")
    similar_entries = memory_graph.find_similar("Python programming concepts", top_k=3)
    
    for entry, similarity in similar_entries:
        print(f"  Similarity {similarity:.3f}: {entry.content[:50]}...")
    
    # Compress entries
    print("\n3. Compressing related entries...")
    compressed_id = memory_graph.compress_entries([entry1_id, entry2_id], compressor)
    print(f"Compressed entries into: {compressed_id}")
    
    # Show remaining entries
    print("\n4. Current memory entries:")
    for entry in memory_graph.get_all_entries():
        print(f"  - {entry.content[:50]}...")
    
    # Database retrieval
    print("\n5. Database operations...")
    all_db_entries = database.get_all_entries()
    print(f"Total entries in database: {len(all_db_entries)}")
    
    # Clean up demo database
    database.clear_database()
    print("Demo database cleared.")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
