"""
Example using OpenAI services with PardusMemory.

This example demonstrates:
1. Using OpenAI embedding service
2. Using OpenAI for knowledge compression
3. Real-world usage with actual LLM services

Note: You need to set OPENAI_API_KEY environment variable or pass api_key parameter.
"""

import os
from pardusmemory import (
    MemoryGraph,
    OpenAIEmbeddingService,
    OpenAICompressor,
    JSONDatabase,
    CosineSimilarity
)


def openai_example():
    """Example using OpenAI services."""
    print("=== OpenAI Integration Example ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this example.")
        return
    
    try:
        # Initialize OpenAI services
        embedding_service = OpenAIEmbeddingService(
            api_key=api_key,
            model="text-embedding-3-small"
        )
        
        compressor = OpenAICompressor(
            api_key=api_key,
            model="gpt-4"
        )
        
        database = JSONDatabase("openai_example_memory.json")
        
        # Create memory graph
        memory_graph = MemoryGraph(
            similarity_function=CosineSimilarity(),
            embedding_service=embedding_service,
            database=database
        )
        
        print("1. Adding memory entries with OpenAI embeddings...")
        
        # Add some technical content
        entries = [
            "React hooks revolutionized functional components by allowing state and lifecycle management without classes.",
            "useState hook manages local component state and triggers re-renders when updated.",
            "useEffect hook handles side effects like API calls, subscriptions, and DOM manipulations.",
            "Custom hooks allow you to extract component logic into reusable functions.",
            "The useContext hook provides access to React context values without prop drilling."
        ]
        
        entry_ids = []
        for content in entries:
            entry_id = memory_graph.add_entry(
                content,
                metadata={"topic": "react", "framework": "javascript"}
            )
            entry_ids.append(entry_id)
            print(f"  Added: {content[:60]}...")
        
        # Find similar entries
        print("\n2. Finding entries similar to 'React state management'...")
        similar = memory_graph.find_similar("React state management", top_k=3)
        
        for entry, similarity in similar:
            print(f"  Similarity {similarity:.3f}: {entry.content}")
        
        # Compress entries using OpenAI
        print("\n3. Compressing React hooks entries with OpenAI...")
        if len(entry_ids) >= 3:
            # Compress first 3 entries
            compressed_id = memory_graph.compress_entries(entry_ids[:3], compressor)
            compressed_entry = memory_graph.get_entry(compressed_id)
            print(f"  Compressed entry ID: {compressed_id}")
            print(f"  Summary: {compressed_entry.content[:200]}...")
        
        print(f"\n4. Total entries after compression: {len(memory_graph.get_all_entries())}")
        
        # Clean up
        database.clear_database()
        print("\nOpenAI example completed successfully!")
        
    except Exception as e:
        print(f"Error during OpenAI integration: {e}")
        print("Make sure your OpenAI API key is valid and has sufficient credits.")


def custom_openai_endpoint_example():
    """Example using custom OpenAI-compatible endpoint."""
    print("\n=== Custom OpenAI Endpoint Example ===\n")
    
    # Example with custom endpoint (e.g., for local LLM or different provider)
    try:
        embedding_service = OpenAIEmbeddingService(
            api_key="your-api-key",
            base_url="https://api.your-custom-provider.com/v1",
            model="your-embedding-model"
        )
        
        compressor = OpenAICompressor(
            api_key="your-api-key", 
            base_url="https://api.your-custom-provider.com/v1",
            model="your-chat-model"
        )
        
        print("Custom endpoint services configured successfully!")
        print("Note: Replace with your actual endpoint and API key to use.")
        
    except Exception as e:
        print(f"Error configuring custom endpoint: {e}")


if __name__ == "__main__":
    openai_example()
    custom_openai_endpoint_example()