#!/usr/bin/env python3
"""
Demonstration of the Knowledge Graph functionality in PardusMemory.

This example shows how to:
1. Create a memory graph with automatic edge formation
2. Add entries and see how they connect based on similarity
3. Analyze the graph structure
4. Visualize the graph (requires networkx and matplotlib)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pardusmemory.core import MemoryGraph, CosineSimilarity
from pardusmemory.embedding_service import MockEmbeddingService
from pardusmemory.compression import MockCompressor


def main():
    print("=== PardusMemory Knowledge Graph Demo ===\n")
    
    # Initialize components
    embedding_service = MockEmbeddingService(dimensions=384)
    similarity_function = CosineSimilarity()
    
    # Create memory graph with similarity threshold
    graph = MemoryGraph(
        similarity_function=similarity_function,
        embedding_service=embedding_service,
        similarity_threshold=0.6  # Edges form when similarity > 0.6
    )
    
    print(f"Created knowledge graph with similarity threshold: {graph.similarity_threshold}")
    print()
    
    # Add related entries about machine learning
    print("Adding entries about machine learning...")
    ml_entries = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn patterns.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Artificial intelligence aims to create machines that can perform human-like tasks.",
        "Supervised learning uses labeled data to train models.",
        "Python is a popular programming language for machine learning."
    ]
    
    entry_ids = []
    for i, content in enumerate(ml_entries):
        entry_id = graph.add_entry(content, metadata={"topic": "machine_learning"})
        entry_ids.append(entry_id)
        print(f"  Added entry {i+1}: {content[:50]}...")
    
    print(f"\nAdded {len(entry_ids)} entries to the graph.")
    
    # Add some unrelated entries
    print("\nAdding unrelated entries...")
    unrelated_entries = [
        "The weather is nice today.",
        "I like to cook pasta on weekends.",
        "Basketball is a popular sport."
    ]
    
    for i, content in enumerate(unrelated_entries):
        entry_id = graph.add_entry(content, metadata={"topic": "general"})
        print(f"  Added entry {i+1}: {content}")
    
    # Get graph statistics
    print("\n=== Graph Statistics ===")
    stats = graph.get_graph_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show connections for a specific entry
    print("\n=== Example Connections ===")
    if entry_ids:
        sample_entry_id = entry_ids[0]
        sample_entry = graph.get_entry(sample_entry_id)
        print(f"Entry: {sample_entry.content}")
        
        connected = graph.get_connected_entries(sample_entry_id)
        if connected:
            print("Connected entries:")
            for entry, weight in connected:
                print(f"  - {entry.content[:50]}... (similarity: {weight:.3f})")
        else:
            print("No connections found.")
    
    # Find connected components
    print("\n=== Connected Components ===")
    components = graph.find_connected_components()
    print(f"Number of connected components: {len(components)}")
    for i, component in enumerate(components):
        print(f"  Component {i+1}: {len(component)} nodes")
        for node_id in list(component)[:3]:  # Show first 3 nodes
            entry = graph.get_entry(node_id)
            print(f"    - {entry.content[:30]}...")
        if len(component) > 3:
            print(f"    ... and {len(component) - 3} more")
    
    # Demonstrate threshold adjustment
    print("\n=== Adjusting Similarity Threshold ===")
    print("Rebuilding graph with higher threshold (0.8)...")
    graph.rebuild_graph(new_threshold=0.8)
    
    new_stats = graph.get_graph_stats()
    print(f"New number of edges: {new_stats['num_edges']}")
    print(f"New number of components: {new_stats['num_components']}")
    
    # Export graph data
    print("\n=== Exporting Graph Data ===")
    graph_data = graph.export_graph_data()
    print(f"Exported {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    # Try visualization (optional)
    print("\n=== Graph Visualization ===")
    try:
        graph.visualize_graph(layout="spring")
        print("Graph visualization displayed!")
    except ImportError:
        print("Visualization requires networkx and matplotlib. Install with:")
        print("  pip install networkx matplotlib")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Demonstrate similarity search
    print("\n=== Similarity Search ===")
    query = "artificial intelligence and neural networks"
    similar_entries = graph.find_similar(query, top_k=3, threshold=0.3)
    
    print(f"Entries similar to '{query}':")
    for entry, similarity in similar_entries:
        print(f"  - {entry.content[:50]}... (similarity: {similarity:.3f})")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()