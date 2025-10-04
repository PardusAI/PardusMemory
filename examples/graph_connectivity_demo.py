#!/usr/bin/env python3
"""
Demonstration of knowledge graph connectivity with different thresholds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pardusmemory.core import MemoryGraph, CosineSimilarity
from pardusmemory.embedding_service import MockEmbeddingService


def main():
    print("=== Knowledge Graph Connectivity Demo ===\n")
    
    # Initialize components
    embedding_service = MockEmbeddingService(dimensions=384)
    similarity_function = CosineSimilarity()
    
    # Test different similarity thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    # Create test data with related content
    test_entries = [
        "Machine learning algorithms learn patterns from data",
        "Deep learning is a subset of machine learning",
        "Neural networks are used in deep learning models",
        "Artificial intelligence includes machine learning",
        "Data science uses machine learning techniques",
        "Python is popular for machine learning development",
        "TensorFlow is a machine learning framework",
        "Natural language processing uses ML models"
    ]
    
    for threshold in thresholds:
        print(f"--- Testing with threshold: {threshold} ---")
        
        # Create graph
        graph = MemoryGraph(
            similarity_function=similarity_function,
            embedding_service=embedding_service,
            similarity_threshold=threshold
        )
        
        # Add entries
        for content in test_entries:
            graph.add_entry(content)
        
        # Get statistics
        stats = graph.get_graph_stats()
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Avg degree: {stats['avg_degree']:.3f}")
        print(f"Components: {stats['num_components']}")
        
        # Show some connections if they exist
        if stats['num_edges'] > 0:
            print("Sample connections:")
            entry_ids = list(graph.entries.keys())[:3]
            for entry_id in entry_ids:
                connected = graph.get_connected_entries(entry_id)
                if connected:
                    entry = graph.get_entry(entry_id)
                    print(f"  '{entry.content[:30]}...' connects to:")
                    for conn_entry, weight in connected[:2]:
                        print(f"    - '{conn_entry.content[:30]}...' (weight: {weight:.3f})")
                    break
        
        print()


if __name__ == "__main__":
    main()