from typing import List, Dict, Any, Callable, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pydantic import BaseModel


@dataclass
class MemoryEntry:
    """A single memory entry (node) in the knowledge graph."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    compressed: bool = False


@dataclass
class GraphEdge:
    """An edge connecting two memory entries in the knowledge graph."""
    source_id: str
    target_id: str
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class SimilarityFunction:
    """Base class for similarity functions."""
    
    def __call__(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        raise NotImplementedError


class CosineSimilarity(SimilarityFunction):
    """Standard cosine similarity between embeddings."""
    
    def __call__(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        if entry1.embedding is None or entry2.embedding is None:
            return 0.0
        
        try:
            dot_product = np.dot(entry1.embedding, entry2.embedding)
            norm1 = np.linalg.norm(entry1.embedding)
            norm2 = np.linalg.norm(entry2.embedding)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # Ensure similarity is in valid range [-1, 1]
            return max(-1.0, min(1.0, similarity))
        except Exception as e:
            raise ValueError(f"Error calculating cosine similarity: {e}")


class CustomSimilarity(SimilarityFunction):
    """Custom similarity function that combines multiple factors."""
    
    def __init__(self, 
                 similarity_weight: float = 0.1,
                 time_weight: float = 0.9,
                 base_similarity: Optional[SimilarityFunction] = None):
        if not (0.0 <= similarity_weight <= 1.0):
            raise ValueError("similarity_weight must be between 0.0 and 1.0")
        if not (0.0 <= time_weight <= 1.0):
            raise ValueError("time_weight must be between 0.0 and 1.0")
        if abs(similarity_weight + time_weight - 1.0) > 0.01:
            raise ValueError("similarity_weight + time_weight must equal 1.0")
        
        self.similarity_weight = similarity_weight
        self.time_weight = time_weight
        self.base_similarity = base_similarity or CosineSimilarity()
    
    def __call__(self, entry1: MemoryEntry, entry2: MemoryEntry) -> float:
        try:
            similarity_score = self.base_similarity(entry1, entry2)
            time_diff = (entry1.timestamp - entry2.timestamp).total_seconds()
            time_diff_normalized = np.tanh(time_diff / (24 * 3600))  # Normalize by day
            
            combined_score = (self.similarity_weight * similarity_score + 
                            self.time_weight * time_diff_normalized)
            # Ensure score is in reasonable range
            return max(-1.0, min(1.0, combined_score))
        except Exception as e:
            raise ValueError(f"Error calculating custom similarity: {e}")


class MemoryGraph:
    """Core knowledge graph system with nodes and edges based on similarity thresholds."""
    
    def __init__(self, 
                 similarity_function: Optional[SimilarityFunction] = None,
                 embedding_service=None,
                 database=None,
                 similarity_threshold: float = 0.7):
        self.similarity_function = similarity_function or CosineSimilarity()
        self.embedding_service = embedding_service
        self.database = database
        self.similarity_threshold = similarity_threshold
        self.entries: Dict[str, MemoryEntry] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency_list: Dict[str, Set[str]] = {}
    
    def add_entry(self, 
                  content: str, 
                  entry_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  auto_connect: bool = True) -> str:
        """Add a new memory entry and automatically create edges based on similarity."""
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        if entry_id is None:
            entry_id = f"entry_{len(self.entries)}_{datetime.now().timestamp()}"
        
        if entry_id in self.entries:
            raise ValueError(f"Entry with ID '{entry_id}' already exists")
        
        embedding = None
        if self.embedding_service:
            try:
                embedding_result = self.embedding_service.get_embedding(content)
                # Ensure embedding is numpy array
                if isinstance(embedding_result, list):
                    embedding = np.array(embedding_result)
                else:
                    embedding = embedding_result
            except Exception as e:
                raise RuntimeError(f"Failed to generate embedding: {e}")
        
        entry = MemoryEntry(
            id=entry_id,
            content=content.strip(),
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.entries[entry_id] = entry
        self.adjacency_list[entry_id] = set()
        
        # Auto-connect to existing entries based on similarity threshold
        if auto_connect and embedding is not None:
            try:
                self._create_edges_for_entry(entry_id)
            except Exception as e:
                raise RuntimeError(f"Failed to create edges for entry: {e}")
        
        if self.database:
            try:
                self.database.store_entry(entry)
            except Exception as e:
                raise RuntimeError(f"Failed to store entry in database: {e}")
        
        return entry_id
    
    def _create_edges_for_entry(self, entry_id: str):
        """Create edges for a new entry based on similarity threshold."""
        if entry_id not in self.entries:
            return
        
        new_entry = self.entries[entry_id]
        if new_entry.embedding is None:
            return
        
        # Check similarity with all existing entries
        for existing_id, existing_entry in self.entries.items():
            if existing_id == entry_id:
                continue
            
            if existing_entry.embedding is None:
                continue
            
            similarity = self.similarity_function(new_entry, existing_entry)
            
            # Create edge if similarity exceeds threshold
            if similarity >= self.similarity_threshold:
                self._create_edge(entry_id, existing_id, similarity)
    
    def _create_edge(self, source_id: str, target_id: str, weight: float):
        """Create an edge between two entries."""
        edge_id = f"{source_id}--{target_id}"
        
        # Avoid duplicate edges
        if edge_id in self.edges or f"{target_id}--{source_id}" in self.edges:
            return
        
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            metadata={"created_by": "similarity_threshold"}
        )
        
        self.edges[edge_id] = edge
        self.adjacency_list[source_id].add(target_id)
        self.adjacency_list[target_id].add(source_id)
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry."""
        return self.entries.get(entry_id)
    
    def find_similar(self, 
                    query: str, 
                    top_k: int = 5,
                    threshold: float = 0.0) -> List[Tuple[MemoryEntry, float]]:
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
    
    def get_neighbors(self, entry_id: str) -> List[str]:
        """Get all neighboring entry IDs for a given entry."""
        return list(self.adjacency_list.get(entry_id, set()))
    
    def get_connected_entries(self, entry_id: str) -> List[Tuple[MemoryEntry, float]]:
        """Get all connected entries with their edge weights."""
        if entry_id not in self.adjacency_list:
            return []
        
        connected = []
        for neighbor_id in self.adjacency_list[entry_id]:
            # Find edge weight
            edge_id = f"{entry_id}--{neighbor_id}"
            reverse_edge_id = f"{neighbor_id}--{entry_id}"
            
            weight = 0.0
            if edge_id in self.edges:
                weight = self.edges[edge_id].weight
            elif reverse_edge_id in self.edges:
                weight = self.edges[reverse_edge_id].weight
            
            if neighbor_id in self.entries:
                connected.append((self.entries[neighbor_id], weight))
        
        return connected
    
    def find_connected_components(self) -> List[Set[str]]:
        """Find all connected components in the graph using DFS."""
        visited = set()
        components = []
        
        def dfs(node_id: str, component: Set[str]):
            visited.add(node_id)
            component.add(node_id)
            
            for neighbor_id in self.adjacency_list.get(node_id, set()):
                if neighbor_id not in visited:
                    dfs(neighbor_id, component)
        
        for entry_id in self.entries:
            if entry_id not in visited:
                component = set()
                dfs(entry_id, component)
                components.append(component)
        
        return components
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        num_nodes = len(self.entries)
        num_edges = len(self.edges)
        
        # Calculate average degree
        total_degree = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        avg_degree = total_degree / num_nodes if num_nodes > 0 else 0
        
        # Find connected components
        components = self.find_connected_components()
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "num_components": len(components),
            "largest_component_size": max(len(comp) for comp in components) if components else 0,
            "similarity_threshold": self.similarity_threshold
        }
    
    def rebuild_graph(self, new_threshold: Optional[float] = None):
        """Rebuild the entire graph with a new similarity threshold."""
        if new_threshold is not None:
            self.similarity_threshold = new_threshold
        
        # Clear existing edges
        self.edges.clear()
        for entry_id in self.adjacency_list:
            self.adjacency_list[entry_id].clear()
        
        # Recreate all edges
        for entry_id in self.entries:
            self._create_edges_for_entry(entry_id)
    
    def visualize_graph(self, output_path: Optional[str] = None, layout: str = "spring"):
        """
        Visualize the knowledge graph using networkx and matplotlib.
        
        Args:
            output_path: If provided, save the visualization to this path
            layout: Layout algorithm ('spring', 'circular', 'random')
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("networkx and matplotlib are required for visualization. Install with: pip install networkx matplotlib") from e
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for entry_id, entry in self.entries.items():
            G.add_node(entry_id, 
                      content=entry.content[:50] + "..." if len(entry.content) > 50 else entry.content,
                      timestamp=entry.timestamp)
        
        # Add edges
        for edge_id, edge in self.edges.items():
            G.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, weight='weight')
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G, weight='weight')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 3 for w in weights], 
                              alpha=0.5, edge_color='gray')
        
        # Draw labels (truncated)
        labels = {node: G.nodes[node]['content'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for external visualization tools."""
        nodes = []
        for entry_id, entry in self.entries.items():
            nodes.append({
                "id": entry_id,
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat(),
                "metadata": entry.metadata
            })
        
        edges = []
        for edge_id, edge in self.edges.items():
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "weight": edge.weight,
                "timestamp": edge.timestamp.isoformat(),
                "metadata": edge.metadata
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "similarity_threshold": self.similarity_threshold,
                "num_nodes": len(nodes),
                "num_edges": len(edges)
            }
        }
    
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