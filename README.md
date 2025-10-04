# PardusMemory

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://openai.com)

A sophisticated memory graph system with custom similarity rules, configurable embeddings, and LLM-based knowledge compression designed for conversational AI agents.

## 🚀 Features

- **Memory Graph System**: Advanced graph-based memory with custom similarity functions
- **Conversation Format Support**: Native support for OpenAI-style conversation messages
- **Knowledge Compression**: LLM-based compression with conversation format return
- **Knowledge Graph Generation**: Automatic knowledge graph creation from text
- **Multiple Embedding Services**: OpenAI, Mock, and custom embedding support
- **Flexible Storage**: JSON database with easy persistence
- **Agent Integration**: Designed for seamless AI agent integration

## 📦 Installation

```bash
pip install pardusmemory
```

### Development Installation

```bash
git clone https://github.com/your-repo/pardusmemory.git
cd pardusmemory
pip install -e .
```

## 🔧 Quick Start

### Basic Memory Usage

```python
from pardusmemory import MemoryGraph, MockEmbeddingService, CosineSimilarity

# Initialize memory graph
memory_graph = MemoryGraph(
    embedding_service=MockEmbeddingService(),
    similarity_function=CosineSimilarity()
)

# Add memories
memory_graph.add_entry("Python is a high-level programming language.")
memory_graph.add_entry("Machine learning uses algorithms to find patterns in data.")

# Search for similar memories
results = memory_graph.find_similar("Python programming", top_k=3)
for entry, similarity in results:
    print(f"Similarity {similarity:.3f}: {entry.content}")
```

### Conversation Compression

```python
from pardusmemory import OpenAICompressor

compressor = OpenAICompressor(api_key="your-openai-key")

conversation = [
    {"role": "user", "content": "What is React?"},
    {"role": "assistant", "content": "React is a JavaScript library for building UIs."},
    {"role": "user", "content": "What are React hooks?"}
]

# Standard compression (returns string)
summary = compressor.compress(conversation)

# Conversation format compression (returns OpenAI format)
compressed_conversation = compressor.compress(conversation, return_as_conversation=True)
# Returns: [{"role": "assistant", "content": "React is a JavaScript library..."}]
```

### Knowledge Graph Generation

```python
from pardusmemory import KnowledgeGraphGenerator

generator = KnowledgeGraphGenerator(
    embedding_service=MockEmbeddingService(),
    similarity_threshold=0.7
)

text = """
Python is a high-level programming language. It supports multiple programming paradigms.
Machine learning is a subset of artificial intelligence. Python is widely used in machine learning.
"""

# Generate knowledge graph
knowledge_graph = generator.generate_knowledge_graph(text)

# Access graph data
nodes = knowledge_graph.get_nodes()
edges = knowledge_graph.get_edges()

print(f"Generated {len(nodes)} nodes and {len(edges)} edges")
```

## 🏗️ Architecture

### Core Components

- **MemoryGraph**: Core memory management with similarity-based retrieval
- **KnowledgeGraph**: Graph structure for semantic relationships
- **EmbeddingService**: Pluggable embedding generation (OpenAI, Mock, Custom)
- **LLMCompressor**: LLM-based knowledge compression
- **Database**: Persistent storage with JSON backend

### Similarity Functions

- **CosineSimilarity**: Standard cosine similarity between embeddings
- **CustomSimilarity**: Weighted combination of similarity and temporal factors

## 📚 Advanced Usage

### Agent Integration

```python
from pardusmemory import MemoryGraph, OpenAICompressor, OpenAIEmbeddingService

class ConversationalAgent:
    def __init__(self, api_key: str):
        self.memory_graph = MemoryGraph(
            embedding_service=OpenAIEmbeddingService(api_key=api_key),
            database=JSONDatabase("agent_memory.json")
        )
        self.compressor = OpenAICompressor(api_key=api_key)
        self.conversation_history = []
    
    def process_message(self, role: str, content: str):
        # Store in memory
        self.memory_graph.add_entry(content, metadata={"role": role})
        self.conversation_history.append({"role": role, "content": content})
        
        # Compress when conversation gets long
        if len(self.conversation_history) > 10:
            compressed = self.compressor.compress(
                self.conversation_history[:-3], 
                return_as_conversation=True
            )
            self.conversation_history = self.conversation_history[-3:] + compressed
```

### Custom Similarity Functions

```python
from pardusmemory import MemoryGraph, CustomSimilarity, CosineSimilarity

# Create custom similarity weighting time more heavily
custom_sim = CustomSimilarity(
    similarity_weight=0.3,  # 30% content similarity
    time_weight=0.7,        # 70% temporal proximity
    base_similarity=CosineSimilarity()
)

memory_graph = MemoryGraph(similarity_function=custom_sim)
```

### Knowledge Graph Analysis

```python
# Analyze knowledge graph structure
graph_stats = knowledge_graph.get_statistics()
print(f"Graph density: {graph_stats['density']:.3f}")
print(f"Average degree: {graph_stats['avg_degree']:.2f}")

# Find central nodes
central_nodes = knowledge_graph.get_central_nodes(top_k=5)
for node_id, centrality in central_nodes:
    node = knowledge_graph.get_node(node_id)
    print(f"Central: {node.content[:50]}... (centrality: {centrality:.3f})")

# Export to different formats
knowledge_graph.export_to_json("knowledge_graph.json")
knowledge_graph.export_to_networkx("graph.graphml")
```

## 🔌 API Reference

### MemoryGraph

```python
class MemoryGraph:
    def add_entry(self, content: str, entry_id: str = None, metadata: dict = None) -> str
    def find_similar(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[MemoryEntry, float]]
    def compress_entries(self, entry_ids: List[str], compressor) -> str
    def compress_conversation(self, conversation: List[Dict], compressor) -> str
    def get_entry(self, entry_id: str) -> MemoryEntry
    def get_all_entries(self) -> List[MemoryEntry]
```

### KnowledgeGraph

```python
class KnowledgeGraph:
    def add_node(self, content: str, node_id: str = None, metadata: dict = None) -> str
    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0, metadata: dict = None) -> str
    def get_nodes(self) -> List[GraphNode]
    def get_edges(self) -> List[GraphEdge]
    def get_neighbors(self, node_id: str) -> List[GraphNode]
    def get_statistics(self) -> dict
    def get_central_nodes(self, top_k: int = 5) -> List[Tuple[str, float]]
    def export_to_json(self, filepath: str) -> None
    def export_to_networkx(self, filepath: str) -> None
```

### LLMCompressor

```python
class LLMCompressor:
    def compress(self, content_list: List[Union[str, Dict]], return_as_conversation: bool = False) -> Union[str, List[Dict]]
    def compress_progressive(self, content_list: List[Union[str, Dict]], chunk_size: int = 5, return_as_conversation: bool = False) -> Union[str, List[Dict]]
```

## 🧪 Examples

The `examples/` directory contains comprehensive examples:

- `basic_usage.py` - Basic memory operations
- `agent_integration.py` - Complete agent workflow
- `conversation_compression_demo.py` - Conversation format compression
- `knowledge_graph_demo.py` - Knowledge graph generation and analysis

Run examples with:

```bash
python examples/basic_usage.py
python examples/agent_integration.py
python examples/knowledge_graph_demo.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the embedding and compression APIs
- NetworkX for graph algorithms inspiration
- The Python community for excellent tools and libraries

## 📞 Support

- 📧 Email: support@pardusmemory.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/pardusmemory/issues)
- 📖 Documentation: [Full Documentation](https://pardusmemory.readthedocs.io)

---

**PardusMemory** - Advanced memory management for the next generation of AI agents. 🚀