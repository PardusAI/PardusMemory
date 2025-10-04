"""
Complete agent workflow demonstrating conversation format compression.

This example shows a real-world agent workflow that:
1. Processes user inputs
2. Maintains conversation memory
3. Compresses conversations when needed
4. Returns compressed results in OpenAI format for API integration
"""

from pardusmemory import (
    MemoryGraph, MockCompressor, MockEmbeddingService, 
    JSONDatabase, CosineSimilarity
)
import time
from typing import List, Dict, Any


class ProductionAgent:
    """
    Production-ready agent with conversation compression.
    
    This agent demonstrates the complete workflow for handling conversations
    with memory compression that returns OpenAI-compatible results.
    """
    
    def __init__(self, compression_threshold: int = 8):
        """Initialize the agent with memory compression capabilities."""
        self.compression_threshold = compression_threshold
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize memory components
        self.memory_graph = MemoryGraph(
            embedding_service=MockEmbeddingService(dimensions=384),
            similarity_function=CosineSimilarity(),
            database=JSONDatabase("production_agent_memory.json")
        )
        
        self.compressor = MockCompressor()
        
        print(f"Production agent initialized (compression threshold: {compression_threshold})")
    
    def process_user_input(self, user_input: str) -> str:
        """
        Process user input with full memory integration.
        
        Args:
            user_input: The user's message
            
        Returns:
            Agent's response
        """
        print(f"\n🔵 Processing: '{user_input}'")
        
        # 1. Search memory for relevant context
        relevant_memories = self.memory_graph.find_similar(user_input, top_k=3)
        context_info = self._build_context(relevant_memories)
        
        # 2. Generate response (simulated LLM call)
        response = self._generate_response(user_input, context_info)
        
        # 3. Store conversation in memory
        self._store_conversation_turn("user", user_input)
        self._store_conversation_turn("assistant", response)
        
        # 4. Check if compression is needed
        if len(self.conversation_history) >= self.compression_threshold:
            self._compress_conversation_memory()
        
        print(f"🟢 Response: {response}")
        return response
    
    def _build_context(self, relevant_memories: List) -> str:
        """Build context string from relevant memories."""
        if not relevant_memories:
            return "No relevant memories found."
        
        context_parts = []
        for entry, similarity in relevant_memories:
            context_parts.append(f"- {entry.content[:100]}... (similarity: {similarity:.2f})")
        
        return "Relevant context:\n" + "\n".join(context_parts)
    
    def _generate_response(self, user_input: str, context: str) -> str:
        """Generate response (simulated LLM call)."""
        # Simple response generation based on input and context
        if "python" in user_input.lower():
            if "data types" in user_input.lower():
                return "Python has several built-in data types including integers, floats, strings, lists, tuples, dictionaries, and sets. Each serves different purposes for data storage and manipulation."
            elif "function" in user_input.lower():
                return "In Python, functions are defined using the 'def' keyword. They can take parameters and return values. Functions help organize code into reusable blocks."
            elif "class" in user_input.lower():
                return "Python classes are blueprints for creating objects. They support inheritance, encapsulation, and polymorphism. Use 'class' keyword to define them."
            else:
                return "Python is a versatile programming language known for its simplicity and readability. It's widely used in web development, data science, AI, and automation."
        elif "help" in user_input.lower():
            return "I'm here to help! I can assist with programming questions, explain concepts, and provide guidance on various topics. What would you like to know?"
        else:
            return f"I understand you're asking about '{user_input}'. Based on our conversation, I can provide relevant information to help you."
    
    def _store_conversation_turn(self, role: str, content: str):
        """Store a conversation turn in memory."""
        # Add to conversation history
        message = {"role": role, "content": content}
        self.conversation_history.append(message)
        
        # Add to memory graph
        self.memory_graph.add_entry(
            content,
            metadata={
                "role": role,
                "turn_number": len(self.conversation_history),
                "timestamp": time.time()
            }
        )
    
    def _compress_conversation_memory(self):
        """Compress conversation memory using conversation format."""
        print(f"\n🔧 Compressing conversation memory (threshold: {self.compression_threshold})")
        
        # Get conversation entries for compression
        all_entries = self.memory_graph.get_all_entries()
        conversation_entries = [
            entry for entry in all_entries 
            if entry.metadata.get("role") in ["user", "assistant"]
        ]
        
        if len(conversation_entries) < 4:
            return
        
        # Sort by turn number and select older entries for compression
        conversation_entries.sort(key=lambda x: x.metadata.get("turn_number", 0))
        
        # Keep recent entries, compress older ones
        keep_recent = 3
        entries_to_compress = conversation_entries[:-keep_recent]
        entry_ids_to_compress = [entry.id for entry in entries_to_compress]
        
        # Format as conversation for compression
        conversation_format = []
        for entry in entries_to_compress:
            conversation_format.append({
                "role": entry.metadata.get("role", "unknown"),
                "content": entry.content
            })
        
        try:
            # Compress using conversation format
            compressed_result = self.compressor.compress(
                conversation_format, 
                return_as_conversation=True
            )
            
            if compressed_result:
                # Extract compressed content
                compressed_content = compressed_result[0]["content"]
                
                # Store compressed summary in memory
                compressed_id = self.memory_graph.add_entry(
                    compressed_content,
                    metadata={
                        "type": "compressed_conversation",
                        "original_messages": len(conversation_format),
                        "compressed": True,
                        "compression_time": time.time()
                    }
                )
                
                # Update conversation history
                recent_messages = self.conversation_history[-keep_recent:]
                compression_message = {
                    "role": "system",
                    "content": f"[Previous conversation compressed to memory entry: {compressed_id}]"
                }
                
                self.conversation_history = recent_messages + [compression_message]
                
                # Remove compressed entries from memory graph
                for entry_id in entry_ids_to_compress:
                    if entry_id in self.memory_graph.entries:
                        del self.memory_graph.entries[entry_id]
                
                print(f"✅ Compressed {len(entry_ids_to_compress)} entries into {compressed_id}")
                print(f"📊 Memory entries: {len(self.memory_graph.get_all_entries())}")
                print(f"💬 Conversation history: {len(self.conversation_history)} items")
                
        except Exception as e:
            print(f"❌ Compression failed: {e}")
    
    def get_conversation_for_api(self) -> List[Dict[str, Any]]:
        """
        Get conversation history formatted for OpenAI API.
        
        Returns:
            List of messages in OpenAI format
        """
        return self.conversation_history.copy()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            "total_memory_entries": len(self.memory_graph.get_all_entries()),
            "conversation_history_length": len(self.conversation_history),
            "compression_threshold": self.compression_threshold,
            "needs_compression": len(self.conversation_history) >= self.compression_threshold
        }


def demonstrate_complete_workflow():
    """Demonstrate the complete agent workflow."""
    print("=== Complete Agent Workflow Demo ===\n")
    
    # Initialize agent
    agent = ProductionAgent(compression_threshold=6)
    
    # Simulate a conversation session
    conversation_inputs = [
        "Hi, I'm learning Python programming. Can you help me?",
        "What are the basic data types in Python?",
        "How do I create functions in Python?",
        "Can you explain classes and objects?",
        "What's the difference between lists and tuples?",
        "How do I handle file I/O in Python?",
        "Tell me about Python decorators",
        "How can I use Python for web development?"
    ]
    
    print("🚀 Starting conversation session...")
    print("=" * 50)
    
    # Process each input
    for i, user_input in enumerate(conversation_inputs, 1):
        print(f"\n--- Turn {i} ---")
        
        # Process user input
        response = agent.process_user_input(user_input)
        
        # Show memory stats
        stats = agent.get_memory_stats()
        print(f"📈 Memory: {stats['total_memory_entries']} entries, "
              f"Conversation: {stats['conversation_history_length']} items")
        
        # Small delay for demo
        time.sleep(0.3)
    
    # Show final state
    print("\n" + "=" * 50)
    print("🏁 Conversation session completed!")
    
    final_stats = agent.get_memory_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Total memory entries: {final_stats['total_memory_entries']}")
    print(f"  Conversation history: {final_stats['conversation_history_length']} items")
    print(f"  Compression threshold: {final_stats['compression_threshold']}")
    
    # Show conversation formatted for API
    print(f"\n🔗 Conversation formatted for OpenAI API:")
    api_conversation = agent.get_conversation_for_api()
    for i, msg in enumerate(api_conversation, 1):
        role = msg["role"].upper()
        content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"  {i}. {role}: {content}")
    
    # Demonstrate API integration
    print(f"\n🔌 API Integration Example:")
    print("```python")
    print("# Get conversation formatted for OpenAI API")
    print("messages = agent.get_conversation_for_api()")
    print("")
    print("# Add system message if needed")
    print("messages.insert(0, {'role': 'system', 'content': 'You are a helpful assistant.'})")
    print("")
    print("# Use in OpenAI API call")
    print("response = openai.chat.completions.create(")
    print("    model='gpt-4',")
    print("    messages=messages")
    print(")")
    print("```")
    
    # Clean up
    if agent.memory_graph.database:
        agent.memory_graph.database.clear_database()
    
    print(f"\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demonstrate_complete_workflow()