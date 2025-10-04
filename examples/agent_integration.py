"""
Example demonstrating agent integration with PardusMemory.

This example shows:
1. Using conversation format for compression
2. Integrating with an AI agent that calls LLM and compresses memories
3. Real-world usage pattern for conversational AI systems
4. Memory management in multi-turn conversations
"""

import os
import time
from typing import List, Dict, Any, Optional
from pardusmemory import (
    MemoryGraph,
    OpenAIEmbeddingService,
    OpenAICompressor,
    JSONDatabase,
    CosineSimilarity,
    MockEmbeddingService,
    MockCompressor
)


class ConversationalAgent:
    """
    Example agent that uses PardusMemory for conversation management.
    
    This agent demonstrates how to integrate memory compression in a conversational AI
    that maintains context over multiple turns.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 use_openai: bool = False,
                 memory_threshold: int = 10):
        """
        Initialize the conversational agent.
        
        Args:
            api_key: OpenAI API key (if using OpenAI services)
            use_openai: Whether to use OpenAI or mock services
            memory_threshold: Number of conversation turns before compression
        """
        self.memory_threshold = memory_threshold
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize services
        if use_openai and api_key:
            self.embedding_service = OpenAIEmbeddingService(api_key=api_key)
            self.compressor = OpenAICompressor(api_key=api_key)
            self.use_openai = True
        else:
            self.embedding_service = MockEmbeddingService(dimensions=384)
            self.compressor = MockCompressor()
            self.use_openai = False
        
        # Initialize memory graph
        self.memory_graph = MemoryGraph(
            similarity_function=CosineSimilarity(),
            embedding_service=self.embedding_service,
            database=JSONDatabase("agent_memory.json")
        )
        
        print(f"Agent initialized with {'OpenAI' if self.use_openai else 'Mock'} services")
        print(f"Memory compression threshold: {self.memory_threshold} turns")
    
    def add_conversation_turn(self, role: str, content: str) -> str:
        """
        Add a conversation turn to memory.
        
        Args:
            role: Role of the speaker (user/assistant/system)
            content: Content of the message
            
        Returns:
            Memory entry ID
        """
        # Create conversation message
        message = {"role": role, "content": content}
        
        # Add to conversation history
        self.conversation_history.append(message)
        
        # Add to memory graph
        entry_id = self.memory_graph.add_entry(
            content,
            metadata={
                "role": role,
                "turn_number": len(self.conversation_history),
                "timestamp": time.time()
            }
        )
        
        print(f"Added {role} message to memory (ID: {entry_id})")
        
        # Check if compression is needed
        if len(self.conversation_history) >= self.memory_threshold:
            self._compress_conversation_memory()
        
        return entry_id
    
    def _compress_conversation_memory(self):
        """Compress conversation history when threshold is reached."""
        print(f"\n--- Compressing conversation memory (threshold: {self.memory_threshold}) ---")
        
        # Get recent conversation entries
        all_entries = self.memory_graph.get_all_entries()
        conversation_entries = [
            entry for entry in all_entries 
            if entry.metadata.get("role") in ["user", "assistant"]
        ]
        
        if len(conversation_entries) < 5:  # Need at least 5 entries to compress
            return
        
        # Sort by turn number and take oldest entries for compression
        conversation_entries.sort(key=lambda x: x.metadata.get("turn_number", 0))
        entries_to_compress = conversation_entries[:-3]  # Keep last 3 entries uncompressed
        entry_ids_to_compress = [entry.id for entry in entries_to_compress]
        
        # Format as conversation for compression
        conversation_format = []
        for entry in entries_to_compress:
            role = entry.metadata.get("role", "unknown")
            conversation_format.append({
                "role": role,
                "content": entry.content
            })
        
        # Compress using conversation format
        try:
            # Get conversation format compression result
            compressed_conversation = self.compressor.compress(conversation_format, return_as_conversation=True)  # type: ignore
            
            # Extract the compressed content
            compressed_content = compressed_conversation[0]["content"] if compressed_conversation else ""  # type: ignore
            
            # Create a memory entry with the compressed content
            compressed_id = self.memory_graph.add_entry(
                compressed_content,
                metadata={
                    "type": "compressed_conversation",
                    "original_messages": len(conversation_format),
                    "compressed": True
                }
            )
            
            # Update conversation history
            self.conversation_history = [
                msg for msg in self.conversation_history 
                if msg not in conversation_format
            ]
            
            # Add compressed summary to history
            compressed_entry = self.memory_graph.get_entry(compressed_id)
            if compressed_entry:
                self.conversation_history.append({
                    "role": "system",
                    "content": f"[COMPRESSED SUMMARY]: {compressed_entry.content}"
                })
            
            print(f"Compressed {len(entry_ids_to_compress)} conversation turns")
            print(f"Remaining conversation turns: {len(self.conversation_history)}")
            
        except Exception as e:
            print(f"Error during compression: {e}")
    
    def search_memory(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search memory for relevant information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant memory contents
        """
        similar_entries = self.memory_graph.find_similar(query, top_k=top_k)
        return [entry.content for entry, _ in similar_entries]
    
    def get_conversation_context(self, max_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation context.
        
        Args:
            max_turns: Maximum number of recent turns to return
            
        Returns:
            Recent conversation history
        """
        return self.conversation_history[-max_turns:]
    
    def simulate_llm_call(self, user_input: str) -> str:
        """
        Simulate an LLM call with memory integration.
        
        In a real agent, this would call an actual LLM API.
        Here we simulate the response and show memory integration.
        """
        print(f"\n--- Agent processing user input: '{user_input}' ---")
        
        # Search memory for relevant context
        relevant_memories = self.search_memory(user_input, top_k=2)
        
        # Get recent conversation context
        recent_context = self.get_conversation_context(max_turns=3)
        
        # Simulate LLM response generation
        response = f"I understand you're asking about '{user_input}'. "
        
        if relevant_memories:
            response += "Based on our previous conversations, I recall that "
            response += "; ".join(relevant_memories[:2]) + ". "
        
        response += "Let me help you with that."
        
        # Add user input and response to memory
        self.add_conversation_turn("user", user_input)
        self.add_conversation_turn("assistant", response)
        
        return response


def demonstrate_conversation_format():
    """Demonstrate conversation format compression."""
    print("=== Conversation Format Compression Demo ===\n")
    
    # Create compressor
    compressor = MockCompressor()
    
    # Example conversation in OpenAI format
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": "Sure! A common example is email spam filtering. The system learns to classify emails as spam or not spam by analyzing thousands of examples."}
    ]
    
    print("Original conversation:")
    for msg in conversation:
        print(f"  {msg['role'].title()}: {msg['content']}")
    
    # Compress the conversation
    compressed = compressor.compress(conversation)  # type: ignore
    
    print(f"\nCompressed conversation:\n{compressed}")


def demonstrate_agent_integration():
    """Demonstrate full agent integration with memory compression."""
    print("\n=== Agent Integration Demo ===\n")
    
    # Create agent
    agent = ConversationalAgent(use_openai=False, memory_threshold=5)
    
    # Simulate conversation
    conversation_turns = [
        "Hi, I'm interested in learning about Python programming.",
        "What are the main features of Python?",
        "Can you explain Python's data types?",
        "How do I create functions in Python?",
        "What about object-oriented programming in Python?",
        "Tell me about Python decorators.",
        "How can I use Python for data analysis?"
    ]
    
    print("Starting conversation with agent...")
    print("(Memory compression will trigger after 5 turns)\n")
    
    for i, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        response = agent.simulate_llm_call(user_input)
        print(f"Agent: {response}")
        
        # Show memory status
        total_entries = len(agent.memory_graph.get_all_entries())
        print(f"Memory entries: {total_entries}")
        
        # Small delay for demo purposes
        time.sleep(0.5)
    
    # Show final memory state
    print(f"\n--- Final Memory State ---")
    print(f"Total conversation turns processed: {len(conversation_turns)}")
    print(f"Current memory entries: {len(agent.memory_graph.get_all_entries())}")
    print(f"Conversation history length: {len(agent.conversation_history)}")
    
    # Demonstrate memory search
    print(f"\n--- Memory Search Demo ---")
    search_results = agent.search_memory("Python functions", top_k=2)
    print(f"Search results for 'Python functions':")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. {result[:100]}...")
    
    # Clean up
    if agent.memory_graph.database:
        agent.memory_graph.database.clear_database()
    print("\nAgent integration demo completed!")


def demonstrate_openai_conversation_compression():
    """Demonstrate conversation compression with OpenAI (if API key available)."""
    print("\n=== OpenAI Conversation Compression Demo ===\n")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Skipping OpenAI demo.")
        print("Set OPENAI_API_KEY environment variable to run this demo.")
        return
    
    try:
        # Create OpenAI compressor
        compressor = OpenAICompressor(api_key=api_key, model="gpt-3.5-turbo")
        
        # Example conversation
        conversation = [
            {"role": "user", "content": "I'm working on a React project and need help with state management."},
            {"role": "assistant", "content": "I'd be happy to help with React state management! What specific aspect are you struggling with? Are you looking at useState, useReducer, Context API, or perhaps external libraries like Redux?"},
            {"role": "user", "content": "I'm trying to decide between useState and useReducer for a complex form with multiple fields."},
            {"role": "assistant", "content": "Great question! For complex forms with multiple related fields, useReducer is often better. It provides a more predictable state transition pattern and makes it easier to handle complex state logic in one place. useState is better for simple, independent state values."},
            {"role": "user", "content": "Can you show me a quick example of useReducer for a form?"},
            {"role": "assistant", "content": "Certainly! Here's a basic example: const [state, dispatch] = useReducer(formReducer, initialState); Your reducer would handle actions like 'UPDATE_FIELD', 'RESET_FORM', etc. This approach centralizes your form logic and makes it easier to add validation or handle complex state transitions."}
        ]
        
        print("Compressing React conversation with OpenAI...")
        compressed = compressor.compress(conversation)  # type: ignore
        print(f"Compressed result:\n{compressed}")
        
    except Exception as e:
        print(f"Error with OpenAI compression: {e}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_conversation_format()
    demonstrate_agent_integration()
    demonstrate_openai_conversation_compression()