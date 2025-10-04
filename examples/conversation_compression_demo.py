"""
Simple demonstration of conversation format compression.

This example shows how to use the enhanced compression functionality
with OpenAI-style conversation messages.
"""

from pardusmemory import MockCompressor, OpenAICompressor
import os


def demonstrate_conversation_compression():
    """Demonstrate conversation format compression."""
    print("=== Conversation Format Compression Demo ===\n")
    
    # Example conversation in OpenAI format
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant specialized in programming."},
        {"role": "user", "content": "I want to learn about React hooks. Can you explain them?"},
        {"role": "assistant", "content": "React hooks are functions that let you use state and other React features in functional components. The most common hooks are useState for managing state and useEffect for handling side effects."},
        {"role": "user", "content": "How does useState work exactly?"},
        {"role": "assistant", "content": "useState returns an array with two elements: the current state value and a function to update it. For example: const [count, setCount] = useState(0). When you call setCount, React re-renders the component with the new value."},
        {"role": "user", "content": "What about useEffect? When should I use it?"},
        {"role": "assistant", "content": "useEffect is for side effects like API calls, subscriptions, or DOM manipulation. It takes two arguments: a function to run and a dependency array. If the dependency array is empty, it runs once after mount. If it contains values, it runs when those values change."}
    ]
    
    print("Original conversation:")
    for i, msg in enumerate(conversation, 1):
        role = msg["role"].title()
        content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"  {i}. {role}: {content}")
    
    print(f"\nTotal messages: {len(conversation)}")
    
    # Compress with MockCompressor
    print("\n--- Compression with MockCompressor ---")
    mock_compressor = MockCompressor()
    
    # Standard compression (returns string)
    mock_compressed = mock_compressor.compress(conversation)  # type: ignore
    print("Mock compressed result (string):")
    print(mock_compressed)
    
    # Conversation format compression (returns OpenAI format)
    mock_compressed_conversation = mock_compressor.compress(conversation, return_as_conversation=True)  # type: ignore
    print("\nMock compressed result (conversation format):")
    print(mock_compressed_conversation)
    
    # Compress with OpenAI if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("\n--- Compression with OpenAI ---")
        try:
            openai_compressor = OpenAICompressor(api_key=api_key, model="gpt-3.5-turbo")
            
            # Standard compression
            openai_compressed = openai_compressor.compress(conversation)  # type: ignore
            print("OpenAI compressed result (string):")
            print(openai_compressed)
            
            # Conversation format compression
            openai_compressed_conversation = openai_compressor.compress(conversation, return_as_conversation=True)  # type: ignore
            print("\nOpenAI compressed result (conversation format):")
            print(openai_compressed_conversation)
            
        except Exception as e:
            print(f"OpenAI compression failed: {e}")
    else:
        print("\n--- OpenAI Compression ---")
        print("Set OPENAI_API_KEY environment variable to see OpenAI compression example.")


def demonstrate_mixed_format():
    """Demonstrate compression with mixed string and conversation format."""
    print("\n=== Mixed Format Compression Demo ===\n")
    
    # Mixed content: some strings, some conversation messages
    mixed_content = [
        "Python is a high-level programming language.",
        {"role": "user", "content": "What are Python's main features?"},
        {"role": "assistant", "content": "Python is known for its simple syntax, dynamic typing, and extensive libraries."},
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        {"role": "user", "content": "Is Python good for data science?"},
        {"role": "assistant", "content": "Yes! Python is excellent for data science with libraries like NumPy, Pandas, and scikit-learn."}
    ]
    
    print("Mixed content (strings + conversation messages):")
    for i, item in enumerate(mixed_content, 1):
        if isinstance(item, str):
            print(f"  {i}. String: {item[:60]}...")
        else:
            print(f"  {i}. {item['role'].title()}: {item['content'][:60]}...")
    
    print("\n--- Compressing mixed format ---")
    compressor = MockCompressor()
    compressed = compressor.compress(mixed_content)
    print("Compressed result:")
    print(compressed)


def demonstrate_agent_workflow():
    """Demonstrate how this would be used in an agent workflow."""
    print("\n=== Agent Workflow Example ===\n")
    
    # Simulate an agent's conversation memory
    agent_memory = []
    
    # Simulate conversation turns
    turns = [
        {"role": "user", "content": "Help me understand machine learning basics."},
        {"role": "assistant", "content": "Machine learning is about training computers to learn patterns from data without explicit programming."},
        {"role": "user", "content": "What's the difference between supervised and unsupervised learning?"},
        {"role": "assistant", "content": "Supervised learning uses labeled data with correct answers, while unsupervised learning finds patterns in unlabeled data."},
        {"role": "user", "content": "Can you give me an example of each?"},
        {"role": "assistant", "content": "Supervised: spam detection (labeled emails). Unsupervised: customer segmentation (grouping similar customers)."}
    ]
    
    print("Agent conversation building up...")
    for turn in turns:
        agent_memory.append(turn)
        print(f"  Added {turn['role']}: {turn['content'][:50]}...")
    
    print(f"\nMemory size: {len(agent_memory)} turns")
    
    # Agent decides to compress when memory gets too large
    if len(agent_memory) > 4:
        print("\n--- Agent compressing memory ---")
        compressor = MockCompressor()
        
        # Keep recent turns, compress older ones
        recent_turns = agent_memory[-2:]  # Keep last 2 turns
        old_turns = agent_memory[:-2]     # Compress older turns
        
        if old_turns:
            compressed_summary = compressor.compress(old_turns)
            
            # New memory structure
            agent_memory = recent_turns + [{
                "role": "system",
                "content": f"[COMPRESSED PREVIOUS CONVERSATION]: {compressed_summary}"
            }]
            
            print(f"Compressed {len(old_turns)} turns into summary")
            print(f"New memory size: {len(agent_memory)} items")
            
            print("\nCompressed memory structure:")
            for i, item in enumerate(agent_memory, 1):
                role = item["role"]
                content = item["content"][:80] + "..." if len(item["content"]) > 80 else item["content"]
                print(f"  {i}. {role.title()}: {content}")


if __name__ == "__main__":
    demonstrate_conversation_compression()
    demonstrate_mixed_format()
    demonstrate_agent_workflow()