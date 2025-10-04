"""
Demonstration of conversation format return from compression.

This example shows how the compression functionality can return
results in OpenAI conversation format: [{"role": "assistant", "content": "..."}]
"""

from pardusmemory import MockCompressor, OpenAICompressor
import os


def demonstrate_conversation_format_return():
    """Demonstrate the new conversation format return feature."""
    print("=== Conversation Format Return Demo ===\n")
    
    # Example conversation
    conversation = [
        {"role": "user", "content": "What is React?"},
        {"role": "assistant", "content": "React is a JavaScript library for building user interfaces, particularly web applications with rich, interactive UIs."},
        {"role": "user", "content": "What are React hooks?"},
        {"role": "assistant", "content": "React hooks are functions that let you use state and other React features in functional components. Common hooks include useState, useEffect, and useContext."},
        {"role": "user", "content": "Can you show me a useState example?"},
        {"role": "assistant", "content": "Sure! Here's a simple useState example: const [count, setCount] = useState(0); This creates a state variable 'count' initialized to 0, and 'setCount' is the function to update it."}
    ]
    
    print("Original conversation:")
    for i, msg in enumerate(conversation, 1):
        print(f"  {i}. {msg['role'].title()}: {msg['content'][:60]}...")
    
    print(f"\nTotal messages: {len(conversation)}")
    
    # Test with MockCompressor
    print("\n--- MockCompressor with Conversation Format Return ---")
    mock_compressor = MockCompressor()
    
    # Standard compression (returns string)
    mock_string = mock_compressor.compress(conversation)  # type: ignore
    print("Standard compression (returns string):")
    print(f"Type: {type(mock_string)}")
    print(f"Content: {mock_string[:100]}...")
    
    # Conversation format compression (returns OpenAI format)
    mock_conversation = mock_compressor.compress(conversation, return_as_conversation=True)  # type: ignore
    print("\nConversation format compression (returns OpenAI format):")
    print(f"Type: {type(mock_conversation)}")
    print(f"Content: {mock_conversation}")
    
    # Test with OpenAI if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("\n--- OpenAICompressor with Conversation Format Return ---")
        try:
            openai_compressor = OpenAICompressor(api_key=api_key, model="gpt-3.5-turbo")
            
            # Standard compression
            openai_string = openai_compressor.compress(conversation)  # type: ignore
            print("Standard compression (returns string):")
            print(f"Type: {type(openai_string)}")
            print(f"Content: {openai_string[:100]}...")
            
            # Conversation format compression
            openai_conversation = openai_compressor.compress(conversation, return_as_conversation=True)  # type: ignore
            print("\nConversation format compression (returns OpenAI format):")
            print(f"Type: {type(openai_conversation)}")
            print(f"Content: {openai_conversation}")
            
        except Exception as e:
            print(f"OpenAI compression failed: {e}")
    else:
        print("\n--- OpenAICompressor ---")
        print("Set OPENAI_API_KEY environment variable to see OpenAI compression example.")


def demonstrate_agent_usage():
    """Demonstrate how this would be used in an agent workflow."""
    print("\n=== Agent Usage Example ===\n")
    
    # Simulate an agent that needs to compress conversation history
    conversation_history = [
        {"role": "user", "content": "I need help with Python programming."},
        {"role": "assistant", "content": "I'd be happy to help you with Python! What specific topic are you interested in?"},
        {"role": "user", "content": "How do I handle file I/O in Python?"},
        {"role": "assistant", "content": "Python provides built-in functions for file I/O. You can use open() to open files, read() to read content, and write() to write content. Always remember to close files or use context managers with 'with' statements."},
        {"role": "user", "content": "Can you show me an example?"},
        {"role": "assistant", "content": "Sure! Here's an example: with open('example.txt', 'r') as file: content = file.read(). This opens the file in read mode and automatically closes it when done."}
    ]
    
    print("Agent conversation history:")
    for i, msg in enumerate(conversation_history, 1):
        print(f"  {i}. {msg['role'].title()}: {msg['content'][:50]}...")
    
    print(f"\nHistory length: {len(conversation_history)} messages")
    
    # Agent decides to compress when history gets too long
    if len(conversation_history) > 4:
        print("\n--- Agent compressing conversation history ---")
        compressor = MockCompressor()
        
        # Compress in conversation format for easy integration
        compressed_result = compressor.compress(conversation_history, return_as_conversation=True)  # type: ignore
        
        print("Compressed result (ready for agent use):")
        print(f"Type: {type(compressed_result)}")
        print(f"Format: {compressed_result}")
        
        # Agent can now use this compressed result directly
        if compressed_result:
            compressed_message = compressed_result[0]
            print(f"\nAgent can now use:")
            print(f"  Role: {compressed_message['role']}")
            print(f"  Content: {compressed_message['content'][:100]}...")
            
            # This can be directly added to conversation history
            new_history = conversation_history[-2:] + [compressed_message]  # Keep last 2 + compressed
            print(f"\nNew conversation history length: {len(new_history)} messages")
            print("Agent successfully compressed and integrated conversation!")


def demonstrate_api_compatibility():
    """Demonstrate compatibility with OpenAI API format."""
    print("\n=== OpenAI API Compatibility Demo ===\n")
    
    # Show how the compressed result can be used with OpenAI API
    conversation = [
        {"role": "user", "content": "Explain machine learning concepts."},
        {"role": "assistant", "content": "Machine learning involves training algorithms to recognize patterns in data and make predictions or decisions without explicit programming."},
        {"role": "user", "content": "What's the difference between classification and regression?"},
        {"role": "assistant", "content": "Classification predicts discrete categories (like spam/not spam), while regression predicts continuous values (like house prices)."}
    ]
    
    print("Original conversation for API compatibility test:")
    for msg in conversation:
        print(f"  {msg['role'].title()}: {msg['content'][:50]}...")
    
    # Compress and show API compatibility
    compressor = MockCompressor()
    compressed = compressor.compress(conversation, return_as_conversation=True)  # type: ignore
    
    print(f"\nCompressed result (OpenAI compatible):")
    print(f"Format: {compressed}")
    
    # Show how this can be used in an API call
    if compressed:
        print(f"\nExample API usage:")
        print("```python")
        print("# Compressed conversation can be used directly in API calls")
        print("messages = [")
        print('    {"role": "system", "content": "You are a helpful assistant."},')
        for msg in compressed:
            print(f'    {msg},')
        print("]")
        print("response = openai.chat.completions.create(")
        print('    model="gpt-4",')
        print("    messages=messages")
        print(")")
        print("```")


if __name__ == "__main__":
    demonstrate_conversation_format_return()
    demonstrate_agent_usage()
    demonstrate_api_compatibility()