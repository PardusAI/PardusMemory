from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from openai import OpenAI


class LLMCompressor(ABC):
    """Abstract base class for LLM-based knowledge compression."""
    
    @abstractmethod
    def compress(self, content_list: List[Union[str, Dict[str, Any]]], return_as_conversation: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Compress a list of content or conversation messages into a detailed summary.
        
        Args:
            content_list: List of content strings or OpenAI-style conversation messages to compress.
            return_as_conversation: If True, returns result in OpenAI conversation format.
            
        Returns:
            Detailed summary preserving as much information as possible.
            If return_as_conversation is True, returns [{"role": "assistant", "content": summary}]
        """
        pass


class OpenAICompressor(LLMCompressor):
    """OpenAI-based LLM compressor for knowledge compression."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-4",
                 max_tokens: Optional[int] = None):
        """
        Initialize OpenAI compressor.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            model: LLM model to use for compression.
            max_tokens: Maximum tokens in the compressed summary.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_tokens = max_tokens
    
    def _format_content(self, item: Union[str, Dict[str, Any]]) -> str:
        """
        Format content item for compression.
        
        Args:
            item: Either a string content or OpenAI-style message dict.
            
        Returns:
            Formatted string content.
        """
        if isinstance(item, str):
            return item
        
        # Handle OpenAI-style conversation format
        if isinstance(item, dict):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            
            # Format as conversation turn
            if role == "user":
                return f"User: {content}"
            elif role == "assistant":
                return f"Assistant: {content}"
            elif role == "system":
                return f"System: {content}"
            else:
                return f"{role.title()}: {content}"
        
        return str(item)
    
    def compress(self, content_list: List[Union[str, Dict[str, Any]]], return_as_conversation: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Compress a list of content or conversation messages into a detailed, information-preserving summary.
        
        Args:
            content_list: List of content strings or OpenAI-style conversation messages to compress.
            return_as_conversation: If True, returns result in OpenAI conversation format.
            
        Returns:
            Detailed summary preserving as much information as possible.
            If return_as_conversation is True, returns [{"role": "assistant", "content": summary}]
        """
        if not content_list:
            return [] if return_as_conversation else ""
        
        if len(content_list) == 1:
            summary = self._format_content(content_list[0])
        else:
            # Format and combine content with clear separators
            formatted_contents = [self._format_content(item) for item in content_list]
            combined_content = "\n\n--- CONTENT SEPARATOR ---\n\n".join(formatted_contents)
            
            # Create detailed compression prompt
            system_prompt = """You are a knowledge compression specialist. Your task is to create a comprehensive, detailed summary of the provided content while preserving as much information as possible.

CRITICAL REQUIREMENTS:
1. Preserve ALL key facts, details, relationships, and context
2. Maintain chronological order and temporal relationships
3. Keep specific names, dates, numbers, and technical details
4. Preserve the original intent and nuance of the content
5. Include important metadata and contextual information
6. Structure the summary logically with clear organization
7. Do NOT lose any critical information during compression

The goal is information preservation, not brevity. Create a summary that someone could use to reconstruct the original content with minimal information loss."""

            user_prompt = f"""Please compress the following content into a detailed, information-preserving summary:

{combined_content}

Remember: Focus on preserving maximum information rather than brevity. Include all important details, facts, and context."""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1  # Low temperature for consistent, factual output
                )
                
                summary = response.choices[0].message.content or ""
                summary = summary.strip()
                
            except Exception as e:
                raise RuntimeError(f"Failed to compress content: {e}")
        
        # Return in requested format
        if return_as_conversation:
            return [{"role": "assistant", "content": summary}]
        else:
            return summary
    
    def compress_progressive(self, content_list: List[Union[str, Dict[str, Any]]], chunk_size: int = 5, return_as_conversation: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Progressively compress large content lists in chunks.
        
        Args:
            content_list: List of content strings or conversation messages to compress.
            chunk_size: Number of items to compress at once.
            return_as_conversation: If True, returns result in OpenAI conversation format.
            
        Returns:
            Final compressed summary.
        """
        if len(content_list) <= chunk_size:
            return self.compress(content_list, return_as_conversation=return_as_conversation)
        
        # Compress in chunks progressively
        current_content = content_list
        
        while len(current_content) > chunk_size:
            next_round = []
            
            # Process in chunks
            for i in range(0, len(current_content), chunk_size):
                chunk = current_content[i:i + chunk_size]
                compressed_chunk = self.compress(chunk, return_as_conversation=False)
                next_round.append(compressed_chunk)
            
            current_content = next_round
        
        # Final compression
        return self.compress(current_content, return_as_conversation=return_as_conversation)


class MockCompressor(LLMCompressor):
    """Mock compressor for testing purposes."""
    
    def compress(self, content_list: List[Union[str, Dict[str, Any]]], return_as_conversation: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """Create a mock summary by concatenating content."""
        if not content_list:
            return [] if return_as_conversation else ""
        
        if len(content_list) == 1:
            summary = self._format_content(content_list[0])
        else:
            # Create a simple mock summary
            summary = "COMPRESSED SUMMARY:\n\n"
            for i, item in enumerate(content_list, 1):
                content = self._format_content(item)
                if content:
                    summary += f"Item {i}: {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
            
            summary += f"\n[Total items compressed: {len(content_list)}]"
        
        # Return in requested format
        if return_as_conversation:
            return [{"role": "assistant", "content": summary}]
        else:
            return summary
    
    def _format_content(self, item: Union[str, Dict[str, Any]]) -> str:
        """Format content item for compression."""
        if isinstance(item, str):
            return item
        
        # Handle OpenAI-style conversation format
        if isinstance(item, dict):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            
            # Format as conversation turn
            if role == "user":
                return f"User: {content}"
            elif role == "assistant":
                return f"Assistant: {content}"
            elif role == "system":
                return f"System: {content}"
            else:
                return f"{role.title()}: {content}"
        
        return str(item)