import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .core import MemoryEntry


class JSONDatabase:
    """JSON-based database for storing and retrieving memory entries."""
    
    def __init__(self, file_path: str = "memory_database.json"):
        """
        Initialize JSON database.
        
        Args:
            file_path: Path to the JSON database file.
        """
        self.file_path = file_path
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Create database file if it doesn't exist."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({"entries": [], "metadata": {"created": datetime.now().isoformat()}}, f)
    
    def store_entry(self, entry: MemoryEntry):
        """Store a memory entry in the JSON database."""
        # Load existing data
        data = self._load_database()
        
        # Convert entry to dict
        entry_dict = {
            "id": entry.id,
            "content": entry.content,
            "embedding": entry.embedding.tolist() if entry.embedding is not None and hasattr(entry.embedding, 'tolist') else entry.embedding if entry.embedding is not None else None,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp.isoformat(),
            "compressed": entry.compressed
        }
        
        # Add or update entry
        existing_index = None
        for i, existing_entry in enumerate(data["entries"]):
            if existing_entry["id"] == entry.id:
                existing_index = i
                break
        
        if existing_index is not None:
            data["entries"][existing_index] = entry_dict
        else:
            data["entries"].append(entry_dict)
        
        # Save back to file
        self._save_database(data)
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific entry from the database."""
        data = self._load_database()
        
        for entry_dict in data["entries"]:
            if entry_dict["id"] == entry_id:
                return self._dict_to_entry(entry_dict)
        
        return None
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries from the database."""
        data = self._load_database()
        return [self._dict_to_entry(entry_dict) for entry_dict in data["entries"]]
    
    def find_similar_entries(self, 
                           query_embedding: List[float], 
                           top_k: int = 5,
                           threshold: float = 0.0) -> List[tuple[MemoryEntry, float]]:
        """Find entries similar to the query embedding."""
        import numpy as np
        
        data = self._load_database()
        similarities = []
        
        query_vec = np.array(query_embedding)
        
        for entry_dict in data["entries"]:
            if entry_dict["embedding"] is None:
                continue
            
            entry_vec = np.array(entry_dict["embedding"])
            
            # Calculate cosine similarity
            dot_product = np.dot(query_vec, entry_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_entry = np.linalg.norm(entry_vec)
            
            if norm_query == 0 or norm_entry == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_entry)
            
            if similarity >= threshold:
                entry = self._dict_to_entry(entry_dict)
                similarities.append((entry, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry from the database."""
        data = self._load_database()
        
        original_length = len(data["entries"])
        data["entries"] = [entry for entry in data["entries"] if entry["id"] != entry_id]
        
        if len(data["entries"]) < original_length:
            self._save_database(data)
            return True
        
        return False
    
    def clear_database(self):
        """Clear all entries from the database."""
        data = {
            "entries": [],
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_cleared": datetime.now().isoformat()
            }
        }
        self._save_database(data)
    
    def _load_database(self) -> Dict[str, Any]:
        """Load database from file."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is corrupted or missing, create new database
            self._ensure_database_exists()
            with open(self.file_path, 'r') as f:
                return json.load(f)
    
    def _save_database(self, data: Dict[str, Any]):
        """Save database to file."""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _dict_to_entry(self, entry_dict: Dict[str, Any]) -> MemoryEntry:
        """Convert dictionary to MemoryEntry object."""
        import numpy as np
        
        embedding = None
        if entry_dict["embedding"] is not None:
            embedding = np.array(entry_dict["embedding"])
        
        timestamp = datetime.fromisoformat(entry_dict["timestamp"])
        
        return MemoryEntry(
            id=entry_dict["id"],
            content=entry_dict["content"],
            embedding=embedding,
            metadata=entry_dict["metadata"],
            timestamp=timestamp,
            compressed=entry_dict["compressed"]
        )