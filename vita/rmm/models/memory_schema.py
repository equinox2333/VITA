from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
import json


@dataclass
class MemoryEntry:
    """Memory entry schema for storing reasoning corrections and insights."""
    
    id: str
    entry_type: str  # "correction", "insight", "fact", etc.
    content: str  # The actual memory content
    confidence: float  # Confidence score (0.0-1.0)
    related_to: List[str] = field(default_factory=list)  # IDs of related memory entries
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory entry to a dictionary."""
        return {
            "id": self.id,
            "entry_type": self.entry_type,
            "content": self.content,
            "confidence": self.confidence,
            "related_to": self.related_to,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert the memory entry to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create a memory entry from a dictionary."""
        return cls(
            id=data["id"],
            entry_type=data["entry_type"],
            content=data["content"],
            confidence=data["confidence"],
            related_to=data.get("related_to", []),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "MemoryEntry":
        """Create a memory entry from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class MemoryBuffer:
    """Container for managing multiple memory entries."""
    
    entries: List[MemoryEntry] = field(default_factory=list)
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the buffer."""
        self.entries.append(entry)
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_entries_by_type(self, entry_type: str) -> List[MemoryEntry]:
        """Get all memory entries of a specific type."""
        return [entry for entry in self.entries if entry.entry_type == entry_type]
    
    def to_prompt_format(self) -> str:
        """Convert the memory buffer to a formatted string for prompting."""
        result = "## Memory Buffer\n\n"
        for entry in self.entries:
            result += f"- [{entry.entry_type.upper()}] {entry.content}\n"
        return result
    
    def clear(self) -> None:
        """Clear all entries from the buffer."""
        self.entries = []