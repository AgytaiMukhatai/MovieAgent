import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests
from datetime import datetime


@dataclass
class Message:
    """Represents a single message in conversation"""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert to dictionary for API calls"""
        msg = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

class ConversationMemory:
    """Custom memory manager for storing conversation history"""
    
    def __init__(self, max_messages: int = 20):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.context: Dict[str, Any] = {}
        self.conversation_summary = ""
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to memory"""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self._trim_memory()
    
    def _trim_memory(self):
        """Keep only recent messages to avoid token overflow"""
        if len(self.messages) > self.max_messages:
            # Keep first message (system) and trim from middle
            self.messages = [self.messages[0]] + self.messages[-(self.max_messages-1):]
    
    def get_messages_for_api(self) -> List[Dict]:
        """Format messages for API call"""
        return [msg.to_dict() for msg in self.messages]
    
    def update_context(self, key: str, value: Any):
        """Store persistent context (user preferences, facts)"""
        self.context[key] = value
    
    def get_context(self, key: str) -> Any:
        """Retrieve context"""
        return self.context.get(key)
    
    def get_recent_movies_discussed(self) -> List[str]:
        """Extract movies discussed in recent conversation"""
        return self.context.get("movies_discussed", [])
    
    def add_movie_to_context(self, movie_title: str):
        """Track movies discussed"""
        movies = self.context.get("movies_discussed", [])
        if movie_title not in movies:
            movies.append(movie_title)
            self.context["movies_discussed"] = movies
    
    def save_to_file(self, filename: str = "memory.json"):
        """Persist memory to disk"""
        data = {
            "messages": [asdict(msg) for msg in self.messages],
            "context": self.context,
            "summary": self.conversation_summary
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str = "memory.json"):
        """Load memory from disk"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.messages = [Message(**msg) for msg in data.get("messages", [])]
                self.context = data.get("context", {})
                self.conversation_summary = data.get("summary", "")
        except FileNotFoundError:
            pass

