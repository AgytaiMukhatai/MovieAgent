import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)



def summarize_messages(messages) -> str:
    """
    messages: list[{"role": "user"/"assistant", "content": "..."}]
    Returns: a concise "memory" of stable user facts and preferences.
    """
    system = {
        "role": "system",
        "content": (
            "You are a memory module for a chatbot.\n"
            "Your job is to extract and maintain long-term memory about the USER.\n\n"
            "From the following dialog between a user and an assistant, "
            "write a short list of facts about the user that will be useful "
            "in future conversations.\n\n"
            "Only include **stable information** about the user: "
            "name, age, origin, location, job, studies, hobbies, preferences, "
            "likes/dislikes, long-term projects, etc.\n\n"
            "Do NOT write a friendly response.\n"
            "Do NOT describe what the assistant said.\n"
            "Do NOT ask the user questions.\n\n"
            "Format as bullet points, like:\n"
            "- Name: ...\n"
            "- From: ...\n"
            "- Studies: ...\n"
            "- Likes: ...\n"
            "If some fields are unknown, just omit them."
        )
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[system] + messages,
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()



Message = Dict[str, str]  # {"role": "user"/"assistant", "content": "..."}


class ConversationSummaryMemory:
    def __init__(self, max_recent_messages: int = 5, summarize_after: int = 10):
        """
        max_recent_messages: how many latest raw messages to keep after summarizing
        summarize_after: when len(history) exceeds this, we summarize
        """
        self.max_recent_messages = max_recent_messages
        self.summarize_after = summarize_after

        self.summary: str = ""           # long-term memory (text)
        self.history: List[Message] = [] # raw messages (no system)
        self.movies_discussed: List[str] = []  # Track movie titles discussed
        self.movie_context: Dict[str, Dict] = {}  # Store movie data by title

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def maybe_summarize(self):
        """
        If the conversation history is long, summarize it and shrink.
        """
        if len(self.history) <= self.summarize_after:
            return  # nothing to do

        # Summarize all current raw history
        new_summary = summarize_messages(self.history)

        if self.summary:
            # Merge old summary + new summary
            merged_summary = (
                self.summary
                + "\n\nUpdates from later conversation:\n"
                + new_summary
            )
            self.summary = merged_summary
        else:
            self.summary = new_summary

        # Keep only the last max_recent_messages raw messages
        self.history = self.history[-self.max_recent_messages:]

    def build_prompt(self, base_system_prompt: str) -> List[Message]:
        """
        Build the messages list to send to the model:
        - base system prompt
        - summary (if exists) as a second system message
        - recent raw history
        """
        messages: List[Message] = []

        # 1) Main system message
        messages.append({
            "role": "system",
            "content": (
                base_system_prompt
                + "\n\nUse the conversation summary (if provided) "
                  "to stay consistent with past discussion."
            ),
        })

        # 2) Summary as another system message (optional)
        if self.summary:
            messages.append({
                "role": "system",
                "content": "Conversation summary so far:\n" + self.summary,
            })

        # 3) Recent raw messages
        messages.extend(self.history)

        return messages
    
    def add_movie_to_context(self, title: str, movie_data: Dict):
        """Add a movie to the context of discussed movies"""
        if title and title not in self.movies_discussed:
            self.movies_discussed.append(title)
        if title:
            self.movie_context[title] = movie_data
    
    def get_recent_movies_discussed(self) -> List[str]:
        """Get list of movies discussed in the conversation"""
        return self.movies_discussed.copy()
    
    def save_to_file(self, filename: str = "conversation.json"):
        """Save conversation history and context to a JSON file"""
        data = {
            "summary": self.summary,
            "history": self.history,
            "movies_discussed": self.movies_discussed,
            "movie_context": self.movie_context
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filename: str = "conversation.json"):
        """Load conversation history and context from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.summary = data.get("summary", "")
            self.history = data.get("history", [])
            self.movies_discussed = data.get("movies_discussed", [])
            self.movie_context = data.get("movie_context", {})
        except FileNotFoundError:
            print(f"⚠️ File {filename} not found. Starting with empty conversation.")
        except json.JSONDecodeError:
            print(f"⚠️ Error reading {filename}. File may be corrupted.")
