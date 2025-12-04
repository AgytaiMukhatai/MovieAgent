import os
import json
import requests
import re
from typing import List, Dict, Any, Optional        
from ConversationSummaryMemory import ConversationSummaryMemory
from ToolExecutor import ToolExecutor
from tools import AVAILABLE_TOOLS


class CinematographyAgent:
    """
    Expert cinematography agent with custom memory and tool calling.
    
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.memory = ConversationSummaryMemory(max_recent_messages=10, summarize_after=20)
        self.tool_executor = ToolExecutor(AVAILABLE_TOOLS)
        
        # System prompt
        self.system_prompt = """You are an expert cinematography analyst with deep knowledge of film production.

Your responsibilities:
1. Retrieve accurate movie data using tools.
2. Provide detailed and accurate information about the movie.
3. Never guess or hallucinate any factual movie information.
4. Never make up any information.
5. Never provide any information that is not present in the tool results.
----------------------------------------------
TOOL USAGE RULES (VERY IMPORTANT)
----------------------------------------------

You MUST ALWAYS call a tool before answering when the user asks for:
- Cast / actors
- Directors / writers
- Plot / synopsis
- Ratings (IMDb, Metacritic, Rotten Tomatoes)
- Runtime, release year, genre
- Awards
- Any other factual movie information
- Searching for a movie by name or keyword
- Comparing movies
- Recommending movies based on genres, actors, directors

Use these tools:
1. `search_movie_by_title` for title-based queries.
2. `get_movie_ratings` when the user asks about ratings.
3. 'search_movies_list' when the user asks about a list of movies.
4. 'compare_movies' when the user asks to compare two movies.
5. 'get_cast_and_authors' when the user asks about the cast and authors of a movie.
6. 'find_movies_by_min_imdb_rating' when the user asks to find movies by minimum IMDb rating.



NEVER answer factual questions from built-in knowledge when a tool exists.
The only time you do NOT call a tool is when:
- The user asks a purely conceptual question about filmmaking (e.g., "What is color grading?")
- The user asks about your reasoning
- The user asks meta-questions about the conversation

---------------------------------------------------------
AFTER GETTING TOOL RESULTS
---------------------------------------------------------
After receiving tool output, use it to:
- Build a detailed answer
- Reference specific scenes
- Compare to other films from the same director or genre

Do NOT invent details that are not present in the tool result.

---------------------------------------------------------
STYLE GUIDELINES
---------------------------------------------------------
Your answers must be:
- Expert-level
- Clear and engaging
- Cinematography-focused when appropriate
- Accurate and grounded in tool data
- Helpful to film students, cinematographers, and enthusiasts

When user asks about the movie's cinematography, include:
- Information about the movie's cinematography style
- Information about the movie's lighting
- Information about the movie's composition
- Information about the movie's lenses
- Information about the movie's movement
- Information about the movie's color palette


If the user expresses a preference or shares personal info (e.g., favorite movie, favorite director), personalize your recommendations.

---------------------------------------------------------
CONTEXT CONTINUITY
---------------------------------------------------------
Remember previous details from the conversation (like the user's name or preferences) unless the session resets.

        Your goal is to deliver accurate, tool-grounded, cinematic expertise.
"""

    def _extract_movie_titles(self, text: str) -> List[str]:
        """
        Simple heuristic to extract potential movie titles from text.
        Looks for capitalized phrases that might be movie titles.
        This is a fallback when tools aren't called.
        """
        titles = []
        
        # Pattern 1: Quoted strings (likely movie titles)
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        titles.extend(quoted)
        
        # Pattern 2: After common verbs like "like", "love", "watch", etc.
        # Matches: "I like Interstellar" -> "Interstellar"
        verb_pattern = r'(?:like|love|enjoy|watch|saw|seen|about|discuss|recommend|prefer|think|thought)\s+([A-Z][a-zA-Z0-9\s]+?)(?:\s|\.|,|$|!|\?)'
        verb_matches = re.findall(verb_pattern, text, re.IGNORECASE)
        titles.extend(verb_matches)
        
        # Pattern 3: Standalone capitalized words/phrases
        # Split text and look for sequences of capitalized words
        words = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\b', text)
        # Filter out common words and very short words
        common_words = {'I', 'The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'From'}
        for word in words:
            if len(word) > 3 and word not in common_words:
                # Check if it's not at the start of sentence (more likely to be a title)
                word_pos = text.find(word)
                if word_pos > 0:  # Not at start
                    titles.append(word)
        
        # Clean and deduplicate
        cleaned_titles = []
        seen = set()
        for title in titles:
            title = title.strip()
            if len(title) > 2:
                title_lower = title.lower()
                if title_lower not in seen and title_lower not in ['the', 'a', 'an', 'i', 'you', 'we', 'they']:
                    seen.add(title_lower)
                    cleaned_titles.append(title)
        
        return cleaned_titles
    
    def _call_llm(self, messages: List[Dict], use_tools: bool = True) -> Dict:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        
        if use_tools:
            payload["tools"] = self.tool_executor.get_tool_schemas()
            payload["tool_choice"] = "auto"
        
        print("\n=== DEBUG: Payload sent to OpenAI ===")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if not response.ok:
            print("\n=== DEBUG: OpenAI error response ===")
            print("Status:", response.status_code)
            print("Body:", response.text)  # <- this is the important part
            response.raise_for_status()
        
        return response.json()
    
    def _call_llm_with_retry(self, messages: List[Dict], use_tools: bool = True, 
                            max_retries: int = 3) -> Dict:
        """Call LLM with exponential backoff retry"""
        import time
        
        for attempt in range(max_retries):
            try:
                return self._call_llm(messages, use_tools)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"LLM API failed after {max_retries} retries: {str(e)}")
                
                wait_time = 2 ** attempt
                print(f"⚠️ API call failed (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def run(self, user_input: str, max_iterations: int = 5, verbose: bool = True) -> str:
        """
        Main agent loop with reasoning and tool calling
        
        Args:
            user_input: User's question or request
            max_iterations: Max tool calling loops to prevent infinite loops
            verbose: Print intermediate steps
        
        Returns:
            Final assistant response
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎬 User: {user_input}")
            print(f"{'='*60}\n")
        
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        self.memory.maybe_summarize()  # <— good place to call this
        messages = self.memory.build_prompt(self.system_prompt)
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            if verbose:
                print(f"🔄 Iteration {iteration}/{max_iterations}")
            
            # Call LLM
            try:
                response = self._call_llm_with_retry(messages, use_tools=True)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.memory.add_message("assistant", error_msg)
                return error_msg
            
            # Parse response
            tool_calls, content = self.tool_executor.parse_tool_calls(response)
            
            # If no tool calls, this is the final answer
            if not tool_calls:
                final_answer = content or "I apologize, I couldn't generate a response."
                self.memory.add_message("assistant", final_answer)
                
                # Try to extract movie titles from user input even if no tools were called
                # This helps track movies mentioned in casual conversation
                potential_movies = self._extract_movie_titles(user_input)
                for movie_title in potential_movies:
                    # Add to discussed movies even without full movie data
                    if movie_title not in self.memory.movies_discussed:
                        self.memory.movies_discussed.append(movie_title)
                
                #if verbose:
                    #print(f"\n✅ Final Answer:\n{final_answer}\n")
                
                return final_answer
            
            # Add assistant message with tool calls to memory
            assistant_msg = response["choices"][0]["message"]
            messages.append(assistant_msg)
            
            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                if verbose:
                    print(f"🔧 Calling tool: {tool_name}")
                    print(f"   Arguments: {json.dumps(tool_args, indent=2)}")
                
                # Execute tool
                result = self.tool_executor.execute_tool(tool_name, tool_args)
                
                if verbose:
                    print(f"   Result: {result[:300]}{'...' if len(result) > 300 else ''}\n")
                
                # Add tool result to messages
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                }
                messages.append(tool_message)
                
                # Track movies in context
                try:
                    result_data = json.loads(result)
                    if result_data.get("success") and "movie" in result_data:
                        movie = result_data["movie"]
                        title = movie.get("Title")
                        if title:
                            self.memory.add_movie_to_context(title, movie)
                except:
                    pass
            
            # Continue loop to get synthesized response
        
        # Max iterations reached
        error_msg = "I apologize, but I couldn't complete the task within the iteration limit."
        self.memory.add_message("assistant", error_msg)
        return error_msg
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history (excluding tools and system)."""
        return [
            msg for msg in self.memory.history
            if msg["role"] not in ("tool", "system")
        ]

    
    def get_discussed_movies(self) -> List[str]:
        """Get list of movies discussed in conversation"""
        return self.memory.get_recent_movies_discussed()
    
    def save_conversation(self, filename: str = "conversation.json"):
        """Save conversation to file"""
        self.memory.save_to_file(filename)
    
    def load_conversation(self, filename: str = "conversation.json"):
        """Load conversation from file"""
        self.memory.load_from_file(filename)


