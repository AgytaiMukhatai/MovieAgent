import os
import json
import requests
from typing import List, Dict, Any, Optional        
from ConversationSummaryMemory import ConversationSummaryMemory
from ToolExecutor import ToolExecutor
from tools import search_movie_by_title, search_movie_by_id, analyze_cinematography, get_movie_ratings

AVAILABLE_TOOLS = [
    search_movie_by_title,
    search_movie_by_id,
    analyze_cinematography,
    get_movie_ratings
]

class CinematographyAgent:
    """
    Expert cinematography agent with custom memory and tool calling.
    No LangChain agent classes - everything built from scratch!
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.memory = ConversationSummaryMemory(max_recent_messages=10, summarize_after=20)
        self.tool_executor = ToolExecutor(AVAILABLE_TOOLS)
        
        # System prompt
        system_prompt = """You are an expert cinematography analyst with deep knowledge of film production.

Your expertise includes:
- Camera techniques, movements, and equipment
- Lighting design, setups, and mood creation
- Color grading, palettes, and visual tone
- Composition, framing, and visual storytelling
- Lens choices and their aesthetic effects
- Famous cinematographers and their signature styles
- Technical specifications and production methods

When analyzing films:
1. Use search_movie_by_title to find movies
2. Use analyze_cinematography to get detailed technical analysis
3. Reference specific cinematographers when known
4. Explain technical concepts in accessible language
5. Draw comparisons to other films when relevant
6. Cite specific scenes or sequences as examples

Always provide insightful, technically accurate explanations suitable for film students, 
cinematographers, and enthusiasts."""

        self.memory.add_message("system", system_prompt)
    
    def _call_llm(self, messages: List[Dict], use_tools: bool = True) -> Dict:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
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
        
        # Get messages for API (system + history)
        messages = self.memory.get_messages_for_api()
        
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
        """Get full conversation history"""
        return [{"role": msg.role, "content": msg.content} 
                for msg in self.memory.messages if msg.role != "tool"]
    
    def get_discussed_movies(self) -> List[str]:
        """Get list of movies discussed in conversation"""
        return self.memory.get_recent_movies_discussed()
    
    def save_conversation(self, filename: str = "conversation.json"):
        """Save conversation to file"""
        self.memory.save_to_file(filename)
    
    def load_conversation(self, filename: str = "conversation.json"):
        """Load conversation from file"""
        self.memory.load_from_file(filename)


