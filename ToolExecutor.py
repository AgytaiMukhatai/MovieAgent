import json
from typing import List, Dict, Any, Optional    
import inspect

class ToolExecutor:
    """Handles parsing and execution of tool calls"""
    
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
    
    def parse_tool_calls(self, response: Dict) -> tuple[List[Dict], Optional[str]]:
        """
        Parse tool calls from LLM response
        
        Returns:
            (tool_calls, content) tuple
        """
        message = response.get("choices", [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls", [])
        content = message.get("content")
        
        parsed_calls = []
        for tool_call in tool_calls:
            try:
                parsed_calls.append({
                    "id": tool_call.get("id"),
                    "name": tool_call["function"]["name"],
                    "arguments": json.loads(tool_call["function"]["arguments"])
                })
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing tool call arguments: {e}")
                continue
        
        return parsed_calls, content
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool and return results"""
        if tool_name not in self.tools:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})
        
        try:
            tool = self.tools[tool_name]
            result = tool.invoke(arguments)
            return result
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
    
    def get_tool_schemas(self) -> List[Dict]:
        """Convert LangChain tools to OpenAI function calling format"""
        schemas = []
        
        # Manual schema definition to ensure OpenAI compatibility
        tool_schemas = {
            "search_movie_by_title": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The movie title to search for (e.g., 'Guardians of the Galaxy')"
                    }
                },
                "required": ["title"]
            },
            "search_movie_by_id": {
                "type": "object",
                "properties": {
                    "imdb_id": {
                        "type": "string",
                        "description": "The IMDb ID (e.g., 'tt3896198')"
                    }
                },
                "required": ["imdb_id"]
            },
            "analyze_cinematography": {
                "type": "object",
                "properties": {
                    "movie_data": {
                        "type": "string",
                        "description": "JSON string of movie data from search results"
                    }
                },
                "required": ["movie_data"]
            },
            "get_movie_ratings": {
                "type": "object",
                "properties": {
                    "movie_data": {
                        "type": "string",
                        "description": "JSON string of movie data"
                    }
                },
                "required": ["movie_data"]
            }
        }
        
        for tool in self.tools.values():
            tool_name = tool.name
            tool_desc = tool.description
            
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": tool_schemas.get(tool_name, {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            schemas.append(schema)
        
        return schemas
         