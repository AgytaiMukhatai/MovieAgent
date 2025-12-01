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
    
    def get_tool_schemas(self):
            tool_schemas = []

            for func in self.tools:
                name = func.__name__
                description = (func.__doc__ or "").strip()

                sig = inspect.signature(func)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    # Default: treat everything as string inputs
                    properties[param_name] = {
                        "type": "string",
                        "description": f"{param_name} parameter"
                    }
                    if param.default is inspect._empty:
                        required.append(param_name)

                schema = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    }
                }

                tool_schemas.append(schema)
            return tool_schemas