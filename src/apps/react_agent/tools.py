import json
import os
import requests
from datetime import datetime
from ddgs import DDGS
from dotenv import load_dotenv
from typing import Annotated, Any, Callable, Dict, List, Literal, Union
from pydantic import BaseModel, Field

# Why Pydantic?
# Pydantic models act as contracts. They:
# 1. Validate incoming data automatically.
# 2. Provide clear error messages when data is invalid.
# 3. Enable IDE autocomplete for better developer experience.
# 4. Generate JSON schemas that modern LLMs can use for structured output.

load_dotenv()

# =========================================================
# TOOL SYSTEM
# =========================================================

class Tool:
    """
    A single callable tool with a name, description, input schema, and function.
    The LLM uses the description and input_schema to know when and how to call it.
    """
    def __init__(
        self,
        name: str,  # A unique identifier
        description: str,   # What the tool does (crucial for the LLM to understand when to use it)
        input_schema: type[BaseModel],   # Defines what parameters the tool expects
        # output_schema: Dict[str, Any],  # What the tool returns
        func: Callable[..., Any], # The actual function that does the work
        ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        # self.output_schema = output_schema
        self.func = func
    
        # The __call__ method makes our Tool instances callable, 
        # so we can use them like regular functions: tool(a=5, b=3)
    def __call__(self, **kwargs):
        return self.func(**kwargs)

    def to_prompt_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }


class ToolRegistry: # use it mainly to register and retrieve tools
    """
    Holds and manages all registered tools.
    Provides lookup, listing, and validated execution.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self.tools.keys())}")
        return self.tools[name]
    # List tools method is particularly important because it generates 
    # a machine-readable description of all available tools.
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
            }
            for tool in self.tools.values()
        ]

    # get_tool_call_args_type creates a Union type of all possible tool argument schemas. 
    # In Python typing, a Union means “one of these types.” 
    # So if you have two tools, it creates: Union[ToolAddArgs, ToolMultiplyArgs].
    # Why does this matter? When the LLM responds with a tool call, 
    # Pydantic will validate that the arguments match one of these schemas. 
    # If the LLM tries to pass {"a": "five", "b": 3} (a string instead of an integer),
    #  Pydantic will catch it before the tool even executes. 
    # This prevents runtime errors and provides clear feedback.
    # def get_tool_call_args_type(self):
    #     input_args_models = [tool.input_schema for tool in self.tools.values()]
    #     return Union[tuple(input_args_models)]
    # get_tool_names generates a Literal type containing only valid tool names: 
    # Literal["add", "multiply"]. This is a powerful constraint, 
    # the LLM can only return tool names that actually exist in the registry.
    # def get_tool_names(self):
    #     return Literal[*self.tools.keys()]
    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self.get(name)
        validated_args = tool.input_schema(**args)
        return tool(**validated_args.model_dump())

    def describe_tools(self) -> str:
        return json.dumps(
            [
            tool.to_prompt_schema()
            for tool in self.tools.values()
            ],
            indent=2,
        )


# =========================================================
# TOOL FUNCTIONS
# =========================================================

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """
    A couple of alternatives worth knowing about if OpenWeatherMap doesn't suit you:

    wttr.in — no API key at all, just requests.get("https://wttr.in/Gdansk?format=3"). 
    Great for quick prototyping.
    Open-Meteo — fully free, no key, more detailed forecasts 
    but requires lat/lon instead of city name (you'd need a geocoding step).
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not set in .env"

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # use "imperial" for °F
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        name = data["name"]
        country = data["sys"]["country"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]

        return (
            f"Weather in {name}, {country}: {description}, "
            f"{temp}°C (feels like {feels_like}°C), "
            f"humidity {humidity}%, wind {wind} m/s"
        )
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return f"City '{city}' not found."
        return f"HTTP error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {e}"

# For a more stable alternative, SerpAPI has a free tier 
# (100 searches/month) with an official API. 
# Same pattern, just needs an API key in .env.
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return f"No results found for: {query}"
        
        formatted = []
        for i, r in enumerate(results, 2):
            formatted.append(
                f"[{i}] {r['title']}\n"
                f"    URL: {r['href']}\n"
                f"    {r['body']}"
            )

        return "\n\n".join(formatted)
    
    except Exception as e:
        return f"Search failed: {e}"

# =========================================================
# INPUT SCHEMAS
# =========================================================

class EmptyArgs(BaseModel):
    pass

class MathArgs(BaseModel):
    a: int
    b: int

class CalculateArgs(BaseModel):
    expression: str = Field(description="A valid Python math expression, e.g. '15 * 0.15'")

class WeatherArgs(BaseModel):
    city: str

class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query to look up")
    max_results: int = Field(default=3, description="Number of results to return (1-5)")

# =========================================================
# LLM RESPONSE SCHEMAS (ReAct pattern)
# =========================================================

# Get type-safe tool names and arguments
# ToolNameLiteral = registry.get_tool_names()
# ToolArgsUnion = registry.get_tool_call_args_type()

class ToolCall(BaseModel):
    action: Literal["tool"]
    thought: str
    tool_name: str
    args: Dict[str, Any]

class FinalAnswer(BaseModel):
    action: Literal["final"]
    answer: str

LLMResponse = Annotated[
    Union[ToolCall, FinalAnswer],
    Field(discriminator="action"),
]

# This structure enforces the ReAct pattern. The LLM must:
# 1. Choose an action type (“tool” or “final”)
# 2. If calling a tool: provide a thought process, tool name, and valid arguments
# 3, If giving a final answer: provide the answer text
# 4. The ToolNameLiteral ensures the LLM can only call tools that actually exist. 
# The ToolArgsUnion ensures arguments match the expected schema 
# for whichever tool is being called.

def _parse_single(data: dict):
    action = data.get("action")
    if action == "tool":
        return ToolCall.model_validate(data)
    elif action == "final":
        return FinalAnswer.model_validate(data)
    else:
        raise ValueError(f"Unknown action: '{action}'. Must be 'tool' or 'final'.")


def parse_llm_response(data):
    # Local models sometimes wrap the response in a list
    # Unwrap list if model returned multiple calls
    if isinstance(data, list):
        if len(data) == 1:
            data = data[0]
        else:
            return [_parse_single(item) for item in data]
    return _parse_single(data)


# =========================================================
# REGISTER TOOLS
# =========================================================

registry = ToolRegistry()

registry.register(Tool(
    name="add",
    description="Add two integers together",
    input_schema=MathArgs,
    func=add,
))

registry.register(Tool(
    name="multiply",
    description="Multiply two integers together",
    input_schema=MathArgs,
    func=multiply,
))

registry.register(Tool(
    name="get_current_date",
    description="Returns the current date and time",
    input_schema=EmptyArgs,
    func=get_current_date,
))

registry.register(Tool(
    name="calculate",
    description="Evaluates a math expression string and returns the numeric result. Use this for percentages, division, or any expression that isn't simple add/multiply.",
    input_schema=CalculateArgs,
    func=calculate,
))

registry.register(Tool(
    name="get_weather",
    description="Gets the current weather for a given city",
    input_schema=WeatherArgs,
    func=get_weather,
))

registry.register(Tool(
        name="web_search",
        description=(
            "Search the web for current information, news, facts, or anything "
            "you don't know. Returns titles, URLs, and snippets of top results."
        ),
        input_schema=WebSearchArgs,
        func=web_search,
    ))
