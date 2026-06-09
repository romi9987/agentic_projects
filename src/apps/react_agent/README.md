# ReAct Agent with local model

A minimal, clean implementation of a **ReAct (Reasoning + Acting)** agent using a local model.

## How it works

ReAct doesn't just generate text. It runs a closed loop:

```
User task
   ↓
LLM thinks → calls a predefined tool with structured input (Action)
   ↓
Tool returns observation -> data (JSON/string)
   ↓
LLM thinks → calls another tool  (until the LLM decides it has enough info)
   ↓
LLM has everything → returns final answer
```

Every LLM response is strict JSON — either a tool call or a final answer.
Pydantic validates each response before anything is executed.

## Project structure

```
react_agent/
├── tools.py        # Tool definitions, input schemas, and registry
├── llm_wrapper.py  # ReactAgent class (the ReAct loop)
├── app.py         # Entry point — single task demo + chat loop
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python app.py
```

## Adding a new tool

1. Write the function in `tools.py`
2. Create a Pydantic input schema for it
3. Register it with the registry

```python
def my_tool(param: str) -> str:
    return f"result for {param}"

class MyToolArgs(BaseModel):
    param: str

registry.register(Tool(
    name="my_tool",
    description="Does something useful",
    input_schema=MyToolArgs,
    func=my_tool,
))
```

That's it — the agent will automatically know about it on next run.

## Switching models

Change `model=` in `main.py` to any model you have:

```python
agent = ReactAgent(client=client, registry=registry, model="<your_model>")
```

Or swap to OpenAI by replacing the client:

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")
agent = ReactAgent(client=client, registry=registry, model="gpt-4o-mini")
```

## Libraries like Langchain

LangChain's @tool decorator handles validation and formatting.

```python
from langchain_core.tools import tool

@tool
def search_flights(destination: str, date: str) -> str:
    """Search for flights to a destination on a specific date."""
    # Replace with real API (Amadeus, Skyscanner, etc.)
    return f"[FLIGHTS] Found 3 options to {destination} on {date}. Cheapest: $420 (Economy)."

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"[WEATHER] {city}: 22°C, Sunny. Pack light layers."
```