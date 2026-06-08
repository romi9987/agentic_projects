# ReAct Agent with Ollama

A minimal, clean implementation of a **ReAct (Reasoning + Acting)** agent using a local Ollama model.

## How it works

The agent follows a loop:

```
User task
   ↓
LLM thinks → calls a tool
   ↓
Tool returns observation
   ↓
LLM thinks → calls another tool  (repeat as needed)
   ↓
LLM has everything → returns final answer
```

Every LLM response is strict JSON — either a tool call or a final answer.
Pydantic validates each response before anything is executed.

## Project structure

```
react_agent/
├── tools.py        # Tool definitions, input schemas, and registry
├── agent.py        # ReactAgent class (the ReAct loop)
├── main.py         # Entry point — single task demo + chat loop
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve

# 3. Pull the model
ollama pull qwen2.5

# 4. Run
python main.py
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

Change `model=` in `main.py` to any model you have pulled in Ollama:

```python
agent = ReactAgent(client=client, registry=registry, model="llama3.2")
```

Or swap to OpenAI by replacing the client:

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")
agent = ReactAgent(client=client, registry=registry, model="gpt-4o-mini")
```
