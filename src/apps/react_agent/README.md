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

## Why Pydantic?
When working with LLMs, one of the biggest challenges is ensuring they return data in a format your code can reliably process. Even though today, in 2026, we have reliable LLM that’s are heavily trained on using tools, but hallucination is still an unsolved problem. That’s why we need type and structure validation.

Pydantic models:
1. Validate incoming data automatically.
2. Provide clear error messages when data is invalid.
3. Enable IDE autocomplete for better developer experience.
4. Generate JSON schemas that modern LLMs can use for structured output.

```python
class ToolCall(BaseModel):
    action: Literal["tool"]
    thought: str
    tool_name: ToolNameLiteral
    args: ToolArgsUnion
class HumanApproval(BaseModel):
    action: Literal["human"]
    reason: str
class FinalAnswer(BaseModel):
    action: Literal["final"]
    answer: str
LLMResponse = Annotated[
    Union[ToolCall, FinalAnswer, HumanApproval],
    Field(discriminator="action"),
]
```
This structure enforces the **ReAct pattern**. The LLM must:
1. Choose an action type: call a tool, request human approval, or provide a final answer
2. If calling a tool: provide a thought process, tool name, and valid arguments
3. If giving a final answer: provide the answer text
4. The ToolNameLiteral type for tool_name ensures the LLM can only call tools that actually exist. The ToolArgsUnion type for args ensures arguments match the expected schema for whichever tool is being called.

This is the essence of the **ReAct pattern**: explicit reasoning (“thought”) followed by explicit actions (“tool_name” + “args”).

In Python typing, a Union, like in LLMResponse means “one of these types.” So if you have two tools, it creates: Union[ToolAddArgs, ToolMultiplyArgs].

Why does this matter? 
When the LLM responds with a tool call, Pydantic will validate that the arguments match one of these schemas. If the LLM tries to pass {"a": "five", "b": 3} (a string instead of an integer), Pydantic will catch it before the tool even executes. This prevents runtime errors and provides clear feedback.

# System Prompt Rules - what the LLM should and shouldn't do:

LLMs are trained to be helpful and will often try to “help” by doing math or reasoning internally. But we want our agent to be observable and reliable. By forcing it to use tools for every operation:
1. We can log and debug each step
2. We can swap tool implementations without changing the agent
3. We can test tools independently
4. We maintain a clear audit trail of actions

# Human-in-the-Loop (HITL) - human approval 
It is a design pattern that creates a checkpoint before critical operations, giving you control over high-stakes decisions.
Good HITL design requires approval for:
1. Irreversible actions: Deleting data, deleting memory, sending emails, making purchases
2. High-cost operations: Running expensive API calls, deploying code
3. Sensitive data access: Reading private files, accessing credentials
4. External communications: Posting to social media, contacting people


## Project structure

```
react_agent/
├── tools.py        # Tool definitions, input schemas, and registry
├── llm_wrapper.py  # Orchestrates the conversation flow
├── logger.py       # Logs for debugging
├── memory.py       # Provides context from past interactions
├── observer.py     # Tracks everything for debugging
├── app.py          # Entry point — single task demo + chat loop
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

# What Could be Next?
- Advanced Memory Systems: Move beyond recency to semantic relevance using embeddings. Implement hierarchical memory (working memory, short-term, long-term). Add memory querying as an explicit tool.
- Sophisticated HITL: Build approval queues for asynchronous review. Implement role-based permissions. Create approval rules engines.
- Production Observability: Integrate with real monitoring systems. Build real-time dashboards. Implement distributed tracing across multiple agents.
- Intelligent Error Handling: Add circuit breakers and fallback strategies. Implement predictive failure detection. Build self-healing capabilities.
- Multi-Agent Systems: Coordinate multiple specialized agents. Implement agent-to-agent communication. Build supervisor agents that manage worker agents.