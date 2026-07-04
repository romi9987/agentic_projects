import asyncio
from datetime import datetime
from rich.console import Console

from agent_state import AgentState
from memory import write_memory
from observer import write_jsonl
from planner_node import planner_tool
from router import choose_tool_and_args
from shared_utils import repair_tool_args
from tools import TOOLS

console = None

def get_console():
    global console
    if console is None:
        console = Console()
    return console

async def run_react_agent(client, user_request: str) -> AgentState:
    state = AgentState(user_request=user_request)

    # First: planner
    planner_result = await planner_tool(client, {"user_request": state.user_request})
    write_jsonl({"event": "tool.result", "tool": "planner", "result": planner_result})

    if "ask_user" in planner_result and planner_result["ask_user"]:
        print(planner_result["ask_user"])
        user_answer = input("> ")
        state.user_request += " " + user_answer
        # re-run planner once
        planner_result = await planner_tool(client, {"user_request": state.user_request})

    state.flight_args = planner_result.get("flight_args") or {}
    state.hotel_args = planner_result.get("hotel_args") or {}
    state.itinerary_skeleton = planner_result.get("itinerary_skeleton") or {}

    state.add_observation({"tool": "planner", "result": planner_result})

    for step in range(25):  # max steps
        console = get_console()
        console.print(f"\n[bold yellow]Step {step}: Calling router...[/bold yellow]")
        
        choice = await choose_tool_and_args(client, state)

        if choice.get("done"):
            state.done = True
            write_jsonl({"event": "agent.done", "step": step})
            break
        
        tool_name = choice.get("tool")
        args = choice.get("args", {})

        if "ask_user" in choice:
            print(choice["ask_user"])
            user_answer = input("> ")
            state.user_request += " " + user_answer
            continue

        # -----------------------------
        # TOOL EXECUTION
        # -----------------------------
        
        if tool_name == "none":
            console.print("[bold yellow]No tool selected, agent done.[/bold yellow]")
            break
        
        # Exclude planner - it's only called once at the start
        valid_tools = {name for name in TOOLS.keys() if name != "planner"}
        
        if tool_name not in valid_tools:
            state.add_error(f"Invalid tool selected: {tool_name}. Valid tools are: {', '.join(sorted(valid_tools))}")
            console.print(f"[bold red]Invalid tool: {tool_name}[/bold red]")
            write_jsonl({"event": "agent.invalid_tool", "tool": tool_name})
            break
        
        console.print(f"[bold cyan]Executing tool: {tool_name}[/bold cyan]")
        
        # Track arguments
        if tool_name == "search_flights":
            state.flight_args = args
        elif tool_name == "search_hotels":
            state.hotel_args = args
        
        tool_fn = TOOLS[tool_name]
        
        # Special case: some tools need the client, others don't
        # Also check if it's async and needs awaiting
        import inspect
        sig = inspect.signature(tool_fn)
        if 'client' in sig.parameters:
            result = await tool_fn(client, args)
        elif asyncio.iscoroutinefunction(tool_fn):
            result = await tool_fn(args)
        else:
            result = tool_fn(args)

        # Record observation
        state.add_observation({"tool": tool_name, "result": result})
        write_jsonl({"event": "agent.observation", "tool": tool_name, "result": result})

        console.print(f"[bold green]✓ Tool completed: {tool_name}[/bold green]")

        # Store last flights/hotels
        if tool_name == "search_flights" and not result.get("errors"):
            state.last_flights = result["result"]["flights"]
        if tool_name == "search_hotels" and not result.get("errors"):
            state.last_hotels = result["result"]["hotels"]

        # Argument repair if errors
        errors = result.get("errors") or []
        if errors and tool_name in ("search_flights", "search_hotels"):
            console.print(f"[bold yellow]Repairing {tool_name} args due to errors...[/bold yellow]")
            # let LLM repair args
            # (you can reuse your existing repair_tool_args)
            fixed_args = await repair_tool_args(client, tool_name, args, errors)

            if tool_name == "search_flights":
                result2 = TOOLS["search_flights"](fixed_args)
            elif tool_name == "search_hotels":
                result2 = TOOLS["search_hotels"](fixed_args)
            else:
                result2 = result  # no repair

            state.add_observation(
                {"tool": tool_name, "result": result2, "repaired": True}
            )
            write_jsonl(
                {
                    "event": "agent.observation_repaired",
                    "tool": tool_name,
                    "result": result2,
                }
            )

    write_memory({
        "user_request": state.user_request,
        "flight_args": state.flight_args,
        "hotel_args": state.hotel_args,
        "itinerary_skeleton": state.itinerary_skeleton,
        "timestamp": datetime.utcnow().isoformat()
    })

    return state
