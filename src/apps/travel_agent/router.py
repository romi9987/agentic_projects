import json
from typing import Dict, Any

from memory import load_memory
from observer import write_jsonl
from agent_state import AgentState
from tools import TOOLS


async def choose_tool_and_args(client, state: AgentState) -> Dict[str, Any]:
    # Exclude planner - it's only called once at the start
    available_tools = {name: func for name, func in TOOLS.items() if name != "planner"}
    
    tools_schema = {
        "tools": [
            {
                "name": name,
                "description": f"Tool: {name}",
                "args_example": {},
            }
            for name in sorted(available_tools.keys())
        ]
    }

    memory = load_memory()
    
    prompt = f"""
        You are a travel React agent.

        Long-term memory:
        {json.dumps(memory, indent=2)}

        User request: {state.user_request}

        Previous observations (JSON):
        {json.dumps(state.observations, default=str)}

        Available tools (JSON):
        {json.dumps(tools_schema)}

        Decide which tool to call next.

        RULES:
        - If flights are missing → call search_flights.
        - If hotels are missing → call search_hotels.
        - If itinerary_skeleton exists and flights+hotels exist → call summarize_trip.
        - If budget context is needed → call check_budget_remaining.
        - If user mentions weather → call get_weather_forecast.
        - If user mentions sightseeing/attractions → call search_attractions.
        - If everything is complete → set done=true and tool="none".

        Return ONLY valid JSON. NO EXPLANATIONS, NO TEXT BEFORE OR AFTER THE JSON.
        The entire response must be a single valid JSON object with the exact format:
        {{
        "thought": "short reasoning about what to do next",
        "tool": "<tool_name>",
        "args": {{ ... }},
        "done": false | true
        }}
        
        Valid tools: {', '.join(sorted(available_tools.keys()))}
        """

    resp = await client.chat.completions.create(
        model="qwen_qwen3-coder-next",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                state.add_error("router: invalid JSON")
                write_jsonl({"event": "router.invalid_json", "raw": raw})
                return {
                    "tool": "none",
                    "args": {},
                    "done": True,
                    "thought": "Failed to parse router output.",
                }
        else:
            state.add_error("router: invalid JSON")
            write_jsonl({"event": "router.invalid_json", "raw": raw})
            return {
                "tool": "none",
                "args": {},
                "done": True,
                "thought": "Failed to parse router output.",
            }
    
    if "ask_user" in parsed:
        return {"ask_user": parsed["ask_user"]}

    state.add_thought(parsed.get("thought", ""))
    write_jsonl({"event": "router.choice", "choice": parsed})
    return parsed
