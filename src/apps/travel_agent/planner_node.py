import json
from datetime import date, timedelta
from typing import Dict, Any

from memory import load_memory
from observer import write_jsonl
from schema import HotelSearchInput, FlightSearchInput

today = date.today()
current_year = today.year

def build_planner_prompt(args: Dict[str, Any]) -> str:
    memory = load_memory()

    return f"""
        You are a travel planning assistant.

        LONG-TERM MEMORY (do NOT use for dates, only for preferences and patterns):
        {json.dumps(memory, indent=2)}

        Your job:
        Convert the user's natural-language request into structured tool arguments.

        RULES YOU MUST FOLLOW:

        1. Convert city names to IATA airport codes.
        Examples:
        - Warsaw → WAW
        - Rennes → RNS
        - Strasbourg → SXB

        2. Infer dates from natural language.

        TODAY = "{today.isoformat()}"
        CURRENT_YEAR = "{current_year}"
        NEXT_YEAR = "{current_year + 1}"

        When the user gives a date without a year (e.g., "August 8th"),
        you MUST infer the year as follows:

        a) If the date has not yet occurred this year → use the CURRENT_YEAR.
        b) If the date has already passed this year → use NEXT_YEAR.
        c) NEVER infer dates in the past.
        d) NEVER infer dates earlier than TODAY.
        e) Do NOT use long-term memory to infer dates.

        CRITICAL: You MUST have BOTH a specific departure date AND a return date.
        - If user says "January next year" without a specific day → ASK FOR CLARIFICATION
        - If user says "10 days" without a start date → ASK FOR CLARIFICATION  
        - If user gives duration but no start date → ASK FOR CLARIFICATION
        - If you cannot determine EXACT dates (YYYY-MM-DD) → ASK FOR CLARIFICATION

        EXAMPLES OF AMBIGUOUS REQUESTS THAT REQUIRE CLARIFICATION:
        - "I want to go in January" → ask: "Which specific date in January?"
        - "Next month for a week" → ask: "What is your departure date?"
        - "January next year" → ask: "What specific day do you want to depart?"

        If the user gives a duration (e.g., "10 days"), 
        you MUST compute return_date = departure_date + duration.

        3. Default passengers = 1 unless the user explicitly says otherwise.
        Default guests = passengers.

        4. You MUST output JSON matching EXACTLY these schemas:

        FlightSearchInput:
        {FlightSearchInput.model_json_schema()}

        HotelSearchInput:
        {HotelSearchInput.model_json_schema()}

        5. You MUST also output an itinerary skeleton:
        - days: list of objects with:
        - day_number (1..N)
        - date (YYYY-MM-DD)
        - theme (short string)

        6. If ANY required field (origin, destination, departure_date, return_date, passengers, city)
        cannot be confidently inferred, DO NOT guess.
        Instead return ONLY:
        {{
            "ask_user": "Your clarification question"
        }}

        7. NEVER output fields not in the schemas (no start_date, end_date, budget_usd, etc.).

        8. NEVER output return_date: null unless the user explicitly says one-way.

        9. Output ONLY valid JSON. No text outside JSON.

        EXAMPLE OF AMBIGUOUS REQUEST (ASK USER):
        {{
            "ask_user": "Which specific day in January would you like to depart? Also, how many days will you stay?"
        }}

        EXAMPLE OF VALID OUTPUT:
        {{
        "flight_args": {{
            "origin": "WAW",
            "destination": "RNS",
            "departure_date": "{today.isoformat()}",
            "return_date": "{(today + timedelta(days=10)).isoformat()}",
            "passengers": 1
        }},
        "hotel_args": {{
            "city": "Rennes",
            "check_in": ""{today.isoformat()}",",
            "check_out": "{(today + timedelta(days=10)).isoformat()}",
            "guests": 1,
            "max_price_per_night_usd": null
        }},
        "itinerary_skeleton": {{
            "days": [
            {{"day_number": 1, "date": "{today.isoformat()}", "theme": "Arrival"}},
            {{"day_number": 2, "date": "{(today + timedelta(days=1)).isoformat()}", "theme": "Sightseeing"}}
            ]
        }},
        "reasoning": "short explanation"
        }}

        USER REQUEST:
        {args["user_request"]}
        """


async def planner_tool(client, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    React-Agent compatible planner.
    - Accepts args (dict)
    - Calls LLM (Qwen Coder)
    - Returns dict
    - Logs JSONL
    """

    write_jsonl({"event": "tool.call", "tool": "planner", "args": args})

    prompt = build_planner_prompt(args)

    resp = await client.chat.completions.create(
        model="qwen_qwen3-coder-next",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.choices[0].message.content
    write_jsonl({"event": "planner.raw", "raw": raw})

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "errors": ["planner: invalid JSON"],
            "flight_args": None,
            "hotel_args": None,
            "itinerary_skeleton": None,
            "reasoning": "",
        }
    
    if "ask_user" in parsed:
        return {
            "errors": [],
            "ask_user": parsed["ask_user"],
            "flight_args": None,
            "hotel_args": None,
            "itinerary_skeleton": None,
            "reasoning": "",
        }

    result = {
        "flight_args": parsed.get("flight_args"),
        "hotel_args": parsed.get("hotel_args"),
        "itinerary_skeleton": parsed.get("itinerary_skeleton"),
        "reasoning": parsed.get("reasoning", ""),
        "errors": [],
    }

    write_jsonl({"event": "tool.result", "tool": "planner", "result": result})
    return result
