import json
from typing import Dict, Any
from openai import AsyncOpenAI

from memory import load_memory
from schema import TripItinerary

def build_summarize_prompt(args: Dict[str, Any]) -> str:
    flights = args.get("flights", [])
    hotels = args.get("hotels", [])
    itinerary_skeleton = args.get("itinerary_skeleton", {})

    memory = load_memory()

    return f"""
        You are a travel itinerary writer.

        Long-term memory:
        {json.dumps(memory, indent=2)}

        Use memory to:
        - personalize itineraries
        - include user preferences
        - avoid repeating mistakes

        You receive context:
        - flights: {json.dumps(flights, indent=2)}
        - hotels: {json.dumps(hotels, indent=2)}
        - itinerary_skeleton: {json.dumps(itinerary_skeleton, indent=2)}

        Your job:
        Produce a COMPLETE TripItinerary object that matches EXACTLY this schema:

        TripItinerary:
        {TripItinerary.model_json_schema()}

        RULES:
        - Output ONLY the TripItinerary JSON.
        - DO NOT wrap it in "result" or "errors".
        - DO NOT include flights/hotels/skeleton in the output.
        - DO NOT include any extra fields.
        - The output MUST validate against TripItinerary.

        Now produce the TripItinerary.
        """


async def summarize_node(client: AsyncOpenAI, args: Dict[str, Any]) -> dict:
    """
    React-Agent compatible summarizer.
    - Accepts dicts, not BookingState
    - Calls LLM (Qwen Coder)
    - Returns raw dict (TripItinerary validated in tool)
    """
    prompt = build_summarize_prompt(args)

    resp = await client.chat.completions.create(
        model="qwen_qwen3-coder-next",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"errors": ["summarize_trip: invalid JSON"], "result": None}
    
    allowed_fields = {"title", "summary", "days", "total_estimated_cost_usd", "packing_tips"}
    cleaned = {k: v for k, v in parsed.items() if k in allowed_fields}
    
    try:
        itinerary = TripItinerary(**cleaned)
        return {"errors": [], "result": itinerary.model_dump()}
    except Exception as e:
        return {"errors": [f"Invalid itinerary: {e}"], "result": None}
