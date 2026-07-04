import os
from typing import Callable, Dict, Any

import httpx

from observer import write_jsonl
from schema import FlightSearchInput, HotelSearchInput, TripItinerary
from summarize_node import summarize_node
from tool_nodes import search_flights_node, search_hotels_node

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]
TOOLS: Dict[str, ToolFn] = {}


def register_tool(name: str):
    def decorator(fn: ToolFn):
        TOOLS[name] = fn
        return fn
    return decorator


@register_tool("planner")
def tool_planner(args: Dict[str, Any]) -> Dict[str, Any]:
    # The router will call this via planner_tool(client, args)
    raise RuntimeError("planner_tool must be called with client")


@register_tool("search_flights")
def tool_search_flights(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    React-Agent tool wrapper for your existing search_flights_node.
    - Validates args using FlightSearchInput
    - Calls your existing node logic
    - Returns JSON-safe dict
    """
    write_jsonl({"event": "tool.call", "tool": "search_flights", "args": args})

    # 1. Validate arguments using Pydantic
    try:
        validated = FlightSearchInput(**args)
    except Exception as e:
        error = str(e)
        write_jsonl({
            "event": "tool.validation_error",
            "tool": "search_flights",
            "error": error
        })
        return {"errors": [error], "result": None}

    # 2. Call your existing node logic
    #    IMPORTANT: your node expects a dict, not a Pydantic model
    result = search_flights_node(validated.model_dump())

    write_jsonl({"event": "tool.result", "tool": "search_flights", "result": result})
    return result


@register_tool("search_hotels")
def tool_search_hotels(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    React-Agent tool wrapper for your existing search_hotels_node.
    - Validates args using HotelSearchInput
    - Calls your existing node logic
    - Returns JSON-safe dict
    """
    write_jsonl({"event": "tool.call", "tool": "search_hotels", "args": args})

    # 1. Validate arguments using Pydantic
    try:
        validated = HotelSearchInput(**args)
    except Exception as e:
        error = str(e)
        write_jsonl({
            "event": "tool.validation_error",
            "tool": "search_hotels",
            "error": error
        })
        return {"errors": [error], "result": None}

    # 2. Call your existing node logic
    #    IMPORTANT: your node expects a dict, not a Pydantic model
    result = search_hotels_node(validated.model_dump())

    write_jsonl({"event": "tool.result", "tool": "search_hotels", "result": result})
    return result


@register_tool("get_weather_forecast")
async def tool_get_weather_forecast(args: Dict[str, Any]) -> Dict[str, Any]:
    write_jsonl({"event": "tool.call", "tool": "get_weather_forecast", "args": args})

    city = args.get("city")
    check_date = args.get("check_date")

    api_key = os.getenv("WEATHERAPI_API_KEY")
    if not api_key:
        return {"errors": ["Missing WEATHERAPI_API_KEY"], "result": None}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.weatherapi.com/v1/forecast.json",
                params={"q": city, "dt": check_date, "key": api_key},
            )

        if r.status_code != 200:
            return {"errors": [f"Weather API returned {r.status_code}"], "result": None}

        data = r.json()
        day = data["forecast"]["forecastday"][0]["day"]

        result = {
            "city": city,
            "date": check_date,
            "condition": day["condition"]["text"],
            "high_c": day["maxtemp_c"],
            "low_c": day["mintemp_c"],
            }

        write_jsonl({"event": "tool.result", "tool": "get_weather_forecast", "result": result})

        return {
            "errors": [],
            "result": result
        }

    except httpx.RequestError as e:
        return {"errors": [f"Weather API unreachable: {e}"], "result": None}


@register_tool("search_attractions")
async def tool_search_attractions(args: Dict[str, Any]) -> Dict[str, Any]:
    write_jsonl({"event": "tool.call", "tool": "search_attractions", "args": args})

    query = args.get("query")
    category = args.get("category")
    destination = args.get("destination")

    base_url = os.getenv("ATTRACTIONS_API_URL")

    if not base_url:
        result = {
            "mock": True,
            "destination": destination,
            "category": category,
            "results": [
                "Ashmolean Museum",
                "Christ Church College",
                "Radcliffe Camera",
                "Bodleian Library",
            ],
        },

        write_jsonl({"event": "tool.result", "tool": "search_attractions", "result": result})
        
        return {
            "errors": [],
            "result": result,
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{base_url.rstrip('/')}/attractions",
                params={"q": query, "near": destination, "category": category},
            )

        if r.status_code != 200:
            return {"errors": [f"Attractions API returned {r.status_code}"], "result": None}
        
        write_jsonl({"event": "tool.result", "tool": "search_attractions", "result": r.json()})

        return {"errors": [], "result": r.json()}

    except httpx.RequestError as e:
        return {"errors": [f"Attractions API unreachable: {e}"], "result": None}


@register_tool("check_budget_remaining")
def tool_check_budget_remaining(args: Dict[str, Any]) -> Dict[str, Any]:
    write_jsonl({"event": "tool.call", "tool": "check_budget_remaining", "args": args})

    total = args.get("total_budget_usd")
    spent = args.get("spent_so_far_usd")

    if total is None or spent is None:
        return {
            "errors": ["Missing required fields: total_budget_usd, spent_so_far_usd"],
            "result": None,
        }

    remaining = total - spent

    if remaining < 0:
        return {"errors": [f"Over budget by ${abs(remaining):.0f}"], "result": None}
    
    result = {"remaining_usd": remaining}

    write_jsonl({"event": "tool.result", "tool": "search_attractions", "result": result})

    return {"errors": [], "result": result}


@register_tool("validate_itinerary")
def tool_validate_itinerary(args: Dict[str, Any]) -> Dict[str, Any]:
    write_jsonl({"event": "tool.call", "tool": "validate_itinerary", "args": args})
    
    itinerary_data = args.get("itinerary", args)
    
    if not isinstance(itinerary_data, dict):
        return {"errors": ["Invalid itinerary: expected dict"], "result": None}
    
    allowed_fields = {"title", "summary", "days", "total_estimated_cost_usd", "packing_tips"}
    cleaned = {k: v for k, v in itinerary_data.items() if k in allowed_fields}
    
    try:
        validated = TripItinerary(**cleaned)
        write_jsonl({"event": "tool.result", "tool": "validate_itinerary", "result": validated.model_dump()})
        return {"errors": [], "result": validated.model_dump()}
    except Exception as e:
        return {"errors": [f"Invalid itinerary: {e}"], "result": None}


@register_tool("summarize_trip")
async def tool_summarize_trip(client, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    React-Agent tool wrapper for summarize_node.
    - async (because summarize_node uses LLM)
    - receives client from react_agent loop
    """

    write_jsonl({"event": "tool.call", "tool": "summarize_trip", "args": args})

    flights = args.get("flights", [])
    hotels = args.get("hotels", [])
    itinerary_skeleton = args.get("itinerary_skeleton", {})

    # 1. Call the LLM summarizer
    raw = await summarize_node(client, {
        "flights": flights,
        "hotels": hotels,
        "itinerary_skeleton": itinerary_skeleton
    })

    write_jsonl({"event": "tool.raw_output", "tool": "summarize_trip", "raw": raw})

    # 2. Validate output using TripItinerary
    # raw is {"errors": [], "result": {...}}, so use raw["result"]
    try:
        itinerary = TripItinerary(**raw["result"])
    except Exception as e:
        error = str(e)
        write_jsonl({
            "event": "tool.validation_error",
            "tool": "summarize_trip",
            "error": error
        })
        return {"errors": [error], "result": None}

    result = {"errors": [], "result": itinerary.model_dump()}
    write_jsonl({"event": "tool.result", "tool": "summarize_trip", "result": result})
    return result
