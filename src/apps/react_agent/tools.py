import json
import os
import re
import requests
import yfinance as yf
from datetime import datetime
from ddgs import DDGS
from dotenv import load_dotenv
from typing import Annotated, Any, Callable, Dict, List, Literal, Union
from pydantic import BaseModel, Field
from tradingview_ta import TA_Handler, Interval

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
    
    # execute_tool checks valid tool names: 
    # Literal["add", "multiply"]. This is a powerful constraint, 
    # the LLM can only return tool names that actually exist in the registry.
    def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'. Available: {list(self.tools.keys())}"
        try:
            validated_args = tool.input_schema(**args)
            return tool(**validated_args.model_dump())
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def describe_tools(self) -> str:
        return json.dumps(self.list_tools(), indent=2)

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

def search_tickets(
    origin: str,
    destination: str,
    outbound_date: str,
    return_date: str = None,
    max_results: int = 3,
) -> str:
    """
    Search Google Flights for cheapest tickets.
    origin/destination: airport IATA codes e.g. GDN, WAW
    outbound_date / return_date: YYYY-MM-DD format
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return "Error: SERPAPI_KEY not set in .env"

    params = {
        "engine":        "google_flights",
        "departure_id":  origin,
        "arrival_id":    destination,
        "outbound_date": outbound_date,
        "currency":      "PLN",
        "hl":            "en",
        "api_key":       api_key,
        "type":          1 if return_date else 2,  # 1=round trip, 2=one way
    }
    if return_date:
        params["return_date"] = return_date

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        flights = data.get("best_flights") or data.get("other_flights", [])
        if not flights:
            return f"No flights found from {origin} to {destination}."

        results = []
        for item in flights[:max_results]:
            price = item.get("price", "N/A")
            for flight in item.get("flights", []):
                dep_airport = flight["departure_airport"]["name"]
                arr_airport = flight["arrival_airport"]["name"]
                departure   = flight["departure_airport"]["time"]
                arrival     = flight["arrival_airport"]["time"]
                airline     = flight.get("airline", "Unknown")
                duration    = flight.get("duration", 0)
                results.append(
                    f"💰 {price} PLN | {dep_airport} → {arr_airport}\n"
                    f"   {airline} | Dep: {departure} | Arr: {arrival}\n"
                    f"   Duration: {duration} min"
                )

        return "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

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
    
def get_newconnect_price(ticker: str) -> str:
    """
    Get NewConnect stock price via TradingView.
    Pass just the ticker symbol e.g. 'APS', 'ROCKGAME'
    """
    try:
        handler = TA_Handler(
            symbol=ticker,
            exchange="NEWCONNECT",
            screener="poland",
            interval=Interval.INTERVAL_1_DAY,
        )
        analysis = handler.get_analysis()
        indicators = analysis.indicators

        price  = indicators.get("close", "N/A")
        change = indicators.get("change", "N/A")
        volume = indicators.get("volume", "N/A")

        return (
            f"{ticker} (NewConnect)\n"
            f"  Price:  {price} PLN\n"
            f"  Change: {change:.2f}%\n"
            f"  Volume: {volume:,.0f}"
        )
    except Exception as e:
        return f"Error fetching '{ticker}' from NewConnect: {e}"
    
def get_stock_price(ticker: str) -> str:
    """
    Get current stock price and key info for a ticker symbol.
    Use suffix for non-US exchanges:
      .WA = Warsaw (GPW), .DE = Frankfurt, .L = London, .PA = Paris
    Examples: CDR.WA, PKN.WA, SAP.DE, AAPL (no suffix for US)
    """
    # One thing to know: Yahoo Finance data has a ~15 minute delay for most exchanges including GPW. 
    # Each exchange on Yahoo Finance has a specified time delay between the live market and what the API returns. 
    # For real-time trading you'd need a paid data provider like Polygon.io or a brokerage API.
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # info can return an empty dict for unknown tickers
        if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
            return f"Ticker '{ticker}' not found or no data available."

        name     = info.get("longName") or info.get("shortName", ticker)
        price    = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        currency = info.get("currency", "N/A")
        change   = info.get("regularMarketChangePercent", 0)
        exchange = info.get("exchange", "N/A")
        high_52w = info.get("fiftyTwoWeekHigh", "N/A")
        low_52w  = info.get("fiftyTwoWeekLow", "N/A")
        mkt_cap  = info.get("marketCap")

        cap_str = f"{mkt_cap:,}" if mkt_cap else "N/A"
        direction = "▲" if change >= 0 else "▼"

        return (
            f"{name} ({ticker})\n"
            f"  Price:      {price} {currency}  {direction} {change:.2f}%\n"
            f"  Exchange:   {exchange}\n"
            f"  52w Range:  {low_52w} – {high_52w} {currency}\n"
            f"  Market Cap: {cap_str} {currency}"
        )

    except Exception as e:
        return f"Error fetching '{ticker}': {e}"

def get_multiple_stock_prices(tickers: str) -> str:
    """
    Get prices for multiple tickers at once.
    Pass comma-separated tickers e.g. 'CDR.WA,PKN.WA,AAPL'
    """
    ticker_list = [t.strip() for t in tickers.split(",")]
    results = []
    for ticker in ticker_list:
        results.append(get_stock_price(ticker))
    return "\n\n".join(results)

def get_stock_history(ticker: str, days: int = 7) -> str:
    """
    Get historical daily prices for a stock over the last N days.
    Use .WA suffix for GPW/NewConnect stocks e.g. CDR.WA
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{days}d")

        if hist.empty:
            return f"No historical data found for '{ticker}'. Check the ticker symbol."

        # Build markdown table directly — LLM will pass it through as-is
        lines = [
            f"**{ticker} — last {days} trading days**\n",
            "| Date | Open | Close | High | Low | Volume | Change |",
            "|------|------|-------|------|-----|--------|--------|",
        ]

        for date, row in hist.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            change = ((row["Close"] - row["Open"]) / row["Open"]) * 100
            direction = "▲" if change >= 0 else "▼"
            lines.append(
                f"| {date_str} "
                f"| {row['Open']:.2f} "
                f"| {row['Close']:.2f} "
                f"| {row['High']:.2f} "
                f"| {row['Low']:.2f} "
                f"| {int(row['Volume']):,} "
                f"| {direction}{abs(change):.2f}% |"
            )

        # Summary
        first_close = hist["Close"].iloc[0]
        last_close  = hist["Close"].iloc[-1]
        total_change = ((last_close - first_close) / first_close) * 100
        direction = "▲" if total_change >= 0 else "▼"
        lines.append(f"  Period change: {direction}{abs(total_change):.2f}%  ({first_close:.2f} → {last_close:.2f})")

        return "\n".join(lines)

    except Exception as e:
        return f"Error fetching history for '{ticker}': {e}"

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

class SearchTicketsArgs(BaseModel):
    origin: str = Field(description="IATA code of departure airport, e.g. GDN for Gdansk")
    destination: str = Field(description="IATA code of destination airport, e.g. WAW for Warsaw")
    outbound_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: str = Field(default=None, description="Return date in YYYY-MM-DD format, omit for one-way")
    max_results: int = Field(default=3, description="Number of results to return")

class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query to look up")
    max_results: int = Field(default=3, description="Number of results to return (1-5)")

class StockPriceArgs(BaseModel):
    ticker: str = Field(
        description=(
            "Stock ticker symbol with exchange suffix for non-US stocks. "
            "Examples: CDR.WA (CD Projekt, Warsaw), PKN.WA (PKN Orlen), "
            "SAP.DE (SAP Frankfurt), AAPL (Apple, US - no suffix needed)"
        )
    )

class MultipleStockPriceArgs(BaseModel):
    tickers: str = Field(
        description="Comma-separated ticker symbols e.g. 'CDR.WA,PKN.WA,AAPL'"
    )

class StockHistoryArgs(BaseModel):
    ticker: str = Field(
        description=(
            "Stock ticker with exchange suffix for non-US stocks. "
            "Examples: CRQ.WA (Creotech Quantum), CRI.WA (Creotech Instruments), CDR.WA (CD Projekt) AAPL (Apple)"
        )
    )
    days: int = Field(
        default=7,
        description="Number of past days to retrieve (e.g. 7, 14, 30)"
    )

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
    elif isinstance(data, str):
        cleaned = re.sub(r"^```(?:json)?\s*", "", data.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: treat entire string as final answer
            return FinalAnswer(answer=data.strip())
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
    name="search_tickets",
    description=(
        "Search for the cheapest tickets between two cities. "
        "Supports flights, trains, and buses. "
        "Requires IATA city/airport codes (e.g. GDN, WAW, KRK, WRO). "
        "Returns prices, times, duration, and booking links."
    ),
    input_schema=SearchTicketsArgs,
    func=search_tickets,
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

registry.register(Tool(
        name="get_newconnect_price",
        description=(
            "Get NewConnect stock price via TradingView. "
            "Pass just the ticker symbol e.g. 'APS', 'ROCKGAME'"
        ),
        input_schema=StockPriceArgs,
        func=get_newconnect_price,
    ))

registry.register(Tool(
    name="get_stock_price",
    description=(
        "Get the current stock price, daily change, 52-week range, and market cap "
        "for a single stock. Use .WA suffix for Warsaw/GPW stocks, .DE for Frankfurt, "
        ".L for London, .PA for Paris. No suffix needed for US stocks."
    ),
    input_schema=StockPriceArgs,
    func=get_stock_price,
))

registry.register(Tool(
    name="get_multiple_stock_prices",
    description=(
        "Get current stock prices for multiple tickers at once. "
        "More efficient than calling get_stock_price repeatedly."
    ),
    input_schema=MultipleStockPriceArgs,
    func=get_multiple_stock_prices,
))

registry.register(Tool(
    name="get_stock_history",
    description=(
        "Get historical daily OHLCV prices for a stock over the last N days. "
        "Returns open, close, high, low, volume, and daily change for each day, "
        "plus the total period change. Use .WA suffix for Polish stocks."
    ),
    input_schema=StockHistoryArgs,
    func=get_stock_history,
))
