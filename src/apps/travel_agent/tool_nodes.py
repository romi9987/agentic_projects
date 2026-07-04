from schema import FlightSearchInput, HotelSearchInput

def search_flights_node(args: dict) -> dict:
    """
    Pure React-Agent node for searching flights.
    - Accepts dict args
    - Validates with FlightSearchInput
    - Returns dict with result + errors
    """

    # 1. Validate input
    try:
        validated = FlightSearchInput(**args)
    except Exception as e:
        return {
            "errors": [f"Invalid flight search args: {e}"],
            "result": None,
        }

    # 2. Call your real flight search logic here
    #    Replace this with your actual API call or DB lookup
    #    For now, we return a placeholder structure
    result = {
        "flights": [
            {
                "airline": "ExampleAir",
                "origin": validated.origin,
                "destination": validated.destination,
                "departure_date": validated.departure_date.isoformat(),
                "return_date": (
                    validated.return_date.isoformat()
                    if validated.return_date
                    else None
                ),
                "price_usd": 750,
            }
        ]
    }

    return {"errors": [], "result": result}


def search_hotels_node(args: dict) -> dict:
    """
    Pure React-Agent node for searching hotels.
    - Accepts dict args
    - Validates with HotelSearchInput
    - Returns dict with result + errors
    """

    # 1. Validate input
    try:
        validated = HotelSearchInput(**args)
    except Exception as e:
        return {
            "errors": [f"Invalid hotel search args: {e}"],
            "result": None,
        }

    # 2. Call your real hotel search logic here
    #    Replace this with your actual API call or DB lookup
    result = {
        "hotels": [
            {
                "name": "Sample Hotel Kyoto",
                "city": validated.city,
                "check_in": validated.check_in.isoformat(),
                "check_out": validated.check_out.isoformat(),
                "price_per_night_usd": 120,
                "guests": validated.guests,
            }
        ]
    }

    return {"errors": [], "result": result}
