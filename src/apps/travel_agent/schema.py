from datetime import date
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Activity(BaseModel):
    # extra="forbid" - prevents LLM hallucinations like:
    #     {
    # "title": "Kyoto Trip",
    # "random_field": "oops"
    # }
    model_config = ConfigDict(extra="forbid")

    time: str = Field(min_length=2, max_length=40)
    name: str = Field(min_length=2, max_length=100)
    description: str = Field(min_length=5, max_length=300)
    estimated_cost_usd: float = Field(ge=0, le=5000)


class DayPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day_number: int = Field(ge=1)
    date: date
    theme: str = Field(min_length=3, max_length=100)
    activities: list[Activity] = Field(min_length=2, max_length=6)
    meals: list[Activity] = Field(min_length=1, max_length=3)

    @field_validator("activities")
    @classmethod
    def ensure_unique_activity_names(cls, v):
        names = [a.name for a in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate activity names in a single day")
        return v


class TripItinerary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=3, max_length=120)
    summary: str = Field(min_length=10, max_length=500)
    days: list[DayPlan] = Field(min_length=1)
    total_estimated_cost_usd: float = Field(ge=0, le=100000)
    packing_tips: list[str] = Field(min_length=1, max_length=5)

    @field_validator("days")
    @classmethod
    def ensure_sequential_days(cls, v):
        expected = list(range(1, len(v) + 1))
        actual = [d.day_number for d in v]
        if expected != actual:
            raise ValueError(f"Day numbers must be sequential: {expected}")
        return v


class FlightSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    departure_date: date
    return_date: date | None = None
    passengers: int = Field(ge=1, le=9)


class HotelSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=2, max_length=80)
    check_in: date
    check_out: date
    guests: int = Field(ge=1, le=10)
    max_price_per_night_usd: float | None = Field(default=None, ge=0)

    @field_validator("check_out")
    @classmethod
    def checkout_after_checkin(cls, v, info):
        if "check_in" in info.data and v <= info.data["check_in"]:
            raise ValueError("check_out must be after check_in")
        return v
