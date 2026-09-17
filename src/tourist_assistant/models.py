from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Category(str, Enum):
    HISTORY = "history"
    ARCHAEOLOGY = "archaeology"
    MUSEUM = "museum"
    RELIGIOUS = "religious"
    PARK = "park"
    BEACH = "beach"
    ARCHITECTURE = "architecture"
    FOOD = "food"
    SHOPPING = "shopping"
    VIEWPOINT = "viewpoint"
    FAMILY = "family"


class TransportMode(str, Enum):
    WALKING = "walking"
    TRANSIT = "transit"
    DRIVING = "driving"


class Pace(str, Enum):
    RELAXED = "relaxed"
    NORMAL = "normal"
    FAST = "fast"


class Attraction(BaseModel):
    id: str
    name: str
    description: str
    categories: list[Category]
    indoor: bool
    lat: float
    lon: float
    opening_hours: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'mon': '08:00-20:00', 'tue': '08:00-20:00', ...}",
    )
    typical_duration_min: int = 60
    source_url: str | None = None
    notes: str | None = None


class WeatherInfo(BaseModel):
    timestamp: datetime
    temperature_c: float
    precipitation_probability: int
    precipitation_mm: float
    wind_speed_kmh: float
    condition: str
    is_day: bool


class Activity(BaseModel):
    attraction_id: str
    name: str
    start_time: datetime
    end_time: datetime
    travel_time_before_min: int = 0
    indoor: bool
    notes: str | None = None


class Itinerary(BaseModel):
    start_time: datetime
    end_time: datetime
    activities: list[Activity]
    feasible: bool
    feasibility_notes: list[str] = Field(default_factory=list)
    weather_summary: str | None = None


class UserPreferences(BaseModel):
    interests: list[Category] = Field(default_factory=list)
    dislikes: list[Category] = Field(default_factory=list)
    pace: Pace = Pace.NORMAL
    transport_mode: TransportMode = TransportMode.WALKING
    has_children: bool = False
    has_car: bool = False
    max_walk_km: float | None = None
    language: str = "en"


class ConversationState(BaseModel):
    session_id: str
    history: list[dict[str, str]] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    current_itinerary: Itinerary | None = None
    visited_attraction_ids: list[str] = Field(default_factory=list)