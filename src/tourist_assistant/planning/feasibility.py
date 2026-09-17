from __future__ import annotations

from datetime import datetime, timedelta

from tourist_assistant.config import settings
from tourist_assistant.models import (
    Activity,
    Attraction,
    Itinerary,
    Pace,
    UserPreferences,
    WeatherInfo,
)
from tourist_assistant.planning.geo import travel_time_minutes
from tourist_assistant.planning.hours import (
    closing_time,
    is_open_at,
    next_opening,
)

# Pace multipliers applied to nominal visit durations.
PACE_MULTIPLIER: dict[Pace, float] = {
    Pace.RELAXED: 1.30,
    Pace.NORMAL: 1.00,
    Pace.FAST: 0.80,
}

# Don't make the user stand around waiting for a museum to open.
MAX_WAIT_MIN = 20

# Outdoor activity under this rain probability (or above) triggers a warning.
RAIN_THRESHOLD_PCT = 60


def _pace_multiplier(prefs: UserPreferences) -> float:
    return PACE_MULTIPLIER[prefs.pace]


def _weather_conflict(
    attraction: Attraction,
    hour: int,
    weather_by_hour: dict[int, WeatherInfo] | None,
) -> WeatherInfo | None:
    if attraction.indoor or not weather_by_hour:
        return None
    w = weather_by_hour.get(hour)
    if w and w.precipitation_probability >= RAIN_THRESHOLD_PCT:
        return w
    return None


def build_itinerary(
    candidates: list[Attraction],
    start_time: datetime,
    end_time: datetime,
    prefs: UserPreferences,
    origin: tuple[float, float] | None = None,
    weather_by_hour: dict[int, WeatherInfo] | None = None,
    strict_weather: bool = False,
) -> Itinerary:
    """
    Sequentially place candidate attractions into the [start_time, end_time] window.

    Respects: travel time, opening hours, visit duration, pace, and (optionally)
    outdoor weather conflicts. Attractions that don't fit are skipped with a
    human-readable note. Deterministic — no LLM in this loop.
    """
    origin = origin or (settings.default_lat, settings.default_lon)
    pace = _pace_multiplier(prefs)

    activities: list[Activity] = []
    notes: list[str] = []
    seen: set[str] = set()

    current_loc = origin
    current_time = start_time

    for attraction in candidates:
        if attraction.id in seen:
            continue

        travel_min = travel_time_minutes(
            current_loc[0], current_loc[1],
            attraction.lat, attraction.lon,
            prefs.transport_mode,
        )
        arrival = current_time + timedelta(minutes=travel_min)

        # --- Opening hours ---
        if not is_open_at(attraction, arrival):
            nxt = next_opening(attraction, arrival)
            if nxt is None or nxt >= end_time:
                notes.append(
                    f"Skipped {attraction.name}: closed during the requested window."
                )
                continue
            wait_min = int((nxt - arrival).total_seconds() / 60)
            if wait_min > MAX_WAIT_MIN:
                notes.append(
                    f"Skipped {attraction.name}: opens at {nxt:%H:%M}, "
                    f"which would require a {wait_min}-minute wait."
                )
                continue
            arrival = nxt

        # --- Visit duration adjusted by pace ---
        duration = int(attraction.typical_duration_min * pace)

        # Clamp to closing time
        close = closing_time(attraction, arrival)
        if close is not None:
            available = int((close - arrival).total_seconds() / 60)
            if available < attraction.typical_duration_min * 0.5:
                notes.append(
                    f"Skipped {attraction.name}: only {available} minutes remain "
                    f"before closing at {close:%H:%M}."
                )
                continue
            duration = min(duration, available)

        visit_end = arrival + timedelta(minutes=duration)

        # --- Window check ---
        if visit_end > end_time:
            notes.append(
                f"Skipped {attraction.name}: does not fit before {end_time:%H:%M}."
            )
            continue

        # --- Weather check for outdoor activities ---
        conflict = _weather_conflict(attraction, arrival.hour, weather_by_hour)
        if conflict is not None:
            msg = (
                f"{attraction.name} is outdoors and rain is likely around "
                f"{arrival:%H:%M} ({conflict.precipitation_probability}% chance)."
            )
            if strict_weather:
                notes.append(f"Skipped: {msg}")
                continue
            notes.append(f"Warning: {msg}")

        activities.append(
            Activity(
                attraction_id=attraction.id,
                name=attraction.name,
                start_time=arrival,
                end_time=visit_end,
                travel_time_before_min=travel_min,
                indoor=attraction.indoor,
            )
        )

        seen.add(attraction.id)
        current_time = visit_end
        current_loc = (attraction.lat, attraction.lon)

    feasible = len(activities) > 0
    if not feasible:
        notes.append("No attraction could be scheduled in the given window.")

    return Itinerary(
        start_time=start_time,
        end_time=end_time,
        activities=activities,
        feasible=feasible,
        feasibility_notes=notes,
    )


def check_itinerary(
    itinerary: Itinerary,
    prefs: UserPreferences,
) -> list[str]:
    """
    Validate an externally-supplied itinerary (e.g. the user proposes 3 places).
    Returns a list of issues; empty list means the plan is feasible.
    """
    issues: list[str] = []
    prev_end: datetime | None = None
    prev_loc: tuple[float, float] | None = None

    for activity in itinerary.activities:
        # Look up the attraction via the retriever to get coordinates + hours.
        from tourist_assistant.rag.retriever import get_retriever

        attraction = get_retriever().get_attraction(activity.attraction_id)
        if attraction is None:
            issues.append(f"Unknown attraction id: {activity.attraction_id}")
            continue

        if prev_end is not None and activity.start_time < prev_end:
            issues.append(
                f"Overlap: {activity.name} starts at {activity.start_time:%H:%M} "
                f"but previous activity ends at {prev_end:%H:%M}."
            )

        if prev_loc is not None:
            needed = travel_time_minutes(
                prev_loc[0], prev_loc[1],
                attraction.lat, attraction.lon,
                prefs.transport_mode,
            )
            if prev_end is not None:
                gap = int((activity.start_time - prev_end).total_seconds() / 60)
                if gap < needed:
                    issues.append(
                        f"{activity.name}: only {gap} min between activities, "
                        f"but ~{needed} min travel is required."
                    )

        if not is_open_at(attraction, activity.start_time):
            issues.append(
                f"{activity.name} is not open at {activity.start_time:%H:%M}."
            )

        prev_end = activity.end_time
        prev_loc = (attraction.lat, attraction.lon)

    return issues


def _score(itinerary: Itinerary) -> float:
    """Higher is better: reward placed activities, penalize warnings."""
    warnings = sum(1 for n in itinerary.feasibility_notes if n.startswith("Warning"))
    return len(itinerary.activities) * 10 - warnings * 2


def plan_day(
    candidates: list[Attraction],
    start_time: datetime,
    end_time: datetime,
    prefs: UserPreferences,
    origin: tuple[float, float] | None = None,
    weather_by_hour: dict[int, WeatherInfo] | None = None,
) -> Itinerary:
    """
    Weather-aware planner.

    Tries several orderings of the candidates (as-is, indoor-first, outdoor-first)
    and returns the one that places the most attractions with the fewest weather
    warnings. This is the deterministic backbone for "change my plan if it rains".
    """
    orderings: list[tuple[str, list[Attraction]]] = [
        ("relevance", list(candidates)),
        ("indoor-first", sorted(candidates, key=lambda a: (not a.indoor,))),
        ("outdoor-first", sorted(candidates, key=lambda a: (a.indoor,))),
    ]

    best: Itinerary | None = None
    best_label = ""
    for label, order in orderings:
        it = build_itinerary(
            order,
            start_time=start_time,
            end_time=end_time,
            prefs=prefs,
            origin=origin,
            weather_by_hour=weather_by_hour,
        )
        if best is None or _score(it) > _score(best):
            best = it
            best_label = label

    assert best is not None
    best.feasibility_notes.insert(0, f"Plan generated using ordering: {best_label}.")
    return best


def summarize(itinerary: Itinerary) -> str:
    """Compact human-readable summary used in logs and the design note."""
    total_visit = sum(
        int((a.end_time - a.start_time).total_seconds() / 60)
        for a in itinerary.activities
    )
    total_travel = sum(a.travel_time_before_min for a in itinerary.activities)
    return (
        f"{len(itinerary.activities)} activities | "
        f"{total_visit} min visiting | {total_travel} min travelling | "
        f"feasible={itinerary.feasible}"
    )