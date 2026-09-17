from __future__ import annotations

from datetime import date, datetime, time, timedelta

from tourist_assistant.models import Attraction

WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _parse_time(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def _parse_range(raw: str | None) -> tuple[time, time] | None:
    if raw is None:
        return None
    if raw.strip().lower() in {"closed", "off", ""}:
        return None
    if "-" not in raw:
        return None
    a, b = raw.split("-", 1)
    return _parse_time(a.strip()), _parse_time(b.strip())


def opening_range(attraction: Attraction, day: date) -> tuple[time, time] | None:
    """Return the (open, close) times for the given calendar day, or None if closed."""
    key = WEEKDAY_KEYS[day.weekday()]
    return _parse_range(attraction.opening_hours.get(key))


def is_open_at(attraction: Attraction, dt: datetime) -> bool:
    rng = opening_range(attraction, dt.date())
    if rng is None:
        return False
    open_t, close_t = rng
    t = dt.time()
    if open_t <= close_t:
        return open_t <= t < close_t
    # Overnight (e.g. 22:00-02:00)
    return t >= open_t or t < close_t


def closing_time(attraction: Attraction, dt: datetime) -> datetime | None:
    """The next closing time on/after `dt`, or None if closed that day."""
    rng = opening_range(attraction, dt.date())
    if rng is None:
        return None
    open_t, close_t = rng
    if open_t <= close_t:
        return datetime.combine(dt.date(), close_t)
    # Overnight: closing is next day
    return datetime.combine(dt.date() + timedelta(days=1), close_t)


def next_opening(
    attraction: Attraction, dt: datetime, horizon_days: int = 7
) -> datetime | None:
    """First moment >= dt when the attraction is open, within the horizon."""
    cursor = dt
    for _ in range(horizon_days):
        rng = opening_range(attraction, cursor.date())
        if rng is not None:
            open_t, _ = rng
            candidate = datetime.combine(cursor.date(), open_t)
            if candidate >= dt:
                return candidate
        cursor = datetime.combine(cursor.date() + timedelta(days=1), time(0, 0))
    return None
