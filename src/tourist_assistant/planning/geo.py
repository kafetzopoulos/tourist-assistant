from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from tourist_assistant.models import TransportMode

EARTH_RADIUS_KM = 6371.0

# Effective city speeds (km/h), already accounting for stops and traffic.
SPEED_KMH: dict[TransportMode, float] = {
    TransportMode.WALKING: 4.5,
    TransportMode.TRANSIT: 15.0,
    TransportMode.DRIVING: 25.0,
}

# Fixed overhead per transfer: waiting for transit, parking, etc.
EXTRA_MIN: dict[TransportMode, int] = {
    TransportMode.WALKING: 0,
    TransportMode.TRANSIT: 5,
    TransportMode.DRIVING: 3,
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def travel_time_minutes(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    mode: TransportMode,
) -> int:
    """
    Approximate travel time between two coordinates.

    Prototype: haversine distance / effective speed + fixed overhead.
    Production: replace with a routing API (OSRM, Google Directions, Transit).
    """
    d = haversine_km(lat1, lon1, lat2, lon2)
    if d < 0.05:
        return 0
    minutes = d / SPEED_KMH[mode] * 60 + EXTRA_MIN[mode]
    return max(1, int(round(minutes)))