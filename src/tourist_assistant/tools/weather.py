from __future__ import annotations

from datetime import datetime
from datetime import date
import httpx

from tourist_assistant.config import settings
from tourist_assistant.models import WeatherInfo


def _weather_code_to_condition(code: int) -> str:
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, "Unknown")


def get_current_weather(lat: float | None = None, lon: float | None = None) -> WeatherInfo:
    lat = lat or settings.default_lat
    lon = lon or settings.default_lon

    url = f"{settings.weather_api_base}/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m,is_day",
        "hourly": "precipitation_probability",
        "timezone": settings.default_timezone,
        "forecast_days": 1,
    }

    with httpx.Client(timeout=10.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    current = data["current"]
    hourly = data["hourly"]

    # Find the precipitation probability for the current hour
    current_time = datetime.fromisoformat(current["time"])
    precip_prob = 0
    for i, t in enumerate(hourly["time"]):
        if datetime.fromisoformat(t).hour == current_time.hour:
            precip_prob = hourly["precipitation_probability"][i]
            break

    return WeatherInfo(
        timestamp=current_time,
        temperature_c=current["temperature_2m"],
        precipitation_probability=precip_prob,
        precipitation_mm=current["precipitation"],
        wind_speed_kmh=current["wind_speed_10m"],
        condition=_weather_code_to_condition(current["weather_code"]),
        is_day=bool(current["is_day"]),
    )


def get_hourly_weather(
    target_date: date | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> dict[int, WeatherInfo]:
    """
    Return a mapping {hour: WeatherInfo} for the given date (today by default).

    Used by the feasibility engine to check outdoor activities against the
    forecast at the exact hour they are scheduled.
    """
    lat = lat or settings.default_lat
    lon = lon or settings.default_lon
    target_date = target_date or date.today()

    url = f"{settings.weather_api_base}/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,precipitation_probability,precipitation,"
            "weather_code,wind_speed_10m,is_day"
        ),
        "timezone": settings.default_timezone,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }

    with httpx.Client(timeout=10.0) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    hourly = data["hourly"]
    result: dict[int, WeatherInfo] = {}
    for i, ts in enumerate(hourly["time"]):
        dt = datetime.fromisoformat(ts)
        result[dt.hour] = WeatherInfo(
            timestamp=dt,
            temperature_c=hourly["temperature_2m"][i],
            precipitation_probability=hourly["precipitation_probability"][i] or 0,
            precipitation_mm=hourly["precipitation"][i] or 0.0,
            wind_speed_kmh=hourly["wind_speed_10m"][i],
            condition=_weather_code_to_condition(hourly["weather_code"][i]),
            is_day=bool(hourly["is_day"][i]),
        )
    return result


if __name__ == "__main__":
    weather = get_current_weather()
    print(weather.model_dump_json(indent=2))