from __future__ import annotations

import json
import math
import ssl
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


def _vapor_pressure_deficit_hpa(
    temperature_celsius: float, relative_humidity_percent: float
) -> float:
    if not math.isfinite(temperature_celsius):
        raise ValueError("Temperature must be finite.")
    if (
        not math.isfinite(relative_humidity_percent)
        or not 0 <= relative_humidity_percent <= 100
    ):
        raise ValueError("Relative humidity must be between 0 and 100.")

    saturation_pressure_kpa = 0.6108 * math.exp(
        17.27 * temperature_celsius / (temperature_celsius + 237.3)
    )
    return saturation_pressure_kpa * (1 - relative_humidity_percent / 100) * 10


def fetch_weather_features(latitude: float, longitude: float) -> dict[str, float]:
    """Fetch current weather and today's forecast aggregates from Open-Meteo."""
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude or longitude is outside its valid range.")

    query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "auto",
            "forecast_days": 1,
        }
    )
    request = Request(
        f"https://api.open-meteo.com/v1/forecast?{query}",
        headers={"User-Agent": "WildfireGuardAI/1.0"},
    )
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=10, context=tls_context) as response:
        payload = json.load(response)

    if not isinstance(payload, dict):
        raise ValueError("Weather service returned an invalid response.")
    if payload.get("error"):
        raise ValueError(f"Weather service error: {payload.get('reason', 'unknown error')}")

    try:
        current = payload["current"]
        daily = payload["daily"]
        temperature = float(current["temperature_2m"])
        humidity = float(current["relative_humidity_2m"])
        features = {
            "tmax": float(daily["temperature_2m_max"][0]),
            "vpd": _vapor_pressure_deficit_hpa(temperature, humidity),
            "ws": float(current["wind_speed_10m"]),
            "rh": humidity,
            "prec": float(daily["precipitation_sum"][0]),
        }
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("Weather service returned incomplete forecast data.") from exc

    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Weather service returned non-finite forecast data.")
    return features
