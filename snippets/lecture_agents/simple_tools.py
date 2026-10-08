"""
Simple, general-purpose tools, shared by the examples of this lecture: plain Python functions, documented for the LLM.
Their names, docstrings, and annotated parameters are what the LLM sees (as a JSON Schema) to decide which tool to call.
"""
import datetime
import json
import urllib.parse
import urllib.request
import zoneinfo
from typing import Annotated
from ddgs import DDGS
from pydantic import Field

instructions = """
You are a helpful assistant. Use the available tools to answer questions about the present,
e.g. the current time, the weather, or recent facts. Ground your answers on the tools' results.
If the tools cannot answer a question, say so.
"""


def get_current_time(
    timezone: Annotated[str, Field(description="IANA time zone name, e.g. 'Europe/Rome' or 'Asia/Tokyo'")],
) -> str:
    """Get the current date and time in a given time zone, in ISO 8601 format, with the weekday."""
    try:
        now = datetime.datetime.now(zoneinfo.ZoneInfo(timezone))
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):  # never trust the LLM's arguments
        raise ValueError(f"Unknown time zone: {timezone!r}. Use IANA names, e.g. 'Asia/Tokyo'.")
    return now.strftime("%A %Y-%m-%dT%H:%M:%S%z")


def get_json(url: str, **params) -> dict:  # not a tool: a helper for get_weather
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=10) as response:
        return json.load(response)


def get_weather(
    location: Annotated[str, Field(description="Name of a city or place, e.g. 'Bologna'")],
) -> dict:
    """Get the current weather in a location: temperature (°C), precipitation (mm), cloud cover (%),
    and wind speed (km/h). The result also reports the location's country and IANA time zone."""
    # 1. geocoding: from the location's name to its coordinates (via Open-Meteo's free API, no key needed)
    places = get_json("https://geocoding-api.open-meteo.com/v1/search", name=location, count=1).get("results")
    if not places:  # errors meaningful to the LLM, which may recover (e.g. by trying another name)
        raise ValueError(f"Unknown location: {location!r}. Try with the name of a nearby city.")
    place = places[0]
    # 2. forecast: current weather at those coordinates
    weather = get_json("https://api.open-meteo.com/v1/forecast", latitude=place["latitude"],
                       longitude=place["longitude"], current="temperature_2m,precipitation,cloud_cover,wind_speed_10m")
    return dict(location=place["name"], country=place.get("country"), timezone=place.get("timezone"), **weather["current"])


def web_search(
    query: Annotated[str, Field(description="Keywords to search for on the Web")],
) -> list[dict]:
    """Search the Web (via DuckDuckGo), returning the title, URL, and snippet of the top 5 results.
    Use it for facts which may have changed recently, or which you are not sure about."""
    return DDGS().text(query, max_results=5)


tools = [get_current_time, get_weather, web_search]  # the tools made available to the agents
