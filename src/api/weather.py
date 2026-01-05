"""Weather API endpoints - real-time global weather data."""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from src.data.sources.weather import get_weather_source
from src.data.sources.climate import get_climate_source


router = APIRouter()


class CurrentWeather(BaseModel):
    """Current weather response."""
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    time_local: str
    is_day: bool
    temperature_c: float
    feels_like_c: float
    humidity_percent: float
    pressure_hpa: float
    wind_speed_kph: float
    wind_direction_deg: float
    wind_gusts_kph: float
    precipitation_mm: float
    rain_mm: float
    snowfall_cm: float
    cloud_cover_percent: float
    weather_code: int
    weather_description: str


class ForecastDay(BaseModel):
    """Single day forecast."""
    date: str
    weather_code: int
    weather_description: str
    temp_max_c: float
    temp_min_c: float
    feels_like_max_c: float
    feels_like_min_c: float
    sunrise: str
    sunset: str
    precipitation_mm: float
    rain_mm: float
    snowfall_cm: float
    wind_max_kph: float
    wind_gusts_kph: float
    wind_direction_deg: float
    uv_index: float


class ForecastResponse(BaseModel):
    """Forecast response."""
    latitude: float
    longitude: float
    days: list[ForecastDay]


@router.get("/current")
async def get_current_weather(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
) -> CurrentWeather:
    """
    Get current weather at a location.
    
    Uses Open-Meteo API (free, no API key required, global coverage).
    """
    try:
        weather_source = get_weather_source()
        weather = await weather_source.get_current(lat, lon)
        
        return CurrentWeather(
            latitude=weather.latitude,
            longitude=weather.longitude,
            elevation_m=weather.elevation_m,
            timezone=weather.timezone,
            time_local=weather.time_local.isoformat(),
            is_day=weather.is_day,
            temperature_c=weather.temperature_c,
            feels_like_c=weather.feels_like_c,
            humidity_percent=weather.humidity_percent,
            pressure_hpa=weather.pressure_hpa,
            wind_speed_kph=weather.wind_speed_kph,
            wind_direction_deg=weather.wind_direction_deg,
            wind_gusts_kph=weather.wind_gusts_kph,
            precipitation_mm=weather.precipitation_mm,
            rain_mm=weather.rain_mm,
            snowfall_cm=weather.snowfall_cm,
            cloud_cover_percent=weather.cloud_cover_percent,
            weather_code=weather.weather_code,
            weather_description=weather.weather_description,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")


@router.get("/forecast")
async def get_forecast(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    days: int = Query(7, ge=1, le=16, description="Forecast days"),
) -> ForecastResponse:
    """
    Get weather forecast for a location.
    
    Up to 16 days forecast available.
    """
    try:
        weather_source = get_weather_source()
        forecast = await weather_source.get_forecast(lat, lon, days)
        
        return ForecastResponse(
            latitude=lat,
            longitude=lon,
            days=[ForecastDay(**day) for day in forecast],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API error: {str(e)}")


@router.get("/cities")
async def get_weather_for_cities() -> dict:
    """
    Get current weather for major Australian cities.
    
    Quick way to see weather across the simulation area.
    """
    cities = {
        "sydney": (-33.8688, 151.2093),
        "melbourne": (-37.8136, 144.9631),
        "brisbane": (-27.4698, 153.0251),
        "perth": (-31.9523, 115.8613),
        "adelaide": (-34.9285, 138.6007),
        "hobart": (-42.8821, 147.3272),
        "darwin": (-12.4634, 130.8456),
        "canberra": (-35.2809, 149.1300),
    }
    
    weather_source = get_weather_source()
    results = {}
    
    for city, (lat, lon) in cities.items():
        try:
            weather = await weather_source.get_current(lat, lon)
            results[city] = {
                "temperature_c": weather.temperature_c,
                "feels_like_c": weather.feels_like_c,
                "weather": weather.weather_description,
                "humidity_percent": weather.humidity_percent,
                "wind_speed_kph": weather.wind_speed_kph,
                "is_day": weather.is_day,
            }
        except Exception:
            results[city] = {"error": "Failed to fetch"}
    
    return results


# =============================================================================
# CLIMATE (Köppen-Geiger classification from historical data)
# =============================================================================

class ClimateResponse(BaseModel):
    """Climate classification response."""
    latitude: float
    longitude: float
    koppen_code: str
    koppen_description: str
    climate_type: str
    annual_mean_temp_c: float
    warmest_month_temp_c: float
    coldest_month_temp_c: float
    temp_range_c: float
    annual_precipitation_mm: float
    wettest_month_mm: float
    driest_month_mm: float
    dry_season: str | None
    has_dry_season: bool
    elevation_m: float
    data_years: int
    source: str


@router.get("/climate")
async def get_climate(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
) -> ClimateResponse:
    """
    Get Köppen-Geiger climate classification for a location.
    
    Computed from 5 years of historical weather data via Open-Meteo.
    This is real data, not a lookup table.
    """
    try:
        climate_source = get_climate_source()
        climate = await climate_source.get_climate(lat, lon)
        
        return ClimateResponse(
            latitude=climate.latitude,
            longitude=climate.longitude,
            koppen_code=climate.koppen_code,
            koppen_description=climate.koppen_description,
            climate_type=climate.climate_type,
            annual_mean_temp_c=climate.annual_mean_temp_c,
            warmest_month_temp_c=climate.warmest_month_temp_c,
            coldest_month_temp_c=climate.coldest_month_temp_c,
            temp_range_c=climate.temp_range_c,
            annual_precipitation_mm=climate.annual_precipitation_mm,
            wettest_month_mm=climate.wettest_month_mm,
            driest_month_mm=climate.driest_month_mm,
            dry_season=climate.dry_season,
            has_dry_season=climate.has_dry_season,
            elevation_m=climate.elevation_m,
            data_years=climate.data_years,
            source=climate.source,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Climate API error: {str(e)}")


@router.get("/climate/cities")
async def get_climate_for_cities() -> dict:
    """
    Get climate classification for major Australian cities.
    
    Shows the variety of climates across the simulation area.
    """
    cities = {
        "sydney": (-33.8688, 151.2093),
        "melbourne": (-37.8136, 144.9631),
        "brisbane": (-27.4698, 153.0251),
        "perth": (-31.9523, 115.8613),
        "adelaide": (-34.9285, 138.6007),
        "hobart": (-42.8821, 147.3272),
        "darwin": (-12.4634, 130.8456),
        "alice_springs": (-23.6980, 133.8807),
    }
    
    climate_source = get_climate_source()
    results = {}
    
    for city, (lat, lon) in cities.items():
        try:
            climate = await climate_source.get_climate(lat, lon)
            results[city] = {
                "koppen_code": climate.koppen_code,
                "description": climate.koppen_description,
                "climate_type": climate.climate_type,
                "annual_mean_temp_c": climate.annual_mean_temp_c,
                "annual_precipitation_mm": climate.annual_precipitation_mm,
            }
        except Exception:
            results[city] = {"error": "Failed to compute"}
    
    return results

