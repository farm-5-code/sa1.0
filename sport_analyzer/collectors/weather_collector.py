import logging
from datetime import datetime
from typing import Optional

from sport_analyzer.collectors.base_collector import BaseCollector
from sport_analyzer.entities.base import WeatherData

logger = logging.getLogger(__name__)


class WeatherCollector(BaseCollector):
    RATE_LIMIT_PER_MINUTE = 60

    def __init__(self, db_path: str = "sport_analyzer.db"):
        super().__init__(db_path=db_path)

    STADIUMS = {
        "Old Trafford": {"lat": 53.4631, "lon": -2.2913},
        "Anfield": {"lat": 53.4308, "lon": -2.9608},
        "Emirates Stadium": {"lat": 51.5549, "lon": -0.1084},
        "Santiago Bernabeu": {"lat": 40.4530, "lon": -3.6883},
        "Camp Nou": {"lat": 41.3809, "lon": 2.1228},
        "Allianz Arena": {"lat": 48.2188, "lon": 11.6248},
        "San Siro": {"lat": 45.4781, "lon": 9.1240},
        "Wembley": {"lat": 51.5560, "lon": -0.2796},
    }

    HOURLY_PARAMS = [
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_gusts_10m",
        "cloudcover",
        "visibility",
        "weathercode",
    ]

    def get_weather_for_match(
        self,
        city: Optional[str] = None,
        stadium: Optional[str] = None,
        match_datetime: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> WeatherData:
        coords = self._resolve_coords(city, stadium, lat, lon)
        if not coords:
            logger.warning("Не удалось определить координаты")
            return WeatherData()
        if match_datetime:
            return self._forecast(coords["lat"], coords["lon"], match_datetime)
        return self._current(coords["lat"], coords["lon"])

    def _resolve_coords(self, city, stadium, lat, lon) -> Optional[dict]:
        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon}
        if stadium and stadium in self.STADIUMS:
            return self.STADIUMS[stadium]
        if city:
            return self._geocode(city)
        return None

    def _geocode(self, city: str) -> Optional[dict]:
        cache_key = f"geo_{city.lower().replace(' ', '_')}"
        cached = self._cache_get(cache_key, max_age_hours=720)
        if cached:
            return cached

        resp = self.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        if resp is None or resp.status_code != 200:
            return None

        results = resp.json().get("results") or []
        if not results:
            return None

        coords = {"lat": results[0]["latitude"], "lon": results[0]["longitude"]}
        self._cache_set(cache_key, coords)
        return coords

    def _current(self, lat: float, lon: float) -> WeatherData:
        resp = self.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": self.HOURLY_PARAMS,
                "timezone": "auto",
            },
        )
        if resp is None or resp.status_code != 200:
            return WeatherData()
        return WeatherData.from_open_meteo_current(resp.json().get("current", {}))

    def _forecast(self, lat: float, lon: float, match_datetime: str) -> WeatherData:
        resp = self.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": self.HOURLY_PARAMS,
                "forecast_days": 14,
                "timezone": "auto",
            },
        )
        if resp is None or resp.status_code != 200:
            return WeatherData()

        hourly = resp.json().get("hourly", {})
        times = hourly.get("time", [])

        try:
            match_dt = datetime.fromisoformat(match_datetime.replace("Z", ""))
            match_str = match_dt.strftime("%Y-%m-%dT%H:00")
        except ValueError:
            match_str = ""

        idx = next((i for i, t in enumerate(times) if t == match_str), 0)
        return WeatherData.from_open_meteo_hourly(hourly, idx)
