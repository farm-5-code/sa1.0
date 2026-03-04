from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List


# ── Погода ────────────────────────────────────────────────────────────

def _weather_code_to_text(code: int) -> str:
    return {
        0:  "Ясно",
        1:  "Преимущественно ясно",
        2:  "Переменная облачность",
        3:  "Пасмурно",
        45: "Туман",
        61: "Лёгкий дождь",
        63: "Умеренный дождь",
        65: "Сильный дождь",
        71: "Лёгкий снег",
        73: "Умеренный снег",
        75: "Сильный снег",
        80: "Ливень",
        95: "Гроза",
        99: "Гроза с градом",
    }.get(code, "Неизвестно")


@dataclass
class WeatherData:
    temperature:   float = 0.0
    precipitation: float = 0.0
    wind_speed:    float = 0.0
    wind_gusts:    float = 0.0
    cloud_cover:   float = 0.0
    visibility:    Optional[float] = None
    condition:     str   = "Неизвестно"
    impact_score:  float = 0.0
    analysis:      List[str] = field(default_factory=list)

    @classmethod
    def from_open_meteo_current(cls, raw: dict) -> WeatherData:
        obj = cls(
            temperature   = float(raw.get("temperature_2m",  0) or 0),
            precipitation = float(raw.get("precipitation",   0) or 0),
            wind_speed    = float(raw.get("wind_speed_10m",  0) or 0),
            wind_gusts    = float(raw.get("wind_gusts_10m",  0) or 0),
            cloud_cover   = float(raw.get("cloudcover",      0) or 0),
            visibility    = raw.get("visibility"),
            condition     = _weather_code_to_text(
                                int(raw.get("weathercode", 0) or 0)),
        )
        obj.impact_score = obj._calc_impact()
        obj.analysis     = obj._build_analysis()
        return obj

    @classmethod
    def from_open_meteo_hourly(cls, hourly: dict, idx: int) -> WeatherData:
        def _v(key: str) -> float:
            lst = hourly.get(key, [])
            return float(lst[idx]) if idx < len(lst) and lst[idx] is not None else 0.0

        obj = cls(
            temperature   = _v("temperature_2m"),
            precipitation = _v("precipitation"),
            wind_speed    = _v("wind_speed_10m"),
            wind_gusts    = _v("wind_gusts_10m"),
            cloud_cover   = _v("cloudcover"),
            visibility    = _v("visibility") or None,
            condition     = _weather_code_to_text(int(_v("weathercode"))),
        )
        obj.impact_score = obj._calc_impact()
        obj.analysis     = obj._build_analysis()
        return obj

    def _calc_impact(self) -> float:
        score = 0.0
        if   self.temperature < 0:    score += 20
        elif self.temperature < 5:    score += 10
        elif self.temperature > 35:   score += 15
        elif self.temperature > 30:   score +=  8
        if   self.precipitation > 5:  score += 25
        elif self.precipitation > 2:  score += 15
        elif self.precipitation > 0.5:score +=  8
        if   self.wind_speed > 50:    score += 20
        elif self.wind_speed > 30:    score += 12
        elif self.wind_speed > 20:    score +=  6
        return min(score, 100.0)

    def _build_analysis(self) -> List[str]:
        notes = []
        if self.precipitation > 2:
            notes.append("🌧️ Дождь усложнит контроль мяча")
        if self.wind_speed > 30:
            notes.append("💨 Сильный ветер повлияет на стандарты")
        if self.temperature < 3:
            notes.append("🥶 Холод может сказаться на физподготовке")
        if self.temperature > 32:
            notes.append("🔥 Жара снизит интенсивность во 2-м тайме")
        if not notes:
            notes.append("✅ Погода благоприятна для игры")
        return notes


# ── Команда ───────────────────────────────────────────────────────────

@dataclass
class TeamStats:
    team_id:             Optional[int]  = None
    name:                str            = ""
    elo:                 float          = 1500.0
    form:                List[str]      = field(default_factory=list)
    form_score:          float          = 50.0
    win_rate:            float          = 50.0
    avg_goals_scored:    float          = 1.4
    avg_goals_conceded:  float          = 1.4
    wins:                int            = 0
    draws:               int            = 0
    losses:              int            = 0
    injuries:            List[str]      = field(default_factory=list)
    suspensions:         List[str]      = field(default_factory=list)


# ── Новости ───────────────────────────────────────────────────────────

@dataclass
class NewsInsight:
    sentiment_score:   float      = 50.0
    sentiment_label:   str        = "😐 Нейтральный"
    has_injuries:      bool       = False
    has_suspensions:   bool       = False
    has_transfers:     bool       = False
    key_topics:        List[str]  = field(default_factory=list)
    injury_players:    List[str]  = field(default_factory=list)
    suspended_players: List[str]  = field(default_factory=list)
    articles_count:    int        = 0
