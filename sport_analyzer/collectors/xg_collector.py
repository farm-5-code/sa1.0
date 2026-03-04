"""sport_analyzer.collectors.xg_collector

xG (Expected Goals) collector.

Scrapes Understat league pages and extracts embedded teamsData JSON.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from sport_analyzer.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)


class XGCollector(BaseCollector):
    RATE_LIMIT_PER_MINUTE = 10

    LEAGUES = {
        "EPL": "EPL",
        "La_liga": "La_liga",
        "Bundesliga": "Bundesliga",
        "Serie_A": "Serie_A",
        "Ligue_1": "Ligue_1",
        "RFPL": "RFPL",
    }

    def get_team_xg(
        self,
        team_name: str,
        league: str = "EPL",
        season: Optional[int] = None,
    ) -> Dict:
        cache_key = (
            f"xg_{team_name.lower().replace(' ', '_')}_{league}_{season or 'cur'}"
        )
        cached = self._cache_get(cache_key, max_age_hours=12.0)
        if cached:
            return cached

        season = season or self._current_season()
        matches = self._fetch_understat_team(team_name, league, season)
        if not matches:
            return self._default_xg()

        result = self._aggregate_xg(matches)
        self._cache_set(cache_key, result)
        return result

    def _fetch_understat_team(
        self, team_name: str, league: str, season: int
    ) -> List[Dict]:
        league = self.LEAGUES.get(league, league)
        url = f"https://understat.com/league/{league}/{season}"
        resp = self.get(url, host_key="understat.com")
        if resp is None or resp.status_code != 200:
            return []
        return self._parse_understat_html(resp.text, team_name)

    def _parse_understat_html(self, html: str, team_name: str) -> List[Dict]:
        pattern = r"var\s+teamsData\s*=\s*JSON\.parse\('(.+?)'\)"
        m = re.search(pattern, html)
        if not m:
            logger.warning("Understat: teamsData not found")
            return []

        try:
            raw_json = m.group(1).encode().decode("unicode_escape")
            data = json.loads(raw_json)
        except Exception as e:
            logger.error(f"Understat JSON parse error: {e}")
            return []

        team_lower = team_name.lower()
        for team_key, team_data in (data or {}).items():
            tk = str(team_key).lower()
            if team_lower in tk or tk in team_lower:
                return list(team_data.get("history", []) or [])
        return []

    def _aggregate_xg(self, matches: List[Dict]) -> Dict:
        xg_for: List[float] = []
        xg_against: List[float] = []
        for m in matches:
            try:
                xgf = float(m.get("xG", 0) or 0)
                xga = float(m.get("xGA", 0) or 0)
            except Exception:
                continue
            xg_for.append(xgf)
            xg_against.append(xga)

        if not xg_for:
            return self._default_xg()

        recent = min(10, len(xg_for))
        avg_xgf = round(sum(xg_for[-recent:]) / recent, 3)
        avg_xga = round(sum(xg_against[-recent:]) / recent, 3)
        return {
            "xg_for": avg_xgf,
            "xg_against": avg_xga,
            "xg_diff": round(avg_xgf - avg_xga, 3),
            "matches": len(xg_for),
            "source": "understat",
        }

    @staticmethod
    def _default_xg() -> Dict:
        return {
            "xg_for": 1.35,
            "xg_against": 1.35,
            "xg_diff": 0.0,
            "matches": 0,
            "source": "default",
        }

    @staticmethod
    def _current_season() -> int:
        from datetime import datetime

        now = datetime.utcnow()
        return now.year if now.month >= 8 else now.year - 1
