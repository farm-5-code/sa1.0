import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sport_analyzer.collectors.base_collector import BaseCollector
from sport_analyzer.entities.base import TeamStats
from sport_analyzer.utils.team_normalizer import normalize_team_name

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10


class SportsCollector(BaseCollector):
    RATE_LIMIT_PER_MINUTE = 9

    def __init__(self, config, db_path: str = "sport_analyzer.db"):
        super().__init__(db_path=db_path)
        self.config = config
        self.session.headers["X-Auth-Token"] = config.FOOTBALL_DATA_KEY

    # ── Матчи ─────────────────────────────────────────────────────────

    def get_matches(self, days_ahead: int = 7) -> List[Dict]:
        date_from = datetime.utcnow().strftime("%Y-%m-%d")
        date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        resp = self.get(
            "https://api.football-data.org/v4/matches",
            params={"dateFrom": date_from, "dateTo": date_to},
        )
        if resp is None or resp.status_code != 200:
            return []

        return [
            {
                "id": m.get("id"),
                "date": m.get("utcDate"),
                "competition": m.get("competition", {}).get("name"),
                "home_team": m.get("homeTeam", {}).get("name"),
                "away_team": m.get("awayTeam", {}).get("name"),
                "home_team_id": m.get("homeTeam", {}).get("id"),
                "away_team_id": m.get("awayTeam", {}).get("id"),
                "stadium": m.get("venue"),
            }
            for m in resp.json().get("matches", [])
        ]

    # ── Статистика команды ────────────────────────────────────────────

    def get_team_stats(self, team_id: int, use_cache: bool = True) -> TeamStats:
        cache_key = f"team_stats_{team_id}"
        if use_cache:
            cached = self._cache_get(cache_key, max_age_hours=3.0)
            if cached:
                return TeamStats(
                    **{
                        k: v
                        for k, v in cached.items()
                        if k in TeamStats.__dataclass_fields__
                    }
                )

        resp = self.get(
            f"https://api.football-data.org/v4/teams/{team_id}/matches",
            params={"limit": 15, "status": "FINISHED"},
        )
        if resp is None or resp.status_code != 200:
            return TeamStats(team_id=team_id)

        stats = self._parse_team_matches(team_id, resp.json().get("matches", []))
        self._cache_set(cache_key, stats.__dict__)
        return stats

    def _parse_team_matches(self, team_id: int, matches: List) -> TeamStats:
        wins = draws = losses = 0
        gs = gc = 0
        form: List[str] = []

        for m in matches:
            score = m.get("score", {}).get("fullTime", {})
            hg = int(score.get("home") or 0)
            ag = int(score.get("away") or 0)
            is_home = m.get("homeTeam", {}).get("id") == team_id
            my, opp = (hg, ag) if is_home else (ag, hg)

            gs += my
            gc += opp
            if my > opp:
                wins += 1
                form.append("W")
            elif my == opp:
                draws += 1
                form.append("D")
            else:
                losses += 1
                form.append("L")

        total = wins + draws + losses or 1
        return TeamStats(
            team_id=team_id,
            form=form[-5:],
            form_score=self._form_score(form[-5:]),
            win_rate=round(wins / total * 100, 1),
            avg_goals_scored=round(gs / total, 2),
            avg_goals_conceded=round(gc / total, 2),
            wins=wins,
            draws=draws,
            losses=losses,
        )

    @staticmethod
    def _form_score(form: List[str]) -> float:
        weights = [1.0, 0.85, 0.70, 0.55, 0.40]
        score = max_s = 0.0
        for i, r in enumerate(reversed(form)):
            w = weights[i] if i < len(weights) else 0.3
            score += (3 if r == "W" else 1 if r == "D" else 0) * w
            max_s += 3 * w
        return round(score / max_s * 100, 1) if max_s else 50.0

    # ── Информация о команде ──────────────────────────────────────────

    def get_team_info(self, team_name: str) -> Dict:
        cache_key = f"team_info_{team_name.lower().replace(' ', '_')}"
        cached = self._cache_get(cache_key, max_age_hours=72)
        if cached:
            return cached

        resp = self.get(
            "https://www.thesportsdb.com/api/v1/json/3/searchteams.php",
            params={"t": team_name},
        )
        if resp is None or resp.status_code != 200:
            return {}

        teams = resp.json().get("teams") or []
        if not teams:
            return {}

        t = teams[0]
        info = {
            "name": t.get("strTeam"),
            "stadium": t.get("strStadium"),
            "stadium_location": t.get("strStadiumLocation"),
            "country": t.get("strCountry"),
        }
        self._cache_set(cache_key, info)
        return info

    # ── Head-to-Head ──────────────────────────────────────────────────

    def get_head_to_head(self, home_name: str, away_name: str) -> List[Dict]:
        cache_key = (
            f"h2h_{home_name.lower().replace(' ', '_')}"
            f"_{away_name.lower().replace(' ', '_')}"
        )
        cached = self._cache_get(cache_key, max_age_hours=24)
        if cached:
            return cached

        resp = self.get(
            "https://www.thesportsdb.com/api/v1/json/3/searchevents.php",
            params={"e": f"{home_name} vs {away_name}"},
        )
        events = []
        if resp and resp.status_code == 200:
            hn_l = home_name.lower()
            an_l = away_name.lower()
            for ev in resp.json().get("event") or []:
                h = (ev.get("strHomeTeam") or "").lower()
                a = (ev.get("strAwayTeam") or "").lower()
                if (hn_l in h or hn_l in a) and (an_l in h or an_l in a):
                    events.append(
                        {
                            "date": ev.get("dateEvent"),
                            "home_team": ev.get("strHomeTeam"),
                            "away_team": ev.get("strAwayTeam"),
                            "home_score": ev.get("intHomeScore"),
                            "away_score": ev.get("intAwayScore"),
                        }
                    )

        self._cache_set(cache_key, events[:10])
        return events[:10]

    def get_h2h_stats(self, home_name: str, away_name: str) -> Dict:
        events = self.get_head_to_head(home_name, away_name)
        if not events:
            return {
                "matches": 0,
                "home_wins": 0,
                "away_wins": 0,
                "draws": 0,
                "home_win_pct": 33.3,
                "draw_pct": 33.3,
                "away_win_pct": 33.3,
            }

        home_wins = away_wins = draws = 0
        hn_l = home_name.lower()

        for ev in events:
            try:
                hs = int(ev.get("home_score") or 0)
                as_ = int(ev.get("away_score") or 0)
            except (TypeError, ValueError):
                continue
            is_our_home = hn_l in (ev.get("home_team") or "").lower()
            my = hs if is_our_home else as_
            opp = as_ if is_our_home else hs
            if my > opp:
                home_wins += 1
            elif my < opp:
                away_wins += 1
            else:
                draws += 1

        total = home_wins + away_wins + draws or 1
        return {
            "matches": total,
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
            "home_win_pct": round(home_wins / total * 100, 1),
            "draw_pct": round(draws / total * 100, 1),
            "away_win_pct": round(away_wins / total * 100, 1),
        }

    # ── OpenLigaDB ────────────────────────────────────────────────────

    @staticmethod
    def _current_season() -> int:
        now = datetime.utcnow()
        return now.year if now.month >= 8 else now.year - 1

    def get_bundesliga_table(self, season: Optional[int] = None) -> List[Dict]:
        season = season or self._current_season()
        resp = self.get(f"https://api.openligadb.de/getbltable/bl1/{season}")
        if resp is None or resp.status_code != 200:
            return []
        return [
            {
                "position": i + 1,
                "team": item.get("teamName"),
                "points": item.get("points"),
                "won": item.get("won"),
                "lost": item.get("lost"),
                "draw": item.get("draw"),
                "goal_diff": item.get("goalDiff"),
            }
            for i, item in enumerate(resp.json())
        ]

    # ── ELO ───────────────────────────────────────────────────────────

    def get_elo(self, team_name: str) -> float:
        name = normalize_team_name(team_name)
        self._ensure_elo_table()
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            row = c.execute("SELECT elo FROM team_elo WHERE name=?", (name,)).fetchone()
        return row[0] if row else 1500.0

    def update_elo(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        league: str = "",
    ) -> None:
        home = normalize_team_name(home)
        away = normalize_team_name(away)
        league = (league or "").strip()

        K = 32
        HOME_ADV = 100

        elo_h = self.get_elo(home)
        elo_a = self.get_elo(away)

        exp_h = 1 / (1 + 10 ** ((elo_a - elo_h - HOME_ADV) / 400))
        actual_h = (
            1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        )

        new_elo_h = round(elo_h + K * (actual_h - exp_h), 1)
        new_elo_a = round(elo_a + K * ((1 - actual_h) - (1 - exp_h)), 1)

        self._ensure_elo_table()
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            for name, elo in ((home, new_elo_h), (away, new_elo_a)):
                c.execute(
                    """
                    INSERT INTO team_elo (name, league, elo) VALUES (?,?,?)
                    ON CONFLICT(name) DO UPDATE SET
                        elo = excluded.elo,
                        league = CASE
                            WHEN excluded.league != '' THEN excluded.league
                            ELSE team_elo.league
                        END
                """,
                    (name, league, elo),
                )

    def _ensure_elo_table(self):
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS team_elo (
                    name    TEXT PRIMARY KEY,
                    league  TEXT NOT NULL DEFAULT '',
                    elo     REAL NOT NULL DEFAULT 1500.0
                )
            """)
