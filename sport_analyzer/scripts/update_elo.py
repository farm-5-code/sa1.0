import sys
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sport_analyzer.config.settings import Config
from sport_analyzer.collectors.sports_collector import SportsCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ELO] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10


class ELOUpdater:
    def __init__(self, config: Config):
        self.config = config
        self.sports = SportsCollector(config, db_path=config.DB_PATH)
        self.db_path = config.DB_PATH
        self._ensure_tables()

    def run(self, days_back: int = 3) -> int:
        logger.info(f"Запуск ELO-обновления (последние {days_back} дня)")
        matches = self._fetch_finished(days_back)
        processed = 0

        for m in matches:
            match_id = m.get("id")
            home_name = m.get("home_team")
            away_name = m.get("away_team")
            home_goals = m.get("home_goals")
            away_goals = m.get("away_goals")
            league = m.get("competition", "")

            if home_goals is None or away_goals is None:
                continue
            if self._already_updated(match_id):
                continue

            elo_bh = self.sports.get_elo(home_name)
            elo_ba = self.sports.get_elo(away_name)

            self.sports.update_elo(home_name, away_name, home_goals, away_goals, league)

            elo_ah = self.sports.get_elo(home_name)
            elo_aa = self.sports.get_elo(away_name)

            self._log_update(
                match_id,
                home_name,
                away_name,
                home_goals,
                away_goals,
                elo_bh,
                elo_ah,
                elo_ba,
                elo_aa,
            )
            logger.info(
                f"{home_name} {home_goals}:{away_goals} {away_name} | "
                f"{home_name}: {elo_bh:.0f}→{elo_ah:.0f} | "
                f"{away_name}: {elo_ba:.0f}→{elo_aa:.0f}"
            )
            processed += 1

        logger.info(f"Готово. Обработано: {processed}")
        return processed

    def _fetch_finished(self, days_back: int) -> list:
        date_from = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.utcnow().strftime("%Y-%m-%d")

        resp = self.sports.get(
            "https://api.football-data.org/v4/matches",
            params={"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
        )
        if resp is None or resp.status_code != 200:
            return []

        result = []
        for m in resp.json().get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            result.append(
                {
                    "id": m.get("id"),
                    "home_team": m.get("homeTeam", {}).get("name"),
                    "away_team": m.get("awayTeam", {}).get("name"),
                    "home_goals": score.get("home"),
                    "away_goals": score.get("away"),
                    "competition": m.get("competition", {}).get("name", ""),
                }
            )
        return result

    def _already_updated(self, match_id: int) -> bool:
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            return bool(
                c.execute(
                    "SELECT 1 FROM elo_updates WHERE match_id=?", (match_id,)
                ).fetchone()
            )

    def _log_update(self, match_id, home, away, hg, ag, ebh, eah, eba, eaa):
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            c.execute(
                """
                INSERT INTO elo_updates
                (match_id,home_team,away_team,home_goals,away_goals,
                 elo_before_home,elo_after_home,elo_before_away,elo_after_away,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    match_id,
                    home,
                    away,
                    hg,
                    ag,
                    ebh,
                    eah,
                    eba,
                    eaa,
                    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    def _ensure_tables(self):
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS elo_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER UNIQUE,
                    home_team TEXT, away_team TEXT,
                    home_goals INTEGER, away_goals INTEGER,
                    elo_before_home REAL, elo_after_home REAL,
                    elo_before_away REAL, elo_after_away REAL,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS team_elo (
                    name TEXT PRIMARY KEY,
                    league TEXT NOT NULL DEFAULT '',
                    elo REAL NOT NULL DEFAULT 1500.0
                );
            """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    ELOUpdater(Config()).run(days_back=args.days)
