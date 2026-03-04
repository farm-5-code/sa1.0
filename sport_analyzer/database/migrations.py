import sqlite3
import logging
from typing import List

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10


def _get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error:
        return []


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone())


def _ensure_migration_log(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """)


def _applied(conn: sqlite3.Connection, mid: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM _migrations WHERE id=?", (mid,)
    ).fetchone())


def _mark_applied(conn: sqlite3.Connection, mid: str):
    conn.execute("INSERT OR IGNORE INTO _migrations(id) VALUES(?)", (mid,))


def run_migrations(db_path: str):
    with sqlite3.connect(db_path, timeout=_CONNECT_TIMEOUT) as conn:
        _ensure_migration_log(conn)
        _m001_team_elo_add_league(conn)
        _m002_team_elo_add_team_id(conn)
        _m003_collector_cache_index(conn)
        _m004_elo_updates_indexes(conn)
        _m005_odds_snapshots(conn)
        _m006_steam_events(conn)
        conn.commit()
    logger.info("Миграции применены")


def _m001_team_elo_add_league(conn):
    mid = "m001_team_elo_add_league"
    if _applied(conn, mid):
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_elo (
            name TEXT PRIMARY KEY,
            league TEXT NOT NULL DEFAULT '',
            elo REAL NOT NULL DEFAULT 1500.0
        )
    """)
    if "league" not in _get_columns(conn, "team_elo"):
        conn.execute("ALTER TABLE team_elo ADD COLUMN league TEXT NOT NULL DEFAULT ''")
        logger.info("M001: добавлена колонка league")
    _mark_applied(conn, mid)


def _m002_team_elo_add_team_id(conn):
    mid = "m002_team_elo_add_team_id"
    if _applied(conn, mid):
        return
    if "team_id" not in _get_columns(conn, "team_elo"):
        conn.execute("ALTER TABLE team_elo ADD COLUMN team_id INTEGER DEFAULT NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS te_team_id ON team_elo(team_id)")
        logger.info("M002: добавлена колонка team_id")
    _mark_applied(conn, mid)


def _m003_collector_cache_index(conn):
    mid = "m003_collector_cache_index"
    if _applied(conn, mid):
        return
    if not _table_exists(conn, "collector_cache"):
        logger.debug("M003: таблица ещё не создана — отложено")
        return
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS cc_ts ON collector_cache(ts)")
        logger.info("M003: индекс создан")
        _mark_applied(conn, mid)
    except sqlite3.Error as e:
        logger.error(f"M003: ошибка создания индекса: {e}")


def _m004_elo_updates_indexes(conn):
    mid = "m004_elo_updates_indexes"
    if _applied(conn, mid):
        return
    if not _table_exists(conn, "elo_updates"):
        logger.debug("M004: таблица ещё не создана — отложено")
        return
    try:
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS eu_match_id ON elo_updates(match_id);
            CREATE INDEX IF NOT EXISTS eu_teams ON elo_updates(home_team, away_team);
        """)
        logger.info("M004: индексы созданы")
        _mark_applied(conn, mid)
    except sqlite3.Error as e:
        logger.error(f"M004: ошибка создания индексов: {e}")


def _m005_odds_snapshots(conn):
    mid = "m005_odds_snapshots"
    if _applied(conn, mid):
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            ts          REAL,
            fixture_id  INTEGER,
            market      TEXT,
            selection   TEXT,
            best_odd    REAL,
            avg_odd     REAL,
            books       INTEGER,
            PRIMARY KEY (ts, fixture_id, market, selection)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_fx_ts ON odds_snapshots(fixture_id, ts)")
    logger.info("M005: odds_snapshots ready")
    _mark_applied(conn, mid)


def _m006_steam_events(conn):
    mid = "m006_steam_events"
    if _applied(conn, mid):
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS steam_events (
            ts           REAL,
            fixture_id   INTEGER,
            market       TEXT,
            selection    TEXT,
            window_min   INTEGER,
            best_start   REAL,
            best_last    REAL,
            best_chg_pct REAL,
            books        INTEGER,
            steam_score  REAL,
            PRIMARY KEY (ts, fixture_id, market, selection, window_min)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steam_fx_ts ON steam_events(fixture_id, ts)")
    logger.info("M006: steam_events ready")
    _mark_applied(conn, mid)
