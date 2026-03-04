import logging
import sqlite3
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from sport_analyzer.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

_API_BASE = "https://v3.football.api-sports.io"


class ApiSportsCollector(BaseCollector):
    """API-Sports (API-Football) collector.

    Требует переменную окружения/API Secret:
      API_FOOTBALL_KEY
    """

    RATE_LIMIT_PER_MINUTE = 25  # мягкий лимит; можно поднять при необходимости

    def __init__(self, config, db_path: str = "sport_analyzer.db"):
        super().__init__(db_path=db_path)
        self.config = config

        key = (getattr(config, "API_FOOTBALL_KEY", "") or "").strip()
        if key:
            # API-Sports header
            self.session.headers["x-apisports-key"] = key
            # На случай RapidAPI-ключа (иногда пользователи кладут его сюда же)
            self.session.headers.setdefault("x-rapidapi-key", key)

    def is_configured(self) -> bool:
        return bool((getattr(self.config, "API_FOOTBALL_KEY", "") or "").strip())

    # ── Fixtures ──────────────────────────────────────────────────────

    def get_fixtures_by_date(
        self,
        date_yyyy_mm_dd: str,
        league: Optional[int] = None,
        season: Optional[int] = None,
        timezone: str = "UTC",
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Список матчей на дату (UTC по умолчанию)."""
        cache_key = f"apisports_fixtures_{date_yyyy_mm_dd}_{league}_{season}_{timezone}"
        if use_cache:
            cached = self._cache_get(cache_key, max_age_hours=0.25)  # 15 минут
            if cached and isinstance(cached, dict) and "items" in cached:
                return cached["items"]

        params: Dict[str, Any] = {"date": date_yyyy_mm_dd, "timezone": timezone}
        if league:
            params["league"] = int(league)
        if season:
            params["season"] = int(season)

        resp = self.get(f"{_API_BASE}/fixtures", params=params)
        if resp is None or resp.status_code != 200:
            return []

        data = resp.json() or {}
        items = []
        for it in data.get("response", []) or []:
            fx = it.get("fixture", {}) or {}
            teams = it.get("teams", {}) or {}
            league_info = it.get("league", {}) or {}

            items.append(
                {
                    "fixture_id": fx.get("id"),
                    "utcDate": fx.get("date"),
                    "timestamp": fx.get("timestamp"),
                    "status": (fx.get("status", {}) or {}).get("short"),
                    "league": league_info.get("name"),
                    "league_id": league_info.get("id"),
                    "season": league_info.get("season"),
                    "round": league_info.get("round"),
                    "home_team": (teams.get("home", {}) or {}).get("name"),
                    "away_team": (teams.get("away", {}) or {}).get("name"),
                    "home_team_id": (teams.get("home", {}) or {}).get("id"),
                    "away_team_id": (teams.get("away", {}) or {}).get("id"),
                }
            )

        if use_cache:
            self._cache_set(cache_key, {"items": items})
        return items

    # ── Odds ─────────────────────────────────────────────────────────

    def get_odds_for_fixture(
        self,
        fixture_id: int,
        bookmakers_limit: int = 25,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Коэфы по матчу.

        Возвращает упрощённую структуру для 3 рынков:
          - 1X2 (Match Winner)  bet_id=1
          - O/U 2.5            bet_id=5 (подвариант 2.5)
          - BTTS               bet_id=8
        """
        cache_key = f"apisports_odds_{fixture_id}"
        if use_cache:
            cached = self._cache_get(cache_key, max_age_hours=0.10)  # 6 минут
            if cached and isinstance(cached, dict) and "markets" in cached:
                return cached

        # API-Sports: odds endpoint
        resp = self.get(f"{_API_BASE}/odds", params={"fixture": int(fixture_id)})
        if resp is None or resp.status_code != 200:
            return {"fixture_id": fixture_id, "markets": {}}

        data = resp.json() or {}
        response = (data.get("response") or [])
        if not response:
            out = {"fixture_id": fixture_id, "markets": {}}
            if use_cache:
                self._cache_set(cache_key, out)
            return out

        # Обычно response[0] относится к fixture, внутри bookmakers -> bets -> values
        root = response[0] or {}
        bookmakers = (root.get("bookmakers") or [])[:bookmakers_limit]

        # helper: collect odds for a given bet_id and optional value filter
        def collect(bet_id: int, value_filter: Optional[List[str]] = None) -> Dict[str, List[float]]:
            out: Dict[str, List[float]] = {}
            for b in bookmakers:
                bets = b.get("bets") or []
                for bet in bets:
                    if bet.get("id") != bet_id:
                        continue
                    for v in bet.get("values") or []:
                        label = str(v.get("value"))
                        odd = v.get("odd")
                        try:
                            odd_f = float(odd)
                        except Exception:
                            continue
                        if value_filter and label not in value_filter:
                            continue
                        out.setdefault(label, []).append(odd_f)
            return out

        markets: Dict[str, Any] = {}

        # 1X2
        mw = collect(1, value_filter=["Home", "Draw", "Away"])
        markets["1X2"] = mw

        # O/U 2.5: bet_id=5 (values like 'Over 2.5', 'Under 2.5')
        ou = collect(5, value_filter=["Over 2.5", "Under 2.5"])
        markets["OU_2_5"] = ou

        # BTTS: bet_id=8 (values 'Yes', 'No')
        btts = collect(8, value_filter=["Yes", "No"])
        markets["BTTS"] = btts

        out = {"fixture_id": fixture_id, "markets": markets}
        if use_cache:
            self._cache_set(cache_key, out)
        return out

# ── Odds history (snapshots) ─────────────────────────────────────

def should_take_snapshot(self, fixture_id: int, interval_seconds: int = 600) -> bool:
    """True если последний snapshot старше interval_seconds."""
    try:
        with sqlite3.connect(self.db_path, timeout=10) as c:
            row = c.execute(
                "SELECT MAX(ts) FROM odds_snapshots WHERE fixture_id=?",
                (int(fixture_id),),
            ).fetchone()
        last = float(row[0]) if row and row[0] is not None else 0.0
        return (time.time() - last) >= float(interval_seconds)
    except Exception:
        # если БД ещё не мигрирована — попробуем снять
        return True

def save_snapshot_from_odds(self, fixture_id: int, odds_payload: Dict[str, Any]):
    """Сохраняет упрощённый snapshot: best/avg/books для рынков 1X2, OU_2_5, BTTS."""
    markets = (odds_payload or {}).get("markets") or {}
    now = time.time()
    rows = []
    for market, values in markets.items():
        for selection, odds in (values or {}).items():
            if not odds:
                continue
            odds_f = []
            for x in odds:
                try:
                    odds_f.append(float(x))
                except Exception:
                    continue
            if not odds_f:
                continue
            rows.append((
                now,
                int(fixture_id),
                str(market),
                str(selection),
                float(max(odds_f)),
                float(sum(odds_f) / len(odds_f)),
                int(len(odds_f)),
            ))

    if not rows:
        return

    with sqlite3.connect(self.db_path, timeout=10) as c:
        c.executemany(
            "INSERT OR REPLACE INTO odds_snapshots (ts, fixture_id, market, selection, best_odd, avg_odd, books) VALUES (?,?,?,?,?,?,?)",
            rows,
        )

def get_snapshot_history(self, fixture_id: int, hours: float = 24.0) -> List[Dict[str, Any]]:
    """История snapshot'ов по матчу за последние hours."""
    since = time.time() - float(hours) * 3600.0
    try:
        with sqlite3.connect(self.db_path, timeout=10) as c:
            cur = c.execute(
                "SELECT ts, fixture_id, market, selection, best_odd, avg_odd, books FROM odds_snapshots WHERE fixture_id=? AND ts>=? ORDER BY ts DESC",
                (int(fixture_id), since),
            )
            out = []
            for ts_, fx, market, sel, best, avg, books in cur.fetchall():
                out.append({
                    "ts": float(ts_),
                    "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(ts_))),
                    "fixture_id": int(fx),
                    "market": market,
                    "selection": sel,
                    "best_odd": float(best) if best is not None else None,
                    "avg_odd": float(avg) if avg is not None else None,
                    "books": int(books) if books is not None else 0,
                })
            return out
    except Exception:
        return []

def save_steam_events(self, events: List[Dict[str, Any]]):
    """Сохраняет события steam в таблицу steam_events."""
    if not events:
        return
    rows = []
    for e in events:
        try:
            rows.append((
                float(e["ts"]),
                int(e["fixture_id"]),
                str(e["market"]),
                str(e["selection"]),
                int(e["window_min"]),
                float(e.get("best_start") or 0.0),
                float(e.get("best_last") or 0.0),
                float(e.get("best_chg_pct") or 0.0),
                int(e.get("books") or 0),
                float(e.get("steam_score") or 0.0),
            ))
        except Exception:
            continue
    if not rows:
        return
    with sqlite3.connect(self.db_path, timeout=10) as c:
        c.executemany(
            "INSERT OR IGNORE INTO steam_events (ts, fixture_id, market, selection, window_min, best_start, best_last, best_chg_pct, books, steam_score) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

def get_steam_events(self, fixture_id: int, hours: float = 24.0) -> List[Dict[str, Any]]:
    since = time.time() - float(hours) * 3600.0
    try:
        with sqlite3.connect(self.db_path, timeout=10) as c:
            cur = c.execute(
                "SELECT ts, fixture_id, market, selection, window_min, best_start, best_last, best_chg_pct, books, steam_score "
                "FROM steam_events WHERE fixture_id=? AND ts>=? ORDER BY ts DESC",
                (int(fixture_id), since),
            )
            out=[]
            for ts_, fx, market, sel, w, bs, bl, pct, books, score in cur.fetchall():
                out.append({
                    "ts": float(ts_),
                    "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(ts_))),
                    "fixture_id": int(fx),
                    "market": market,
                    "selection": sel,
                    "window_min": int(w),
                    "best_start": float(bs),
                    "best_last": float(bl),
                    "best_chg_pct": float(pct),
                    "books": int(books),
                    "steam_score": float(score),
                })
            return out
    except Exception:
        return []

# ── Team context / fatigue ───────────────────────────────────────

def get_team_recent_fixtures(self, team_id: int, last: int = 20, timezone: str = "UTC", use_cache: bool = True) -> List[Dict[str, Any]]:
    """Последние матчи команды (API-Sports)."""
    cache_key = f"apisports_team_last_{team_id}_{last}_{timezone}"
    if use_cache:
        cached = self._cache_get(cache_key, max_age_hours=0.5)  # 30 минут
        if cached and isinstance(cached, dict) and "items" in cached:
            return cached["items"]

    resp = self.get(f"{_API_BASE}/fixtures", params={"team": int(team_id), "last": int(last), "timezone": timezone})
    if resp is None or resp.status_code != 200:
        return []
    data = resp.json() or {}
    items=[]
    for it in data.get("response", []) or []:
        fx = it.get("fixture", {}) or {}
        league_info = it.get("league", {}) or {}
        teams = it.get("teams", {}) or {}
        items.append({
            "fixture_id": fx.get("id"),
            "utcDate": fx.get("date"),
            "timestamp": fx.get("timestamp"),
            "status": (fx.get("status", {}) or {}).get("short"),
            "league": league_info.get("name"),
            "league_id": league_info.get("id"),
            "season": league_info.get("season"),
            "home_team": (teams.get("home", {}) or {}).get("name"),
            "away_team": (teams.get("away", {}) or {}).get("name"),
            "goals_home": (it.get("goals", {}) or {}).get("home"),
            "goals_away": (it.get("goals", {}) or {}).get("away"),
        })
    if use_cache:
        self._cache_set(cache_key, {"items": items})
    return items

def compute_fatigue_metrics(self, team_id: int, match_utc_iso: str, timezone: str = "UTC") -> Dict[str, Any]:
    """Базовые метрики нагрузки/усталости:
    - rest_days: дней отдыха с предыдущего матча
    - matches_7d / matches_14d: количество матчей за 7/14 дней до матча
    - last_match_utc: дата предыдущего матча
    """
    try:
        match_dt = datetime.fromisoformat(str(match_utc_iso).replace("Z", "+00:00"))
    except Exception:
        match_dt = datetime.utcnow()

    # берем последние 20 игр и считаем окна
    fixtures = self.get_team_recent_fixtures(int(team_id), last=25, timezone=timezone, use_cache=True)
    # timestamps of played games before match_dt
    played_ts=[]
    for f in fixtures:
        try:
            dt = datetime.fromisoformat(str(f.get("utcDate")).replace("Z", "+00:00"))
        except Exception:
            continue
        # only before match
        if dt < match_dt:
            played_ts.append(dt)

    played_ts.sort()
    last_match = played_ts[-1] if played_ts else None
    rest_days = None
    if last_match:
        rest_days = (match_dt - last_match).total_seconds() / 86400.0

    def count_within(days: int) -> int:
        cutoff = match_dt - timedelta(days=days)
        return sum(1 for dt in played_ts if dt >= cutoff)

    out = {
        "team_id": int(team_id),
        "match_utc": match_dt.isoformat(),
        "last_match_utc": last_match.isoformat() if last_match else None,
        "rest_days": round(rest_days, 2) if rest_days is not None else None,
        "matches_7d": int(count_within(7)),
        "matches_14d": int(count_within(14)),
        "matches_30d": int(count_within(30)),
    }
    return out

def compute_team_form_metrics(self, team_id: int, match_utc_iso: str, last: int = 10, timezone: str = "UTC") -> Dict[str, Any]:
    """Форма команды по последним last матчам до match_utc_iso."""
    try:
        match_dt = datetime.fromisoformat(str(match_utc_iso).replace("Z", "+00:00"))
    except Exception:
        match_dt = datetime.utcnow()

    fixtures = self.get_team_recent_fixtures(int(team_id), last=max(25, last*3), timezone=timezone, use_cache=True)

    games=[]
    for f in fixtures:
        try:
            dt = datetime.fromisoformat(str(f.get("utcDate")).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= match_dt:
            continue
        gh = f.get("goals_home")
        ga = f.get("goals_away")
        if gh is None or ga is None:
            continue
        h_id = f.get("home_team_id")
        a_id = f.get("away_team_id")
        if h_id is None or a_id is None:
            continue

        is_home = int(h_id) == int(team_id)
        if not is_home and int(a_id) != int(team_id):
            continue

        gf = int(gh) if is_home else int(ga)
        ga_ = int(ga) if is_home else int(gh)

        res = "D"
        if gf > ga_:
            res = "W"
        elif gf < ga_:
            res = "L"

        games.append({
            "dt": dt,
            "is_home": is_home,
            "gf": gf,
            "ga": ga_,
            "res": res,
        })

    games = sorted(games, key=lambda x: x["dt"])[-int(last):]
    n = len(games)
    if n == 0:
        return {"team_id": int(team_id), "n": 0}

    w = sum(1 for g in games if g["res"]=="W")
    d = sum(1 for g in games if g["res"]=="D")
    l = sum(1 for g in games if g["res"]=="L")
    pts = 3*w + d
    gf = sum(g["gf"] for g in games)
    ga_ = sum(g["ga"] for g in games)
    cs = sum(1 for g in games if g["ga"]==0)
    btts = sum(1 for g in games if (g["gf"]>0 and g["ga"]>0))
    over25 = sum(1 for g in games if (g["gf"]+g["ga"]>=3))

    home_games = [g for g in games if g["is_home"]]
    away_games = [g for g in games if not g["is_home"]]

    def pack(arr):
        if not arr:
            return {"n": 0}
        w_ = sum(1 for g in arr if g["res"]=="W")
        d_ = sum(1 for g in arr if g["res"]=="D")
        l_ = sum(1 for g in arr if g["res"]=="L")
        gf_ = sum(g["gf"] for g in arr)
        ga__ = sum(g["ga"] for g in arr)
        return {
            "n": len(arr),
            "W": int(w_), "D": int(d_), "L": int(l_),
            "pts": int(3*w_ + d_),
            "GF": int(gf_), "GA": int(ga__),
            "avg_GF": round(gf_/len(arr), 2),
            "avg_GA": round(ga__/len(arr), 2),
        }

    out = {
        "team_id": int(team_id),
        "n": int(n),
        "W": int(w),
        "D": int(d),
        "L": int(l),
        "pts": int(pts),
        "GF": int(gf),
        "GA": int(ga_),
        "avg_GF": round(gf/n, 2),
        "avg_GA": round(ga_/n, 2),
        "clean_sheets": int(cs),
        "btts_rate": round(btts/n, 3),
        "over2_5_rate": round(over25/n, 3),
        "home": pack(home_games),
        "away": pack(away_games),
        "last_match_utc": games[-1]["dt"].isoformat(),
    }
    return out

def get_h2h(self, home_team_id: int, away_team_id: int, last: int = 10, timezone: str = "UTC", use_cache: bool = True) -> List[Dict[str, Any]]:
    """Head-to-head матчи между командами."""
    cache_key = f"apisports_h2h_{home_team_id}_{away_team_id}_{last}_{timezone}"
    if use_cache:
        cached = self._cache_get(cache_key, max_age_hours=6)
        if cached and isinstance(cached, dict) and "items" in cached:
            return cached["items"]

    resp = self.get(f"{_API_BASE}/fixtures/headtohead", params={"h2h": f"{int(home_team_id)}-{int(away_team_id)}", "last": int(last), "timezone": timezone})
    if resp is None or resp.status_code != 200:
        return []
    data = resp.json() or {}
    items=[]
    for it in data.get("response", []) or []:
        fx = it.get("fixture", {}) or {}
        teams = it.get("teams", {}) or {}
        league_info = it.get("league", {}) or {}
        goals = it.get("goals", {}) or {}
        items.append({
            "utcDate": fx.get("date"),
            "league": league_info.get("name"),
            "home": (teams.get("home", {}) or {}).get("name"),
            "away": (teams.get("away", {}) or {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
        })
    if use_cache:
        self._cache_set(cache_key, {"items": items})
    return items

# ── Lineups / injuries ───────────────────────────────────────────

def get_fixture_lineups(self, fixture_id: int, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Lineups for a fixture (if available)."""
    cache_key = f"apisports_lineups_{fixture_id}"
    if use_cache:
        cached = self._cache_get(cache_key, max_age_hours=0.25)  # 15 min
        if cached and isinstance(cached, dict) and "items" in cached:
            return cached["items"]

    resp = self.get(f"{_API_BASE}/fixtures/lineups", params={"fixture": int(fixture_id)})
    if resp is None or resp.status_code != 200:
        return []
    data = resp.json() or {}
    items = data.get("response") or []
    # Keep only essentials
    out=[]
    for it in items:
        team = it.get("team") or {}
        out.append({
            "team": {"id": team.get("id"), "name": team.get("name")},
            "formation": it.get("formation"),
            "coach": (it.get("coach") or {}).get("name"),
            "startXI": [
                {
                    "name": (p.get("player") or {}).get("name"),
                    "number": (p.get("player") or {}).get("number"),
                    "pos": (p.get("player") or {}).get("pos"),
                    "grid": (p.get("player") or {}).get("grid"),
                }
                for p in (it.get("startXI") or [])
            ],
            "substitutes": [
                {
                    "name": (p.get("player") or {}).get("name"),
                    "number": (p.get("player") or {}).get("number"),
                    "pos": (p.get("player") or {}).get("pos"),
                }
                for p in (it.get("substitutes") or [])
            ],
        })
    if use_cache:
        self._cache_set(cache_key, {"items": out})
    return out

def get_fixture_injuries(self, fixture_id: int, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Injuries for a fixture (if API provides)."""
    cache_key = f"apisports_injuries_{fixture_id}"
    if use_cache:
        cached = self._cache_get(cache_key, max_age_hours=1.0)
        if cached and isinstance(cached, dict) and "items" in cached:
            return cached["items"]

    resp = self.get(f"{_API_BASE}/injuries", params={"fixture": int(fixture_id)})
    if resp is None or resp.status_code != 200:
        return []
    data = resp.json() or {}
    items = data.get("response") or []
    out=[]
    for it in items:
        team = it.get("team") or {}
        player = it.get("player") or {}
        out.append({
            "team": {"id": team.get("id"), "name": team.get("name")},
            "player": {"id": player.get("id"), "name": player.get("name")},
            "type": it.get("type"),
            "reason": it.get("reason"),
        })
    if use_cache:
        self._cache_set(cache_key, {"items": out})
    return out

# ── Standings / motivation ───────────────────────────────────────

def get_standings(self, league_id: int, season: int, use_cache: bool = True) -> Dict[str, Any]:
    """Standings for league+season."""
    cache_key = f"apisports_standings_{league_id}_{season}"
    if use_cache:
        cached = self._cache_get(cache_key, max_age_hours=6)
        if cached and isinstance(cached, dict) and "data" in cached:
            return cached["data"]

    resp = self.get(f"{_API_BASE}/standings", params={"league": int(league_id), "season": int(season)})
    if resp is None or resp.status_code != 200:
        return {}
    data = resp.json() or {}
    payload = (data.get("response") or [])
    out = payload[0] if payload else {}
    if use_cache:
        self._cache_set(cache_key, {"data": out})
    return out

def compute_motivation_metrics(self, standings_payload: Dict[str, Any], team_id: int) -> Dict[str, Any]:
    """Extract table position, points, goals, and distance to key zones."""
    try:
        leagues = standings_payload.get("league") or {}
        standings = (leagues.get("standings") or [[]])[0]  # main table
    except Exception:
        standings = []

    row = None
    for r in standings or []:
        if int((r.get("team") or {}).get("id") or 0) == int(team_id):
            row = r
            break
    if not row:
        return {"team_id": int(team_id), "found": False}

    rank = int(row.get("rank") or 0)
    pts = int((row.get("points") or 0))
    played = int((row.get("all") or {}).get("played") or 0)
    gf = int((row.get("all") or {}).get("goals", {}).get("for") or 0)
    ga = int((row.get("all") or {}).get("goals", {}).get("against") or 0)
    gd = int(row.get("goalsDiff") or (gf-ga))

    # define key targets (heuristics):
    # - top4 (UCL-ish), top6 (Europe-ish), bottom3 (relegation)
    def points_at(pos: int) -> int:
        for r in standings or []:
            if int(r.get("rank") or 0) == int(pos):
                return int(r.get("points") or 0)
        return pts

    top4_pts = points_at(4)
    top6_pts = points_at(6)
    bottom3_rank = max(1, len(standings) - 2)
    releg_pts = points_at(bottom3_rank)

    dist_top4 = top4_pts - pts
    dist_top6 = top6_pts - pts
    dist_releg = pts - releg_pts

    # simple motivation flags
    near_top4 = dist_top4 <= 3 and rank > 4
    near_top6 = dist_top6 <= 3 and rank > 6
    near_releg = dist_releg <= 3 and rank < bottom3_rank

    return {
        "team_id": int(team_id),
        "found": True,
        "rank": rank,
        "points": pts,
        "played": played,
        "GF": gf,
        "GA": ga,
        "GD": gd,
        "dist_top4": int(dist_top4),
        "dist_top6": int(dist_top6),
        "dist_releg": int(dist_releg),
        "near_top4": bool(near_top4),
        "near_top6": bool(near_top6),
        "near_releg": bool(near_releg),
    }

def compute_trend_streaks(self, team_id: int, match_utc_iso: str, last: int = 10, timezone: str = "UTC") -> Dict[str, Any]:
    """Серии/тренды по последним матчам (до match_utc_iso).
    Возвращает длины текущих streaks (до первого разрыва).
    """
    form = self.compute_team_form_metrics(team_id=int(team_id), match_utc_iso=match_utc_iso, last=int(last), timezone=timezone)
    # For streaks we need ordered games; reuse internal data by re-fetching and building list
    try:
        match_dt = datetime.fromisoformat(str(match_utc_iso).replace("Z", "+00:00"))
    except Exception:
        match_dt = datetime.utcnow()

    fixtures = self.get_team_recent_fixtures(int(team_id), last=max(30, last*4), timezone=timezone, use_cache=True)
    games=[]
    for f in fixtures:
        try:
            dt = datetime.fromisoformat(str(f.get("utcDate")).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt >= match_dt:
            continue
        gh = f.get("goals_home")
        ga = f.get("goals_away")
        if gh is None or ga is None:
            continue
        h_id = f.get("home_team_id")
        a_id = f.get("away_team_id")
        if h_id is None or a_id is None:
            continue

        is_home = int(h_id) == int(team_id)
        if not is_home and int(a_id) != int(team_id):
            continue

        gf = int(gh) if is_home else int(ga)
        ga_ = int(ga) if is_home else int(gh)

        res = "D"
        if gf > ga_:
            res = "W"
        elif gf < ga_:
            res = "L"

        games.append({
            "dt": dt,
            "gf": gf,
            "ga": ga_,
            "res": res,
            "is_home": is_home,
            "total": gf + ga_,
        })

    games = sorted(games, key=lambda x: x["dt"], reverse=True)[:int(last)]  # most recent first
    if not games:
        return {"team_id": int(team_id), "n": 0}

    def streak(predicate):
        s=0
        for g in games:
            if predicate(g):
                s += 1
            else:
                break
        return s

    out = {
        "team_id": int(team_id),
        "n": int(len(games)),
        "streak_scoring": int(streak(lambda g: g["gf"] > 0)),
        "streak_conceding": int(streak(lambda g: g["ga"] > 0)),
        "streak_clean_sheets": int(streak(lambda g: g["ga"] == 0)),
        "streak_btts": int(streak(lambda g: g["gf"] > 0 and g["ga"] > 0)),
        "streak_over2_5": int(streak(lambda g: g["total"] >= 3)),
        "streak_under2_5": int(streak(lambda g: g["total"] <= 2)),
        "streak_unbeaten": int(streak(lambda g: g["res"] in ("W","D"))),
        "streak_winless": int(streak(lambda g: g["res"] in ("L","D"))),
        "streak_wins": int(streak(lambda g: g["res"] == "W")),
        "streak_losses": int(streak(lambda g: g["res"] == "L")),
    }
    # add simple momentum: last5 points
    pts_last5 = 0
    for g in games[:5]:
        pts_last5 += 3 if g["res"]=="W" else (1 if g["res"]=="D" else 0)
    out["points_last5"] = int(pts_last5)
    return out

# ── Travel / away-run metrics ────────────────────────────────────
def compute_travel_metrics(self, team_id: int, match_utc_iso: str, last: int = 10, timezone: str = "UTC") -> Dict[str, Any]:
    """Оценка нагрузки поездок:
    - away_streak: подряд выездные матчи
    - home_streak: подряд домашние матчи
    - away_ratio_lastN
    - venue_switches (частота смены home/away)
    """
    try:
        match_dt = datetime.fromisoformat(str(match_utc_iso).replace("Z","+00:00"))
    except Exception:
        match_dt = datetime.utcnow()

    fixtures = self.get_team_recent_fixtures(int(team_id), last=max(25,last*3), timezone=timezone, use_cache=True)

    games=[]
    for f in fixtures:
        try:
            dt = datetime.fromisoformat(str(f.get("utcDate")).replace("Z","+00:00"))
        except Exception:
            continue
        if dt >= match_dt:
            continue

        h_id = f.get("home_team_id")
        a_id = f.get("away_team_id")
        if h_id is None or a_id is None:
            continue

        if int(h_id)==int(team_id):
            venue="home"
        elif int(a_id)==int(team_id):
            venue="away"
        else:
            continue

        games.append({"dt":dt,"venue":venue})

    games = sorted(games,key=lambda x:x["dt"], reverse=True)[:int(last)]
    if not games:
        return {"team_id":int(team_id),"n":0}

    def streak(kind):
        s=0
        for g in games:
            if g["venue"]==kind:
                s+=1
            else:
                break
        return s

    away_count=sum(1 for g in games if g["venue"]=="away")
    switches=0
    for i in range(1,len(games)):
        if games[i]["venue"]!=games[i-1]["venue"]:
            switches+=1

    return {
        "team_id":int(team_id),
        "n":len(games),
        "away_streak":streak("away"),
        "home_streak":streak("home"),
        "away_ratio_lastN":round(away_count/len(games),3),
        "venue_switches":int(switches),
    }

