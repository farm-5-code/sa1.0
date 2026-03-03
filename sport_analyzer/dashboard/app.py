"""Sport Analyzer — Streamlit dashboard."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from sport_analyzer.analyzers.match_analyzer import MatchAnalyzer
from sport_analyzer.collectors.api_sports_collector import ApiSportsCollector
from sport_analyzer.collectors.news_collector import NewsCollector
from sport_analyzer.collectors.sports_collector import SportsCollector
from sport_analyzer.collectors.weather_collector import WeatherCollector
from sport_analyzer.config.settings import Config
from sport_analyzer.database.migrations import run_migrations
from sport_analyzer.utils.team_normalizer import normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sport_analyzer.dashboard")

PAGES = [
    "Анализ",
    "Расписание",
    "История",
    "Opportunities",
    "Insights",
    "Сигналы",
    "Диагностика",
]
SPORTS = ["Футбол", "Баскетбол"]

TOP_LEAGUE_KEYWORDS = {
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "champions league",
    "europa league",
    "europa conference league",
}


def init_state() -> None:
    st.session_state.setdefault("result", {})
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_run_at", "")


def ensure_db(path: str) -> None:
    try:
        run_migrations(path)
    except Exception as exc:  # pragma: no cover
        logger.exception("run_migrations failed: %s", exc)

    try:
        with sqlite3.connect(path, timeout=10) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT (datetime('now')),
                    sport TEXT DEFAULT 'football',
                    match TEXT,
                    datetime TEXT,
                    prediction TEXT,
                    confidence REAL,
                    analysis_json TEXT
                )
                """
            )
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()
            ]
            if "sport" not in cols:
                conn.execute(
                    "ALTER TABLE analyses ADD COLUMN sport TEXT DEFAULT 'football'"
                )
            conn.commit()
    except Exception as exc:  # pragma: no cover
        logger.exception("ensure_db failed: %s", exc)


def save_analysis(path: str, result: dict[str, Any], sport: str) -> None:
    try:
        match = result.get("match") or (
            f"{result.get('home_team', '')} vs {result.get('away_team', '')}"
        )
        dt_value = str(result.get("datetime", ""))
        prediction = str(result.get("best_pick", result.get("prediction", "")))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        raw = json.dumps(result, ensure_ascii=False)

        with sqlite3.connect(path, timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO analyses(sport, match, datetime, prediction, confidence, analysis_json)
                VALUES(?,?,?,?,?,?)
                """,
                (sport, match, dt_value, prediction, confidence, raw),
            )
            conn.commit()
    except Exception as exc:  # pragma: no cover
        logger.exception("save_analysis failed: %s", exc)


def load_analyses(path: str, limit: int = 300) -> pd.DataFrame:
    try:
        with sqlite3.connect(path, timeout=10) as conn:
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()
            ]
            if "sport" in cols:
                query = """
                SELECT id, created_at, sport, match, datetime, prediction, confidence
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
                """
            else:
                query = """
                SELECT id, created_at, match, datetime, prediction, confidence
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
                """
            return pd.read_sql_query(query, conn, params=(int(limit),))
    except Exception as exc:  # pragma: no cover
        logger.exception("load_analyses failed: %s", exc)
        return pd.DataFrame()


def render_result(result: dict[str, Any]) -> None:
    if not result:
        st.info("Пока нет результата. Запустите анализ.")
        return

    home = str(result.get("home_team", "Home"))
    away = str(result.get("away_team", "Away"))
    confidence = float(result.get("confidence", 0.0) or 0.0)

    st.markdown(f"## 🧾 Результат: **{home} vs {away}**")
    st.metric("Уверенность", f"{confidence:.1f}%")

    probs = result.get("final_probs") or {}
    if isinstance(probs, dict) and probs:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("🏠 Home")
            st.progress(float(probs.get("home_win", 0.0) or 0.0))
        with c2:
            st.write("🤝 Draw")
            st.progress(float(probs.get("draw", 0.0) or 0.0))
        with c3:
            st.write("✈️ Away")
            st.progress(float(probs.get("away_win", 0.0) or 0.0))

    recommendations = result.get("recommendations") or []
    if recommendations:
        with st.expander("💡 Рекомендации", expanded=True):
            for recommendation in recommendations:
                st.write(f"• {recommendation}")

    with st.expander("🔎 Raw (debug)"):
        st.json(result)


def _league_is_top(league_name: str) -> bool:
    raw = (league_name or "").strip().lower()
    return any(key in raw for key in TOP_LEAGUE_KEYWORDS)


def _collect_upcoming_football_matches(
    sports: SportsCollector,
    api: ApiSportsCollector,
    days: int,
    include_minor: bool,
) -> list[dict[str, Any]]:
    if api.is_configured():
        rows: list[dict[str, Any]] = []
        base_day = datetime.utcnow().date()
        for i in range(days):
            day = (base_day + timedelta(days=i)).isoformat()
            for m in api.get_fixtures_by_date(day, timezone="UTC", use_cache=True):
                league_name = str(m.get("league") or "")
                if not include_minor and not _league_is_top(league_name):
                    continue
                rows.append(
                    {
                        "id": m.get("fixture_id"),
                        "date": m.get("utcDate"),
                        "competition": league_name,
                        "home_team": m.get("home_team"),
                        "away_team": m.get("away_team"),
                        "home_team_id": m.get("home_team_id"),
                        "away_team_id": m.get("away_team_id"),
                    }
                )
        return rows

    rows = sports.get_matches(days_ahead=days) or []
    if include_minor:
        return rows
    return [m for m in rows if _league_is_top(str(m.get("competition") or ""))]


def _collect_upcoming_basketball_matches(days: int) -> list[dict[str, Any]]:
    try:
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.static import teams
    except Exception as exc:  # pragma: no cover
        logger.warning("nba_api unavailable: %s", exc)
        return []

    team_map = {int(t["id"]): str(t["full_name"]) for t in teams.get_teams()}
    rows: list[dict[str, Any]] = []
    base_day = datetime.utcnow().date()

    for i in range(days):
        d = base_day + timedelta(days=i)
        game_date = d.strftime("%m/%d/%Y")
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=game_date)
            headers = sb.game_header.get_dict().get("data", [])
        except Exception as exc:
            logger.warning("NBA scoreboard failed for %s: %s", game_date, exc)
            continue

        for g in headers:
            home_id = int(g[6]) if g[6] is not None else None
            away_id = int(g[7]) if g[7] is not None else None
            status = str(g[4] or "")
            rows.append(
                {
                    "id": g[2],
                    "date": str(g[0]),
                    "competition": "NBA",
                    "home_team": team_map.get(home_id or -1, str(home_id or "")),
                    "away_team": team_map.get(away_id or -1, str(away_id or "")),
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "status": status,
                }
            )

    return rows


def _analyze_basketball_match(home: str, away: str, match_dt: str) -> dict[str, Any]:
    return {
        "sport": "basketball",
        "home_team": home,
        "away_team": away,
        "datetime": match_dt,
        "prediction": "Победа будет определена после загрузки расширенной NBA-статистики",
        "best_pick": "Lean: Home",
        "confidence": 50.0,
        "recommendations": [
            "🏀 Баскетбольный модуль включён (beta).",
            "📊 Добавим продвинутый расчёт (pace/ORtg/DRtg/injuries) в следующих итерациях.",
        ],
    }


def page_analyze(analyzer: MatchAnalyzer, cfg: Config, selected_sport: str) -> None:
    st.title(f"🏆 Анализ матча ({selected_sport})")

    home = st.text_input(
        "Home", "Arsenal" if selected_sport == "Футбол" else "Boston Celtics"
    )
    away = st.text_input(
        "Away", "Chelsea" if selected_sport == "Футбол" else "Miami Heat"
    )
    city = st.text_input("Город (опционально)", "")

    c1, c2 = st.columns(2)
    with c1:
        match_date = st.date_input("Дата", value=date.today())
    with c2:
        match_time = st.time_input("Время (UTC)", value=datetime.utcnow().time())

    neutral = st.checkbox(
        "Нейтральное поле", value=False, disabled=selected_sport != "Футбол"
    )

    if st.button("🚀 Анализировать", type="primary"):
        st.session_state["last_error"] = ""
        try:
            match_dt = (
                datetime.combine(match_date, match_time)
                .replace(microsecond=0)
                .isoformat()
            )
            if selected_sport == "Футбол":
                kwargs: dict[str, Any] = {
                    "home_team": normalize_team_name(home),
                    "away_team": normalize_team_name(away),
                    "match_datetime": match_dt,
                    "neutral_field": neutral,
                }
                if city.strip():
                    kwargs["city"] = city.strip()
                result = analyzer.analyze_match(**kwargs)
                sport_key = "football"
            else:
                result = _analyze_basketball_match(home.strip(), away.strip(), match_dt)
                sport_key = "basketball"

            st.session_state["result"] = result
            st.session_state["last_run_at"] = datetime.utcnow().isoformat()
            save_analysis(cfg.DB_PATH, result, sport=sport_key)
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            logger.exception("analyze failed: %s", exc)

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    render_result(st.session_state.get("result") or {})


def page_schedule(
    sports: SportsCollector, api: ApiSportsCollector, selected_sport: str
) -> None:
    st.title(f"📅 Расписание ({selected_sport})")

    if selected_sport == "Футбол":
        c1, c2 = st.columns(2)
        with c1:
            days = st.slider("Дней вперёд", 1, 14, 7)
        with c2:
            include_minor = st.checkbox("Показывать небольшие турниры", value=True)

        with st.spinner("Загружаем матчи…"):
            matches = _collect_upcoming_football_matches(
                sports, api, days=days, include_minor=include_minor
            )
    else:
        days = st.slider("Дней вперёд", 1, 10, 3)
        with st.spinner("Загружаем NBA матчи…"):
            matches = _collect_upcoming_basketball_matches(days=days)

    if not matches:
        st.warning("Матчи не найдены. Проверьте ключи API и доступ к сети.")
        return

    leagues = sorted(
        {str(m.get("competition") or "") for m in matches if m.get("competition")}
    )
    selected_leagues = st.multiselect(
        "Фильтр турниров", options=leagues, default=leagues[:12]
    )

    filtered = [
        m
        for m in matches
        if not selected_leagues or str(m.get("competition") or "") in selected_leagues
    ]
    df = pd.DataFrame(filtered)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
            "%d.%m %H:%M"
        )
    st.caption(f"Найдено матчей: {len(filtered)}")
    st.dataframe(df, use_container_width=True)


def page_opportunities(
    analyzer: MatchAnalyzer,
    sports: SportsCollector,
    api: ApiSportsCollector,
    selected_sport: str,
) -> None:
    st.title(f"💎 Opportunities ({selected_sport})")

    if selected_sport != "Футбол":
        st.info(
            "Для баскетбола сейчас доступен базовый скан матчей без продвинутого рейтинга."
        )
        days = st.slider("Окно (дней)", 1, 7, 2)
        rows = _collect_upcoming_basketball_matches(days=days)
        if not rows:
            st.warning("NBA-матчи не удалось загрузить.")
            return
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
                "%d.%m %H:%M"
            )
        st.dataframe(df, use_container_width=True)
        return

    st.caption("Авто-скан футбольных матчей с фильтрацией по турнирам и confidence.")

    c1, c2, c3 = st.columns(3)
    with c1:
        days = st.slider("Окно (дней)", 1, 7, 2)
    with c2:
        limit_matches = st.slider("Лимит матчей", 5, 80, 20)
    with c3:
        include_minor = st.checkbox("Включая небольшие турниры", value=True)

    autoref = st.checkbox("Автообновление (30 сек)", value=False)
    if autoref:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore

            st_autorefresh(interval=30_000, key="opportunities_refresh")
        except Exception:
            st.warning(
                "Пакет streamlit-autorefresh не установлен. "
                "Используй кнопку «Запустить авто-скан» для ручного обновления."
            )

    if st.button("🚀 Запустить авто-скан", type="primary", use_container_width=True):
        st.session_state["op_rows"] = None

    cached = st.session_state.get("op_rows")
    if cached is None:
        with st.spinner("Сканируем матчи..."):
            matches = _collect_upcoming_football_matches(
                sports,
                api,
                days=days,
                include_minor=include_minor,
            )
            matches = matches[:limit_matches]

            rows: list[dict[str, Any]] = []
            for m in matches:
                home = str(m.get("home_team") or "")
                away = str(m.get("away_team") or "")
                if not home or not away:
                    continue

                try:
                    res = analyzer.analyze_match(
                        home_team=normalize_team_name(home),
                        away_team=normalize_team_name(away),
                        match_datetime=str(m.get("date") or ""),
                        home_team_id=m.get("home_team_id"),
                        away_team_id=m.get("away_team_id"),
                        competition=str(m.get("competition") or ""),
                    )
                except Exception as exc:
                    logger.warning("scan failed for %s vs %s: %s", home, away, exc)
                    continue

                rows.append(
                    {
                        "date": m.get("date"),
                        "league": m.get("competition"),
                        "match": f"{home} vs {away}",
                        "prediction": res.get("best_pick")
                        or res.get("prediction")
                        or "",
                        "confidence": float(res.get("confidence") or 0.0),
                    }
                )

            rows.sort(key=lambda x: x["confidence"], reverse=True)
            st.session_state["op_rows"] = rows
            st.session_state["op_meta"] = datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            cached = rows

    if not cached:
        st.info("Нет данных для сканирования. Измени фильтры и запусти скан ещё раз.")
        return

    threshold = st.slider("Мин. уверенность (%)", 40, 90, 55)
    filtered = [r for r in cached if float(r.get("confidence") or 0.0) >= threshold]

    leagues = sorted({str(r.get("league") or "") for r in filtered if r.get("league")})
    selected_leagues = st.multiselect("Турниры", options=leagues, default=leagues[:10])
    filtered = [
        r
        for r in filtered
        if not selected_leagues or str(r.get("league") or "") in selected_leagues
    ]

    st.caption(
        f"Последнее обновление: {st.session_state.get('op_meta', '—')} | матчей: {len(filtered)}"
    )
    out = pd.DataFrame(filtered)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime(
            "%d.%m %H:%M"
        )
    st.dataframe(out, use_container_width=True)


def page_history(cfg: Config) -> None:
    st.title("🕘 История анализов")
    limit = st.slider("Последних записей", 10, 500, 100, step=10)
    df = load_analyses(cfg.DB_PATH, limit=limit)
    if df.empty:
        st.info("История пока пустая.")
        return
    st.dataframe(df, use_container_width=True)


def page_diagnostics(cfg: Config, api: ApiSportsCollector) -> None:
    st.title("🧪 Диагностика")
    st.write(f"DB_PATH: `{cfg.DB_PATH}`")
    st.write("API-Sports configured:", "✅" if api.is_configured() else "❌")
    st.write("Last run:", st.session_state.get("last_run_at") or "—")
    if st.session_state.get("last_error"):
        st.code(st.session_state["last_error"])


def page_insights() -> None:
    st.title("🧠 Insights")
    st.info("Раздел в разработке.")


def page_signals(_api: ApiSportsCollector) -> None:
    st.title("📡 Сигналы")
    st.info("Раздел в разработке.")


def main() -> None:
    init_state()

    cfg = Config()
    ensure_db(cfg.DB_PATH)

    sports = SportsCollector(cfg)
    api = ApiSportsCollector(cfg)
    weather = WeatherCollector(cfg)
    news = NewsCollector(cfg)
    analyzer = MatchAnalyzer(cfg, sports=sports, weather=weather, news=news)

    st.sidebar.title("🏆 Sport Analyzer")
    selected_sport = st.sidebar.selectbox("Вид спорта", SPORTS)
    page = st.sidebar.radio("Раздел", PAGES)

    if page == "Анализ":
        page_analyze(analyzer, cfg, selected_sport)
    elif page == "Расписание":
        page_schedule(sports, api, selected_sport)
    elif page == "История":
        page_history(cfg)
    elif page == "Opportunities":
        page_opportunities(analyzer, sports, api, selected_sport)
    elif page == "Insights":
        page_insights()
    elif page == "Сигналы":
        page_signals(api)
    else:
        page_diagnostics(cfg, api)
