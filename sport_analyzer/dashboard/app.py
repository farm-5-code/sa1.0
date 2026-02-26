"""
Sport Analyzer — Streamlit Dashboard (incremental build)

Правило сборки:
- Ты вставляешь следующий блок кода ВСЕГДА в самый конец файла.
- Мы не правим середину, не ищем/заменяем.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, date
from typing import Any, Dict

import pandas as pd
import streamlit as st

from sport_analyzer.config.settings import Config
from sport_analyzer.collectors.sports_collector import SportsCollector
from sport_analyzer.collectors.api_sports_collector import ApiSportsCollector
from sport_analyzer.collectors.weather_collector import WeatherCollector
from sport_analyzer.collectors.news_collector import NewsCollector
from sport_analyzer.analyzers.match_analyzer import MatchAnalyzer
from sport_analyzer.database.migrations import run_migrations
from sport_analyzer.utils.team_normalizer import normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sport_analyzer.dashboard")


# ---------- STATE ----------
def init_state():
    st.session_state.setdefault("result", {})
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_run_at", "")


# ---------- DB ----------
def ensure_db(path: str):
    # project migrations (safe)
    try:
        run_migrations(path)
    except Exception as e:
        logger.exception("run_migrations failed: %s", e)

    # minimal table for history
    try:
        with sqlite3.connect(path, timeout=10) as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT (datetime('now')),
                    match TEXT,
                    datetime TEXT,
                    prediction TEXT,
                    confidence REAL,
                    analysis_json TEXT
                )
                """
            )
            c.commit()
    except Exception as e:
        logger.exception("ensure_db failed: %s", e)


def save_analysis(path: str, result: Dict[str, Any]):
    try:
        match = result.get("match") or f"{result.get('home_team','')} vs {result.get('away_team','')}"
        dt = str(result.get("datetime", ""))
        pred = str(result.get("best_pick", result.get("prediction", "")))
        conf = float(result.get("confidence", 0.0) or 0.0)
        raw = json.dumps(result, ensure_ascii=False)

        with sqlite3.connect(path, timeout=10) as c:
            c.execute(
                """
                INSERT INTO analyses(match, datetime, prediction, confidence, analysis_json)
                VALUES(?,?,?,?,?)
                """,
                (match, dt, pred, conf, raw),
            )
            c.commit()
    except Exception as e:
        logger.exception("save_analysis failed: %s", e)


def load_analyses(path: str, limit: int = 300) -> pd.DataFrame:
    try:
        with sqlite3.connect(path, timeout=10) as c:
            return pd.read_sql_query(
                f"""
                SELECT id, created_at, match, datetime, prediction, confidence
                FROM analyses
                ORDER BY id DESC
                LIMIT {int(limit)}
                """,
                c,
            )
    except Exception as e:
        logger.exception("load_analyses failed: %s", e)
        return pd.DataFrame()


# ---------- UI helpers ----------
def render_result(result: Dict[str, Any]):
    if not result:
        st.info("Пока нет результата. Запусти анализ.")
        return

    home = str(result.get("home_team", "Home"))
    away = str(result.get("away_team", "Away"))
    conf = float(result.get("confidence", 0.0) or 0.0)

    st.markdown(f"## 🧾 Результат: **{home} vs {away}**")
    st.metric("Уверенность", f"{conf:.1f}%")

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

    recs = result.get("recommendations") or []
    if recs:
        with st.expander("💡 Рекомендации", expanded=True):
            for r in recs:
                st.write(f"• {r}")

    with st.expander("🔎 Raw (debug)", expanded=False):
        st.json(result)


# ---------- PAGES (stubs for now) ----------
def page_analyze(analyzer: MatchAnalyzer, sports: SportsCollector, cfg: Config, api: ApiSportsCollector):
    st.title("🏆 Анализ (скелет)")
    st.caption("Дальше добавим выбор матча, odds, value и т.д.")

    home = st.text_input("Home", "Arsenal")
    away = st.text_input("Away", "Chelsea")
    match_date = st.date_input("Дата", value=date.today())
    match_time = st.time_input("Время (UTC)", value=datetime.utcnow().time())

    if st.button("🚀 Анализировать", type="primary"):
        st.session_state["last_error"] = ""
        try:
            result = analyzer.analyze_match(
                home_team=normalize_team_name(home),
                away_team=normalize_team_name(away),
                match_datetime=f"{match_date}T{match_time}:00",
            )
            st.session_state["result"] = result
            st.session_state["last_run_at"] = datetime.utcnow().isoformat()
            save_analysis(cfg.DB_PATH, result)
        except Exception as e:
            st.session_state["last_error"] = str(e)
            logger.exception("analyze failed: %s", e)

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    render_result(st.session_state.get("result") or {})


def page_schedule(sports: SportsCollector):
    st.title("📅 Расписание (скоро)")
    st.info("Добавим в следующей части.")


def page_history(cfg: Config):
    st.title("🕘 История (скоро)")
    st.info("Добавим в следующей части.")


def page_diagnostics(cfg: Config, api: ApiSportsCollector):
    st.title("🧪 Диагностика (скелет)")
    st.write(f"DB_PATH: `{cfg.DB_PATH}`")
    st.write("API-Sports configured:", "✅" if api.is_configured() else "❌")
    st.write("Last run:", st.session_state.get("last_run_at") or "—")
    if st.session_state.get("last_error"):
        st.code(st.session_state["last_error"])


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="Sport Analyzer", page_icon="🏆", layout="wide")
    init_state()

    cfg = Config()
    ensure_db(cfg.DB_PATH)

    sports = SportsCollector(cfg)
    api = ApiSportsCollector(cfg)
    weather = WeatherCollector(cfg)
    news = NewsCollector(cfg)

    # Важно: не передаём xg= (мы уже ловили ошибку), пока не согласуем сигнатуру
    analyzer = MatchAnalyzer(cfg, sports=sports, weather=weather, news=news)

    st.sidebar.title("🏆 Sport Analyzer")
    page = st.sidebar.radio("Раздел", ["Анализ", "Расписание", "История", "Диагностика"], index=0)

    if page == "Анализ":
        page_analyze(analyzer, sports, cfg, api)
    elif page == "Расписание":
        page_schedule(sports)
    elif page == "История":
        page_history(cfg)
    else:
        page_diagnostics(cfg, api)


if __name__ == "__main__":
    main()
