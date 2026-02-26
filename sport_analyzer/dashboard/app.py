"""
Sport Analyzer — Dashboard rebuild (STEP 2)

Добавлено:
✅ Расписание
✅ История анализов
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
logger = logging.getLogger("dashboard")


# ---------------- STATE ----------------

def init_state():
    st.session_state.setdefault("result", {})
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("last_run_at", "")


# ---------------- DB ----------------

def ensure_db(path: str):
    run_migrations(path)
    with sqlite3.connect(path) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS analyses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            match TEXT,
            datetime TEXT,
            prediction TEXT,
            confidence REAL,
            analysis_json TEXT
        )
        """)
        c.commit()


def save_analysis(path: str, result: Dict[str, Any]):
    with sqlite3.connect(path) as c:
        c.execute("""
        INSERT INTO analyses(match,datetime,prediction,confidence,analysis_json)
        VALUES(?,?,?,?,?)
        """, (
            f"{result.get('home_team')} vs {result.get('away_team')}",
            str(result.get("datetime","")),
            str(result.get("prediction","")),
            float(result.get("confidence",0)),
            json.dumps(result, ensure_ascii=False)
        ))
        c.commit()


def load_analyses(path: str, limit: int = 300) -> pd.DataFrame:
    try:
        with sqlite3.connect(path) as c:
            return pd.read_sql_query(
                f"SELECT * FROM analyses ORDER BY id DESC LIMIT {limit}",
                c
            )
    except Exception:
        return pd.DataFrame()


# ---------------- UI ----------------

def render_result(result: Dict[str, Any]):
    if not result:
        st.info("Запусти анализ.")
        return

    st.markdown(
        f"## {result.get('home_team')} vs {result.get('away_team')}"
    )

    conf = float(result.get("confidence", 0))
    st.metric("Confidence", f"{conf:.1f}%")

    probs = result.get("final_probs") or {}

    c1, c2, c3 = st.columns(3)
    c1.progress(float(probs.get("home_win", 0)))
    c2.progress(float(probs.get("draw", 0)))
    c3.progress(float(probs.get("away_win", 0)))

    with st.expander("RAW"):
        st.json(result)


# ---------------- PAGES ----------------

def page_analyze(analyzer: MatchAnalyzer, sports: SportsCollector, cfg: Config):
    st.title("🏆 Анализ")

    home = st.text_input("Home", "Arsenal")
    away = st.text_input("Away", "Chelsea")

    run = st.button("Анализировать", type="primary")

    if run:
        try:
            result = analyzer.analyze_match(
                home_team=normalize_team_name(home),
                away_team=normalize_team_name(away),
                match_datetime=datetime.utcnow().isoformat(),
            )

            st.session_state.result = result
            save_analysis(cfg.DB_PATH, result)

        except Exception as e:
            st.session_state.last_error = str(e)

    render_result(st.session_state.result)


def page_schedule(sports: SportsCollector):
    st.title("📅 Расписание")

    days = st.slider("Days ahead", 1, 14, 7)

    matches = sports.get_matches(days_ahead=days) or []

    if not matches:
        st.warning("Нет матчей")
        return

    df = pd.DataFrame(matches)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_history(cfg: Config):
    st.title("🕘 История")

    df = load_analyses(cfg.DB_PATH)

    if df.empty:
        st.info("История пустая")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)


def page_diagnostics(cfg: Config, api: ApiSportsCollector):
    st.title("🧪 Диагностика")

    st.write("DB:", cfg.DB_PATH)
    st.write("API-Sports:", "✅" if api.is_configured() else "❌")

    if st.session_state.last_error:
        st.code(st.session_state.last_error)


# ---------------- MAIN ----------------

def main():
    st.set_page_config(page_title="Sport Analyzer", layout="wide")

    init_state()

    cfg = Config()
    ensure_db(cfg.DB_PATH)

    sports = SportsCollector(cfg)
    api = ApiSportsCollector(cfg)
    weather = WeatherCollector(cfg)
    news = NewsCollector(cfg)

    analyzer = MatchAnalyzer(cfg, sports=sports, weather=weather, news=news)

    page = st.sidebar.radio(
        "Раздел",
        ["Анализ", "Расписание", "История", "Диагностика"]
    )

    if page == "Анализ":
        page_analyze(analyzer, sports, cfg)
    elif page == "Расписание":
        page_schedule(sports)
    elif page == "История":
        page_history(cfg)
    else:
        page_diagnostics(cfg, api)


if __name__ == "__main__":
    main()
