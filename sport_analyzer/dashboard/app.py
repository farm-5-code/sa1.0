"""
Sport Analyzer — Streamlit Dashboard (rebuilt step-by-step)

Step 1: Base shell + Analyze + Diagnostics + DB init
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, date
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from sport_analyzer.config.settings import Config
from sport_analyzer.collectors.sports_collector import SportsCollector
from sport_analyzer.collectors.api_sports_collector import ApiSportsCollector
from sport_analyzer.collectors.weather_collector import WeatherCollector
from sport_analyzer.collectors.news_collector import NewsCollector
from sport_analyzer.collectors.xg_collector import XGCollector
from sport_analyzer.analyzers.match_analyzer import MatchAnalyzer
from sport_analyzer.database.migrations import run_migrations
from sport_analyzer.utils.team_normalizer import normalize_team_name

logger = logging.getLogger("sport_analyzer.dashboard")
logging.basicConfig(level=logging.INFO)


# -----------------------------
# Session State
# -----------------------------
def _init_state():
    if "result" not in st.session_state:
        st.session_state["result"] = {}
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = ""
    if "last_run_at" not in st.session_state:
        st.session_state["last_run_at"] = ""


# -----------------------------
# DB init (safe)
# -----------------------------
def _ensure_db(db_path: str):
    # project migrations
    try:
        run_migrations(db_path)
    except Exception as e:
        logger.exception("run_migrations failed: %s", e)

    # minimal tables so history pages won’t crash later
    try:
        with sqlite3.connect(db_path, timeout=10) as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
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
        logger.exception("DB init failed: %s", e)


def _save_analysis(db_path: str, result: Dict[str, Any]):
    try:
        match = result.get("match") or f"{result.get('home_team','')} vs {result.get('away_team','')}"
        dt = str(result.get("datetime", ""))
        pred = str(result.get("best_pick", result.get("prediction", "")))
        conf = float(result.get("confidence", 0.0) or 0.0)
        raw = json.dumps(result, ensure_ascii=False)

        with sqlite3.connect(db_path, timeout=10) as c:
            c.execute(
                """
                INSERT INTO analyses (match, datetime, prediction, confidence, analysis_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (match, dt, pred, conf, raw),
            )
            c.commit()
    except Exception as e:
        logger.exception("save_analysis failed: %s", e)


# -----------------------------
# UI helpers
# -----------------------------
def _kpi_conf_label(conf: float) -> str:
    if conf >= 60:
        return "✅ высокая"
    if conf >= 48:
        return "🟡 средняя"
    return "⚠️ низкая"


def render_result(result: Dict[str, Any]):
    if not result:
        st.info("Пока нет результата. Запусти анализ.")
        return

    home = str(result.get("home_team", "Home"))
    away = str(result.get("away_team", "Away"))
    conf = float(result.get("confidence", 0.0) or 0.0)

    st.markdown(f"## 🧾 Результат: **{home} vs {away}**")
    st.metric("Уверенность", f"{conf:.1f}% ({_kpi_conf_label(conf)})")

    probs = result.get("final_probs") or {}
    if isinstance(probs, dict) and probs:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("🏠 Home win")
            st.progress(min(1.0, max(0.0, float(probs.get("home_win", 0.0) or 0.0))))
        with c2:
            st.write("🤝 Draw")
            st.progress(min(1.0, max(0.0, float(probs.get("draw", 0.0) or 0.0))))
        with c3:
            st.write("✈️ Away win")
            st.progress(min(1.0, max(0.0, float(probs.get("away_win", 0.0) or 0.0))))

    recs = result.get("recommendations") or []
    if recs:
        with st.expander("💡 Рекомендации", expanded=True):
            for r in recs:
                st.write(f"• {r}")

    with st.expander("🔎 Raw result (debug)", expanded=False):
        st.json(result)


def page_analyze(analyzer: MatchAnalyzer, sports: SportsCollector):
    st.markdown("# 🏆 Sport Analyzer — Анализ матча")

    left, right = st.columns([1, 1.4])

    with left:
        mode = st.radio("Источник матча", ["Вручную", "Из football-data (ближайшие)"], horizontal=True)

        fixture_row = None
        fixture_id = None

        if mode == "Из football-data (ближайшие)":
            days = st.slider("Дней вперёд", 1, 14, 7)
            with st.spinner("Загружаем матчи…"):
                matches = sports.get_matches(days_ahead=days) or []
            if matches:
                df = pd.DataFrame(matches)
                df["label"] = df.apply(
                    lambda r: f"{r.get('date','')} — {r.get('competition','')} — {r.get('home_team','')} vs {r.get('away_team','')}",
                    axis=1,
                )
                pick = st.selectbox("Матч", df["label"].tolist(), index=0)
                fixture_row = df[df["label"] == pick].iloc[0].to_dict()
                home_raw = fixture_row.get("home_team", "")
                away_raw = fixture_row.get("away_team", "")
                h_id = int(fixture_row.get("home_team_id") or 0)
                a_id = int(fixture_row.get("away_team_id") or 0)
                fixture_id = fixture_row.get("fixture_id")
                st.caption(f"fixture_id: {fixture_id}")
                st.text_input("🏠 Домашняя команда", value=home_raw, disabled=True)
                st.text_input("✈️ Гостевая команда", value=away_raw, disabled=True)
            else:
                st.warning("Матчи не найдены. Проверь FOOTBALL_DATA_KEY.")
                home_raw = st.text_input("🏠 Домашняя команда")
                away_raw = st.text_input("✈️ Гостевая команда")
                h_id = st.number_input("ID хозяев (football-data)", value=0, step=1)
                a_id = st.number_input("ID гостей (football-data)", value=0, step=1)
        else:
            home_raw = st.text_input("🏠 Домашняя команда", placeholder="Arsenal")
            away_raw = st.text_input("✈️ Гостевая команда", placeholder="Chelsea")
            h_id = st.number_input("ID хозяев (football-data)", value=0, step=1)
            a_id = st.number_input("ID гостей (football-data)", value=0, step=1)

        dcol, tcol = st.columns(2)
        with dcol:
            match_date = st.date_input("📅 Дата", value=date.today())
        with tcol:
            match_time = st.time_input("🕐 Время (UTC)", value=datetime.utcnow().time())

        city = st.text_input("📍 Город (необязательно)", placeholder="London")
        neutral = st.checkbox("Нейтральное поле", value=False)

        run_btn = st.button("🚀 Анализировать", type="primary", use_container_width=True)

    with right:
        if run_btn:
            st.session_state["last_error"] = ""

            if not home_raw or not away_raw:
                st.error("Введите обе команды.")
                return

            home = normalize_team_name(home_raw)
            away = normalize_team_name(away_raw)

            try:
                with st.spinner("Считаю прогноз…"):
                    result = analyzer.analyze_match(
                        home_team=home,
                        away_team=away,
                        match_datetime=f"{match_date}T{match_time}:00",
                        city=(city or None),
                        home_team_id=int(h_id) if int(h_id) else None,
                        away_team_id=int(a_id) if int(a_id) else None,
                        neutral_field=neutral,
                    )
                st.session_state["result"] = result
                st.session_state["last_run_at"] = datetime.utcnow().isoformat()

                cfg = Config()
                _save_analysis(cfg.DB_PATH, result)

            except Exception as e:
                st.session_state["last_error"] = str(e)
                logger.exception("Analyze failed: %s", e)

        if st.session_state.get("last_error"):
            st.error("Ошибка анализа:")
            st.code(st.session_state["last_error"])

        render_result(st.session_state.get("result") or {})


def page_diagnostics(cfg: Config, api_sports: ApiSportsCollector):
    st.markdown("# 🧪 Диагностика")
    st.write(f"DB_PATH: `{cfg.DB_PATH}`")

    st.markdown("### Ключи (только наличие)")
    for key in ["FOOTBALL_DATA_KEY", "API_FOOTBALL_KEY", "NEWS_API_KEY", "GNEWS_KEY"]:
        if hasattr(cfg, key):
            st.write(f"{key}: {'✅' if bool(getattr(cfg, key)) else '❌'}")

    st.markdown("### API-Sports")
    st.write("Configured:", "✅" if api_sports.is_configured() else "❌")

    st.markdown("### Последний запуск")
    st.write(st.session_state.get("last_run_at") or "—")

    st.markdown("### Последняя ошибка")
    if st.session_state.get("last_error"):
        st.code(st.session_state["last_error"])
    else:
        st.write("—")


def main():
    st.set_page_config(page_title="Sport Analyzer", page_icon="🏆", layout="wide")
    _init_state()

    cfg = Config()
    _ensure_db(cfg.DB_PATH)

    sports = SportsCollector(cfg)
    api_sports = ApiSportsCollector(cfg)
    weather = WeatherCollector(cfg)
    news = NewsCollector(cfg)
    # IMPORTANT: MatchAnalyzer in your project does NOT accept xg= keyword
    # If it needs xg collector, it likely uses xg_collector=... internally in its own signature.
    analyzer = MatchAnalyzer(cfg, sports=sports, weather=weather, news=news)

    st.sidebar.title("🏆 Sport Analyzer")
    page = st.sidebar.radio("Раздел", ["Анализ", "Диагностика"], index=0)

    if page == "Анализ":
        page_analyze(analyzer, sports)
    else:
        page_diagnostics(cfg, api_sports)


if __name__ == "__main__":
    main()
