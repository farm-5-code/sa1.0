"""
Sport Analyzer — Streamlit Dashboard (stable)

Локальный запуск:
  streamlit run app.py
или
  streamlit run sport_analyzer/dashboard/app.py

Для Streamlit Cloud:
  - main file path: app.py (в корне) или sport_analyzer/dashboard/app.py
  - requirements.txt в корне
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
# Session state init (critical)
# -----------------------------
if "result" not in st.session_state:
    st.session_state["result"] = {}
if "last_error" not in st.session_state:
    st.session_state["last_error"] = ""


# -----------------------------
# UI helpers
# -----------------------------
def _kpi_conf_label(conf: float) -> Tuple[str, str]:
    if conf >= 60:
        return "Высокая", "✅"
    if conf >= 48:
        return "Средняя", "🟡"
    return "Низкая", "⚠️"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _prob_bar(label: str, p: float):
    p = max(0.0, min(1.0, _safe_float(p)))
    st.write(f"**{label}** — {p*100:.1f}%")
    st.progress(p)


def _plot_odds_history(df_hist: pd.DataFrame, market: str, selection: str):
    if df_hist.empty:
        st.info("История пустая.")
        return

    d = df_hist[(df_hist["market"] == market) & (df_hist["selection"] == selection)].copy()
    if d.empty:
        st.info("Нет данных для выбранной комбинации.")
        return

    d["ts"] = pd.to_numeric(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts")
    if d.empty:
        st.info("Нет валидных timestamp в истории.")
        return

    x = pd.to_datetime(d["ts"], unit="s", utc=True)

    fig, ax = plt.subplots()
    ax.plot(x, d["best_odd"].astype(float), label="best")
    ax.plot(x, d["avg_odd"].astype(float), label="avg")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("odds")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# -----------------------------
# DB (history)
# -----------------------------
def _ensure_db():
    cfg = Config()
    run_migrations(cfg.DB_PATH)
    # Create minimal tables if migrations are missing
    try:
        with sqlite3.connect(cfg.DB_PATH, timeout=10) as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match TEXT,
                    datetime TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    prediction TEXT,
                    confidence REAL,
                    analysis_json TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER,
                    home_team TEXT,
                    away_team TEXT,
                    market TEXT,
                    selection TEXT,
                    best_odd REAL,
                    avg_odd REAL,
                    books INTEGER
                )
                """
            )
            c.commit()
    except Exception as e:
        logger.exception("DB init failed: %s", e)


def _save_analysis_to_db(result: Dict[str, Any]):
    cfg = Config()
    try:
        match = result.get("match") or f"{result.get('home_team','')} vs {result.get('away_team','')}"
        dt = _safe_str(result.get("datetime", ""))
        conf = _safe_float(result.get("confidence", 0.0))
        pred = _safe_str(result.get("best_pick", result.get("prediction", "")))

        with sqlite3.connect(cfg.DB_PATH, timeout=10) as c:
            c.execute(
                """
                INSERT INTO analyses (match, datetime, prediction, confidence, analysis_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (match, dt, pred, conf, json.dumps(result, ensure_ascii=False)),
            )
            c.commit()
    except Exception as e:
        logger.exception("DB write analyses failed: %s", e)


def _load_analyses(limit: int = 200) -> pd.DataFrame:
    cfg = Config()
    try:
        with sqlite3.connect(cfg.DB_PATH, timeout=10) as c:
            df = pd.read_sql_query(
                f"""
                SELECT id, created_at, match, datetime, prediction, confidence
                FROM analyses
                ORDER BY id DESC
                LIMIT {int(limit)}
                """,
                c,
            )
        return df
    except Exception as e:
        logger.exception("DB read analyses failed: %s", e)
        return pd.DataFrame()


def _load_odds_history(home: str, away: str, limit: int = 2000) -> pd.DataFrame:
    cfg = Config()
    try:
        with sqlite3.connect(cfg.DB_PATH, timeout=10) as c:
            df = pd.read_sql_query(
                """
                SELECT ts, home_team, away_team, market, selection, best_odd, avg_odd, books
                FROM odds_history
                WHERE home_team = ? AND away_team = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                c,
                params=(home, away, int(limit)),
            )
        return df
    except Exception as e:
        logger.exception("DB read odds_history failed: %s", e)
        return pd.DataFrame()


# -----------------------------
# Rendering blocks
# -----------------------------
def render_result(result: Dict[str, Any]):
    if not result:
        st.info("Пока нет результата. Запусти анализ слева.")
        return

    home = _safe_str(result.get("home_team", "Home"))
    away = _safe_str(result.get("away_team", "Away"))
    conf = _safe_float(result.get("confidence", 0.0))
    conf_label, conf_emoji = _kpi_conf_label(conf)

    st.markdown(f"## 🧾 Результат: **{home} vs {away}**")
    st.metric("Уверенность", f"{conf:.1f}% ({conf_emoji} {conf_label})")

    probs = result.get("final_probs") or {}
    poisson = result.get("poisson") or {}

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Вероятности 1X2")
        _prob_bar(f"🏠 {home}", probs.get("home_win", 0))
        _prob_bar("🤝 Ничья", probs.get("draw", 0))
        _prob_bar(f"✈️ {away}", probs.get("away_win", 0))
    with c2:
        st.markdown("### Poisson / Totals")
        st.write(f"λ_home: **{_safe_float(poisson.get('lambda_h', 0)):.2f}**")
        st.write(f"λ_away: **{_safe_float(poisson.get('lambda_a', 0)):.2f}**")
        st.write(f"Over 2.5: **{_safe_float(poisson.get('over_2_5', 0))*100:.1f}%**")
        st.write(f"BTTS: **{_safe_float(poisson.get('both_score', 0))*100:.1f}%**")

    recs = result.get("recommendations") or []
    if recs:
        with st.expander("💡 Рекомендации", expanded=True):
            for r in recs:
                st.write(f"• {r}")

    # extra info
    with st.expander("🧩 Источники данных / фолбэки", expanded=False):
        av = result.get("availability") or {}
        notes = result.get("notes") or []
        if av:
            for k, v in av.items():
                ok = bool((v or {}).get("ok", True))
                if ok:
                    st.write(f"✅ {k}")
                else:
                    st.write(f"⚠️ {k}: {(v or {}).get('error')}")
        if notes:
            st.markdown("**Примечания:**")
            for n in notes:
                st.caption(n)

    # weather/news blocks (optional)
    weather = result.get("weather") or {}
    news = result.get("news") or {}
    if weather:
        with st.expander("🌤️ Погода", expanded=False):
            st.json(weather)
    if news:
        with st.expander("📰 Новости", expanded=False):
            st.json(news)


def render_page_analyze(
    analyzer: MatchAnalyzer,
    sports: SportsCollector,
    api_sports: ApiSportsCollector,
):
    st.markdown("# 🏆 Sport Analyzer — Анализ матча")

    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("### Ввод")
        mode = st.radio("Источник матча", ["Вручную", "Из football-data (ближайшие)"], horizontal=True)

        fixture_id = None
        match_date = date.today()
        match_time = datetime.utcnow().time()

        if mode == "Из football-data (ближайшие)":
            days = st.slider("Дней вперёд", 1, 14, 7)
            with st.spinner("Загружаем матчи…"):
                matches = sports.get_matches(days_ahead=days) or []
            if not matches:
                st.warning("Матчи не найдены. Проверь FOOTBALL_DATA_KEY.")
                home_raw = st.text_input("🏠 Домашняя команда")
                away_raw = st.text_input("✈️ Гостевая команда")
                h_id = st.number_input("ID хозяев (football-data)", value=0, step=1)
                a_id = st.number_input("ID гостей (football-data)", value=0, step=1)
            else:
                df = pd.DataFrame(matches)
                df["label"] = df.apply(lambda r: f"{r.get('date','')} — {r.get('competition','')} — {r.get('home_team','')} vs {r.get('away_team','')}", axis=1)
                pick = st.selectbox("Выбери матч", df["label"].tolist(), index=0)
                row = df[df["label"] == pick].iloc[0].to_dict()
                home_raw = row.get("home_team", "")
                away_raw = row.get("away_team", "")
                h_id = int(row.get("home_team_id") or 0)
                a_id = int(row.get("away_team_id") or 0)
                fixture_id = row.get("fixture_id")

                # parse datetime
                try:
                    dt = row.get("utcDate") or row.get("date") or ""
                    dtp = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
                    match_date = dtp.date()
                    match_time = dtp.time()
                except Exception:
                    pass

                st.caption(f"fixture_id: {fixture_id}")
                st.text_input("🏠 Домашняя команда", value=home_raw, disabled=True)
                st.text_input("✈️ Гостевая команда", value=away_raw, disabled=True)

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

                # Optional: API-Sports odds/injuries/lineups if configured
                if api_sports.is_configured() and fixture_id:
                    try:
                        odds = api_sports.get_odds_for_fixture(int(fixture_id))
                        result["api_sports"] = {"fixture_id": int(fixture_id), "odds": odds}
                    except Exception as e:
                        result["api_sports"] = {"fixture_id": int(fixture_id), "odds_error": str(e)}

                st.session_state["result"] = result
                st.session_state["last_error"] = ""
                _save_analysis_to_db(result)

            except Exception as e:
                st.session_state["last_error"] = str(e)
                logger.exception("Analyze failed: %s", e)

        if st.session_state.get("last_error"):
            st.error("Ошибка анализа:")
            st.code(st.session_state["last_error"])

        render_result(st.session_state.get("result") or {})


def render_page_schedule(sports: SportsCollector):
    st.markdown("# 📅 Расписание")
    days = st.slider("Дней вперёд", 1, 14, 7)
    with st.spinner("Загружаю…"):
        matches = sports.get_matches(days_ahead=days) or []
    if not matches:
        st.warning("Матчи не найдены. Проверь FOOTBALL_DATA_KEY.")
        return
    df = pd.DataFrame(matches)
    # best effort columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%d.%m %H:%M")
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_page_history():
    st.markdown("# 🕘 История анализов")
    df = _load_analyses(limit=300)
    if df.empty:
        st.info("История пуста.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_page_odds_history():
    st.markdown("# 📈 Odds history (если пишется в БД)")
    r = st.session_state.get("result") or {}
    home = _safe_str(r.get("home_team", ""))
    away = _safe_str(r.get("away_team", ""))
    if not home or not away:
        st.info("Сначала сделай анализ матча — тогда появятся команды для фильтра.")
        return

    df_hist = _load_odds_history(home, away, limit=5000)
    if df_hist.empty:
        st.info("Нет данных odds_history для этих команд.")
        return

    markets = sorted(df_hist["market"].dropna().unique().tolist())
    if not markets:
        st.info("Нет рынков в истории.")
        return

    m = st.selectbox("Market", markets, index=0)
    sels = sorted(df_hist[df_hist["market"] == m]["selection"].dropna().unique().tolist())
    if not sels:
        st.info("Нет selection для выбранного market.")
        return

    s = st.selectbox("Selection", sels, index=0)
    _plot_odds_history(df_hist, m, s)

    with st.expander("Raw history"):
        st.dataframe(df_hist, use_container_width=True, hide_index=True)


def render_page_diagnostics(cfg: Config, api_sports: ApiSportsCollector):
    st.markdown("# 🧪 Диагностика")

    st.markdown("### Конфиг")
    st.write(f"DB_PATH: `{cfg.DB_PATH}`")

    st.markdown("### Ключи / доступность (без раскрытия значений)")
    def has(x: str) -> str:
        v = getattr(cfg, x, None)
        return "✅" if bool(v) else "❌"

    # Поля могут отличаться — показываем безопасно:
    for key_name in ["FOOTBALL_DATA_KEY", "API_FOOTBALL_KEY", "NEWS_API_KEY", "GNEWS_KEY"]:
        if hasattr(cfg, key_name):
            st.write(f"{key_name}: {has(key_name)}")

    st.markdown("### API-Sports")
    st.write("Configured:", "✅" if api_sports.is_configured() else "❌")

    st.markdown("### Последняя ошибка")
    if st.session_state.get("last_error"):
        st.code(st.session_state["last_error"])
    else:
        st.write("—")


# -----------------------------
# App start
# -----------------------------
def main():
    st.set_page_config(page_title="Sport Analyzer", page_icon="🏆", layout="wide")

    _ensure_db()

    cfg = Config()

    sports = SportsCollector(cfg)
    api_sports = ApiSportsCollector(cfg)
    weather = WeatherCollector(cfg)  # used inside analyzer
    news = NewsCollector(cfg)        # used inside analyzer
    xg = XGCollector(cfg)            # used inside analyzer
    analyzer = MatchAnalyzer(cfg, sports=sports, weather=weather, news=news, xg=xg)

    st.sidebar.title("🏆 Sport Analyzer")
    page = st.sidebar.radio(
        "Раздел",
        ["Анализ", "Расписание", "История", "Odds history", "Диагностика"],
        index=0,
    )

    if page == "Анализ":
        render_page_analyze(analyzer, sports, api_sports)
    elif page == "Расписание":
        render_page_schedule(sports)
    elif page == "История":
        render_page_history()
    elif page == "Odds history":
        render_page_odds_history()
    else:
        render_page_diagnostics(cfg, api_sports)


if __name__ == "__main__":
    main()
