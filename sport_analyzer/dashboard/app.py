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

    if page == "Анализ":
        page_analyze(analyzer, sports, cfg, api)
    elif page == "Insights":
        page_insights()
    elif page == "Opportunities":
        page_opportunities(analyzer, sports)
    elif page == "Сигналы":
        page_signals(api)
    elif page == "Расписание":
        page_schedule(sports)
    elif page == "История":
        page_history(cfg)
    else:
        page_diagnostics(cfg, api)
    
# ============================================================
# STEP 2 EXTENSION — Schedule + History implementation
# (вставляется В КОНЕЦ файла, ничего выше менять не нужно)
# ============================================================


def page_schedule(sports: SportsCollector):
    st.title("📅 Расписание")

    days = st.slider("Дней вперёд", 1, 14, 7)

    with st.spinner("Загружаем матчи…"):
        matches = sports.get_matches(days_ahead=days) or []

    if not matches:
        st.warning("Матчи не найдены. Проверь FOOTBALL_DATA_KEY.")
        return

    df = pd.DataFrame(matches)

    # best-effort formatting
    for col in ["date", "utcDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%d.%m %H:%M")

    st.dataframe(df, use_container_width=True, hide_index=True)



def page_history(cfg: Config):
    st.title("🕘 История анализов")

    limit = st.slider("Сколько записей показать", 50, 1000, 300, step=50)

    df = load_analyses(cfg.DB_PATH, limit=int(limit))

    if df.empty:
        st.info("История пока пустая.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    # просмотр raw json результата
    with st.expander("📦 Открыть сохранённый результат", expanded=False):
        pick_id = st.number_input("ID анализа", value=int(df.iloc[0]["id"]), step=1)

        try:
            with sqlite3.connect(cfg.DB_PATH) as c:
                row = c.execute(
                    "SELECT analysis_json FROM analyses WHERE id=?",
                    (int(pick_id),),
                ).fetchone()

            if not row:
                st.warning("Запись не найдена.")
                return

            st.json(json.loads(row[0]))

        except Exception as e:
            st.error(str(e))

# ============================================================
# STEP 3 — SIGNALS ENGINE
# ============================================================

def _movement_from_history(df_hist: pd.DataFrame, window_min: int) -> pd.DataFrame:
    if df_hist.empty:
        return pd.DataFrame()

    df = df_hist.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])

    if df.empty:
        return pd.DataFrame()

    latest_ts = df["ts"].max()
    cutoff = latest_ts - window_min * 60

    base = df[df["ts"] >= cutoff].copy()
    if base.empty:
        return pd.DataFrame()

    base = base.sort_values("ts")

    grp = ["market", "selection"]

    first = base.groupby(grp, as_index=False).first()
    last = base.groupby(grp, as_index=False).last()

    merged = first.merge(last, on=grp, suffixes=("_start", "_last"))

    for col in ["best_odd", "avg_odd"]:
        if f"{col}_start" in merged and f"{col}_last" in merged:
            merged[f"{col}_chg_pct"] = (
                merged[f"{col}_last"].astype(float)
                / (merged[f"{col}_start"].astype(float) + 1e-9)
                - 1
            ) * 100

    merged["steam_score"] = (-merged.get("best_odd_chg_pct", 0)).clip(lower=0)

    return merged


def page_signals(api: ApiSportsCollector):
    st.title("📡 Сигналы (движение линии)")

    if not api.is_configured():
        st.error("API_FOOTBALL_KEY не задан в Secrets.")
        return

    date_sel = st.date_input("Дата", value=datetime.utcnow().date())

    with st.spinner("Загрузка fixtures…"):
        fixtures = api.get_fixtures_by_date(
            date_sel.strftime("%Y-%m-%d"),
            timezone="UTC",
            use_cache=True,
        )

    if not fixtures:
        st.warning("Fixtures не найдены.")
        return

    df_fx = pd.DataFrame(fixtures)
    df_fx["match"] = df_fx["home_team"] + " — " + df_fx["away_team"]

    options = {
        f"{r['league']} • {r['match']}": r["fixture_id"]
        for _, r in df_fx.iterrows()
    }

    label = st.selectbox("Матч", list(options.keys()))
    fixture_id = options[label]

    if st.button("📸 Снять snapshot"):
        odds = api.get_odds_for_fixture(int(fixture_id), use_cache=False)
        api.save_snapshot_from_odds(int(fixture_id), odds)
        st.success("Snapshot сохранён")

    hist = api.get_snapshot_history(int(fixture_id), hours=24)

    if not hist:
        st.info("История snapshot пока пустая.")
        return

    dfh = pd.DataFrame(hist)

    frames = []
    for w in (10, 30, 60):
        frames.append(_movement_from_history(dfh, w))

    dfm = pd.concat(frames, ignore_index=True)

    if dfm.empty:
        st.info("Недостаточно snapshot.")
        return

    st.dataframe(
        dfm.sort_values("steam_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
# ============================================================
# STEP 4 — OPPORTUNITIES (48h) without API-Sports
# ============================================================

def _safe_prob(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def _pick_from_probs(home: float, draw: float, away: float):
    m = max(home, draw, away)
    if m == home:
        return "Home", m
    if m == draw:
        return "Draw", m
    return "Away", m


def page_opportunities(analyzer: MatchAnalyzer, sports: SportsCollector):
    st.title("💎 Opportunities (48h)")
    st.caption("Список ближайших матчей и сильнейшие прогнозы по модели (без odds / без API-Sports).")

    days = st.slider("Дней вперёд", 1, 7, 2)
    top_n = st.slider("Показать топ", 5, 50, 15)

    with st.spinner("Загружаю матчи…"):
        matches = sports.get_matches(days_ahead=int(days)) or []

    if not matches:
        st.warning("Матчи не найдены. Проверь FOOTBALL_DATA_KEY.")
        return

    df = pd.DataFrame(matches)
    if df.empty:
        st.warning("Пустой список матчей.")
        return

    # Best effort columns
    # expecting: home_team, away_team, home_team_id, away_team_id, date/utcDate
    rows = []
    prog = st.progress(0.0)
    total = len(df)

    for i, r in enumerate(df.to_dict("records"), start=1):
        prog.progress(i / max(1, total))

        home_raw = str(r.get("home_team", "") or "")
        away_raw = str(r.get("away_team", "") or "")
        if not home_raw or not away_raw:
            continue

        home = normalize_team_name(home_raw)
        away = normalize_team_name(away_raw)

        match_dt = r.get("utcDate") or r.get("date") or ""
        match_dt = str(match_dt)

        h_id = r.get("home_team_id") or r.get("homeTeamId") or 0
        a_id = r.get("away_team_id") or r.get("awayTeamId") or 0

        try:
            res = analyzer.analyze_match(
                home_team=home,
                away_team=away,
                match_datetime=match_dt if match_dt else datetime.utcnow().isoformat(),
                home_team_id=int(h_id) if int(h_id) else None,
                away_team_id=int(a_id) if int(a_id) else None,
            )
        except Exception:
            # не валим весь список из-за одного матча
            continue

        probs = res.get("final_probs") or {}
        ph = _safe_prob(probs.get("home_win", 0))
        pdw = _safe_prob(probs.get("draw", 0))
        pa = _safe_prob(probs.get("away_win", 0))

        pick, pmax = _pick_from_probs(ph, pdw, pa)
        conf = float(res.get("confidence", pmax * 100) or (pmax * 100))

        rows.append({
            "datetime": match_dt,
            "league": r.get("competition") or r.get("league") or "",
            "home": home_raw,
            "away": away_raw,
            "pick": pick,
            "p_pick": round(pmax, 4),
            "confidence_%": round(conf, 1),
            "p_home": round(ph, 4),
            "p_draw": round(pdw, 4),
            "p_away": round(pa, 4),
        })

    prog.empty()

    out = pd.DataFrame(rows)
    if out.empty:
        st.info("Не удалось построить прогнозы по матчам (возможно, не хватает данных/ключа).")
        return

    out = out.sort_values(["confidence_%", "p_pick"], ascending=False).head(int(top_n))
    st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption("Подсказка: кликай по строкам/фильтруй таблицу — это живой рейтинг силы прогнозов.")
    # ============================================================
# STEP 5 — MODEL INSIGHTS
# ============================================================

import matplotlib.pyplot as plt


def page_insights():
    st.title("🧠 Model Insights")

    result = st.session_state.get("result")

    if not result:
        st.info("Сначала выполни анализ матча.")
        return

    probs = result.get("final_probs") or {}

    home = float(probs.get("home_win", 0))
    draw = float(probs.get("draw", 0))
    away = float(probs.get("away_win", 0))

    st.subheader("Вероятности исходов")

    fig = plt.figure()
    plt.bar(["Home", "Draw", "Away"], [home, draw, away])
    plt.ylabel("Probability")

    st.pyplot(fig, clear_figure=True)

    st.subheader("Уверенность модели")

    conf = float(result.get("confidence", 0))
    st.metric("Confidence", f"{conf:.1f}%")

    if conf > 60:
        st.success("Высокая уверенность модели")
    elif conf > 48:
        st.warning("Средняя уверенность")
    else:
        st.error("Низкая уверенность")

    recs = result.get("recommendations") or []

    if recs:
        st.subheader("Рекомендации")
        for r in recs:
            st.write("•", r)

    # если анализатор отдаёт факторы — покажем
    factors = result.get("factors") or result.get("model_factors")

    if isinstance(factors, dict) and factors:
        st.subheader("Факторы модели")
        st.json(factors)

# ============================================================
# STEP 5 — INSIGHTS (Explainability)
# ============================================================

import matplotlib.pyplot as plt


def page_insights():
    st.title("🧠 Insights")

    result = st.session_state.get("result") or {}
    if not result:
        st.info("Сначала запусти Анализ, чтобы появился результат.")
        return

    home_team = str(result.get("home_team", "Home"))
    away_team = str(result.get("away_team", "Away"))
    st.caption(f"{home_team} vs {away_team}")

    probs = result.get("final_probs") or {}
    p_home = float(probs.get("home_win", 0.0) or 0.0)
    p_draw = float(probs.get("draw", 0.0) or 0.0)
    p_away = float(probs.get("away_win", 0.0) or 0.0)

    st.subheader("Вероятности 1X2")
    fig, ax = plt.subplots()
    ax.bar(["Home", "Draw", "Away"], [p_home, p_draw, p_away])
    ax.set_ylabel("Probability")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    conf = float(result.get("confidence", 0.0) or 0.0)
    st.subheader("Уверенность")
    st.metric("Confidence", f"{conf:.1f}%")
    if conf >= 60:
        st.success("Высокая уверенность модели")
    elif conf >= 48:
        st.warning("Средняя уверенность модели")
    else:
        st.info("Низкая уверенность модели")

    recs = result.get("recommendations") or []
    if recs:
        st.subheader("Рекомендации")
        for r in recs:
            st.write("•", r)

    # если в результатах есть какие-то факторы — покажем
    factors = result.get("factors") or result.get("model_factors") or {}
    if isinstance(factors, dict) and factors:
        with st.expander("Факторы модели", expanded=False):
            st.json(factors)

    with st.expander("Raw result", expanded=False):
        st.json(result)


# ----------------- override MAIN with Insights tab -----------------

def main():
    init_state()

    cfg = Config()
    ensure_db(cfg.DB_PATH)

    sports = SportsCollector(cfg)
    api = ApiSportsCollector(cfg)
    weather = WeatherCollector(cfg)
    news = NewsCollector(cfg)

    analyzer = MatchAnalyzer(cfg, sports=sports, weather=weather, news=news)

    st.sidebar.title("🏆 Sport Analyzer")
    page = st.sidebar.radio(
        "Раздел",
        [
        "🏆 Анализ",
        "🔥 Opportunities",
        "🧠 Insights",
        "📡 Сигналы",
        "📅 Расписание",
        "📚 История",
        "🧪 Диагностика",
        ],
        index=0,
        key="main_navigation_radio",
    )

if page == "🏆 Анализ":
    page_analyze(analyzer, sports, cfg, api)

elif page == "🔥 Opportunities":
    page_opportunities(analyzer, sports)

elif page == "🧠 Insights":
    page_insights()

elif page == "📡 Сигналы":
    page_signals(api)

elif page == "📅 Расписание":
    page_schedule(sports)

elif page == "📚 История":
    page_history(cfg)

else:
    page_diagnostics(cfg, api)
