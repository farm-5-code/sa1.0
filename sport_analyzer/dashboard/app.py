"""Streamlit dashboard.

Локальный запуск:
  streamlit run sport_analyzer/dashboard/app.py

Для деплоя (Streamlit Community Cloud / similar):
    - этот файл должен быть выбран как entrypoint  
    - requirements.txt должен лежать в корне репозитория  
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, date, timedelta
import time
from typing import Dict

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

logger = logging.getLogger("sport_analyzer.dashboard")
# --- Streamlit state init ---
if "result" not in st.session_state:
 ул. сессия_государство["Результат"]  =  {}
  
# Fallback: make sure repo root is on sys.path when запуск идет из подпапки
try:
    import sport_analyzer  # noqa: F401
except Exception as e:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from sport_analyzer.config.settings import Config
from sport_analyzer.collectors.sports_collector import SportsCollector
from sport_analyzer.collectors.api_sports_collector import ApiSportsCollector
from sport_analyzer.collectors.weather_collector import WeatherCollector
from sport_analyzer.collectors.news_collector import NewsCollector
from sport_analyzer.collectors.xg_collector import XGCollector
from sport_analyzer.analyzers.match_analyzer import MatchAnalyzer
from sport_analyzer.database.migrations import run_migrations
from sport_analyzer.utils.team_normalizer import normalize_team_name



def _plot_odds_history(df_hist: pd.DataFrame, market: str, selection: str):
    """Line chart for best/avg odds over time."""
    if df_hist.empty:
         st.info("История пустая.") info("История пустая.")
        return
    d = df_hist[(df_hist["market"] == market) & (df_hist["selection"] == selection)].copy()
    if d.empty:
         st.info("Нет данных для выбранной комбинации.") info("Нет данных для выбранной комбинации.")
        return
    d["ts"] = pd.to_numeric(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts"]).sort_values("ts")
    x = pd.to_datetime(d["ts"], unit="s", utc=True)
    fig = plt.figure()
    plt.plot(x, d["best_odd"].astype(float), label="best")
    plt.plot(x, d["avg_odd"].astype(float), label="avg")
    plt.legend()
    plt.xlabel("time (UTC)")
    plt.ylabel("odds")
    st.pyplot(fig, clear_figure=True)

# ── Odds / signals helpers ───────────────────────────────────────────

def _summarize_market(values: Dict[str, list]) -> Dict[str, Dict]:
    """values: {label: [odds,...]} -> {label: {best, avg, n}}"""
    out = {}
    for label, odds in (values or {}).items():
        if not odds:
            continue
        try:
            odds_f = [float(x) for x in odds if x]
        except Exception as e:
            continue
        if not odds_f:
            continue
        out[label] = {
            "best": max(odds_f),
            "avg": sum(odds_f) / len(odds_f),
            "n": len(odds_f),
        }
    return out


def _movement_from_history(df_hist: pd.DataFrame, window_min: int) -> pd.DataFrame:
    """Compute movement for each (market, selection) between latest and earliest >= window."""
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
    merged["window_min"] = window_min

    for col in ["best_odd", "avg_odd"]:
        if f"{col}_start" not in merged.columns or f"{col}_last" not in merged.columns:
            continue
        merged[f"{col}_chg"] = merged[f"{col}_last"].astype(float) - merged[f"{col}_start"].astype(float)
        merged[f"{col}_chg_pct"] = (
            merged[f"{col}_last"].astype(float) / (merged[f"{col}_start"].astype(float)   +   1e-9) - 1.0
        ) * 100.0

    # normalize books column name
    if "books_last" not in merged.columns:
        if "books" in merged.columns:
            merged["books_last"] = merged["books"]
        elif "books_start" in merged.columns:
            merged["books_last"] = merged["books_start"]
        else:
            merged["books_last"] = 0

    return merged

def _calc_value_signals_all(result: Dict, odds_markets: Dict) -> pd.DataFrame:
    """Value/EV по рынкам 1X2, OU 2.5, BTTS на основе модели.
    EV = P_model * best_odd - 1
    """
    rows = []
    final_probs = result.get("final_probs") or {}
    poisson = result.get("poisson") or {}

    # 1X2
    prob_map_1x2 = {
        "Home": float(final_probs.get("home_win", 0.0)),
        "Draw": float(final_probs.get("draw", 0.0)),
        "Away": float(final_probs.get("away_win", 0.0)),
    }
    mw = _summarize_market((odds_markets or {}).get("1X2"))
    for k, v in mw.items():
        p = prob_map_1x2.get(k)
        if p is None:
            continue
        ev = p * float(v["best"]) - 1.0
        rows.append({
            "market": "1X2",
            "selection": k,
            "model_p": p,
            "best_odd": float(v["best"]),
            "avg_odd": float(v["avg"]),
            "books": int(v["n"]),
            "ev": ev,
        })

    # OU 2.5 (из poisson)
    over_p = float(poisson.get("over_2_5", 0.0))
    under_p = max(0.0, 1.0 - over_p)
    ou = _summarize_market((odds_markets or {}).get("OU_2_5"))
    ou_map = {"Over 2.5": over_p, "Under 2.5": under_p}
    for k, v in ou.items():
        p = ou_map.get(k)
        if p is None:
            continue
        ev = p * float(v["best"]) - 1.0
        rows.append({
            "market": "OU_2_5",
            "selection": k,
            "model_p": p,
            "best_odd": float(v["best"]),
            "avg_odd": float(v["avg"]),
            "books": int(v["n"]),
            "ev": ev,
        })

    # BTTS (из poisson)
    yes_p = float(poisson.get("both_score", 0.0))
    no_p = max(0.0, 1.0 - yes_p)
    btts = _summarize_market((odds_markets or {}).get("BTTS"))
    bt_map = {"Yes": yes_p, "No": no_p}
    for k, v in btts.items():
        p = bt_map.get(k)
        if p is None:
            continue
        ev = p * float(v["best"]) - 1.0
        rows.append({
            "market": "BTTS",
            "selection": k,
            "model_p": p,
            "best_odd": float(v["best"]),
            "avg_odd": float(v["avg"]),
            "books": int(v["n"]),
            "ev": ev,
        })

    df = pd.DataFrame(rows)
    "coerce" df.empty:
        return df
    df["ev_pct"] = (df["ev"] * 100).round(2)
    df["model_p_pct"] = (df["model_p"] * 100).round(1)
    df["best_odd"] = df["best_odd"].round(3)
    df["avg_odd"] = df["avg_odd"].round(3)
    df["value_vs_avg"] = (df["best_odd"] / (df["avg_odd"].replace(0, np.nan))).round(3)
    df["value_vs_avg"] = df["value_vs_avg"].fillna(0.0)
    return df.sort_values(["ev"], ascending=False)

def _calc_movement_signals(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Return movement table for 10/30/60 minutes."""
    out=[]
    for w in (10,30,60):
        m = _movement_from_history(df_hist, w)
        if not m.empty:
            out.append(m)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    # steam score: stronger when best moves down (odds shorten) significantly and books high
    # negative best_odd_chg => odds decreased => market moving towards selection
    df["steam_score"] = (-df["best_odd_chg_pct"]).clip(lower=0) * (df["books_last"].fillna(0).astype(float).clip(lower=1)/5.0)
    df["steam_score"] = df["steam_score"].round(2)
    df["best_odd_chg"] = df["best_odd_chg"].round(3)
    df["avg_odd_chg"] = df["avg_odd_chg"].round(3)
    df["best_odd_chg_pct"] = df["best_odd_chg_pct"].round(2)
    df["avg_odd_chg_pct"] = df["avg_odd_chg_pct"].round(2)
    return df.sort_values(["steam_score","best_odd_chg_pct"], ascending=[False, True])



st.set_page_config(
    page_title="Sport Analyzer",
    page_icon="🏆",
    layout="wide",
)

# ── Стили ─────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
.main-header { font-size: 28px; font-weight: 800; margin: 0 0 8px 0; }
.sub-header  { font-size: 18px; font-weight: 700; margin: 12px 0 6px 0; }
.card { border:1px solid #e8e8e8; border-radius: 14px; padding: 14px; margin-bottom: 10px; }
.rec-item { padding: 10px 12px; border-radius: 12px; border: 1px solid #efefef; margin: 8px 0; }
.kpi { font-size: 22px; font-weight: 800; }
.small { color: #777; font-size: 12px; }
.conf-high   { color:#2f9e44; font-weight:700; }
.conf-medium { color:#e67700; font-weight:700; }
.conf-low    { color:#c92a2a; font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Вспомогательные функции ───────────────────────────────────────────

def _save_to_db(result: dict):
    """
     Сохраняет результат анализа в SQLite. 
    Исправлено: правильные отступы + всё внутри `with`.
    """
    cfg = Config()
    poisson = result.get("poisson", {})
    p = result.get("final_probs") or {
        "home_win": poisson.get("home_win", 0),
        "draw": poisson.get("draw", 0),
        "away_win": poisson.get("away_win", 0),
    }

    best = max(
        [
            ("home_win", p.get("home_win", 0)),
            ("draw", p.get("draw", 0)),
            ("away_win", p.get("away_win", 0)),
        ],
        key=lambda x: x[1],
    )

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
                INSERT INTO analyses (match, datetime, prediction, confidence, analysis_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.get("match", ""),
                    result.get("datetime", ""),
                    best[0],
                    float(result.get("confidence", 0.0)),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            c.commit()
    except Exception as e:
        logger.exception("DB write failed in _save_to_history: %s", e)
        return


def _load_history() -> pd.DataFrame:
    try:
        with sqlite3.connect(Config.DB_PATH, timeout=10) as c:
            return pd.read_sql(
                """
                SELECT created_at, match, prediction, confidence
                FROM analyses
                ORDER BY id DESC
                LIMIT 500
                """,
                c,
            )
    except Exception as e:
        logger.exception("DB read failed: %s", e)
        return pd.DataFrame()


def _load_elo() -> pd.DataFrame:
    try:
        with sqlite3.connect(Config.DB_PATH, timeout=10) as c:
            return pd.read_sql(
                "SELECT name, league, elo FROM team_elo ORDER BY elo DESC",
                c,
            )
    except Exception as e:
        logger.exception("DB read failed: %s", e)
        return pd.DataFrame()


def _prob_bar(label: str, prob: float, color: str):
    width = int(prob * 280)
    st.markdown(
        f'<div style="margin:4px 0">'
        f'<span style="font-weight:600;width:170px;display:inline-block">{label}</span>'
        f'<span style="background:{color};display:inline-block;'
        f'height:20px;border-radius:4px;vertical-align:middle;'
        f'width:{width}px"></span>'
        f'&nbsp;<b>{prob*100:.1f}%</b></div>',
        unsafe_allow_html=True,
    )


# ── Рендер результата ─────────────────────────────────────────────────

# ── Confidence Engine ────────────────────────────────────────────
def _safe_num(x, default=0.0):
    try:
        if x is None: return default
        return float(x)
    except Exception as e:
        return default

def _compute_confidence_engine(result: dict) -> dict:
    """Combine all signals into a single confidence score (0–100)."""
    score = 50.0  # neutral baseline
    reasons = []

    # ---- FORM ----
    tf = result.get("team_form") or {}
    try:
        h5 = (tf.get("home_last5") or {}).get("pts", 0)
        a5 = (tf.get("away_last5") or {}).get("pts", 0)
        delta = _safe_num(h5) - _safe_num(a5)
        score += delta * 1.2
        if abs(delta) >= 4:
            reasons.append("form edge")
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)

    # ---- FATIGUE ----
    fat = result.get("fatigue") or {}
    try:
        hr = _safe_num((fat.get("home") or {}).get("rest_days"))
        ar = _safe_num((fat.get("away") or {}).get("rest_days"))
        score += (hr - ar) * 1.5
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)

    # ---- MOTIVATION ----
    mot = result.get("motivation") or {}
    try:
        hflag = (mot.get("home") or {}).get("near_releg") or (mot.get("home") or {}).get("near_top4")
        aflag = (mot.get("away") or {}).get("near_releg") or (mot.get("away") or {}).get("near_top4")
        if hflag and not aflag:
            score += 4
            reasons.append("home motivation")
        elif aflag and not hflag:
            score -= 4
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)

    # ---- TRENDS ----
    tr = result.get("trends") or {}
    try:
        hs = _safe_num((tr.get("home") or {}).get("streak_unbeaten"))
        as_ = _safe_num((tr.get("away") or {}).get("streak_unbeaten"))
        score += (hs - as_) * 0.8
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)

    # ---- TRAVEL ----
    trv = result.get("travel") or {}
    try:
        away_ratio_home = _safe_num((trv.get("home") or {}).get("away_ratio_lastN"))
        away_ratio_away = _safe_num((trv.get("away") or {}).get("away_ratio_lastN"))
        score += (away_ratio_away - away_ratio_home) * 6
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)

    # ---- MARKET (value/steam proxy) ----
    api = result.get("api_sports") or {}
    odds = (api.get("odds") or {})
    if odds:
        score += 2
        reasons.append("market data present")

    # normalize
    score = max(0, min(100, score))

    risk = "MED"
    if score >= 70:
        risk = "LOW"
    elif score <= 40:
        risk = "HIGH"

    return {
        "confidence": round(score,1),
        "risk": risk,
        "reasons": reasons,
    }

def _detect_bet_profile(result: dict) -> dict:
    """Detect bet profile: VALUE / STEAM / TREND / AVOID.
    Returns dict with profile, tag, reasons, and optional suggestions.
    """
    ce = (result.get("confidence_engine") or {})
    conf = _safe_num(ce.get("confidence"), 50.0)

    api = result.get("api_sports") or {}
    odds = (api.get("odds") or {})
    markets = (odds.get("markets") or {}) if isinstance(odds, dict) else {}

    # Compute value table if odds present
    dfv = pd.DataFrame()
    if markets:
        try:
            dfv = _calc_value_signals_all(result, markets)
        except Exception as e:
            dfv = pd.DataFrame()

    # Steam proxy: if we have steam events in DB already attached? not; use odds history if present in DB via render_result chart block.
    # In opportunities we already compute steam separately; here we use a lightweight heuristic: if odds present and books high + best>avg indicates market inefficiency.
    reasons=[]
    profile="TREND PLAY"
    tag="🟡"
    suggestions=[]

    # Value candidate
    best_value = None
    if not dfv.empty:
        cand = dfv.sort_values(["ev"], ascending=False).head(1)
        if not cand.empty:
            best_value = cand.iloc[0].to_dict()
            ev_pct = float(best_value.get("ev_pct", 0.0))
            books = int(best_value.get("books", 0))
            if ev_pct >= 5.0 and books >= 6 and conf >= 55:
                profile="VALUE BET"
                tag="🟢"
                reasons.append(f"EV {ev_pct:.1f}%")
                if books >= 10:
                    reasons.append("liquid market")
                suggestions.append(f"{best_value.get('market')} / {best_value.get('selection')}")

    # Trend-based: use streaks signals
    tr = result.get("trends") or {}
    if profile != "VALUE BET":
        try:
            hs = int((tr.get("home") or {}).get("streak_scoring") or 0)
            as_ = int((tr.get("away") or {}).get("streak_scoring") or 0)
            hbtts = int((tr.get("home") or {}).get("streak_btts") or 0)
            abtts = int((tr.get("away") or {}).get("streak_btts") or 0)
            if (hs >= 5 and as_ >= 3) or (hbtts >= 4 and abtts >= 3):
                profile="TREND PLAY"
                tag="🟡"
                reasons.append("strong scoring/BTTS streaks")
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

    # Steam-follow: if we have movement signals cached in result (not by default). We'll approximate steam-follow if best_value exists and best<<avg (sharp) OR books high and value_vs_avg notable.
    if profile != "VALUE BET" and best_value is not None:
        try:
            val_vs_avg = float(best_value.get("value_vs_avg", 1.0))
            books = int(best_value.get("books", 0))
            ev_pct = float(best_value.get("ev_pct", 0.0))
            # If best significantly better than avg (arbitrage-ish), it's usually because a few books moved faster -> "steam follow"
            if books >= 8 and val_vs_avg >= 1.03 and ev_pct >= 2.0:
                profile="STEAM FOLLOW"
                tag="🔵"
                reasons.append("market dispersion (best vs avg)")
                suggestions.append(f"{best_value.get('market')} / {best_value.get('selection')}")

        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

    # Trap risk: anti-trap heuristics
    # - low liquidity (books<4)
    # - very low rest days combined with high travel
    # - confidence low
    trap=False
    if best_value is not None:
        try:
            books = int(best_value.get("books", 0))
            if books <= 3:
                trap=True
                reasons.append("low liquidity (few books)")
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)
    try:
        fat = result.get("fatigue") or {}
        hr = _safe_num((fat.get("home") or {}).get("rest_days"), 3.0)
        ar = _safe_num((fat.get("away") or {}).get("rest_days"), 3.0)
        trv = result.get("travel") or {}
        har = _safe_num((trv.get("home") or {}).get("away_ratio_lastN"), 0.5)
        aar = _safe_num((trv.get("away") or {}).get("away_ratio_lastN"), 0.5)
        # very short rest + heavy away schedule increases variance
        if (hr <= 2 and har >= 0.7) or (ar <= 2 and aar >= 0.7):
            trap=True
            reasons.append("schedule stress (rest+travel)")
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)
    if conf <= 40:
        trap=True
        reasons.append("low confidence")

    if trap:
        profile="AVOID"
        tag="🔴"

    return {
        "profile": profile,
        "tag": tag,
        "reasons": reasons[:5],
        "suggestions": suggestions[:3],
    }

def _classify_profile(ev_pct: float, steam: float, books: int, confidence: float) -> str:
    # simple, deterministic labels for table
    if confidence <= 40 or books <= 3:
        return "AVOID"
    if ev_pct >= 5 and books >= 6 and confidence >= 55:
        return "VALUE BET"
    if steam >= 3 and books >= 6:
        return "STEAM FOLLOW"
    return "TREND PLAY"

def _trap_flags(ev_pct: float, steam: float, books: int, confidence: float, rest_home: float = None, rest_away: float = None) -> str:
    flags=[]
    try:
        if books <= 3:
            flags.append("low_liquidity")
        if confidence <= 40:
            flags.append("low_conf")
        if abs(ev_pct) >= 15 and books <= 5:
            flags.append("spike_ev")
        # steam but low books can be fake
        if steam >= 3 and books <= 4:
            flags.append("steam_low_books")
        if rest_home is not None and rest_home <= 2:
            flags.append("home_short_rest")
        if rest_away is not None and rest_away <= 2:
            flags.append("away_short_rest")
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)
    return ",".join(flags) if flags else "—"

def _bet_size_units(profile: str, confidence: float, ev_pct: float, flags: str) -> float:
    # Conservative staking (units): 0..1
    if profile == "AVOID":
        return 0.0
    base = 0.25
    if profile == "TREND PLAY":
        base = 0.25
    elif profile == "STEAM FOLLOW":
        base = 0.5
    elif profile == "VALUE BET":
        base = 0.75

    # scale by confidence
    c = max(0.0, min(100.0, float(confidence)))
    scale = 0.5 + (c/100.0)  # 0.5..1.5
    u = base * scale

    # bonus for strong EV
    if ev_pct >= 8:
        u *= 1.1
    if ev_pct >= 12:
        u *= 1.15

    # penalties for flags
    if flags and flags != "—":
        u *= 0.7

    # clamp
    return round(max(0.0, min(1.0, u)), 2)

    def _risk_level(confidence: float, flags: str = "—") -> str:
        c = _safe_num(confidence, 50.0)
        if flags and flags != "—":
            # any flags raise risk by one notch
            if c >= 75:
                return "MED"
            return "HIGH"
        if c >= 70:
            return "LOW"
        if c >= 50:
            return "MED"
        return "HIGH"

    def _fmt_pct(x, digits=1):
        try:
            return f"{float(x)*100:.{digits}f}%"
        except Exception as e:
            return "—"

    def _generate_plain_forecast(result: dict) -> str:
        """Human-readable forecast in Russian based on all computed blocks."""
        teams = result.get("teams") or {}
        home = teams.get("home") or result.get("home_team") or "Home"
        away = teams.get("away") or result.get("away_team") or "Away"

        ce = result.get("confidence_engine") or {}
        bp = result.get("bet_profile") or {}

        lines = []
        lines.append(f"### 🧾 Прогноз простым языком")
        lines.append(f"**Матч:** {home} — {away}")

        # headline
        conf = ce.get("confidence")
        risk = ce.get("risk")
        if conf is not None:
            lines.append(f"**Уверенность модели:** {conf}/100 · **Риск:** {risk}")
        if bp:
            lines.append(f"**Профиль ставки:** {bp.get('tag','')} **{bp.get('profile','')}**")

        # key factors
        lines.append("")
        lines.append("#### Почему модель так думает")
        bullets = []
        tf = result.get("team_form") or {}
        try:
            h5 = (tf.get("home_last5") or {}).get("pts")
            a5 = (tf.get("away_last5") or {}).get("pts")
            if h5 is not None and a5 is not None:
                bullets.append(f"Форма (посл. 5): **{home} {h5} очк.** vs **{away} {a5} очк.**")
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

        fat = result.get("fatigue") or {}
        try:
            hr = (fat.get("home") or {}).get("rest_days")
            ar = (fat.get("away") or {}).get("rest_days")
            if hr is not None and ar is not None:
                bullets.append(f"Отдых: **{home} {hr} дн.** vs **{away} {ar} дн.**")
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

        mot = result.get("motivation") or {}
        try:
            hm = mot.get("home") or {}
            am = mot.get("away") or {}
            if hm.get("found") and am.get("found"):
                bullets.append(f"Таблица: **{home} #{hm.get('rank')} ({hm.get('points')} очк.)** vs **{away} #{am.get('rank')} ({am.get('points')} очк.)**")
                flags=[]
                if hm.get("near_top4") or hm.get("near_top6") or hm.get("near_releg"):
                    flags.append(f"{home}: мотивация/зона (top/releg)")
                if am.get("near_top4") or am.get("near_top6") or am.get("near_releg"):
                    flags.append(f"{away}: мотивация/зона (top/releg)")
                if flags:
                    bullets.append("Мотивация: " + "; ".join(flags))
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

        tr = result.get("trends") or {}
        try:
            ht = tr.get("home") or {}
            at = tr.get("away") or {}
            if ht and at:
                bullets.append(f"Серии (last10): {home} unbeaten **{ht.get('streak_unbeaten')}**, {away} unbeaten **{at.get('streak_unbeaten')}**")
                bullets.append(f"BTTS/O2.5 streak: {home} BTTS **{ht.get('streak_btts')}**, O2.5 **{ht.get('streak_over2_5')}** · {away} BTTS **{at.get('streak_btts')}**, O2.5 **{at.get('streak_over2_5')}**")
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

        trv = result.get("travel") or {}
        try:
            htrv = trv.get("home") or {}
            atrv = trv.get("away") or {}
            if htrv and atrv and htrv.get("n"):
                bullets.append(f"Календарь/выезды: {home} away_ratio **{htrv.get('away_ratio_lastN')}**, {away} away_ratio **{atrv.get('away_ratio_lastN')}**")
        except Exception as e:
            logger.debug("Ignored exception: %s", e, exc_info=True)

        if not bullets:
            bullets.append("Контекстных данных пока мало — прогноз основан в основном на базовой модели и рынке.")

        lines.extend([f"- {b}" for b in bullets])

        # market & suggestion
        api = result.get("api_sports") or {}
        odds = (api.get("odds") or {})
        markets = (odds.get("markets") or {}) if isinstance(odds, dict) else {}
        if markets:
            try:
                dfv = _calc_value_signals_all(result, markets)
                if not dfv.empty:
                    best = dfv.sort_values("ev", ascending=False).head(1).iloc[0].to_dict()
                    lines.append("")
                    lines.append("#### Что выглядит лучше по рынку")
                    lines.append(f"- Лучшее value: **{best.get('market')} / {best.get('selection')}**")
                    lines.append(f"- Model P: **{best.get('model_p_pct'):.1f}%** · Best odd: **{best.get('best_odd'):.2f}** · EV: **{best.get('ev_pct'):.2f}%** · Books: **{int(best.get('books',0))}**")
            except Exception as e:
                logger.debug("Ignored exception: %s", e, exc_info=True)
        else:
            lines.append("")
            lines.append("#### Рынок")
            lines.append("- Коэффициенты по API-Sports пока недоступны/пустые для этого матча.")

        # caveats
        lines.append("")
        lines.append("#### Важно помнить")
        cave=[]
        if (api.get("lineups") or []) == []:
            cave.append("Составы могут появиться ближе к началу матча — до этого риск выше.")
        if bp and bp.get("profile")=="AVOID":
            cave.append("Система пометила матч как AVOID — лучше пропустить или снизить размер ставки.")
        if not cave:
            cave.append("Это вероятностная оценка, а не гарантия исхода.")
        lines.extend([f"- {c}" for c in cave])

        return "\n".join(lines)

# ── Explain trap flags (human language) ──────────────────────────
_FLAG_EXPLANATIONS = {
    "low_liquidity": "Мало букмекерских контор дают линию → рынок слабый, коэффициент может быть нестабильным.",
    "low_conf": "Модель не уверена в исходе — слишком много противоречивых сигналов.",
    "spike_ev": "Слишком высокий EV при малом рынке — часто ложный value (перекос линии).",
    "steam_low_books": "Движение коэффициента есть, но рынок маленький — возможно ложный сигнал.",
    "home_short_rest": "Домашняя команда играла недавно → возможна усталость.",
    "away_short_rest": "Гостевая команда играла недавно → повышенный риск ошибок.",
}

def _explain_flags_human(flags: str) -> list:
    if not flags or flags == "—":
        return []
    out=[]
    for f in str(flags).split(","):
        f=f.strip()
        if not f:
            continue
        out.append(_FLAG_EXPLANATIONS.get(f, f))
    return out

def _badge(profile: str) -> str:
    p = (profile or "").upper()
    if "VALUE" in p:
        return "🟢"
    if "STEAM" in p:
        return "🔵"
    if "AVOID" in p:
        return "🔴"
    return "🟡"

def _one_line_summary(result: dict) -> str:
    teams = result.get("teams") or {}
    home = teams.get("home") or result.get("home_team") or "Home"
    away = teams.get("away") or result.get("away_team") or "Away"

    ce = result.get("confidence_engine") or {}
    bp = result.get("bet_profile") or {}
    conf = ce.get("confidence")
    risk = ce.get("risk") or _risk_level(conf if conf is not None else 50.0, "—")
    profile = bp.get("profile") or "TREND PLAY"
    tag = bp.get("tag") or _badge(profile)

    # suggested market/selection (if present)
    sugg = ""
    try:
        if bp.get("suggestions"):
            sugg = str(bp["suggestions"][0])
    except Exception as e:
        sugg = ""

    # top reason (short)
    why = ""
    try:
        rs = bp.get("reasons") or []
        if rs:
            why = str(rs[0])
    except Exception as e:
        logger.debug("Ignored exception: %s", e, exc_info=True)

    conf_txt = f"{conf}" if conf is not None else "—"
    parts = [
        f"{tag} {profile}",
        f"{home}—{away}",
    ]
    if sugg:
        parts.append(f"→ {sugg}")
    parts.append(f"Conf {conf_txt}")
    parts.append(f"Risk {risk}")
    if why:
        parts.append(f"· {why}")
    return " | ".join(parts)

def render_result(result: dict):
    home = result.get("home_team", "Home")
    away = result.get("away_team", "Away")

    probs = result.get("final_probs", {})
    poisson = result.get("poisson", {})
    weather = result.get("weather", {})
    h2h = result.get("h2h", {})
    news = result.get("news", {})
    conf = float(result.get("confidence", 0))
    conf_l = result.get("confidence_label", "")


    # ── AI Summary Mode ──────────────────────────────────────────────
    st.info(_one_line_summary(result))

    st.markdown("### Вероятности исходов")
    _prob_bar(f"🏠 {home}", probs.get("home_win", 0), "#51cf66")
    _prob_bar("🤝 Ничья", probs.get("draw", 0), "#ffd43b")
    _prob_bar(f"✈️ {away}", probs.get("away_win", 0), "#ff6b6b")

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚽ Голы (хоз.)", poisson.get("lambda_h", 0))
    c2.metric("⚽ Голы (гост.)", poisson.get("lambda_a", 0))
    c3.metric("📊 Тотал Б 2.5", f"{poisson.get('over_2_5', 0) * 100:.1f}%")
    c4.metric("✅ Обе забьют", f"{poisson.get('both_score', 0) * 100:.1f}%")

    st.markdown("---")
    t1, t2, t3 = st.columns(3)
    t1.metric("Тотал Б 1.5", f"{poisson.get('over_1_5', 0) * 100:.1f}%")
    t2.metric("Тотал Б 2.5", f"{poisson.get('over_2_5', 0) * 100:.1f}%")
    t3.metric("Тотал Б 3.5", f"{poisson.get('over_3_5', 0) * 100:.1f}%")

    st.markdown("---")
    css = "conf-high" if conf >= 60 else ("conf-medium" if conf >= 48 else "conf-low")
    st.markdown(
        f'<span class="{css}">Уверенность: {conf:.1f}% — {conf_l}</span>',
        unsafe_allow_html=True,
    )

    if weather.get("temperature") is not None:
        with st.expander("🌤️ Погода на матч"):
            w1, w2, w3 = st.columns(3)
            w1.metric("🌡️ Температура", f"{weather.get('temperature')}°C")
            w2.metric("🌧️ Осадки", f"{weather.get('precipitation', 0)} мм")
            w3.metric("💨 Ветер", f"{weather.get('wind_speed', 0)} км/ч")
            st.info(
                f"{weather.get('condition', '')} | "
                f"Влияние: {weather.get('impact_score', 0)}/100"
            )
            for note in weather.get("analysis", []):
                st.caption(note)

    if h2h.get("matches", 0) > 0:
        with st.expander(f"📋 Личные встречи ({h2h['matches']} матчей)"):
            h1, h2c, h3 = st.columns(3)
            h1.metric(f"🏠 {home}", f"{h2h.get('home_win_pct', 0)}%")
            h2c.metric("🤝 Ничья", f"{h2h.get('draw_pct', 0)}%")
            h3.metric(f"✈️ {away}", f"{h2h.get('away_win_pct', 0)}%")

    hn = news.get("home", {})
    an = news.get("away", {})
    if hn or an:
        with st.expander("📰 Новостной фон"):
            n1, n2 = st.columns(2)
            with n1:
                st.markdown(f"**🏠 {home}**")
                st.write(hn.get("sentiment_label", "N/A"))
                for t in hn.get("key_topics", []):
                    st.caption(t)
            with n2:
                st.markdown(f"**✈️ {away}**")
                st.write(an.get("sentiment_label", "N/A"))
                for t in an.get("key_topics", []):
                    st.caption(t)

with st.expander("🧩 Источники данных и фолбэки", expanded=False):
    result = st.session_state.get("result", {})
av = result.get("availability") or {}
    notes = result.get("notes") or []
    if av:
        for k, v in av.items():
            ok = bool(v.get("ok", True))
            if ok:
                st.write(f"✅ {k}")
            else:
                st.write(f"⚠️ {k}: {v.get('error')}")
    if notes:
        st.markdown("**Примечания:**")
        for n in notes:
            st.caption(n)

    with st.expander("💡 Рекомендации", expanded=True):
        for rec in result.get("recommendations", []):
            st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)





# ── Standings / motivation ───────────────────────────────────────
mot = result.get("motivation") or {}
st_payload = (result.get("standings") or {}).get("payload") if isinstance(result.get("standings"), dict) else None
if mot and isinstance(mot, dict):
    with st.expander("🏁 Таблица и мотивация (API-Sports)", expanded=False):

        def _mcard(title: str, d: dict):
            if not d or not d.get("found"):
                st.caption(f"{title}: нет данных таблицы")
                return
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("rank", d.get("rank"))
            with c2:
                st.metric("pts", d.get("points"))
            with c3:
                st.metric("played", d.get("played"))
            with c4:
                st.metric("GD", d.get("GD"))
            with c5:
                flags = []
                if d.get("near_top4"): flags.append("near top4")
                if d.get("near_top6"): flags.append("near top6")
                if d.get("near_releg"): flags.append("near releg")
                st.metric("flags", ", ".join(flags) if flags else "—")
            st.caption(f"dist_top4: {d.get('dist_top4')} · dist_top6: {d.get('dist_top6')} · dist_releg: {d.get('dist_releg')}")

        _mcard("🏠 Home", mot.get("home") or {})
        _mcard("✈️ Away", mot.get("away") or {})

        if st_payload:
            with st.expander("Показать таблицу (raw)", expanded=False):
                try:
                    # Extract a compact table
                    league = st_payload.get("league") or {}
                    table = (league.get("standings") or [[]])[0]
                    df = pd.DataFrame([{
                        "rank": r.get("rank"),
                        "team": (r.get("team") or {}).get("name"),
                        "pts": r.get("points"),
                        "played": (r.get("all") or {}).get("played"),
                        "W": (r.get("all") or {}).get("win"),
                        "D": (r.get("all") or {}).get("draw"),
                        "L": (r.get("all") or {}).get("lose"),
                        "GF": ((r.get("all") or {}).get("goals") or {}).get("for"),
                        "GA": ((r.get("all") or {}).get("goals") or {}).get("against"),
                        "GD": r.get("goalsDiff"),
                    } for r in table])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception as e:
                    logger.debug("Ignored exception: %s", e, exc_info=True)




# ── Confidence Engine ────────────────────────────────────────────
ce = result.get("confidence_engine") or {}
if ce:
    with st.expander("🔥 Confidence Engine", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Confidence", ce.get("confidence"))
        with c2:
            st.metric("Risk", ce.get("risk"))

bp = result.get("bet_profile") or {}
if bp:
    st.markdown(f"### {bp.get('tag','')} Bet Profile: **{bp.get('profile','')}**")
    if bp.get("reasons"):
        st.caption("Why: " + ", ".join(bp.get("reasons")))
    if bp.get("suggestions"):
        st.caption("Suggested: " + " · ".join(bp.get("suggestions")))

# Explain flags human-readable
flags = (result.get("bet_profile") or {}).get("reasons", [])
if flags:
    with st.expander("Explain flags", expanded=False):
        for fl in flags:
            txts = _explain_flags_human(fl)
            if txts:
                for t in txts:
                    st.write("• " + t)
            else:
                st.write("• " + fl)

        if ce.get("reasons"):
            st.caption("Signals: " + ", ".join(ce.get("reasons")))

# ── Travel / away-run ────────────────────────────────────────────
trv = result.get("travel") or {}
if trv and isinstance(trv, dict):
    with st.expander("✈️ Travel / Away-run", expanded=False):

        def _card(title, d):
            if not d or int(d.get("n",0))==0:
                st.caption(f"{title}: нет данных")
                return
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("away_streak", d.get("away_streak"))
            with c2: st.metric("home_streak", d.get("home_streak"))
            with c3: st.metric("away_ratio", d.get("away_ratio_lastN"))
            with c4: st.metric("venue_switches", d.get("venue_switches"))
            st.divider()

        _card("🏠 Home", trv.get("home") or {})
        _card("✈️ Away", trv.get("away") or {})

# ── Trends / streaks ─────────────────────────────────────────────
tr = result.get("trends") or {}
if tr and isinstance(tr, dict):
    with st.expander("📈 Тренды и серии (last 10)", expanded=False):

        def _tcard(title: str, d: dict):
            if not d or int(d.get("n", 0)) == 0:
                st.caption(f"{title}: нет данных")
                return
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("scoring", d.get("streak_scoring"))
                st.metric("conceding", d.get("streak_conceding"))
            with c2:
                st.metric("unbeaten", d.get("streak_unbeaten"))
                st.metric("winless", d.get("streak_winless"))
            with c3:
                st.metric("BTTS", d.get("streak_btts"))
                st.metric("O2.5", d.get("streak_over2_5"))
            with c4:
                st.metric("wins", d.get("streak_wins"))
                st.metric("losses", d.get("streak_losses"))
            st.caption(f"points_last5: {d.get('points_last5')} · clean_sheets_streak: {d.get('streak_clean_sheets')}")
            st.divider()

        _tcard("🏠 Home", tr.get("home") or {})
        _tcard("✈️ Away", tr.get("away") or {})

# ── Extra team context ───────────────────────────────────────────
tf = result.get("team_form") or {}
h2h = result.get("h2h") or []
if (tf and isinstance(tf, dict)) or (h2h and isinstance(h2h, list)):
    with st.expander("📊 Доп. данные по командам (форма / H2H)", expanded=False):

        def _show_form(title: str, d: dict):
            if not d or not isinstance(d, dict) or int(d.get("n", 0)) == 0:
                st.caption(f"{title}: нет данных")
                return
            st.markdown(f"**{title}** (n={d.get('n')})")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("W-D-L", f"{d.get('W')}-{d.get('D')}-{d.get('L')}")
            with c2:
                st.metric("pts", d.get("pts"))
            with c3:
                st.metric("GF/GA", f"{d.get('GF')}/{d.get('GA')}")
            with c4:
                st.metric("BTTS", f"{round(float(d.get('btts_rate',0))*100,1)}%" )
            with c5:
                st.metric("O2.5", f"{round(float(d.get('over2_5_rate',0))*100,1)}%" )
            st.caption(f"avg_GF={d.get('avg_GF')} avg_GA={d.get('avg_GA')} clean_sheets={d.get('clean_sheets')}")
            if isinstance(d.get("home"), dict) and int(d["home"].get("n",0))>0:
                st.caption(f"home: {d['home'].get('W')}-{d['home'].get('D')}-{d['home'].get('L')} pts {d['home'].get('pts')} GF/GA {d['home'].get('GF')}/{d['home'].get('GA')}")
            if isinstance(d.get("away"), dict) and int(d["away"].get("n",0))>0:
                st.caption(f"away: {d['away'].get('W')}-{d['away'].get('D')}-{d['away'].get('L')} pts {d['away'].get('pts')} GF/GA {d['away'].get('GF')}/{d['away'].get('GA')}")
            st.divider()

        # Home/away forms
        _show_form("🏠 Home last 5", tf.get("home_last5") or {})
        _show_form("✈️ Away last 5", tf.get("away_last5") or {})
        _show_form("🏠 Home last 10", tf.get("home_last10") or {})
        _show_form("✈️ Away last 10", tf.get("away_last10") or {})

        if h2h:
            st.markdown("**🤝 H2H (последние 10)**")
            try:
                dfh = pd.DataFrame(h2h)
                if not dfh.empty:
                    st.dataframe(dfh, use_container_width=True, hide_index=True)
            except Exception as e:
                logger.debug("Ignored exception: %s", e, exc_info=True)

# ── Team load / fatigue ──────────────────────────────────────────
fat = result.get("fatigue") or {}
if fat and isinstance(fat, dict):
    with st.expander("🧠 Нагрузка / усталость команд (API-Sports)", expanded=False):
        def _card(title: str, d: dict):
            if not d:
                st.info(f"{title}: нет данных")
                return
            st.markdown(f"**{title}**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("rest_days", d.get("rest_days"))
            with c2:
                st.metric("matches_7d", d.get("matches_7d"))
            with c3:
                st.metric("matches_14d", d.get("matches_14d"))
            with c4:
                st.metric("matches_30d", d.get("matches_30d"))
            st.caption(f"last_match_utc: {d.get('last_match_utc')}")

        _card("🏠 Home", fat.get("home") or {})
        _card("✈️ Away", fat.get("away") or {})


# ── Lineups / injuries ───────────────────────────────────────────
api_block = result.get("api_sports") or {}
if isinstance(api_block, dict) and (api_block.get("lineups") or api_block.get("injuries")):
    with st.expander("🧾 Составы и травмы (API-Sports)", expanded=False):
        lineups = api_block.get("lineups") or []
        injuries = api_block.get("injuries") or []

        if lineups:
            st.markdown("### 🧩 Lineups")
            for lu in lineups:
                team = (lu.get("team") or {}).get("name") or "Team"
                st.markdown(f"**{team}** · formation: {lu.get('formation')} · coach: {lu.get('coach')}")
                sx = lu.get("startXI") or []
                if sx:
                    df = pd.DataFrame(sx)
                    # nicer ordering
                    cols = [c for c in ["number", "pos", "name", "grid"] if c in df.columns]
                    st.dataframe(df[cols], use_container_width=True, hide_index=True)
                subs = lu.get("substitutes") or []
                if subs:
                    with st.expander(f"Запас ({team})", expanded=False):
                        df2 = pd.DataFrame(subs)
                        cols2 = [c for c in ["number", "pos", "name"] if c in df2.columns]
                        st.dataframe(df2[cols2], use_container_width=True, hide_index=True)
                st.divider()
        else:
            st.info("Lineups пока не доступны для этого матча (обычно появляются ближе к старту).")

        if injuries:
            st.markdown("### 🩹 Injuries")
            df = pd.DataFrame(injuries)
            if not df.empty:
                # flatten
                if "team" in df.columns:
                    df["team"] = df["team"].apply(lambda x: (x or {}).get("name"))
                if "player" in df.columns:
                    df["player"] = df["player"].apply(lambda x: (x or {}).get("name"))
                cols = [c for c in ["team","player","type","reason"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.info("Травмы по fixture могут быть пустыми (зависит от доступности в API).")

# ── Odds / value bets (API-Sports) ───────────────────────────────
api = result.get("api_sports") or {}
odds = (api.get("odds") or {})
markets = (odds.get("markets") or {}) if isinstance(odds, dict) else {}

if markets:
    with st.expander("🧾 Коэффициенты и value (API-Sports)", expanded=False):
        st.caption("EV считается как P(модель) × best_odd − 1. Рынки: 1X2, OU 2.5, BTTS.")
        try:
            dfv = _calc_value_signals_all(result, markets)
            if dfv.empty:
                st.info("Коэффициенты есть, но value не найдено (или рынки пустые).")
            else:
                st.dataframe(
                    dfv,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "model_p_pct": st.column_config.NumberColumn("Model P (%)", format="%.1f"),
                        "ev_pct": st.column_config.NumberColumn("EV (%)", format="%.2f"),
                        "best_odd": st.column_config.NumberColumn("Best", format="%.3f"),
                        "avg_odd": st.column_config.NumberColumn("Avg", format="%.3f"),
                    },
                )
        except Exception as e:
            st.warning(f"Не удалось посчитать value: {e}")

        # ── Charts + steam log for this fixture (from DB) ───────────
        try:
            fx_id = int((result.get("api_sports") or {}).get("fixture_id") or 0)
        except Exception as e:
            fx_id = 0

        if fx_id:
            try:
                import sqlite3, time as _time
                db_path = Config.DB_PATH
                since = _time.time() - 24 * 3600

                with sqlite3.connect(db_path, timeout=10) as c:
                    rows = c.execute(
                        "SELECT ts, market, selection, best_odd, avg_odd, books "
                        "FROM odds_snapshots WHERE fixture_id=? AND ts>=? ORDER BY ts ASC",
                        (fx_id, since),
                    ).fetchall()

                    if rows:
                        df_hist = pd.DataFrame(rows, columns=["ts", "market", "selection", "best_odd", "avg_odd", "books"])
                        st.markdown("#### 📉 График odds по матчу (best/avg)")
                        m1, m2 = st.columns(2)
                        with m1:
                            chart_market = st.selectbox(
                                "Market",
                                ["1X2", "OU_2_5", "BTTS"],
                                index=0,
                                key=f"res_chart_market_{fx_id}",
                            )
                        with m2:
                            sel_opts = {
                                "1X2": ["Home", "Draw", "Away"],
                                "OU_2_5": ["Over 2.5", "Under 2.5"],
                                "BTTS": ["Yes", "No"],
                            }.get(chart_market, [])
                            chart_sel = st.selectbox(
                                "Selection",
                                sel_opts,
                                index=0,
                                key=f"res_chart_sel_{fx_id}",
                            )
                        _plot_odds_history(df_hist, chart_market, chart_sel)

                        with st.expander("🧾 Steam events (по этому матчу)", expanded=False):
                            evs = c.execute(
                                "SELECT ts, market, selection, window_min, best_start, best_last, best_chg_pct, books, steam_score "
                                "FROM steam_events WHERE fixture_id=? AND ts>=? ORDER BY ts DESC LIMIT 50",
                                (fx_id, since),
                            ).fetchall()
                            if not evs:
                                st.info("Событий steam пока нет.")
                            else:
                                dfe = pd.DataFrame(
                                    evs,
                                    columns=[
                                        "ts",
                                        "market",
                                        "selection",
                                        "window_min",
                                        "best_start",
                                        "best_last",
                                        "best_chg_pct",
                                        "books",
                                        "steam_score",
                                    ],
                                )
                                dfe["datetime"] = pd.to_datetime(dfe["ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
                                st.dataframe(
                                    dfe[
                                        [
                                            "datetime",
                                            "window_min",
                                            "market",
                                            "selection",
                                            "best_start",
                                            "best_last",
                                            "best_chg_pct",
                                            "books",
                                            "steam_score",
                                        ]
                                    ],
                                    use_container_width=True,
                                    hide_index=True,
                                )
                    else:
                        st.info("История odds для графика пока не собрана. Открой вкладку 📡 Сигналы и сними пару snapshot'ов.")
            except Exception as _e:
                st.warning(f"Графики/steam log недоступны: {_e}")

with st.expander("🔧 Raw JSON"):
        st.json(result)


# ── Страницы ──────────────────────────────────────────────────────────

def render_page_analyze(analyzer: MatchAnalyzer, sports: SportsCollector, api_sports: ApiSportsCollector):
    st.markdown('<div class="main-header">🔍 Анализ матча</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Режим выбора матча",
        ["📅 Матчи на дату (реальные)", "✍️ Ручной ввод"],
        horizontal=True,
    )

    col_f, col_r = st.columns([1, 1.6], gap="large")

    # ── LEFT: inputs ────────────────────────────────────────────────
    with col_f:
        st.markdown("### Параметры")

        fixture_id = None
        home_raw = ""
        away_raw = ""
        h_id = 0
        a_id = 0

        if mode.startswith("📅"):
            date_sel = st.date_input("📅 Дата", value=datetime.today())
            source = st.selectbox(
                "Источник матчей",
                ["football-data (рекомендуется для статистики)", "API-Sports (fixtures)"],
                index=0,
            )

            matches = []
            if source.startswith("football-data"):
                # football-data: используем dateFrom=dateTo = выбранная дата
                with st.spinner("Загружаем матчи (football-data)…"):
                    dt = date_sel.strftime("%Y-%m-%d")
                    # переиспользуем коллектор: запрашиваем 0 дней вперёд, затем фильтруем точную дату через новый запрос
                    resp = sports.get(
                        "https://api.football-data.org/v4/matches",
                        params={"dateFrom": dt, "dateTo": dt},
                    )
                    if resp is not None and resp.status_code == 200:
                        for m in (resp.json() or {}).get("matches", []) or []:
                            matches.append({
                                "fixture_id": m.get("id"),
                                "utcDate": m.get("utcDate"),
                                "league": (m.get("competition") or {}).get("name"),
                                "home_team": (m.get("homeTeam") or {}).get("name"),
                                "away_team": (m.get("awayTeam") or {}).get("name"),
                                "home_team_id": (m.get("homeTeam") or {}).get("id"),
                                "away_team_id": (m.get("awayTeam") or {}).get("id"),
                                "stadium": m.get("venue"),
                            })
            else:
                if not api_sports.is_configured():
                    st.warning("API_FOOTBALL_KEY не задан. Добавь его в Secrets/ENV, чтобы получать fixtures из API-Sports.")
                else:
                    with st.spinner("Загружаем матчи (API-Sports)…"):
                        matches = api_sports.get_fixtures_by_date(date_sel.strftime("%Y-%m-%d"))

            if not matches:
                st.info("Матчи не найдены на выбранную дату (или нет ключа). Можно перейти на ручной ввод.")
            else:
                def _label(m):
                    d = (m.get("utcDate") or "")[11:16] if m.get("utcDate") else ""
                    lg = m.get("league") or ""
                    return f"{d} • {lg} • {m.get('home_team')} — {m.get('away_team')}"
                opts = { _label(m): m for m in matches if m.get("home_team") and m.get("away_team") }
                sel = st.selectbox("Матч", list(opts.keys()))
                chosen = opts.get(sel) or {}

                fixture_id = chosen.get("fixture_id")
                home_raw = chosen.get("home_team") or ""
                away_raw = chosen.get("away_team") or ""
                h_id = int(chosen.get("home_team_id") or 0)
                a_id = int(chosen.get("away_team_id") or 0)

                # время матча
                try:
                    dt = chosen.get("utcDate") or ""
                    match_date = datetime.fromisoformat(dt.replace("Z", "+00:00")).date() if dt else datetime.today().date()
                    match_time = datetime.fromisoformat(dt.replace("Z", "+00:00")).time() if dt else datetime.utcnow().time()
                except Exception as e:
                    match_date = date_sel
                    match_time = datetime.utcnow().time()

                st.caption(f"Fixture ID: {fixture_id}")
                st.text_input("🏠 Домашняя команда", value=home_raw, disabled=True)
                st.text_input("✈️ Гостевая команда", value=away_raw, disabled=True)
                city = st.text_input("📍 Город (необязательно)", placeholder="London")
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.date_input("📅 Дата", value=match_date, disabled=True)
                with t_col2:
                    st.time_input("🕐 Время UTC", value=match_time, disabled=True)

                if source.startswith("API-Sports"):
                    st.info("ℹ️ ID команд здесь из API-Sports. Для части твоей текущей статистики лучше использовать football-data, либо позже сделаем маппинг.")
        else:
            home_raw = st.text_input("🏠 Домашняя команда", placeholder="Arsenal")
            away_raw = st.text_input("✈️ Гостевая команда", placeholder="Chelsea")
            city = st.text_input("📍 Город (необязательно)", placeholder="London")
            d_col, t_col = st.columns(2)
            with d_col:
                match_date = st.date_input("📅 Дата", value=datetime.today())
            with t_col:
                match_time = st.time_input("🕐 Время UTC")
            h_id = st.number_input("ID хозяев (football-data)", value=0, step=1)
            a_id = st.number_input("ID гостей (football-data)", value=0, step=1)

        neutral = st.checkbox("Нейтральное поле")
        run_btn = st.button("🚀 Анализировать", use_container_width=True, type="primary")

    # ── RIGHT: results ──────────────────────────────────────────────
    with col_r:
        if run_btn:
            if not home_raw or not away_raw:
                st.error("Введите обе команды")
                return

            home = normalize_team_name(home_raw)
            away = normalize_team_name(away_raw)

            with st.spinner("Анализируем…"):
                result = analyzer.analyze_match(
                    home_team=home,
                    away_team=away,
                    match_datetime=f"{match_date}T{match_time}:00",
                    city=(locals().get("city") or None),
                    home_team_id=int(h_id) if h_id else None,
                    away_team_id=int(a_id) if a_id else None,
                    neutral_field=neutral,
                )

            # ── Odds + signals (API-Sports) ───────────────────────────
            if api_sports.is_configured() and fixture_id:
                try:
                    odds = api_sports.get_odds_for_fixture(int(fixture_id))
                    result["api_sports"] = {"fixture_id": int(fixture_id), "odds": odds}

                    # lineups + injuries (если доступны)
                    try:
                        result["api_sports"]["lineups"] = api_sports.get_fixture_lineups(int(fixture_id), use_cache=True)
                        result["api_sports"]["injuries"] = api_sports.get_fixture_injuries(int(fixture_id), use_cache=True)
                    except Exception as e:
                        logger.debug("Ignored exception: %s", e, exc_info=True)


                    # team context: fatigue + form + H2H (API-Sports)
                    try:
                        match_iso = str(chosen.get("utcDate") or "") or f"{match_date}T{match_time}:00"
                        hid = chosen.get("home_team_id")
                        aid = chosen.get("away_team_id")
                        if hid and aid:
                            result["fatigue"] = {
                                "home": api_sports.compute_fatigue_metrics(int(hid), match_iso),
                                "away": api_sports.compute_fatigue_metrics(int(aid), match_iso),
                            }
                            result["team_form"] = {
                                "home_last5": api_sports.compute_team_form_metrics(int(hid), match_iso, last=5),
                                "away_last5": api_sports.compute_team_form_metrics(int(aid), match_iso, last=5),
                                "home_last10": api_sports.compute_team_form_metrics(int(hid), match_iso, last=10),
                                "away_last10": api_sports.compute_team_form_metrics(int(aid), match_iso, last=10),
                            }
                            result["trends"] = {
                                "home": api_sports.compute_trend_streaks(int(hid), match_iso, last=10),
                                "away": api_sports.compute_trend_streaks(int(aid), match_iso, last=10),
                            }
                            result["travel"] = {
                                "home": api_sports.compute_travel_metrics(int(hid), match_iso, last=10),
                                "away": api_sports.compute_travel_metrics(int(aid), match_iso, last=10),
                            }
                            result["h2h"] = api_sports.get_h2h(int(hid), int(aid), last=10)
                            # standings / motivation (league table)
                            try:
                                lid = chosen.get("league_id")
                                season = chosen.get("season")
                                if lid and season:
                                    st_payload = api_sports.get_standings(int(lid), int(season), use_cache=True)
                                    result["standings"] = {"league_id": int(lid), "season": int(season), "payload": st_payload}
                                    result["motivation"] = {
                                        "home": api_sports.compute_motivation_metrics(st_payload, int(hid)),
                                        "away": api_sports.compute_motivation_metrics(st_payload, int(aid)),
                                    }
                            except Exception as e:
                                logger.debug("Ignored exception: %s", e, exc_info=True)
                    except Exception as e:
                        logger.debug("Ignored exception: %s", e, exc_info=True)
                except Exception as e:
                    result["api_sports"] = {"fixture_id": int(fixture_id), "odds_error": str(e)}

            st.session_state["last_result"] = result
            _save_to_db(result)
            render_result(result)

        elif "last_result" in st.session_state:
            render_result(st.session_state["last_result"])


def render_page_schedule(sports: SportsCollector, analyzer: MatchAnalyzer):
    st.markdown('<div class="main-header">📅 Расписание</div>', unsafe_allow_html=True)

    days = st.slider("Дней вперёд", 1, 14, 7)

    with st.spinner("Загружаем…"):
        matches = sports.get_matches(days_ahead=days)

    if not matches:
        st.warning("Матчи не найдены. Проверьте FOOTBALL_DATA_KEY.")
        return

    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%d.%m %H:%M")

    disp = df[["date", "competition", "home_team", "away_team"]].rename(
        columns={
            "date": "Дата",
            "competition": "Лига",
            "home_team": "Хозяева",
            "away_team": "Гости",
        }
    )

    leagues = ["Все"] + sorted(disp["Лига"].dropna().unique().tolist())
    sel_l = st.selectbox("Фильтр по лиге", leagues)
    if sel_l != "Все":
        disp = disp[disp["Лига"] == sel_l]

    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    filtered_idx = disp.index.tolist()
    if not filtered_idx:
        return

    sel = st.selectbox(
        "Выберите матч для анализа",
        options=filtered_idx,
        format_func=lambda i: (
            f"{disp.loc[i,'Дата']} | {disp.loc[i,'Хозяева']} vs {disp.loc[i,'Гости']}"
        ),
    )

    if st.button("🚀 Анализировать матч"):
        orig = matches[sel]
        with st.spinner("Анализ…"):
            result = analyzer.analyze_match(
                home_team=normalize_team_name(orig["home_team"]),
                away_team=normalize_team_name(orig["away_team"]),
                match_datetime=orig.get("date"),
                home_team_id=orig.get("home_team_id"),
                away_team_id=orig.get("away_team_id"),
            )
        st.session_state["last_result"] = result
        _save_to_db(result)
        result['confidence_engine'] = _compute_confidence_engine(result)
        result['bet_profile'] = _detect_bet_profile(result)
        render_result(result)


def render_page_history():
    st.markdown('<div class="main-header">📊 История прогнозов</div>', unsafe_allow_html=True)
    df = _load_history()
    if df.empty:
        st.info("История пуста — сделайте первый анализ")
        return

    s1, s2, s3 = st.columns(3)
    s1.metric("Всего", len(df))
    s2.metric("Средняя уверенность", f"{df['confidence'].mean():.1f}%")
    s3.metric(">60% уверенность", int((df["confidence"] > 60).sum()))

    st.dataframe(
        df.rename(
            columns={
                "created_at": "Дата",
                "match": "Матч",
                "prediction": "Прогноз",
                "confidence": "Уверенность %",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    chart = df[["created_at", "confidence"]].copy()
    chart["created_at"] = pd.to_datetime(chart["created_at"])
    st.line_chart(chart.set_index("created_at"))


def render_page_elo():
    st.markdown('<div class="main-header">📈 ELO рейтинги</div>', unsafe_allow_html=True)
    st.info("ELO обновляется через `python scripts/update_elo.py`. Начальный рейтинг: 1500.")

    df = _load_elo()
    if df.empty:
        st.warning("Рейтинги ещё не сформированы")
        return

    df = df.sort_values("elo", ascending=False).reset_index(drop=True)
    df.index += 1

    leagues = ["Все"] + sorted(df["league"].dropna().unique().tolist())
    sel = st.selectbox("Фильтр по лиге", leagues)
    if sel != "Все":
        df = df[df["league"] == sel]

    left, right = st.columns([1, 1.4])
    with left:
        st.dataframe(
            df.rename(columns={"name": "Команда", "league": "Лига", "elo": "ELO"}),
            use_container_width=True,
        )
    with right:
        if not df.empty:
            st.bar_chart(df.head(20).set_index("name")["elo"])


# ════════════════════════════════════════════════════════════════════
# Инициализация
# ════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_resources():
    cfg = Config()
    run_migrations(cfg.DB_PATH)
    sports = SportsCollector(cfg, db_path=cfg.DB_PATH)
    api_sports = ApiSportsCollector(cfg, db_path=cfg.DB_PATH)
    weather = WeatherCollector(db_path=cfg.DB_PATH)
    news = NewsCollector(cfg)
    xg = XGCollector(db_path=cfg.DB_PATH)
    analyzer = MatchAnalyzer(cfg, sports, weather, news, xg_collector=xg)
    return analyzer, sports, api_sports


analyzer, sports, api_sports = init_resources()

d
st.markdown("### 📉 График odds (best/avg)")
mcol1, mcol2 = st.columns(2)
with mcol1:
    chart_market = st.selectbox("Market", ["1X2","OU_2_5","BTTS"], index=0, key="chart_market")
with mcol2:
    # selections depend on market
    sel_opts = {
        "1X2": ["Home","Draw","Away"],
        "OU_2_5": ["Over 2.5","Under 2.5"],
        "BTTS": ["Yes","No"],
    }.get(chart_market, [])
    chart_sel = st.selectbox("Selection", sel_opts, index=0, key="chart_sel")

_plot_odds_history(dfh, chart_market, chart_sel)

with st.expander("🧾 Steam events log", expanded=False):
    evs = api_sports.get_steam_events(int(fixture_id), hours=float(hours))
    if not evs:
        st.info("Событий steam пока нет.")
    else:
        dfe = pd.DataFrame(evs)
        st.dataframe(
            dfe[["datetime","window_min","market","selection","best_start","best_last","best_chg_pct","books","steam_score"]],
            use_container_width=True,
            hide_index=True,
        )

def render_page_signals(api_sports: ApiSportsCollector):
    st.markdown('<div class="main-header">📡 Сигналы (API-Sports)</div>', unsafe_allow_html=True)

    if not api_sports.is_configured():
        st.error("API_FOOTBALL_KEY не задан. Добавь его в Secrets Streamlit Cloud, чтобы получать коэффициенты/сигналы.")
        st.code('API_FOOTBALL_KEY="..."')
        return

    st.caption("Режим **безопасный**: снимаем snapshots только по выбранному матчу (не по всем матчам дня). Частота — раз в 10 минут.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        date_sel = st.date_input("📅 Дата", value=datetime.today())
    with c2:
        interval_min = st.selectbox("Интервал snapshot", [10, 15, 30], index=0)
    with c3:
        hours = st.slider("Показывать историю (часов)", 1, 48, 24)

    with st.spinner("Загружаем fixtures…"):
        fixtures = api_sports.get_fixtures_by_date(date_sel.strftime("%Y-%m-%d"), use_cache=True)

    if not fixtures:
        st.warning("На выбранную дату fixtures не найдены (или лимит/ключ).")
        return

    df_fx = pd.DataFrame(fixtures)
    df_fx["time"] = df_fx["utcDate"].astype(str).str[11:16]
    df_fx["match"] = df_fx["home_team"].astype(str) + " — " + df_fx["away_team"].astype(str)
    df_fx = df_fx.sort_values(["time", "league", "match"])

    # select one fixture
    def _label(r):
        return f"{r.get('time','')} • {r.get('league','')} • {r.get('match','')}"

    options = { _label(r): int(r.get("fixture_id")) for r in df_fx.to_dict("records") if r.get("fixture_id") }
    sel_label = st.selectbox("Матч (для snapshot и истории)", list(options.keys()))
    fixture_id = options.get(sel_label)

    # snapshot controls
    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        force = st.button("📸 Снять snapshot сейчас", use_container_width=True)
    with b2:
        show_table = st.checkbox("Показать таблицу истории", value=True)
    with b3:
        st.caption("Snapshot сохраняет best/avg/books по рынкам 1X2, OU 2.5, BTTS.")

    interval_seconds = int(interval_min) * 60
    take = force or api_sports.should_take_snapshot(int(fixture_id), interval_seconds=interval_seconds)

    if take:
        with st.spinner("Снимаем snapshot odds…"):
            odds = api_sports.get_odds_for_fixture(int(fixture_id), use_cache=False)
            api_sports.save_snapshot_from_odds(int(fixture_id), odds)
        st.success("Snapshot сохранён ✅")
    else:
        st.info("Последний snapshot свежий — ждём интервал 10 минут (или нажми кнопку форс).")

    # history
    hist = api_sports.get_snapshot_history(int(fixture_id), hours=float(hours))
    if not hist:
        st.warning("История пока пустая (возможно, рынок odds ещё не доступен или snapshot не сохранился).")
        return

    dfh = pd.DataFrame(hist)
    # user-friendly formatting
    dfh["best_odd"] = dfh["best_odd"].round(3)
    dfh["avg_odd"] = dfh["avg_odd"].round(3)

    st.caption(f"Снимков строк: {len(dfh)} за последние {hours}ч")

    # quick latest snapshot view
    latest_ts = dfh["ts"].max()
    latest = dfh[dfh["ts"] == latest_ts].copy()
    st.markdown("### Последний snapshot (агрегировано)")
    st.dataframe(
        latest[["datetime","market","selection","best_odd","avg_odd","books"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "best_odd": st.column_config.NumberColumn("Best", format="%.3f"),
            "avg_odd": st.column_config.NumberColumn("Avg", format="%.3f"),
        },
    )

# movement / steam
    st.markdown("### 📈 Движение линии (10/30/60 мин)")
    dfm = _calc_movement_signals(dfh)
    # Anti-trap: исключаем шум/ловушки (настройки ниже)
    if dfm.empty:
        st.info("Недостаточно истории для движения линии. Нужно минимум 2 snapshot в пределах окна.")
    else:
        # filters
        
        with st.expander("🧯 Anti-trap фильтры", expanded=False):
            cta1, cta2, cta3 = st.columns(3)
            with cta1:
                max_abs_best_pct = st.slider("Макс. |Best Δ%| (шумовой спайк)", 1.0, 50.0, 25.0, 1.0)
            with cta2:
                require_avg_same_dir = st.checkbox("Требовать подтверждение avg (в ту же сторону)", value=True)
            with cta3:
                ignore_reverse = st.checkbox("Игнорировать reverse steam (best ↑, но avg ↓)", value=True)
        
        f1, f2, f3 = st.columns([1,1,1])
        with f1:
            min_books = st.slider("Мин. букмекеров (в последнем snapshot)", 1, 25, 5, key="mv_min_books")
        with f2:
            min_steam = st.slider("Мин. steam score", 0.0, 50.0, 2.0, 0.5, key="mv_min_steam")
        with f3:
            market_filter = st.multiselect("Рынки", ["1X2","OU_2_5","BTTS"], default=["1X2","OU_2_5","BTTS"], key="mv_markets")

        view = dfm.copy()
        view = view[view["books_last"].fillna(0).astype(int) >= int(min_books)]
        view = view[view["steam_score"].fillna(0) >= float(min_steam)]
        view = view[view["market"].isin(market_filter)]
        # anti-trap filters
        view = view[view["best_odd_chg_pct"].abs() <= float(max_abs_best_pct)]
        if require_avg_same_dir:
            # sign(best) should match sign(avg)
            view = view[np.sign(view["best_odd_chg_pct"]) == np.sign(view["avg_odd_chg_pct"])]
        if ignore_reverse:
            # reverse steam: best gets worse (↑) but avg improves (↓) or vice-versa (conflict)
            conflict = (np.sign(view["best_odd_chg_pct"]) != np.sign(view["avg_odd_chg_pct"]))
            view = view[~conflict]
        if view.empty:
            st.info("По текущим фильтрам сигналов движения нет.")
        else:
            st.dataframe(
                view[[
                    "window_min","market","selection",
                    "best_odd_start","best_odd_last","best_odd_chg","best_odd_chg_pct",
                    "avg_odd_start","avg_odd_last","avg_odd_chg","avg_odd_chg_pct",
                    "books_last","steam_score"
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "best_odd_chg_pct": st.column_config.NumberColumn("Best Δ%", format="%.2f"),
                    "avg_odd_chg_pct": st.column_config.NumberColumn("Avg Δ%", format="%.2f"),
                    "steam_score": st.column_config.NumberColumn("Steam", format="%.2f"),
                },
            )

            st.markdown("### 🚨 Steam (топ 8)")
            top = view.sort_values("steam_score", ascending=False).head(8)
            events = []
            for _, rr in top.iterrows():
                direction = "⬇️" if rr["best_odd_chg"] < 0 else "⬆️"
                events.append({
                    "ts": float(time.time()),
                    "fixture_id": int(fixture_id),
                    "market": rr["market"],
                    "selection": rr["selection"],
                    "window_min": int(rr["window_min"]),
                    "best_start": float(rr["best_odd_start"]),
                    "best_last": float(rr["best_odd_last"]),
                    "best_chg_pct": float(rr["best_odd_chg_pct"]),
                    "books": int(rr["books_last"]),
                    "steam_score": float(rr["steam_score"]),
                })
                st.markdown(
                    f"**{rr['market']} / {rr['selection']}** · окно {int(rr['window_min'])}м · books {int(rr['books_last'])} · steam {rr['steam_score']:.2f}\\n"\
                    f"{direction} best: {rr['best_odd_start']:.2f} → {rr['best_odd_last']:.2f} ({rr['best_odd_chg_pct']:+.2f}%) · avg: {rr['avg_odd_start']:.2f} → {rr['avg_odd_last']:.2f}"
                )

            api_sports.save_steam_events(events)

    if show_table:
        st.markdown("### История snapshots")
        st.dataframe(
            dfh[["datetime","market","selection","best_odd","avg_odd","books"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "best_odd": st.column_config.NumberColumn("Best", format="%.3f"),
                "avg_odd": st.column_config.NumberColumn("Avg", format="%.3f"),
            },
        )

def render_page_opportunities(analyzer: MatchAnalyzer, sports: SportsCollector, api_sports: ApiSportsCollector):
    st.markdown('<div class="main-header">🔥 Opportunities (48h)</div>', unsafe_allow_html=True)

    if not api_sports.is_configured():
        st.error("API_FOOTBALL_KEY не задан. Добавь его в Secrets, иначе odds/fixtures не будут работать.")
        return

    # date range: today + next day (≈48h)
    tz_note = "UTC"
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        start_date = st.date_input("Старт", value=datetime.date.today())
    with c2:
        days = st.selectbox("Горизонт", [1, 2], index=1)  # 2 дня ~ 48h
    with c3:
        min_ev = st.slider("Мин. EV (%)", -5.0, 20.0, 3.0, 0.5)
    with c4:
        min_books = st.slider("Мин. букмекеров", 1, 25, 5)

    
    profile_filter = st.multiselect(
    "Профили",
    ["VALUE BET", "STEAM FOLLOW", "TREND PLAY", "AVOID"],
    default=["VALUE BET", "STEAM FOLLOW", "TREND PLAY"],
)
    only_no_traps = st.checkbox("Скрыть AVOID/с trap flags", value=False)

    markets_sel = st.multiselect("Рынки", ["1X2", "OU_2_5", "BTTS"], default=["1X2", "OU_2_5", "BTTS"])

    limit = st.slider("Лимит матчей для расчёта (скорость)", 5, 40, 20)
    run = st.button("🔎 Найти opportunities", type="primary", use_container_width=True)

    st.caption(f"Источник fixtures/odds: API-Sports. Источник team_id/stat: football-data (если удастся сопоставить по названиям). Таймзона: {tz_note}.")

    if not run:
        st.info("Нажми кнопку, чтобы посчитать opportunities (это может занять немного времени).")
        return

    # ── Gather fixtures for 48h ─────────────────────────────────────
    dates = [start_date + datetime.timedelta(days=i) for i in range(int(days))]
    fx_all = []
    with st.spinner("Загружаем fixtures (48h)…"):
        for d in dates:
            fx_all.extend(api_sports.get_fixtures_by_date(d.strftime("%Y-%m-%d"), timezone="UTC", use_cache=True))

    if not fx_all:
        st.warning("Fixtures не найдены.")
        return

    # keep upcoming/not started
    df_fx = pd.DataFrame(fx_all)
    if df_fx.empty:
        st.warning("Fixtures пустые.")
        return
    df_fx["time"] = df_fx["utcDate"].astype(str).str[11:16]
    df_fx["match"] = df_fx["home_team"].astype(str) + " — " + df_fx["away_team"].astype(str)
    df_fx = df_fx.sort_values(["utcDate", "league", "match"])
    # limit for speed
    df_fx = df_fx.head(int(limit))

    # ── Build football-data mapping (same date range) ────────────────
    def norm(s: str) -> str:
        try:
            return normalize_team_name(str(s))
        except Exception as e:
            return str(s).strip().lower()

    fd_map = {}
    with st.spinner("Сопоставляем team_id (football-data)…"):
        for d in dates:
            dt = d.strftime("%Y-%m-%d")
            resp = sports.get("https://api.football-data.org/v4/matches", params={"dateFrom": dt, "dateTo": dt})
            if resp is None or resp.status_code != 200:
                continue
            for m in (resp.json() or {}).get("matches", []) or []:
                home = (m.get("homeTeam") or {}).get("name")
                away = (m.get("awayTeam") or {}).get("name")
                utc = m.get("utcDate") or ""
                key = (utc[:13], norm(home), norm(away))  # hour bucket
                fd_map[key] = {
                    "home_id": (m.get("homeTeam") or {}).get("id"),
                    "away_id": (m.get("awayTeam") or {}).get("id"),
                    "competition": (m.get("competition") or {}).get("name") or "",
                }

    # ── Compute opportunities ───────────────────────────────────────
    rows = []
    with st.spinner("Считаем модель + odds + EV…"):
        for _, r in df_fx.iterrows():
            fx_id = int(r["fixture_id"])
            home = str(r.get("home_team") or "")
            away = str(r.get("away_team") or "")
            utc = str(r.get("utcDate") or "")
            match_dt = utc.replace("Z", "")
            comp = str(r.get("league") or "")

            # try map to football-data ids
            key = (utc[:13], norm(home), norm(away))
            ids = fd_map.get(key) or {}
            h_id = ids.get("home_id")
            a_id = ids.get("away_id")
            comp2 = ids.get("competition") or comp

            try:
                result = analyzer.analyze_match(
                    home_team=norm(home),
                    away_team=norm(away),
                    match_datetime=match_dt,
                    home_team_id=int(h_id) if h_id else None,
                    away_team_id=int(a_id) if a_id else None,
                    competition=comp2,
                )
            except Exception as e:
                continue

            # odds (no snapshot forcing here; keep cache)
            odds = api_sports.get_odds_for_fixture(fx_id, use_cache=True)
            markets = (odds.get("markets") or {})
            dfv = _calc_value_signals_all(result, markets)
            if dfv.empty:
                continue

            # filters and compute edge score
            dfv = dfv[dfv["books"] >= int(min_books)]
            if dfv.empty:
                continue

            dfv = dfv[dfv["market"].isin(markets_sel)]
            if dfv.empty:
                continue

            dfv = dfv[dfv["ev_pct"] >= float(min_ev)]
            if dfv.empty:
                continue

            confidence = float(result.get("confidence") or 0.0)
            # add steam from history if exists
            hist = api_sports.get_snapshot_history(fx_id, hours=6.0)
            steam_map = {}
            if hist:
                dfh = pd.DataFrame(hist)
                if not dfh.empty:
                    dfm = _calc_movement_signals(dfh)
    # Anti-trap: исключаем шум/ловушки (настройки ниже)
                    if not dfm.empty:
                        # use 30-min window as main
                        df30 = dfm[dfm["window_min"] == 30]
                        for _, mr in df30.iterrows():
                            steam_map[(mr["market"], mr["selection"])] = float(mr.get("steam_score") or 0.0)

            for _, vr in dfv.head(6).iterrows():  # top per match
                steam = float(steam_map.get((vr["market"], vr["selection"]), 0.0))
                books = int(vr["books"])
                # edge score (простая, но полезная): EV% + 0.6*steam + 0.2*(books-5) + confidence/20
                edge = float(vr["ev_pct"]) + 0.6*steam + 0.2*max(0, books-5) + (confidence/20.0)
                rows.append({
                    "utc": utc,
                    "time": utc[11:16] if len(utc) >= 16 else "",
                    "league": comp2,
                    "match": f"{home} — {away}",
                    "fixture_id": fx_id,
                    "market": vr["market"],
                    "selection": vr["selection"],
                    "model_p(%)": vr["model_p_pct"],
                    "best": float(vr["best_odd"]),
                    "avg": float(vr["avg_odd"]),
                    "books": books,
                    "EV(%)": float(vr["ev_pct"]),
                    "steam": round(steam, 2),
                    "confidence": round(confidence, 1),
                    "EDGE": round(edge, 2),
                    "profile": _classify_profile(float(vr["ev_pct"]), float(steam), int(books), float(confidence)),
                    "trap_flags": _trap_flags(float(vr["ev_pct"]), float(steam), int(books), float(confidence)),
                    "bet_units": _bet_size_units(
                        _classify_profile(float(vr["ev_pct"]), float(steam), int(books), float(confidence)),
                        float(confidence),
                        float(vr["ev_pct"]),
                        _trap_flags(float(vr["ev_pct"]), float(steam), int(books), float(confidence)),
                    ),
                    "risk_level": _risk_level(float(confidence), _trap_flags(float(vr["ev_pct"]), float(steam), int(books), float(confidence))),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Opportunities не найдено по текущим фильтрам. Попробуй снизить порог EV или min_books.")
        return

    df = df.sort_values(["EDGE", "EV(%)"], ascending=False)

    st.markdown("### 🧾 Top opportunities")
    cols_show = ["time","league","match","market","selection","profile","EV(%)","steam","books","confidence","EDGE","bet_units","risk_level","trap_flags","fixture_id"]
    cols_show = [c for c in cols_show if c in df.columns]
    st.dataframe(df[cols_show], use_container_width=True, hide_index=True)

# Explain flags for selected row
st.markdown("#### 🧠 Explain flags (выбери строку)")
sel_idx = st.number_input("Index строки", min_value=0, max_value=max(len(df)-1,0), value=0, step=1)
if len(df) > 0:
    row = df.iloc[int(sel_idx)]
    flags = row.get("trap_flags")
    explanations = _explain_flags_human(flags)
    if explanations:
        for e in explanations:
            st.write("• " + e)
    else:
        st.caption("Нет предупреждений для выбранной ставки.")


st.markdown("### 🚨 Alerts summary (48h)")
s1, s2, s3 = st.columns(3)
with s1:
    min_steam = st.slider("Мин. steam", 0.0, 50.0, 2.0, 0.5, key="opp_min_steam")
with s2:
    min_conf = st.slider("Мин. confidence", 0.0, 100.0, 50.0, 1.0, key="opp_min_conf")
with s3:
    show_only_positive_ev = st.checkbox("Только EV > 0", value=True, key="opp_pos_ev")

view = df.copy()
view = view[view["confidence"] >= float(min_conf)]
if show_only_positive_ev:
    view = view[view["EV(%)"] > 0]

# steam-first list
steam_df = view[view["steam"] >= float(min_steam)].sort_values(["steam","EDGE"], ascending=False).head(15)
ev_df = view.sort_values(["EV(%)","EDGE"], ascending=False).head(15)
edge_df = view.sort_values(["EDGE","EV(%)"], ascending=False).head(15)

t1, t2, t3 = st.tabs(["Steam", "EV", "EDGE"])
with t1:
    if steam_df.empty:
        st.info("Нет steam-алертов по фильтрам (возможно, ещё мало snapshots).")
    else:
        st.dataframe(steam_df, use_container_width=True, hide_index=True)
with t2:
    st.dataframe(ev_df, use_container_width=True, hide_index=True)
with t3:
    st.dataframe(edge_df, use_container_width=True, hide_index=True)

    

st.markdown("### 📦 Портфель на день")
p1, p2, p3 = st.columns(3)
with p1:
    max_total_units = st.slider("Лимит units на день", 1.0, 10.0, 4.0, 0.25)
with p2:
    max_bets = st.slider("Макс. ставок", 1, 20, 8, 1)
with p3:
    exclude_high = st.checkbox("Исключить HIGH risk", value=True)

port = df.copy()
if "risk_level" in port.columns and exclude_high:
    port = port[port["risk_level"] != "HIGH"]
if "bet_units" in port.columns:
    port = port.sort_values(["bet_units","EDGE"], ascending=False).head(int(max_bets))
    total_units = float(port["bet_units"].sum())
    st.metric("Итого units", round(total_units, 2))
    if total_units > float(max_total_units):
        st.warning("Портфель превышает дневной лимит — уменьшай max ставок или фильтруй профили.")
    st.dataframe(port, use_container_width=True, hide_index=True)
else:
    st.info("Нет bet_units — проверь, что opportunities рассчитаны.")


st.markdown("### ⚡ Быстрые summary")
n_sum = st.slider("Сколько показать", 3, 30, 10, 1)
if len(df) == 0:
    st.info("Пока нет opportunities.")
else:
    # Use available columns to build simple one-liners
    view_sum = df.copy()
    if "bet_units" in view_sum.columns:
        view_sum = view_sum.sort_values(["bet_units","EDGE"], ascending=False)
    view_sum = view_sum.head(int(n_sum))

    for _, r in view_sum.iterrows():
        tag = _badge(str(r.get("profile","")))
        match = str(r.get("match",""))
        pick = f"{r.get('market','')} / {r.get('selection','')}"
        conf = r.get("confidence","—")
        risk = r.get("risk_level","—") if "risk_level" in view_sum.columns else "—"
        why = ""
        flags = r.get("trap_flags","—")
        if flags and flags != "—":
            why = f"flags: {flags}"
        st.write(f"{tag} {r.get('profile','')} | {match} → {pick} | Conf {conf} | Risk {risk}" + (f" · {why}" if why else ""))

st.markdown("### 🎯 Top 10 карточки")
for _, rr in df.head(10).iterrows():
    st.markdown(
        f"**{rr['match']}** · {rr['time']} · {rr['league']}\n"
        f"🧩 {rr['market']} / {rr['selection']} · EV {rr['EV(%)']:.2f}% · steam {rr['steam']:.2f} · books {int(rr['books'])} · conf {rr['confidence']:.0f}\n"
        f"⭐ EDGE {rr['EDGE']:.2f} · best {rr['best']:.2f} (avg {rr['avg']:.2f})"
    )


with st.sidebar:
    st.markdown("## 🏆 Sport Analyzer")
    st.markdown("---")
    page = st.radio(
        "Навигация",
        ["🔍 Анализ матча", "🔥 Opportunities (48h)", "📡 Сигналы (API-Sports)", "📅 Расписание", "📊 История", "📈 ELO рейтинги"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("football-data.org · open-meteo · thesportsdb")

if page == "🔍 Анализ матча":
    render_page_analyze(analyzer, sports, api_sports)
elif page == "🔥 Opportunities (48h)":
    render_page_opportunities(analyzer, sports, api_sports)
elif page == "📡 Сигналы (API-Sports)":
    render_page_signals(api_sports)
elif page == "📅 Расписание":
    render_page_schedule(sports, analyzer)
elif page == "📊 История":
    render_page_history()
elif page == "📈 ELO рейтинги":
    render_page_elo()
