import logging
from typing import Dict, List, Optional

import numpy as np

from sport_analyzer.entities.base import TeamStats, WeatherData, NewsInsight
from sport_analyzer.analyzers.poisson_elo import calculate_poisson, confidence_level
from sport_analyzer.analyzers.dixon_coles import build_dc_matrix
from sport_analyzer.analyzers.form_weighted import calculate_weighted_form
from sport_analyzer.analyzers.fatigue import (
    calculate_fatigue,
    fatigue_to_lambda_factor,
    fatigue_to_concede_factor,
    get_match_importance,
)
from sport_analyzer.analyzers.injury_impact import injury_factors
from sport_analyzer.models.ml_predictor import MatchPredictor, build_features

logger = logging.getLogger(__name__)


class MatchAnalyzer:

    def __init__(self, config, sports, weather, news, xg_collector=None):
        self.config = config
        self.sports = sports
        self.weather = weather
        self.news = news
        self.xg = xg_collector
        self.ml = MatchPredictor()

    def analyze_match(
        self,
        home_team: str,
        away_team: str,
        match_datetime: Optional[str] = None,
        stadium: Optional[str] = None,
        city: Optional[str] = None,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None,
        neutral_field: bool = False,
        competition: str = "",
    ) -> Dict:

        logger.info(f"Анализ: {home_team} vs {away_team}")

        availability = {
            "team_info": {"ok": True, "error": None},
            "weather": {"ok": True, "error": None},
            "news_home": {"ok": True, "error": None},
            "news_away": {"ok": True, "error": None},
            "h2h": {"ok": True, "error": None},
            "xg": {"ok": True, "error": None},
            "fatigue": {"ok": True, "error": None},
            "ml": {"ok": True, "error": None},
        }

        def safe(name: str, fn, default):
            try:
                return fn()
            except Exception as e:
                availability[name]["ok"] = False
                availability[name]["error"] = f"{type(e).__name__}: {e}"
                logger.warning(f"Источник данных '{name}' недоступен: {e}")
                return default

        # ── Team stats / ELO
        home_stats = self._build_team_stats(home_team, home_team_id)
        away_stats = self._build_team_stats(away_team, away_team_id)

        # Взвешенная форма (сила соперников)
        home_stats = self._apply_weighted_form(home_stats, home_team_id)
        away_stats = self._apply_weighted_form(away_stats, away_team_id)

        # ── Venue / Weather
        team_info = safe("team_info", lambda: self.sports.get_team_info(home_team), {})
        stadium = stadium or team_info.get("stadium")
        city = city or team_info.get("stadium_location")

        weather_data = safe(
            "weather",
            lambda: self.weather.get_weather_for_match(city=city, stadium=stadium, match_datetime=match_datetime),
            WeatherData(),
        )

        # ── News + H2H
        home_news = safe("news_home", lambda: self._build_news_insight(home_team), NewsInsight())
        away_news = safe("news_away", lambda: self._build_news_insight(away_team), NewsInsight())
        h2h = safe("h2h", lambda: self.sports.get_h2h_stats(home_team, away_team), {})

        # ── xG
        xg_home = safe("xg", lambda: self._get_xg(home_team), 0.0)
        xg_away = safe("xg", lambda: self._get_xg(away_team), 0.0)

        # ── Fatigue
        fatigue_home = safe("fatigue", lambda: self._get_fatigue(home_team_id, match_datetime), 0.0)
        fatigue_away = safe("fatigue", lambda: self._get_fatigue(away_team_id, match_datetime), 0.0)

        # ── Match importance
        importance = get_match_importance(competition)

        # ── Apply xG blend
        home_stats = self._apply_xg(home_stats, xg_home)
        away_stats = self._apply_xg(away_stats, xg_away)

        # ── Apply news factors
        home_stats = self._apply_news_factors(home_stats, home_news)
        away_stats = self._apply_news_factors(away_stats, away_news)

        # ── Взвешенное влияние травм/дисквалификаций (по игрокам, если распознаны)
        ih = injury_factors(home_team, home_news.injury_players, home_news.suspended_players)
        ia = injury_factors(away_team, away_news.injury_players, away_news.suspended_players)
        home_stats.avg_goals_scored   *= ih["attack"]
        home_stats.avg_goals_conceded *= ih["defense"]
        away_stats.avg_goals_scored   *= ia["attack"]
        away_stats.avg_goals_conceded *= ia["defense"]

        # ── Apply fatigue factors
        home_stats.avg_goals_scored *= fatigue_to_lambda_factor(fatigue_home)
        home_stats.avg_goals_conceded *= fatigue_to_concede_factor(fatigue_home)
        away_stats.avg_goals_scored *= fatigue_to_lambda_factor(fatigue_away)
        away_stats.avg_goals_conceded *= fatigue_to_concede_factor(fatigue_away)

        # ── Base Poisson
        poisson = calculate_poisson(home_stats, away_stats, weather_data, h2h, neutral_field)

        # ── Dixon–Coles correction on outcome/totals
        dc_matrix = build_dc_matrix(poisson.lambda_h, poisson.lambda_a, rho=-0.13)
        poisson = self._apply_dc_correction(poisson, dc_matrix)

        # ── ML correction (optional if model exists)
        ml_probs = None
        if self.ml.is_trained:
            feats = build_features(
                lambda_h=poisson.lambda_h,
                lambda_a=poisson.lambda_a,
                elo_home=home_stats.elo,
                elo_away=away_stats.elo,
                form_home=home_stats.form_score,
                form_away=away_stats.form_score,
                xg_home=xg_home,
                xg_away=xg_away,
                weather=weather_data.__dict__,
                h2h=h2h,
                fatigue_home=fatigue_home,
                fatigue_away=fatigue_away,
                match_importance=importance,
            )
            ml_probs = safe("ml", lambda: self.ml.predict(feats), None)

        final_probs = self._ensemble(poisson, ml_probs)
        confidence, conf_lbl = confidence_level(poisson)

        notes = []
        for k, v in availability.items():
            if not v.get("ok"):
                notes.append(f"Источник '{k}' недоступен — использован безопасный дефолт.")

        return {
            "match": f"{home_team} vs {away_team}",
            "datetime": match_datetime,
            "home_team": home_team,
            "away_team": away_team,
            "team_stats": {"home": home_stats.__dict__, "away": away_stats.__dict__},
            "h2h": h2h,
            "weather": weather_data.__dict__,
            "news": {"home": home_news.__dict__, "away": away_news.__dict__},
            "xg": {"home": xg_home, "away": xg_away},
            "fatigue": {"home": round(fatigue_home, 3), "away": round(fatigue_away, 3)},
            "poisson": poisson.__dict__,
            "final_probs": final_probs,
            "ml_used": ml_probs is not None,
            "availability": availability,
            "notes": notes,
            "confidence": confidence,
            "confidence_label": conf_lbl,
            "recommendations": self._recommendations(
                final_probs,
                home_team,
                away_team,
                poisson,
                weather_data,
                home_news,
                away_news,
                h2h,
                fatigue_home,
                fatigue_away,
                xg_home,
                xg_away,
            ),
        }

    # ── helpers

    def _build_team_stats(self, name: str, team_id: Optional[int]) -> TeamStats:
        stats = TeamStats(name=name)
        if team_id:
            stats = self.sports.get_team_stats(team_id)
            stats.name = name
        stats.elo = self.sports.get_elo(name)
        return stats

    def _build_news_insight(self, team_name: str) -> NewsInsight:
        raw = self.news.get_team_news(team_name)
        return NewsInsight(
            sentiment_score=raw.get("sentiment_score", 50.0),
            sentiment_label=raw.get("sentiment_label", "😐 Нейтральный"),
            has_injuries=raw.get("has_injuries", False),
            has_suspensions=raw.get("has_suspensions", False),
            has_transfers=raw.get("has_transfers", False),
            key_topics=raw.get("key_topics", []),
            injury_players=raw.get("injury_players", []),
            suspended_players=raw.get("suspended_players", []),
            articles_count=raw.get("articles_count", 0),
        )

    def _get_xg(self, team_name: str) -> Dict:
        if self.xg is not None:
            try:
                return self.xg.get_team_xg(team_name)
            except Exception as e:
                logger.debug(f"xG error for {team_name}: {e}")
        return {"xg_for": 1.35, "xg_against": 1.35, "xg_diff": 0.0, "matches": 0, "source": "default"}

    def _get_recent_matches(self, team_id: Optional[int], limit: int = 10) -> List[Dict]:
        """Возвращает последние FINISHED матчи команды для расчёта усталости/формы."""
        if not team_id:
            return []
        try:
            resp = self.sports.get(
                f"https://api.football-data.org/v4/teams/{team_id}/matches",
                params={"limit": limit, "status": "FINISHED"},
            )
            if not (resp and resp.status_code == 200):
                return []

            out: List[Dict] = []
            for m in resp.json().get("matches", []):
                home_id = m.get("homeTeam", {}).get("id")
                away_id = m.get("awayTeam", {}).get("id")
                is_home = home_id == team_id
                opp = (
                    m.get("awayTeam", {}).get("name")
                    if is_home else
                    m.get("homeTeam", {}).get("name")
                )

                score = (m.get("score", {}) or {}).get("fullTime", {}) or {}
                hg = int(score.get("home") or 0)
                ag = int(score.get("away") or 0)
                my = hg if is_home else ag
                their = ag if is_home else hg
                result = "W" if my > their else "D" if my == their else "L"

                out.append({
                    "date": (m.get("utcDate") or "")[:10],
                    "is_home": is_home,
                    "opponent_name": opp or "",
                    "result": result,
                })
            return out
        except Exception as e:
            logger.debug(f"Recent matches fetch error: {e}")
            return []

    def _get_fatigue(self, team_id: Optional[int], match_datetime: Optional[str]) -> float:
        recent = self._get_recent_matches(team_id, limit=12)
        if not recent:
            return 0.0
        return calculate_fatigue(
            [{"date": m["date"], "is_away": not m.get("is_home", False)} for m in recent],
            match_datetime,
        )

    def _apply_weighted_form(self, stats: TeamStats, team_id: Optional[int]) -> TeamStats:
        """Обновляет stats.form_score, учитывая силу соперников."""
        recent = self._get_recent_matches(team_id, limit=12)
        if not recent:
            return stats
        try:
            w_form = calculate_weighted_form(recent, self.sports.get_elo, max_matches=8)
            stats.form_score = round(0.6 * w_form + 0.4 * float(stats.form_score or 50.0), 1)
        except Exception as e:
            logger.debug(f"Weighted form error: {e}")
        return stats

    @staticmethod
    def _apply_xg(stats: TeamStats, xg: Dict) -> TeamStats:
        # Blend 60% xG + 40% actual goals when enough matches.
        if xg.get("matches", 0) >= 5:
            blend = 0.6
            stats.avg_goals_scored = blend * float(xg.get("xg_for", 1.35)) + (1 - blend) * stats.avg_goals_scored
            stats.avg_goals_conceded = blend * float(xg.get("xg_against", 1.35)) + (1 - blend) * stats.avg_goals_conceded
        return stats

    @staticmethod
    def _apply_news_factors(stats: TeamStats, news: NewsInsight) -> TeamStats:
        if news.has_injuries:
            stats.avg_goals_scored *= 0.93
            stats.avg_goals_conceded *= 1.05
        if news.has_suspensions:
            stats.avg_goals_scored *= 0.96
            stats.avg_goals_conceded *= 1.03
        return stats

    @staticmethod
    def _apply_dc_correction(poisson, dc_matrix: np.ndarray):
        from sport_analyzer.analyzers.poisson_elo import _calc_over, _calc_both_score

        n = dc_matrix.shape[0]
        idx = np.arange(n)
        p_draw = float(dc_matrix[idx, idx].sum())
        p_home = float(np.tril(dc_matrix, k=-1).sum())
        p_away = float(np.triu(dc_matrix, k=1).sum())
        total = p_home + p_draw + p_away or 1.0

        poisson.home_win = round(p_home / total, 4)
        poisson.draw = round(p_draw / total, 4)
        poisson.away_win = round(p_away / total, 4)
        poisson.both_score = round(_calc_both_score(dc_matrix), 4)
        poisson.over_1_5 = round(_calc_over(dc_matrix, 1.5), 4)
        poisson.over_2_5 = round(_calc_over(dc_matrix, 2.5), 4)
        poisson.over_3_5 = round(_calc_over(dc_matrix, 3.5), 4)
        return poisson

    @staticmethod
    def _ensemble(poisson, ml_probs: Optional[Dict]) -> Dict:
        if ml_probs is None:
            return {
                "home_win": poisson.home_win,
                "draw": poisson.draw,
                "away_win": poisson.away_win,
                "source": "poisson+dc",
            }

        w_p, w_m = 0.70, 0.30
        home = w_p * poisson.home_win + w_m * ml_probs.get("home_win", 0.33)
        draw = w_p * poisson.draw + w_m * ml_probs.get("draw", 0.33)
        away = w_p * poisson.away_win + w_m * ml_probs.get("away_win", 0.33)
        tot = home + draw + away or 1.0
        return {
            "home_win": round(home / tot, 4),
            "draw": round(draw / tot, 4),
            "away_win": round(away / tot, 4),
            "source": "poisson+dc+ml",
        }

    def _recommendations(
        self,
        final_probs: Dict,
        home: str,
        away: str,
        poisson,
        weather: WeatherData,
        home_news: NewsInsight,
        away_news: NewsInsight,
        h2h: Dict,
        fatigue_home: float,
        fatigue_away: float,
        xg_home: Dict,
        xg_away: Dict,
    ) -> List[str]:
        recs: List[str] = []

        outcomes = {
            "home_win": (final_probs.get("home_win", poisson.home_win), f"🏠 {home}"),
            "draw": (final_probs.get("draw", poisson.draw), "🤝 Ничья"),
            "away_win": (final_probs.get("away_win", poisson.away_win), f"✈️ {away}"),
        }
        best_key = max(outcomes, key=lambda k: outcomes[k][0])
        best_prob, label = outcomes[best_key]
        src = final_probs.get("source", "poisson")
        recs.append(f"🎯 {label} — {best_prob*100:.1f}% [{src}]")

        if poisson.over_2_5 > 0.60:
            recs.append(f"⚽ Тотал Б 2.5 — {poisson.over_2_5*100:.1f}%")
        elif poisson.over_2_5 < 0.40:
            recs.append(f"🔒 Тотал М 2.5 — {(1-poisson.over_2_5)*100:.1f}%")

        if poisson.both_score > 0.60:
            recs.append(f"✅ Обе забьют — {poisson.both_score*100:.1f}%")
        elif poisson.both_score < 0.35:
            recs.append(f"❌ Обе не забьют — {(1-poisson.both_score)*100:.1f}%")

        # xG insight
        if xg_home.get("matches", 0) >= 5 and xg_away.get("matches", 0) >= 5:
            xgd_h = float(xg_home.get("xg_diff", 0.0))
            xgd_a = float(xg_away.get("xg_diff", 0.0))
            if xgd_h > 0.30:
                recs.append(f"📈 {home} создаёт больше моментов (xG+{xgd_h:.2f})")
            if xgd_a > 0.30:
                recs.append(f"📈 {away} создаёт больше моментов (xG+{xgd_a:.2f})")

        # Fatigue
        if fatigue_home > 0.6:
            recs.append(f"😓 {home}: высокая усталость ({fatigue_home:.0%})")
        if fatigue_away > 0.6:
            recs.append(f"😓 {away}: высокая усталость ({fatigue_away:.0%})")
        if fatigue_home > 0.4 and fatigue_away < 0.2:
            recs.append(f"⚡ {away} свежее — физическое преимущество гостей")

        if h2h.get("matches", 0) >= 5:
            recs.append(
                f"📊 H2H ({h2h['matches']} матчей): "
                f"{home} {h2h['home_win_pct']}% | "
                f"Ничья {h2h['draw_pct']}% | "
                f"{away} {h2h['away_win_pct']}%"
            )

        recs.extend(weather.analysis)

        if home_news.has_injuries:
            msg = f"⚠️ {home}: травмы"
            if home_news.injury_players:
                msg += f" ({', '.join(home_news.injury_players[:2])})"
            recs.append(msg)
        if away_news.has_injuries:
            msg = f"⚠️ {away}: травмы"
            if away_news.injury_players:
                msg += f" ({', '.join(away_news.injury_players[:2])})"
            recs.append(msg)

        return recs