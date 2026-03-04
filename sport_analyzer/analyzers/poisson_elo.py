import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from sport_analyzer.entities.base import TeamStats, WeatherData

_LEAGUE_AVG_HOME = 1.50
_LEAGUE_AVG_AWAY = 1.15
_HOME_FIELD_K = 1.10
_MAX_GOALS = 10


@dataclass
class PoissonResult:
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0
    lambda_h: float = 0.0
    lambda_a: float = 0.0
    total_exp: float = 0.0
    both_score: float = 0.0
    over_1_5: float = 0.0
    over_2_5: float = 0.0
    over_3_5: float = 0.0

    @property
    def best_outcome(self) -> str:
        return max(
            [
                ("home_win", self.home_win),
                ("draw", self.draw),
                ("away_win", self.away_win),
            ],
            key=lambda x: x[1],
        )[0]

    @property
    def best_prob(self) -> float:
        return max(self.home_win, self.draw, self.away_win)


def _pmf_array(lam: float, max_k: int) -> np.ndarray:
    lam = max(lam, 1e-9)
    k = np.arange(max_k + 1, dtype=np.float64)
    log_fact = np.zeros(max_k + 1)
    for i in range(1, max_k + 1):
        log_fact[i] = log_fact[i - 1] + math.log(i)
    log_pmf = k * math.log(lam) - lam - log_fact
    return np.exp(log_pmf)


def _calc_over(matrix: np.ndarray, threshold: float) -> float:
    n = matrix.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(n))
    return float(np.clip(matrix[idx > threshold].sum(), 0.0, 1.0))


def _calc_both_score(matrix: np.ndarray) -> float:
    p_home_0 = float(matrix[0, :].sum())
    p_away_0 = float(matrix[:, 0].sum())
    p_both_0 = float(matrix[0, 0])
    return max(0.0, 1.0 - p_home_0 - p_away_0 + p_both_0)


def _elo_expected_home(elo_h: float, elo_a: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_a - elo_h - 100) / 400))


def _weather_home_factor(weather: WeatherData) -> float:
    reduction = (weather.impact_score / 100.0) * (_HOME_FIELD_K - 1.0)
    return max(1.0, _HOME_FIELD_K - reduction)


def _form_adjustment(form_score: float) -> float:
    return 1.0 + (form_score - 50.0) / 500.0


def calculate_poisson(
    home: TeamStats,
    away: TeamStats,
    weather: WeatherData,
    h2h: Dict,
    neutral_field: bool = False,
) -> PoissonResult:
    home_field = 1.0 if neutral_field else _weather_home_factor(weather)
    elo_exp = _elo_expected_home(home.elo, away.elo)
    elo_adj = 1.0 + (elo_exp - 0.5) * 0.30
    form_adj_h = _form_adjustment(home.form_score)
    form_adj_a = _form_adjustment(away.form_score)

    attack_h = max(home.avg_goals_scored, 0.3) / _LEAGUE_AVG_HOME
    attack_a = max(away.avg_goals_scored, 0.3) / _LEAGUE_AVG_AWAY
    defense_h = max(home.avg_goals_conceded, 0.3) / _LEAGUE_AVG_HOME
    defense_a = max(away.avg_goals_conceded, 0.3) / _LEAGUE_AVG_AWAY

    lambda_h = float(
        np.clip(
            _LEAGUE_AVG_HOME * attack_h * defense_a * home_field * elo_adj * form_adj_h,
            0.3,
            6.0,
        )
    )
    lambda_a = float(
        np.clip(
            _LEAGUE_AVG_AWAY * attack_a * defense_h / home_field / elo_adj * form_adj_a,
            0.3,
            6.0,
        )
    )

    if h2h.get("matches", 0) >= 3:
        rate = h2h.get("home_win_pct", 50) / 100.0
        blend = 0.12
        if rate > 0.55:
            lambda_h *= 1 + blend
            lambda_a *= 1 - blend * 0.5
        elif rate < 0.35:
            lambda_h *= 1 - blend
            lambda_a *= 1 + blend * 0.5

    pmf_h = _pmf_array(lambda_h, _MAX_GOALS)
    pmf_a = _pmf_array(lambda_a, _MAX_GOALS)
    matrix = np.outer(pmf_h, pmf_a)

    n = matrix.shape[0]
    indices = np.arange(n)
    p_draw = float(matrix[indices, indices].sum())
    p_home = float(np.tril(matrix, k=-1).sum())
    p_away = float(np.triu(matrix, k=1).sum())
    total = p_home + p_draw + p_away or 1.0

    return PoissonResult(
        home_win=round(p_home / total, 4),
        draw=round(p_draw / total, 4),
        away_win=round(p_away / total, 4),
        lambda_h=round(lambda_h, 2),
        lambda_a=round(lambda_a, 2),
        total_exp=round(lambda_h + lambda_a, 2),
        both_score=round(_calc_both_score(matrix), 4),
        over_1_5=round(_calc_over(matrix, 1.5), 4),
        over_2_5=round(_calc_over(matrix, 2.5), 4),
        over_3_5=round(_calc_over(matrix, 3.5), 4),
    )


def confidence_level(result: PoissonResult) -> Tuple[float, str]:
    pct = round(result.best_prob * 100, 1)
    if pct >= 60:
        label = "🟢 Высокая"
    elif pct >= 48:
        label = "🟡 Средняя"
    else:
        label = "🔴 Низкая"
    return pct, label
