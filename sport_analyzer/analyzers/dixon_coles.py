"""sport_analyzer.analyzers.dixon_coles

Dixon–Coles correction for low-score dependence in football.

A plain independent Poisson model tends to under-estimate the frequency
of low-score outcomes like 0:0, 1:0, 0:1 and 1:1.

This module applies the classic Dixon–Coles (1997) tau correction to the
scoreline probability matrix built from (lambda_home, lambda_away).
"""

from __future__ import annotations

from typing import List

import numpy as np


def _tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Tau correction for low scores (0-1 goals each)."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def build_dc_matrix(
    lambda_h: float,
    lambda_a: float,
    max_goals: int = 10,
    rho: float = -0.13,
) -> np.ndarray:
    """Build scoreline probability matrix with Dixon–Coles correction."""

    # Local import to avoid circular imports at module load time.
    from sport_analyzer.analyzers.poisson_elo import _pmf_array

    pmf_h = _pmf_array(lambda_h, max_goals)
    pmf_a = _pmf_array(lambda_a, max_goals)
    matrix = np.outer(pmf_h, pmf_a)

    # Apply tau correction only for 0/1 scores.
    for x in range(min(2, max_goals + 1)):
        for y in range(min(2, max_goals + 1)):
            corr = _tau(x, y, lambda_h, lambda_a, rho)
            matrix[x, y] = max(0.0, float(matrix[x, y]) * float(corr))

    total = float(matrix.sum())
    if total > 0:
        matrix /= total
    return matrix


def estimate_rho(historical_matches: List[dict]) -> float:
    """Estimate rho via a lightweight MLE if SciPy is available."""
    if not historical_matches or len(historical_matches) < 50:
        return -0.13

    try:
        from scipy.optimize import minimize_scalar  # type: ignore
    except Exception:
        return -0.13

    def neg_log_likelihood(rho: float) -> float:
        total_ll = 0.0
        for m in historical_matches:
            try:
                hg = int(m.get("home_goals") or 0)
                ag = int(m.get("away_goals") or 0)
                lh = float(m.get("lambda_h") or 0)
                la = float(m.get("lambda_a") or 0)
            except Exception:
                continue
            tau = _tau(hg, ag, lh, la, rho)
            if tau <= 0:
                return 1e9
            total_ll -= float(np.log(max(tau, 1e-10)))
        return total_ll

    res = minimize_scalar(neg_log_likelihood, bounds=(-0.5, 0.0), method="bounded")
    return float(res.x) if getattr(res, "success", False) else -0.13
