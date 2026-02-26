"""sport_analyzer.analyzers.fatigue

Simple team fatigue and match-importance helpers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
import math


def calculate_fatigue(
    recent_matches: List[Dict],
    current_date: Optional[str] = None,
) -> float:
    """Return fatigue score in [0.0, 1.0]."""
    if not recent_matches:
        return 0.0

    try:
        ref_date = (
            datetime.strptime(current_date[:10], "%Y-%m-%d")
            if current_date
            else datetime.utcnow()
        )
    except Exception:
        ref_date = datetime.utcnow()

    total = 0.0
    decay_days = 7.0

    for m in recent_matches:
        try:
            match_date = datetime.strptime((m.get("date") or "")[:10], "%Y-%m-%d")
        except Exception:
            continue

        days_ago = (ref_date - match_date).days
        if days_ago < 0 or days_ago > 30:
            continue

        base = 1.0 * (1.3 if m.get("is_away", False) else 1.0)
        total += base * math.exp(-days_ago / decay_days)

    normalized = total / 5.0
    return float(min(max(normalized, 0.0), 1.0))


def fatigue_to_lambda_factor(fatigue: float) -> float:
    f = float(min(max(fatigue, 0.0), 1.0))
    return 1.0 - f * 0.08


def fatigue_to_concede_factor(fatigue: float) -> float:
    f = float(min(max(fatigue, 0.0), 1.0))
    return 1.0 + f * 0.10


def get_match_importance(
    competition: str,
    match_week: Optional[int] = None,
    total_weeks: Optional[int] = None,
    position_home: Optional[int] = None,
    position_away: Optional[int] = None,
) -> float:
    """Heuristic importance factor in [0.5, 2.0]."""
    importance = 1.0
    comp_lower = (competition or "").lower()

    if any(kw in comp_lower for kw in ("final", "playoff", "play-off")):
        importance *= 1.8
    elif any(kw in comp_lower for kw in ("cup", "кубок")):
        importance *= 1.3

    if match_week and total_weeks and total_weeks > 0:
        remaining = total_weeks - match_week
        if remaining <= 4:
            importance *= 1.5
        elif remaining <= 8:
            importance *= 1.2

    if position_home and position_away:
        positions = sorted([position_home, position_away])
        if positions[0] <= 4 and positions[1] <= 6:
            importance *= 1.3
        if positions[0] >= 15:
            importance *= 1.4

    return float(min(max(importance, 0.5), 2.0))
