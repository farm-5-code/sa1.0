"""analyzers/form_weighted.py

Взвешенная форма команды с учётом силы соперника и дома/в гостях.

Идея:
- Победа над сильным соперником должна давать больше "формы", чем победа над слабым.
- Выездные матчи обычно сложнее, поэтому результат там ценнее.

На вход подаём список последних матчей (FINISHED) как dict-ы:
{
  "date": "YYYY-MM-DD",
  "is_home": bool,
  "opponent_name": str,
  "result": "W"|"D"|"L",
}
и функцию get_elo(opponent_name) -> float.

На выходе: score 0..100 (как и текущий form_score).
"""

from __future__ import annotations

from typing import Callable, Dict, List


def calculate_weighted_form(
    recent_matches: List[Dict],
    get_elo: Callable[[str], float],
    *,
    base_elo: float = 1500.0,
    max_matches: int = 8,
) -> float:
    """Возвращает взвешенную форму 0..100.

    - Берём до max_matches последних матчей (желательно 6-10).
    - Вес матча = (elo_opponent / base_elo) * home_away_factor
    - home_away_factor: дома 1.05, в гостях 1.15 (чуть выше ценность выезда)
    - Очки: W=3, D=1, L=0.
    """
    if not recent_matches:
        return 50.0

    matches = recent_matches[-max_matches:]
    weights: List[float] = []
    points: List[float] = []

    for m in matches:
        r = (m.get("result") or "").upper()
        p = 3.0 if r == "W" else 1.0 if r == "D" else 0.0

        opp_name = (m.get("opponent_name") or "").strip()
        opp_elo = float(get_elo(opp_name) or base_elo) if opp_name else base_elo

        is_home = bool(m.get("is_home"))
        ha = 1.05 if is_home else 1.15

        w = (opp_elo / base_elo) * ha
        weights.append(max(0.15, min(w, 2.5)))
        points.append(p)

    # Нормируем на максимум (3 очка в каждом матче)
    max_points = sum(3.0 * w for w in weights) or 1.0
    got_points = sum(p * w for p, w in zip(points, weights))
    score = (got_points / max_points) * 100.0

    # Центрируем в разумный диапазон
    return round(float(max(0.0, min(score, 100.0))), 1)
