"""analyzers/injury_impact.py

Более точная модель влияния травм/дисквалификаций.

Проблема:
- "есть травмы" ≠ "травмирован лидер атаки".

Решение:
- Держим маленькую базу ключевых игроков (можно расширять),
  где для каждого игрока задан вклад в xG/создание моментов.
- Преобразуем список найденных имён (из NewsCollector) в коэффициенты,
  которые масштабируют ожидаемые голы.

Важно:
- Источник имён из новостей шумный, поэтому:
  - сравнение по подстроке (case-insensitive),
  - ограничиваем максимальный штраф.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


# Минимальный стартовый словарь (расширяйте при желании).
# Значения: "attack" — уменьшает xG_for, "defense" — ухудшает защиту (увеличивает xG_against).
KEY_PLAYERS: Dict[str, Dict[str, Dict[str, float]]] = {
    "Manchester City": {
        "Haaland": {"attack": 0.42},
        "De Bruyne": {"attack": 0.30},
        "Rodri": {"defense": 0.18},
    },
    "Liverpool": {
        "Salah": {"attack": 0.30},
        "Van Dijk": {"defense": 0.22},
    },
    "Real Madrid": {
        "Bellingham": {"attack": 0.24},
        "Vinícius": {"attack": 0.28},
    },
}


def _match_players(known: Dict[str, Dict[str, float]], injured: Iterable[str]) -> List[Dict[str, float]]:
    injured_l = [p.lower() for p in injured if p]
    matched: List[Dict[str, float]] = []
    for name, impact in known.items():
        nl = name.lower()
        if any(nl in p or p in nl for p in injured_l):
            matched.append(impact)
    return matched


def injury_factors(
    team_name: str,
    injured_players: Iterable[str],
    suspended_players: Iterable[str] = (),
    *,
    max_attack_penalty: float = 0.35,   # до -35% атаки
    max_defense_penalty: float = 0.25,  # до +25% к пропускаемости
) -> Dict[str, float]:
    """Возвращает коэффициенты {'attack': factor<=1, 'defense': factor>=1}."""
    injured = [p for p in (injured_players or []) if isinstance(p, str) and p.strip()]
    suspended = [p for p in (suspended_players or []) if isinstance(p, str) and p.strip()]
    all_absent = injured + suspended

    known = KEY_PLAYERS.get(team_name, {})
    impacts = _match_players(known, all_absent)

    # Базовые штрафы если данных по игрокам нет (но флаг травм/дискв есть)
    generic_attack = 0.07 if injured else 0.0
    generic_defense = 0.05 if injured else 0.0
    generic_attack += 0.05 if suspended else 0.0
    generic_defense += 0.03 if suspended else 0.0

    attack_pen = generic_attack
    defense_pen = generic_defense

    for imp in impacts:
        attack_pen += float(imp.get("attack", 0.0) or 0.0) * 1.2
        defense_pen += float(imp.get("defense", 0.0) or 0.0) * 1.2

    attack_pen = min(float(attack_pen), max_attack_penalty)
    defense_pen = min(float(defense_pen), max_defense_penalty)

    attack_factor = max(0.65, 1.0 - attack_pen)
    defense_factor = 1.0 + defense_pen
    return {"attack": round(attack_factor, 3), "defense": round(defense_factor, 3), "matched": len(impacts)}
