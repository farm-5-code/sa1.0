"""Пользовательские настройки Sport Analyzer.

Это файл *для тебя*. Здесь можно безопасно менять параметры, не трогая код.
Если сделать ошибку — приложение продолжит работать на дефолтах (оно не упадёт),
но лучше сохранять корректный Python-синтаксис.

Совет: меняй по одному параметру и проверяй через:
    python -m sport_analyzer.main --self-check
"""

from __future__ import annotations


# ── Пример: режим "строже/стабильнее" ───────────────────────────────
# Чем выше MIN_CONFIDENCE, тем реже приложение будет давать "жирные" рекомендации.
USER_MIN_CONFIDENCE: int | None = None   # например 55

# Сколько последних матчей учитывать в форме команд
USER_FORM_MATCHES: int | None = None     # например 6

# Веса факторов (сумма желательно 1.0). Если None — используется дефолт.
USER_WEIGHTS: dict[str, float] | None = None
# пример:
# USER_WEIGHTS = {
#     "team_form":      0.30,
#     "head_to_head":   0.20,
#     "home_advantage": 0.15,
#     "player_stats":   0.15,
#     "injuries":       0.10,
#     "weather":        0.05,
#     "news_sentiment": 0.03,
#     "odds_movement":  0.02,
# }

# Насколько доверять "внешним" источникам, если они доступны.
# 1.0 = как сейчас, 0.0 = полностью игнорировать (но данные всё равно собираются).
USER_EXTERNAL_SIGNALS_STRENGTH: float | None = None   # например 0.7


def apply_user_overrides(Config):  # noqa: N802 (это намеренно)
    """Применяет пользовательские настройки к Config."""

    if USER_MIN_CONFIDENCE is not None:
        Config.MIN_CONFIDENCE = int(USER_MIN_CONFIDENCE)

    if USER_FORM_MATCHES is not None:
        Config.FORM_MATCHES = int(USER_FORM_MATCHES)

    if USER_WEIGHTS is not None:
        Config.WEIGHTS = dict(USER_WEIGHTS)

    # Этот параметр можно использовать в будущем для
    # "усиления/ослабления" влияния новостей/погоды/и т.п.
    if USER_EXTERNAL_SIGNALS_STRENGTH is not None:
        Config.EXTERNAL_SIGNALS_STRENGTH = float(USER_EXTERNAL_SIGNALS_STRENGTH)
