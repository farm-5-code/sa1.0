import os

# dotenv — опционально (на хостингах может не быть установлен)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


class Config:
    # API Keys
    FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
    API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    GNEWS_KEY = os.getenv("GNEWS_KEY", "")

    # Database
    DB_PATH = os.getenv("DB_PATH", "sport_analyzer.db")

    # Analysis weights
    WEIGHTS = {
        "team_form": 0.25,
        "head_to_head": 0.20,
        "home_advantage": 0.15,
        "player_stats": 0.15,
        "injuries": 0.10,
        "weather": 0.05,
        "news_sentiment": 0.05,
        "odds_movement": 0.05,
    }

    MIN_CONFIDENCE = 48
    FORM_MATCHES = 5

    @classmethod
    def validate(cls) -> list[str]:
        """Проверки, которые делают поведение приложения предсказуемым.

        Возвращает список предупреждений (не валим приложение),
        но критичные ошибки бросаем явно.
        """
        warnings: list[str] = []

        # ── веса
        if not isinstance(cls.WEIGHTS, dict) or not cls.WEIGHTS:
            raise ValueError("Config.WEIGHTS должен быть непустым словарём")
        s = float(sum(float(v) for v in cls.WEIGHTS.values()))
        if abs(s - 1.0) > 1e-6:
            warnings.append(
                f"Сумма весов WEIGHTS = {s:.3f} (желательно 1.000). "
                "Пайплайн всё равно работает, но интерпретация вкладов ухудшается."
            )

        # ── ключи API: это не критично (есть фолбэки), но лучше подсветить
        if not cls.FOOTBALL_DATA_KEY:
            warnings.append(
                "FOOTBALL_DATA_KEY не задан — список матчей и часть статистики "
                "могут быть недоступны."
            )
        if not (cls.NEWS_API_KEY or cls.GNEWS_KEY):
            warnings.append(
                "NEWS_API_KEY/GNEWS_KEY не заданы — анализ новостей будет работать "
                "в урезанном режиме."
            )

        return warnings


# ── Пользовательские настройки (необязательно) ─────────────────────────
try:
    from sport_analyzer.config.settings_user import apply_user_overrides  # type: ignore

    apply_user_overrides(Config)
except Exception:
    # если файла нет или есть ошибка — работаем с дефолтом
    pass
