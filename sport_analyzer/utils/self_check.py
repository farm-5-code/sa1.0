from __future__ import annotations

import os
import sys
import sqlite3
import platform
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_self_check(cfg) -> dict[str, Any]:
    """Быстрая диагностика, чтобы понимать: всё ли в порядке.

    Ничего не ломает и не требует ключей.
    Возвращает структурированный отчёт (для CLI/дашборда).
    """
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "db_path": getattr(cfg, "DB_PATH", "sport_analyzer.db"),
        "checks": [],
        "warnings": [],
        "ok": True,
    }

    # 1) Config.validate()
    try:
        warnings = list(cfg.validate())
        report["warnings"].extend(warnings)
        report["checks"].append({"name": "config.validate", "ok": True})
    except Exception as e:
        report["checks"].append({"name": "config.validate", "ok": False, "error": f"{type(e).__name__}: {e}"})
        report["ok"] = False

    # 2) DB connect
    db_path = report["db_path"]
    try:
        t0 = time.time()
        with sqlite3.connect(db_path, timeout=5) as c:
            c.execute("SELECT 1;")
        report["checks"].append({"name": "db.connect", "ok": True, "ms": int((time.time()-t0)*1000)})
    except Exception as e:
        report["checks"].append({"name": "db.connect", "ok": False, "error": f"{type(e).__name__}: {e}"})
        report["ok"] = False

    # 3) Imports (чтобы не было скрытых ModuleNotFoundError)
    try:
        from sport_analyzer.collectors.sports_collector import SportsCollector  # noqa: F401
        from sport_analyzer.collectors.weather_collector import WeatherCollector  # noqa: F401
        from sport_analyzer.collectors.news_collector import NewsCollector  # noqa: F401
        from sport_analyzer.collectors.xg_collector import XGCollector  # noqa: F401
        from sport_analyzer.analyzers.match_analyzer import MatchAnalyzer  # noqa: F401
        report["checks"].append({"name": "imports.core", "ok": True})
    except Exception as e:
        report["checks"].append({"name": "imports.core", "ok": False, "error": f"{type(e).__name__}: {e}"})
        report["ok"] = False

    # 4) Optional: интернет (не критично)
    # Мы не делаем запросы к платным API, только проверяем, что сеть вообще есть.
    try:
        import urllib.request
        t0 = time.time()
        urllib.request.urlopen("https://example.com", timeout=3)  # nosec B310
        report["checks"].append({"name": "internet.basic", "ok": True, "ms": int((time.time()-t0)*1000)})
    except Exception as e:
        report["checks"].append({"name": "internet.basic", "ok": False, "error": f"{type(e).__name__}: {e}"})
        report["warnings"].append("Интернет недоступен/ограничен — приложение будет работать, но без обновлений внешних данных.")

    return report
