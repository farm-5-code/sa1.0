import os
import sys
import time
import logging
import sqlite3
import schedule
from datetime import timezone, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Scheduler] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def job_update_elo():
    logger.info(f"[{_utc_now()}] ▶ ELO update")
    try:
        from sport_analyzer.config.settings import Config
        from scripts.update_elo import ELOUpdater

        n = ELOUpdater(Config()).run(days_back=2)
        logger.info(f"✔ ELO: {n} матчей")
    except Exception as e:
        logger.error(f"✖ ELO error: {e}", exc_info=True)


def job_clean_cache():
    logger.info(f"[{_utc_now()}] ▶ Cache clean")
    try:
        from sport_analyzer.config.settings import Config

        cfg = Config()
        threshold = time.time() - 24 * 3600
        with sqlite3.connect(cfg.DB_PATH, timeout=10) as c:
            deleted = c.execute(
                "DELETE FROM collector_cache WHERE ts < ?", (threshold,)
            ).rowcount
        logger.info(f"✔ Cache: удалено {deleted} записей")
    except Exception as e:
        logger.error(f"✖ Cache error: {e}")


def main():
    tz = os.environ.get("TZ", "local")
    logger.info(f"Scheduler запущен (TZ={tz})")
    if tz != "UTC":
        logger.warning("Для UTC: TZ=UTC python scripts/scheduler.py")

    schedule.every().day.at("06:00").do(job_update_elo)
    schedule.every(30).minutes.do(job_clean_cache)

    job_update_elo()
    job_clean_cache()

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
