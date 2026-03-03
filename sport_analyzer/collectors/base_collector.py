import time
import random
import sqlite3
import logging
import requests
from typing import Optional

from sport_analyzer.collectors.rate_limiter import get_global_limiter

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10


class BaseCollector:
    RATE_LIMIT_PER_MINUTE = 60
    RETRYABLE_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        db_path: str = "sport_analyzer.db",
        timeout: int = 12,
        retries: int = 4,
        backoff: float = 1.5,
    ):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "SportAnalyzer/2.0"
        self._limiter = get_global_limiter()

    def get(
        self,
        url: str,
        host_key: Optional[str] = None,
        **kwargs,
    ) -> Optional[requests.Response]:
        import urllib.parse

        host = host_key or urllib.parse.urlparse(url).netloc

        self._limiter.wait_until_allowed(host, self.RATE_LIMIT_PER_MINUTE)
        kwargs.setdefault("timeout", self.timeout)

        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, **kwargs)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in self.RETRYABLE_CODES:
                    sleep_t = (self.backoff**attempt) + random.random() * 0.3
                    logger.warning(
                        f"[{host}] HTTP {resp.status_code} "
                        f"попытка {attempt + 1}/{self.retries}, "
                        f"жду {sleep_t:.1f}s"
                    )
                    time.sleep(sleep_t)
                    continue
                logger.error(f"[{host}] HTTP {resp.status_code}: {url}")
                return resp
            except requests.exceptions.Timeout:
                logger.warning(f"[{host}] Timeout попытка {attempt + 1}")
                time.sleep(self.backoff**attempt)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[{host}] ConnectionError: {e}")
                time.sleep(self.backoff**attempt)

        logger.error(f"[{host}] Все {self.retries} попытки провалились")
        return None

    # ── Кэш ──────────────────────────────────────────────────────────

    def _ensure_cache_table(self):
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS collector_cache (
                    key       TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    ts        REAL NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS cc_ts ON collector_cache(ts)")

    def _cache_get(self, key: str, max_age_hours: float = 6.0) -> Optional[dict]:
        import json

        self._ensure_cache_table()
        threshold = time.time() - max_age_hours * 3600
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            row = c.execute(
                "SELECT data_json FROM collector_cache WHERE key=? AND ts>=?",
                (key, threshold),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_set(self, key: str, data: dict):
        import json

        self._ensure_cache_table()
        with sqlite3.connect(self.db_path, timeout=_CONNECT_TIMEOUT) as c:
            c.execute(
                "INSERT OR REPLACE INTO collector_cache VALUES (?,?,?)",
                (key, json.dumps(data, ensure_ascii=False, default=str), time.time()),
            )
