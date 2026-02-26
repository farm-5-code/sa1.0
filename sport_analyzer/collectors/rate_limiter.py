import time
import threading
import collections
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """
    Sliding-window rate limiter.
    Thread-safe. Без SQLite на каждый запрос.
    """

    def __init__(self):
        self._windows: Dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def _cleanup(self, window: collections.deque, now: float):
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

    def wait_until_allowed(self, host: str, limit_per_minute: int):
        while True:
            with self._lock:
                now = time.monotonic()
                if host not in self._windows:
                    self._windows[host] = collections.deque()
                window = self._windows[host]
                self._cleanup(window, now)

                if len(window) < limit_per_minute:
                    window.append(now)
                    return

                oldest = window[0]

            sleep_for = max(0.1, 60.0 - (time.monotonic() - oldest) + 0.05)
            logger.debug(f"Rate limit [{host}]: жду {sleep_for:.1f}s")
            time.sleep(min(sleep_for, 2.0))

    def stats(self) -> Dict[str, int]:
        with self._lock:
            now = time.monotonic()
            result = {}
            for host, window in self._windows.items():
                self._cleanup(window, now)
                result[host] = len(window)
            return result


_GLOBAL_LIMITER = InMemoryRateLimiter()


def get_global_limiter() -> InMemoryRateLimiter:
    return _GLOBAL_LIMITER
