import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional

from sport_analyzer.collectors.base_collector import BaseCollector

logger = logging.getLogger(__name__)

_INJURY_KEYWORDS = {
    "injured",
    "injury",
    "ruled out",
    "out for",
    "out until",
    "out injured",
    "fitness doubt",
    "fitness concern",
    "doubtful",
    "hamstring",
    "muscle strain",
    "ankle",
    "knee",
    "absent",
    "unavailable",
    "miss the match",
    "miss out",
    "set to miss",
}

_SUSPENSION_KEYWORDS = {
    "suspended",
    "suspension",
    "banned",
    "ban",
    "red card",
    "accumulated",
    "disciplinary",
    "sent off",
}

_TRANSFER_KEYWORDS = {
    "transfer",
    "signing",
    "signed",
    "deal",
    "fee",
    "bid",
    "offer",
    "contract",
    "loan",
}

_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "https://www.theguardian.com/football/rss",
    "https://www.skysports.com/rss/12040",
]


@dataclass
class RawArticle:
    title: str = ""
    description: str = ""
    published: str = ""
    source: str = ""
    lang: str = "en"

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.description}".strip()

    @property
    def full_text_lower(self) -> str:
        return self.full_text.lower()


class NewsCollector(BaseCollector):
    RATE_LIMIT_PER_MINUTE = 30

    def __init__(self, config):
        super().__init__(db_path=config.DB_PATH)
        self.config = config
        self._vader = self._load_vader()

    @staticmethod
    def _load_vader():
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            return SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment не установлен")
            return None

    def _sentiment_score(self, text: str, lang: str = "en") -> float:
        if lang != "en" or not text.strip():
            return 50.0
        if self._vader:
            scores = self._vader.polarity_scores(text)
            return round((scores["compound"] + 1) / 2 * 100, 1)
        try:
            from textblob import TextBlob

            return round((TextBlob(text).sentiment.polarity + 1) / 2 * 100, 1)
        except Exception:
            return 50.0

    def get_team_news(self, team_name: str) -> Dict:
        cache_key = f"news_{team_name.lower().replace(' ', '_')}"
        cached = self._cache_get(cache_key, max_age_hours=2.0)
        if cached:
            return cached

        articles = self._collect_all(team_name)

        if not articles:
            result = {
                "team": team_name,
                "articles_count": 0,
                "sentiment_score": 50.0,
                "sentiment_label": "😐 Нейтральный",
                "has_injuries": False,
                "has_suspensions": False,
                "has_transfers": False,
                "key_topics": [],
                "injury_players": [],
                "suspended_players": [],
            }
            self._cache_set(cache_key, result)
            return result

        sentiment = self._aggregate_sentiment(articles)
        events = self._extract_events(articles)

        result = {
            "team": team_name,
            "articles_count": len(articles),
            "sentiment_score": sentiment,
            "sentiment_label": self._sentiment_label(sentiment),
            **events,
        }
        self._cache_set(cache_key, result)
        return result

    def _collect_all(self, team_name: str) -> List[RawArticle]:
        articles: List[RawArticle] = []

        if self.config.GNEWS_KEY:
            articles.extend(self._fetch_gnews(team_name))
        if self.config.NEWS_API_KEY:
            articles.extend(self._fetch_newsapi(team_name))
        articles.extend(self._fetch_rss(team_name))

        seen: set = set()
        unique: List[RawArticle] = []
        for a in articles:
            key = a.title.lower()[:60]
            if key not in seen and key:
                seen.add(key)
                unique.append(a)
        return unique

    def _fetch_gnews(self, team_name: str) -> List[RawArticle]:
        resp = self.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": f"{team_name} football",
                "lang": "en",
                "max": 10,
                "token": self.config.GNEWS_KEY,
            },
        )
        if resp is None or resp.status_code != 200:
            return []
        return [
            RawArticle(
                title=a.get("title", ""),
                description=a.get("description", ""),
                published=a.get("publishedAt", ""),
                source=a.get("source", {}).get("name", "GNews"),
            )
            for a in resp.json().get("articles", [])
        ]

    def _fetch_newsapi(self, team_name: str) -> List[RawArticle]:
        resp = self.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": team_name,
                "sortBy": "publishedAt",
                "pageSize": 10,
                "language": "en",
                "apiKey": self.config.NEWS_API_KEY,
            },
        )
        if resp is None or resp.status_code != 200:
            return []
        return [
            RawArticle(
                title=a.get("title", ""),
                description=a.get("description", ""),
                published=a.get("publishedAt", ""),
                source=a.get("source", {}).get("name", "NewsAPI"),
            )
            for a in resp.json().get("articles", [])
        ]

    def _fetch_rss(self, team_name: str) -> List[RawArticle]:
        articles: List[RawArticle] = []
        team_lower = team_name.lower()
        for feed_url in _RSS_FEEDS:
            resp = self.get(feed_url, host_key="rss")
            if resp is None or resp.status_code != 200:
                continue
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                continue
            for item in root.findall(".//item"):
                title = item.findtext("title", "") or ""
                desc = item.findtext("description", "") or ""
                if team_lower not in title.lower() and team_lower not in desc.lower():
                    continue
                articles.append(
                    RawArticle(
                        title=title,
                        description=desc,
                        published=item.findtext("pubDate", ""),
                        source="RSS",
                    )
                )
        return articles

    def _aggregate_sentiment(self, articles: List[RawArticle]) -> float:
        scores = [self._sentiment_score(a.full_text, a.lang) for a in articles[:10]]
        return round(sum(scores) / len(scores), 1) if scores else 50.0

    def _extract_events(self, articles: List[RawArticle]) -> Dict:
        has_injuries = has_suspensions = has_transfers = False
        injury_players: List[str] = []
        suspended_players: List[str] = []
        key_topics: List[str] = []

        for a in articles:
            text_l = a.full_text_lower
            if any(kw in text_l for kw in _INJURY_KEYWORDS):
                has_injuries = True
                p = self._extract_player_name(a.title)
                if p and p not in injury_players:
                    injury_players.append(p)
            if any(kw in text_l for kw in _SUSPENSION_KEYWORDS):
                has_suspensions = True
                p = self._extract_player_name(a.title)
                if p and p not in suspended_players:
                    suspended_players.append(p)
            if any(kw in text_l for kw in _TRANSFER_KEYWORDS):
                has_transfers = True

        if has_injuries:
            label = "⚠️ Травмы"
            if injury_players:
                label += f" ({', '.join(injury_players[:2])})"
            key_topics.append(label)
        if has_suspensions:
            label = "🚫 Дисквалификации"
            if suspended_players:
                label += f" ({', '.join(suspended_players[:2])})"
            key_topics.append(label)
        if has_transfers:
            key_topics.append("📋 Трансферные слухи")

        return {
            "has_injuries": has_injuries,
            "has_suspensions": has_suspensions,
            "has_transfers": has_transfers,
            "injury_players": injury_players,
            "suspended_players": suspended_players,
            "key_topics": key_topics,
        }

    @staticmethod
    def _extract_player_name(title: str) -> Optional[str]:
        words = title.split()
        parts = []
        for w in words[:3]:
            clean = w.strip(".,!?:;\"'")
            if clean and clean[0].isupper() and len(clean) > 2:
                parts.append(clean)
            else:
                break
        return " ".join(parts) if 1 <= len(parts) <= 2 else None

    @staticmethod
    def _sentiment_label(score: float) -> str:
        if score >= 70:
            return "😊 Позитивный"
        elif score >= 58:
            return "🙂 Умеренно позитивный"
        elif score >= 42:
            return "😐 Нейтральный"
        elif score >= 30:
            return "😟 Умеренно негативный"
        else:
            return "😞 Негативный"
