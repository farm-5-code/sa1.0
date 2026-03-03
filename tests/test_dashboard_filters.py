from sport_analyzer.dashboard.app import (
    _analyze_basketball_match,
    _collect_upcoming_football_matches,
    _league_is_top,
)


class _ApiMock:
    def __init__(self):
        self.calls = 0

    def is_configured(self):
        return True

    def get_fixtures_by_date(self, *_args, **_kwargs):
        self.calls += 1
        return [
            {
                "fixture_id": 1,
                "utcDate": "2026-01-01T20:00:00Z",
                "league": "Premier League",
                "home_team": "A",
                "away_team": "B",
                "home_team_id": 10,
                "away_team_id": 20,
            },
            {
                "fixture_id": 2,
                "utcDate": "2026-01-01T21:00:00Z",
                "league": "Finland Ykkonen",
                "home_team": "C",
                "away_team": "D",
                "home_team_id": 30,
                "away_team_id": 40,
            },
        ]


class _SportsMock:
    def get_matches(self, days_ahead=7):
        return []


def test_league_is_top():
    assert _league_is_top("English Premier League")
    assert not _league_is_top("Finland Ykkonen")


def test_collect_matches_filters_minor_leagues():
    rows = _collect_upcoming_football_matches(
        sports=_SportsMock(), api=_ApiMock(), days=1, include_minor=False
    )
    assert len(rows) == 1
    assert rows[0]["competition"] == "Premier League"


def test_basketball_stub_result_shape():
    res = _analyze_basketball_match(
        "Boston Celtics", "Miami Heat", "2026-01-01T20:00:00"
    )
    assert res["sport"] == "basketball"
    assert isinstance(res.get("confidence"), float)
