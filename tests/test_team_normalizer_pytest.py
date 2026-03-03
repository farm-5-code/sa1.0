from sport_analyzer.utils.team_normalizer import (
    normalize_team_name,
    strip_legal_suffix,
    teams_are_same,
)


def test_normalize_known_aliases() -> None:
    assert normalize_team_name("man city") == "Manchester City"
    assert normalize_team_name("MAN UTD") == "Manchester United"


def test_strip_legal_suffix() -> None:
    assert strip_legal_suffix("Arsenal FC") == "Arsenal"
    assert strip_legal_suffix("Juventus S.p.A.") == "Juventus"


def test_teams_are_same_case_and_suffix() -> None:
    assert teams_are_same("Arsenal", "Arsenal FC")
    assert teams_are_same("man city", "Manchester City")
    assert not teams_are_same("Arsenal", "Chelsea")
