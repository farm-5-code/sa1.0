"""Запуск: python -m utils.test_normalizer"""

import sys
from pathlib import Path

# Allow running this file directly: `python sport_analyzer/utils/test_normalizer.py`
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sport_analyzer.utils.team_normalizer import normalize_team_name, strip_legal_suffix, teams_are_same, _to_title


def run_tests() -> bool:
    cases = [
        ("man city", "Manchester City"),
        ("MAN CITY", "Manchester City"),
        ("Arsenal FC", "Arsenal"),
        ("FC Barcelona", "FC Barcelona"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Juventus S.p.A.", "Juventus"),
        ("1. fc köln", "1. FC Köln"),
    ]

    suffix_cases = [
        ("Arsenal FC", "Arsenal"),
        ("FC Barcelona", "FC Barcelona"),
        ("Juventus S.p.A.", "Juventus"),
    ]

    same_cases = [
        ("man city", "Manchester City", True),
        ("Arsenal FC", "Arsenal", True),
        ("Arsenal", "Chelsea", False),
    ]

    title_cases = [
        ("1. fc köln", "1. FC Köln"),
        ("sc freiburg", "SC Freiburg"),
        ("as roma", "AS Roma"),
        ("rb leipzig", "RB Leipzig"),
    ]

    total = passed = 0

    for raw, exp in cases:
        total += 1
        got = normalize_team_name(raw)
        if got == exp:
            passed += 1
        else:
            print(f"FAIL normalize: {raw!r} -> {got!r}, expected {exp!r}")

    for raw, exp in suffix_cases:
        total += 1
        got = strip_legal_suffix(raw)
        if got == exp:
            passed += 1
        else:
            print(f"FAIL suffix: {raw!r} -> {got!r}, expected {exp!r}")

    for a, b, exp in same_cases:
        total += 1
        got = teams_are_same(a, b)
        if got == exp:
            passed += 1
        else:
            print(f"FAIL same: {a!r}, {b!r} -> {got!r}, expected {exp!r}")

    for raw, exp in title_cases:
        total += 1
        got = _to_title(raw)
        if got == exp:
            passed += 1
        else:
            print(f"FAIL title: {raw!r} -> {got!r}, expected {exp!r}")

    failed = total - passed
    if failed == 0:
        print(f"OK: {total} tests passed")
        return True
    print(f"FAILED: {failed}/{total}")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
