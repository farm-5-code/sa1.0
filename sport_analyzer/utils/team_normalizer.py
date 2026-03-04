from __future__ import annotations
import re
from typing import Optional

_ABBR: frozenset[str] = frozenset({
    "fc", "cf", "sc", "as", "ac", "rc",
    "sd", "gd", "fk", "sk", "bk", "if",
    "rb", "sb", "sv", "bv", "vf",
})


def _to_title(s: str) -> str:
    out = []
    for word in s.split():
        core = word.strip(".").lower()
        if core in _ABBR:
            out.append(word.upper())
        elif word.isupper() and len(word) > 1:
            out.append(word)
        else:
            out.append(word.capitalize())
    return " ".join(out)


_SUFFIX_PATTERN = re.compile(
    r"""
    ,?\s*\b(
        f\.?c\.? | c\.?f\.? | s\.?c\.? |
        a\.?s\.? | a\.?c\.? | r\.?c\.? |
        s\.?d\.? | s\.?a\.?d | s\.?p\.?a |
        fk | sk | bk | gd | spa
    )\b\.?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def strip_legal_suffix(name: str) -> str:
    return _SUFFIX_PATTERN.sub("", name).strip()


_ALIASES: dict[str, str] = {
    "man city": "Manchester City", "man utd": "Manchester United",
    "man united": "Manchester United", "manchester utd": "Manchester United",
    "spurs": "Tottenham Hotspur", "tottenham": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers", "newcastle": "Newcastle United",
    "saints": "Southampton", "hammers": "West Ham United",
    "west ham": "West Ham United", "foxes": "Leicester City",
    "leicester": "Leicester City", "villa": "Aston Villa",
    "villans": "Aston Villa", "toffees": "Everton",
    "gunners": "Arsenal", "reds": "Liverpool",
    "brighton": "Brighton & Hove Albion", "palace": "Crystal Palace",
    "sheffield utd": "Sheffield United", "sheff utd": "Sheffield United",
    "sheffield wed": "Sheffield Wednesday", "wednesday": "Sheffield Wednesday",
    "forest": "Nottingham Forest", "nottm forest": "Nottingham Forest",
    "luton": "Luton Town", "boro": "Middlesbrough",
    "afc bournemouth": "Bournemouth", "bournemouth": "Bournemouth",
    "afc wimbledon": "AFC Wimbledon",
    "barca": "FC Barcelona", "barcelona": "FC Barcelona",
    "atletico": "Atletico Madrid", "atletico madrid": "Atletico Madrid",
    "atleti": "Atletico Madrid", "betis": "Real Betis",
    "sevilla": "Sevilla", "celta": "Celta Vigo",
    "sociedad": "Real Sociedad", "bilbao": "Athletic Club",
    "athletic bilbao": "Athletic Club", "villarreal": "Villarreal",
    "fcb": "Bayern Munich", "bayern": "Bayern Munich",
    "fc bayern": "Bayern Munich", "fc bayern münchen": "Bayern Munich",
    "fc bayern munich": "Bayern Munich",
    "bvb": "Borussia Dortmund", "dortmund": "Borussia Dortmund",
    "gladbach": "Borussia Mönchengladbach",
    "mgladbach": "Borussia Mönchengladbach",
    "leverkusen": "Bayer Leverkusen", "bayer": "Bayer Leverkusen",
    "frankfurt": "Eintracht Frankfurt", "eintracht": "Eintracht Frankfurt",
    "leipzig": "RB Leipzig", "rb leipzig": "RB Leipzig",
    "vfb stuttgart": "VfB Stuttgart", "stuttgart": "VfB Stuttgart",
    "vfl wolfsburg": "VfL Wolfsburg", "wolfsburg": "VfL Wolfsburg",
    "vfl bochum": "VfL Bochum", "bochum": "VfL Bochum",
    "sc freiburg": "SC Freiburg", "freiburg": "SC Freiburg",
    "fsv mainz 05": "FSV Mainz 05", "mainz": "FSV Mainz 05",
    "tsg hoffenheim": "TSG Hoffenheim", "hoffenheim": "TSG Hoffenheim",
    "fc augsburg": "FC Augsburg", "augsburg": "FC Augsburg",
    "werder": "Werder Bremen", "werder bremen": "Werder Bremen",
    "hamburger sv": "Hamburger SV", "hamburg": "Hamburger SV", "hsv": "Hamburger SV",
    "köln": "1. FC Köln", "koln": "1. FC Köln", "cologne": "1. FC Köln",
    "1. fc köln": "1. FC Köln", "heidenheim": "1. FC Heidenheim",
    "young boys": "BSC Young Boys", "bsc young boys": "BSC Young Boys",
    "milan": "AC Milan", "ac milan": "AC Milan",
    "inter": "Inter Milan", "inter milan": "Inter Milan",
    "internazionale": "Inter Milan", "nerazzurri": "Inter Milan",
    "juve": "Juventus", "la vecchia signora": "Juventus",
    "roma": "AS Roma", "as roma": "AS Roma",
    "lazio": "SS Lazio", "ss lazio": "SS Lazio",
    "ssc napoli": "SSC Napoli", "napoli": "SSC Napoli",
    "atalanta": "Atalanta", "fiorentina": "ACF Fiorentina",
    "acf fiorentina": "ACF Fiorentina", "torino": "Torino",
    "hellas": "Hellas Verona", "hellas verona": "Hellas Verona",
    "verona": "Hellas Verona", "lecce": "US Lecce", "us lecce": "US Lecce",
    "salernitana": "US Salernitana",
    "psg": "Paris Saint-Germain", "paris": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "lyon": "Olympique Lyonnais", "ol": "Olympique Lyonnais",
    "marseille": "Olympique de Marseille", "om": "Olympique de Marseille",
    "monaco": "AS Monaco", "as monaco": "AS Monaco",
    "nice": "OGC Nice", "ogc nice": "OGC Nice",
    "lille": "Lille OSC", "losc": "Lille OSC", "losc lille": "Lille OSC",
    "lens": "RC Lens", "rc lens": "RC Lens",
    "rennes": "Stade Rennais", "nantes": "FC Nantes",
    "brest": "Stade Brestois", "reims": "Stade de Reims",
    "benfica": "SL Benfica", "sl benfica": "SL Benfica",
    "porto": "FC Porto", "fc porto": "FC Porto",
    "sporting": "Sporting CP", "sporting cp": "Sporting CP",
    "braga": "SC Braga", "sc braga": "SC Braga",
    "ajax": "AFC Ajax", "afc ajax": "AFC Ajax",
    "psv": "PSV Eindhoven", "psv eindhoven": "PSV Eindhoven",
    "eindhoven": "PSV Eindhoven", "feyenoord": "Feyenoord",
    "az": "AZ Alkmaar", "alkmaar": "AZ Alkmaar",
    "twente": "FC Twente", "fc twente": "FC Twente",
    "цска": "ЦСКА Москва", "спартак": "Спартак Москва",
    "зенит": "Зенит", "локо": "Локомотив Москва",
    "локомотив": "Локомотив Москва", "динамо": "Динамо Москва",
    "краснодар": "Краснодар",
    "if elfsborg": "IF Elfsborg", "ifk göteborg": "IFK Göteborg",
    "if göteborg": "IFK Göteborg",
}


def normalize_team_name(raw: str) -> str:
    if not raw or not raw.strip():
        return raw
    stripped = raw.strip()
    lower    = stripped.lower()
    if lower in _ALIASES:
        return _ALIASES[lower]
    stripped_name = strip_legal_suffix(stripped)
    if stripped_name.lower() in _ALIASES:
        return _ALIASES[stripped_name.lower()]
    if stripped_name != stripped:
        return _to_title(stripped_name)
    return _to_title(stripped)


def teams_are_same(name1: str, name2: str) -> bool:
    return normalize_team_name(name1) == normalize_team_name(name2)
