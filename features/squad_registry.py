"""Tournament squad registry: maps (tournament_year, team) -> squad quality features."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Maps (tournament_name_fragment, year) to tournament_key for squad lookup.
# tournament_name_fragment matches substrings of Kaggle CSV 'tournament' column.
TOURNAMENT_KEY_MAP: list[tuple[str, int, str]] = [
    ("FIFA World Cup", 2026, "wc2026"),
    ("FIFA World Cup", 2022, "wc2022"),
    ("FIFA World Cup", 2018, "wc2018"),
    ("FIFA World Cup", 2014, "wc2014"),
    ("UEFA Euro", 2024, "euro2024"),
    ("UEFA Euro", 2021, "euro2021"),  # Euro 2020 played in 2021
    ("UEFA Euro", 2016, "euro2016"),
]

# Big 5 league clubs for the 2024-25 season.
# Used to compute top5_share: fraction of squad players whose club is in this set.
# Both common Wikipedia name variants are included for robustness.
BIG5_CLUBS: frozenset[str] = frozenset(
    [
        # Premier League (20)
        "Arsenal",
        "Aston Villa",
        "Bournemouth",
        "Brentford",
        "Brighton",
        "Brighton & Hove Albion",
        "Chelsea",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Ipswich Town",
        "Ipswich",
        "Leicester City",
        "Leicester",
        "Liverpool",
        "Manchester City",
        "Man City",
        "Manchester United",
        "Man Utd",
        "Man United",
        "Newcastle United",
        "Newcastle",
        "Nottingham Forest",
        "Southampton",
        "Tottenham Hotspur",
        "Tottenham",
        "Spurs",
        "West Ham United",
        "West Ham",
        "Wolverhampton Wanderers",
        "Wolves",
        # La Liga (20)
        "Athletic Club",
        "Athletic Bilbao",
        "Atletico Madrid",
        "Atlético Madrid",
        "Atl. Madrid",
        "Barcelona",
        "Celta Vigo",
        "Celta",
        "Deportivo Alaves",
        "Alavés",
        "Alaves",
        "Espanyol",
        "Getafe",
        "Girona",
        "Las Palmas",
        "Leganes",
        "Leganés",
        "Mallorca",
        "Osasuna",
        "Rayo Vallecano",
        "Rayo",
        "Real Betis",
        "Betis",
        "Real Madrid",
        "Real Sociedad",
        "Real Valladolid",
        "Valladolid",
        "Sevilla",
        "Valencia",
        "Villarreal",
        # Bundesliga (18)
        "Augsburg",
        "FC Augsburg",
        "Bayer Leverkusen",
        "Leverkusen",
        "Bayern Munich",
        "FC Bayern Munich",
        "Bayern",
        "Borussia Dortmund",
        "Dortmund",
        "BVB",
        "Borussia Monchengladbach",
        "Borussia M'gladbach",
        "Borussia Mönchengladbach",
        "Monchengladbach",
        "Eintracht Frankfurt",
        "Frankfurt",
        "FC Heidenheim",
        "Heidenheim",
        "FC St. Pauli",
        "St. Pauli",
        "Freiburg",
        "SC Freiburg",
        "Hamburg",
        "Hamburger SV",
        "HSV",
        "Holstein Kiel",
        "Kiel",
        "Mainz",
        "FSV Mainz 05",
        "RB Leipzig",
        "Leipzig",
        "Stuttgart",
        "VfB Stuttgart",
        "Union Berlin",
        "1. FC Union Berlin",
        "VfL Bochum",
        "Bochum",
        "VfL Wolfsburg",
        "Wolfsburg",
        "Werder Bremen",
        "Bremen",
        # Serie A (20)
        "AC Milan",
        "Milan",
        "Atalanta",
        "Bologna",
        "Cagliari",
        "Como",
        "Empoli",
        "Fiorentina",
        "Genoa",
        "Hellas Verona",
        "Verona",
        "Inter Milan",
        "Inter",
        "Internazionale",
        "FC Internazionale",
        "Juventus",
        "Lazio",
        "SS Lazio",
        "Lecce",
        "Monza",
        "Napoli",
        "SSC Napoli",
        "Parma",
        "Roma",
        "AS Roma",
        "Torino",
        "Udinese",
        "Venezia",
        # Ligue 1 (18)
        "Angers",
        "Auxerre",
        "AJ Auxerre",
        "Brest",
        "Stade Brestois",
        "Le Havre",
        "Lens",
        "RC Lens",
        "Lille",
        "LOSC Lille",
        "Lyon",
        "Olympique Lyonnais",
        "Marseille",
        "Olympique de Marseille",
        "Monaco",
        "AS Monaco",
        "Montpellier",
        "Nantes",
        "FC Nantes",
        "Nice",
        "OGC Nice",
        "PSG",
        "Paris Saint-Germain",
        "Paris SG",
        "Reims",
        "Stade de Reims",
        "Rennes",
        "Stade Rennais",
        "Saint-Etienne",
        "Saint-Étienne",
        "AS Saint-Étienne",
        "Strasbourg",
        "RC Strasbourg",
        "Toulouse",
        "TFC",
    ]
)


def _resolve_tournament_key(tournament: str, year: int) -> str | None:
    """Return tournament_key for a (tournament fragment, year) pair, or None if not found."""
    for fragment, t_year, key in TOURNAMENT_KEY_MAP:
        if fragment in tournament and t_year == year:
            return key
    return None


class SquadRegistry:
    """Registry mapping (tournament_key, team) -> squad quality features."""

    def __init__(self) -> None:
        self._cache: dict[str, pd.DataFrame] = {}  # tournament_key -> squad df
        self._features: dict[
            tuple[str, str], dict[str, float]
        ] = {}  # (tournament_key, team) -> features

    def load_tournament(self, tournament_key: str, force_refresh: bool = False) -> None:
        """Load squad data for one tournament key and compute per-team features."""
        from data.ingest.squads import TOURNAMENT_SQUAD_URLS, load_squads

        if tournament_key not in TOURNAMENT_SQUAD_URLS:
            raise ValueError(f"Unknown tournament_key {tournament_key!r}")

        df = load_squads(tournament_key=tournament_key, force_refresh=force_refresh)
        self._cache[tournament_key] = df
        self._compute_features(tournament_key, df)

    def load_all(self, force_refresh: bool = False) -> None:
        """Load all supported tournaments. Fails silently per tournament (prints warning)."""
        from data.ingest.squads import TOURNAMENT_SQUAD_URLS

        for tournament_key in TOURNAMENT_SQUAD_URLS:
            try:
                self.load_tournament(tournament_key, force_refresh=force_refresh)
            except Exception as exc:
                logger.warning(
                    "[SquadRegistry] Could not load %r: %s — squad features will be 0.0 for this tournament.",
                    tournament_key,
                    exc,
                )

    def _compute_features(self, tournament_key: str, df: pd.DataFrame) -> None:
        """Compute top5_share and avg_caps_norm per team and store in _features."""
        has_caps = "caps" in df.columns

        for team, group in df.groupby("team"):
            n_squad = len(group)
            if n_squad == 0:
                top5_share = 0.0
                avg_caps_norm = 0.0
            else:
                # top5_share: fraction of players whose club is in BIG5_CLUBS
                top5_count = group["club"].apply(lambda c: c in BIG5_CLUBS).sum()
                top5_share = float(top5_count) / n_squad

                # avg_caps_norm: mean caps / 100
                if has_caps:
                    caps_vals = pd.to_numeric(group["caps"], errors="coerce").fillna(0.0)
                    avg_caps_norm = float(caps_vals.mean()) / 100.0
                else:
                    avg_caps_norm = 0.0

            self._features[(tournament_key, str(team))] = {
                "top5_share": top5_share,
                "avg_caps_norm": avg_caps_norm,
            }

    def get_features(self, team: str, year: int, tournament: str) -> dict[str, float]:
        """Return {top5_share, avg_caps_norm} for a team in a tournament.

        Returns zeros if the tournament/team is not found in the registry.
        """
        _default: dict[str, float] = {"top5_share": 0.0, "avg_caps_norm": 0.0}

        tournament_key = _resolve_tournament_key(tournament, year)
        if tournament_key is None:
            return _default

        result = self._features.get((tournament_key, team))
        if result is None:
            return _default

        return result

    @classmethod
    def build(cls, force_refresh: bool = False) -> SquadRegistry:
        """Build registry by loading all tournament squads. Fails silently per tournament."""
        registry = cls()
        registry.load_all(force_refresh=force_refresh)
        return registry
