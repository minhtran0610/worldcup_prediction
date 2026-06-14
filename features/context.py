from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from features.squad_registry import SquadRegistry

FRIENDLY_SAMPLE_WEIGHT: float = 0.2
FRIENDLY_TOURNAMENT: str = "Friendly"

KNOCKOUT_TOURNAMENTS: frozenset[str] = frozenset(
    {
        "FIFA World Cup",
        "UEFA Euro",
        "Copa America",
        "Africa Cup of Nations",
        "AFC Asian Cup",
        "CONCACAF Gold Cup",
    }
)

WC_2026_HOSTS: frozenset[str] = frozenset({"United States", "Canada", "Mexico"})


def derive_context(
    results: pd.DataFrame,
    squad_registry: SquadRegistry | None = None,
) -> pd.DataFrame:
    last_match: dict[str, pd.Timestamp] = {}

    rest_days_home: list[float] = []
    rest_days_away: list[float] = []

    for row in results.itertuples(index=False):
        home: str = row.home_team
        away: str = row.away_team
        match_date: pd.Timestamp = row.date

        rest_h = float((match_date - last_match[home]).days) if home in last_match else float("nan")
        rest_a = float((match_date - last_match[away]).days) if away in last_match else float("nan")

        rest_days_home.append(rest_h)
        rest_days_away.append(rest_a)

        last_match[home] = match_date
        last_match[away] = match_date

    out = results.copy()
    out["rest_days_home"] = rest_days_home
    out["rest_days_away"] = rest_days_away
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]
    out["is_knockout"] = out["tournament"].isin(KNOCKOUT_TOURNAMENTS)
    out["is_host_home"] = out["home_team"].isin(WC_2026_HOSTS)
    out["is_host_away"] = out["away_team"].isin(WC_2026_HOSTS)
    out["sample_weight"] = out["tournament"].apply(
        lambda t: FRIENDLY_SAMPLE_WEIGHT if t == FRIENDLY_TOURNAMENT else 1.0
    )

    # Squad quality features (default 0.0; populated if registry is provided)
    out["squad_top5_home"] = 0.0
    out["squad_top5_away"] = 0.0
    out["squad_caps_home"] = 0.0
    out["squad_caps_away"] = 0.0

    if squad_registry is not None:
        top5_home_vals: list[float] = []
        top5_away_vals: list[float] = []
        caps_home_vals: list[float] = []
        caps_away_vals: list[float] = []

        for row in out.itertuples(index=False):
            year: int = int(row.date.year)
            tournament: str = row.tournament

            feats_h = squad_registry.get_features(row.home_team, year, tournament)
            feats_a = squad_registry.get_features(row.away_team, year, tournament)

            top5_home_vals.append(feats_h["top5_share"])
            top5_away_vals.append(feats_a["top5_share"])
            caps_home_vals.append(feats_h["avg_caps_norm"])
            caps_away_vals.append(feats_a["avg_caps_norm"])

        out["squad_top5_home"] = top5_home_vals
        out["squad_top5_away"] = top5_away_vals
        out["squad_caps_home"] = caps_home_vals
        out["squad_caps_away"] = caps_away_vals

    return out
