import pandas as pd

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


def derive_context(results: pd.DataFrame) -> pd.DataFrame:
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
    return out
