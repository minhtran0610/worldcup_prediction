import pandas as pd

INITIAL_RATING: float = 1500.0
HOME_ADVANTAGE: float = 100.0
ELO_SCALE: float = 400.0
INACTIVITY_THRESHOLD_DAYS: int = 365
INACTIVITY_DECAY_RATE: float = 0.1

COMPETITION_K: dict[str, float] = {
    "FIFA World Cup": 60,
    "UEFA Euro": 50,
    "Copa America": 50,
    "Africa Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "CONCACAF Gold Cup": 40,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "Friendly": 20,
}
DEFAULT_K: float = 35

_DAYS_PER_YEAR: float = 365.0


def get_competition_k(tournament: str) -> float:
    if tournament in COMPETITION_K:
        return COMPETITION_K[tournament]
    # Sort by key length descending so the most specific prefix wins
    for key, k in sorted(COMPETITION_K.items(), key=lambda x: len(x[0]), reverse=True):
        if tournament.startswith(key):
            return k
    return DEFAULT_K


def _apply_inactivity_decay(
    rating: float,
    last_date: pd.Timestamp | None,
    match_date: pd.Timestamp,
) -> float:
    if last_date is None:
        return rating
    days_gap = (match_date - last_date).days
    if days_gap <= INACTIVITY_THRESHOLD_DAYS:
        return rating
    years_inactive = days_gap / _DAYS_PER_YEAR
    # Pulls rating toward INITIAL_RATING; stronger decay for longer absences
    return rating - INACTIVITY_DECAY_RATE * years_inactive * (rating - INITIAL_RATING)


def _run_elo_pass(
    results: pd.DataFrame,
    *,
    record_pre: bool,
) -> tuple[dict[str, float], list[float], list[float]]:
    ratings: dict[str, float] = {}
    last_played: dict[str, pd.Timestamp | None] = {}
    elo_home_pre: list[float] = []
    elo_away_pre: list[float] = []

    for row in results.itertuples(index=False):
        home: str = row.home_team
        away: str = row.away_team
        match_date: pd.Timestamp = row.date

        rating_h = _apply_inactivity_decay(
            ratings.get(home, INITIAL_RATING), last_played.get(home), match_date
        )
        rating_a = _apply_inactivity_decay(
            ratings.get(away, INITIAL_RATING), last_played.get(away), match_date
        )

        if record_pre:
            elo_home_pre.append(rating_h)
            elo_away_pre.append(rating_a)

        effective_h = rating_h + (0.0 if row.neutral else HOME_ADVANTAGE)
        effective_a = rating_a

        e_h = 1.0 / (1.0 + 10.0 ** ((effective_a - effective_h) / ELO_SCALE))
        e_a = 1.0 - e_h

        home_score: int = row.home_score
        away_score: int = row.away_score
        if home_score > away_score:
            actual_h, actual_a = 1.0, 0.0
        elif home_score == away_score:
            actual_h, actual_a = 0.5, 0.5
        else:
            actual_h, actual_a = 0.0, 1.0

        k = get_competition_k(row.tournament)
        ratings[home] = rating_h + k * (actual_h - e_h)
        ratings[away] = rating_a + k * (actual_a - e_a)

        last_played[home] = match_date
        last_played[away] = match_date

    return ratings, elo_home_pre, elo_away_pre


def compute_elo_ratings(results: pd.DataFrame) -> pd.DataFrame:
    _, elo_home_pre, elo_away_pre = _run_elo_pass(results, record_pre=True)
    out = results.copy()
    out["elo_home_pre"] = elo_home_pre
    out["elo_away_pre"] = elo_away_pre
    return out


def get_current_ratings(results: pd.DataFrame) -> dict[str, float]:
    final_ratings, _, _ = _run_elo_pass(results, record_pre=False)
    return final_ratings


def extend_elo_through_matches(
    base_results: pd.DataFrame,
    new_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Carry Elo forward through new_matches, starting from the end-state of base_results.

    Returns new_matches with elo_home_pre, elo_away_pre, elo_diff columns added.
    new_matches must already have: date, home_team, away_team, home_score, away_score,
    neutral, tournament (needed for home-advantage and K-factor in the Elo pass).
    """
    keep = ["date", "home_team", "away_team", "home_score", "away_score", "neutral", "tournament"]
    n_new = len(new_matches)

    new_slim = new_matches[keep].reset_index(drop=True)
    base_tagged = base_results[keep].assign(_src=0)
    new_tagged = new_slim.assign(_src=range(1, n_new + 1))

    combined = (
        pd.concat([base_tagged, new_tagged], ignore_index=True)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )

    _, elo_h_all, elo_a_all = _run_elo_pass(combined[keep], record_pre=True)
    combined["_elo_h"] = elo_h_all
    combined["_elo_a"] = elo_a_all

    new_rows = combined[combined["_src"] > 0].sort_values("_src")
    out = new_matches.copy().reset_index(drop=True)
    out["elo_home_pre"] = new_rows["_elo_h"].values
    out["elo_away_pre"] = new_rows["_elo_a"].values
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]
    return out
