from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from data.ingest.results import is_wc2026_match

if TYPE_CHECKING:
    from features.squad_registry import SquadRegistry

FRIENDLY_SAMPLE_WEIGHT: float = 0.2
FRIENDLY_TOURNAMENT: str = "Friendly"
RECENCY_HALF_LIFE_DAYS: float = 365.0 * 3
WC2026_BOOST: float = 1.0

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


def compute_sample_weight(results: pd.DataFrame) -> pd.Series:
    """Return per-row training weight: tournament_base x recency x wc2026_boost.

    tournament_base: FRIENDLY_SAMPLE_WEIGHT for friendlies, 1.0 otherwise.
    recency: exponential decay with a RECENCY_HALF_LIFE_DAYS half-life,
      relative to the most recent date in `results` — a gentle decay
      (years, not days) since this trains once over ~150 years of history,
      unlike Dixon-Coles' much steeper per-refit xi decay.
    wc2026_boost: WC2026_BOOST multiplier on WC 2026 matches, on top of
      recency. Left at 1.0 (no-op) as of 2026-07-09: a chronological sweep
      (train on history + completed group stage, boost values 1/4/8/12/16/24,
      eval RPS/NLL/accuracy on the 26 held-out completed knockout matches)
      showed accuracy degrading monotonically from 65% at boost=1.0 to 54%
      at boost=24 — plain recency already weights same-week matches near 1.0,
      so the extra multiplier only let a small group-stage sample dominate
      gradients and overfit to group-stage-specific patterns that don't
      transfer to knockout football. Re-validate with the same sweep
      methodology if this is revisited.
    """
    base = results["tournament"].apply(
        lambda t: FRIENDLY_SAMPLE_WEIGHT if t == FRIENDLY_TOURNAMENT else 1.0
    )
    dates = results["date"]
    latest = dates.max()
    days_ago = (latest - dates).dt.days.clip(lower=0)
    recency = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    boost = np.where(is_wc2026_match(results), WC2026_BOOST, 1.0)
    return base * recency * boost


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
    out["sample_weight"] = compute_sample_weight(out)

    # Squad quality features (default 0.0; populated if registry is provided)
    out["squad_top5_home"] = 0.0
    out["squad_top5_away"] = 0.0
    out["squad_caps_home"] = 0.0
    out["squad_caps_away"] = 0.0
    out["squad_goals_home"] = 0.0
    out["squad_goals_away"] = 0.0

    if squad_registry is not None:
        top5_home_vals: list[float] = []
        top5_away_vals: list[float] = []
        caps_home_vals: list[float] = []
        caps_away_vals: list[float] = []
        goals_home_vals: list[float] = []
        goals_away_vals: list[float] = []

        for row in out.itertuples(index=False):
            year: int = int(row.date.year)
            tournament: str = row.tournament

            feats_h = squad_registry.get_features(row.home_team, year, tournament)
            feats_a = squad_registry.get_features(row.away_team, year, tournament)

            top5_home_vals.append(feats_h["top5_share"])
            top5_away_vals.append(feats_a["top5_share"])
            caps_home_vals.append(feats_h["avg_caps_norm"])
            caps_away_vals.append(feats_a["avg_caps_norm"])
            goals_home_vals.append(feats_h["intl_goals_per_cap"])
            goals_away_vals.append(feats_a["intl_goals_per_cap"])

        out["squad_top5_home"] = top5_home_vals
        out["squad_top5_away"] = top5_away_vals
        out["squad_caps_home"] = caps_home_vals
        out["squad_caps_away"] = caps_away_vals
        out["squad_goals_home"] = goals_home_vals
        out["squad_goals_away"] = goals_away_vals

    return out
