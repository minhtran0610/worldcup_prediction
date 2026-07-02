"""Reconstruct 90-minute regulation scores for matches whose stored score
includes extra time.

1X2 betting markets settle on the regulation-time result, but the Kaggle
results CSV stores the full-time score INCLUDING extra time for some
historical knockout matches (verified: Croatia 2-1 England, 2018 WC
semifinal, was 1-1 after 90 minutes — Mandzukic's winner came in the 109th
minute).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

GOALSCORERS_CSV_DEFAULT: Path = Path("data/raw/goalscorers.csv")

_ET_MINUTE_THRESHOLD: int = 90


def correct_extra_time_scores(results: pd.DataFrame, goalscorers: pd.DataFrame) -> pd.DataFrame:
    """Return results with home_score/away_score corrected to the 90-minute
    regulation score for matches that went to extra time.

    goalscorers must have columns: date, home_team, away_team, team, minute.
    A match is corrected only if goalscorers has coverage for it (matched on
    date/home_team/away_team) AND at least one goal has minute > 90 — i.e. it
    went to extra time. Matches without goalscorer coverage, or with no goal
    past minute 90, are returned unchanged.

    own_goal rows in goalscorers.csv already credit the benefiting team in
    the `team` column (verified: Argentina 1-0 Chile, 1917 — the sole goal
    row is team=Argentina, own_goal=True, scored by a Chilean player), so no
    special-casing for own goals is needed.
    """
    gs = goalscorers.copy()
    gs["date"] = pd.to_datetime(gs["date"])

    out = results.copy()
    out["date"] = pd.to_datetime(out["date"])

    if gs.empty:
        return out

    max_minute = gs.groupby(["date", "home_team", "away_team"])["minute"].max()
    et_keys = max_minute[max_minute > _ET_MINUTE_THRESHOLD].index

    if len(et_keys) == 0:
        return out

    reg_goals = gs[gs["minute"] <= _ET_MINUTE_THRESHOLD]
    reg_counts = (
        reg_goals.groupby(["date", "home_team", "away_team", "team"])
        .size()
        .rename("n")
        .reset_index()
    )

    n_corrected = 0
    for date, home, away in et_keys:
        mask = (out["date"] == date) & (out["home_team"] == home) & (out["away_team"] == away)
        if not mask.any():
            continue

        home_rows = reg_counts[
            (reg_counts["date"] == date)
            & (reg_counts["home_team"] == home)
            & (reg_counts["away_team"] == away)
            & (reg_counts["team"] == home)
        ]
        away_rows = reg_counts[
            (reg_counts["date"] == date)
            & (reg_counts["home_team"] == home)
            & (reg_counts["away_team"] == away)
            & (reg_counts["team"] == away)
        ]
        new_home_score = int(home_rows["n"].iloc[0]) if len(home_rows) else 0
        new_away_score = int(away_rows["n"].iloc[0]) if len(away_rows) else 0

        out.loc[mask, "home_score"] = new_home_score
        out.loc[mask, "away_score"] = new_away_score
        n_corrected += 1

    print(
        f"[extra_time] Corrected {n_corrected} match(es) to 90-minute regulation score.",
        file=sys.stderr,
    )
    return out
