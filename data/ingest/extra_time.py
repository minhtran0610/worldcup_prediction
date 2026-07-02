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

    Known false-positive class: goalscorers.csv's minute column has a tail
    at 91-96 that mixes genuine early-extra-time goals with 2nd-half
    stoppage-time goals recorded literally (e.g. "90+3" stored as 93). A
    single-leg match can only reach extra time if it is level at 90 minutes,
    so any correction that produces a NON-DRAW 90-minute score for a match
    that isn't a two-legged tie is almost certainly this misread, not a
    genuine ET case. We do not attempt to detect two-legged ties here (no
    reliable signal in this data) — instead we just warn loudly on every
    non-draw correction so a human can eyeball it.
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
    non_draw_warnings: list[str] = []
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

        old_home_score = out.loc[mask, "home_score"].iloc[0]
        old_away_score = out.loc[mask, "away_score"].iloc[0]

        out.loc[mask, "home_score"] = new_home_score
        out.loc[mask, "away_score"] = new_away_score
        n_corrected += 1

        if new_home_score != new_away_score:
            date_str = pd.Timestamp(date).date().isoformat()
            non_draw_warnings.append(
                f"[extra_time] WARNING: non-draw correction for {date_str} {home} vs {away}: "
                f"{old_home_score}-{old_away_score} -> {new_home_score}-{new_away_score} "
                "(verify this is a genuine two-legged extra-time tie, not a "
                "stoppage-time-goal misread)"
            )

    for warning in non_draw_warnings:
        print(warning, file=sys.stderr)

    print(
        f"[extra_time] Corrected {n_corrected} match(es) to 90-minute regulation score.",
        file=sys.stderr,
    )
    return out
