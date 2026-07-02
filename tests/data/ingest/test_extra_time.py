from __future__ import annotations

import pandas as pd

from data.ingest.extra_time import correct_extra_time_scores


def _results_row(date: str, home: str, away: str, hs: int, as_: int) -> dict:
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": "FIFA World Cup",
        "neutral": True,
        "country": "Russia",
    }


def _goal_row(
    date: str, home: str, away: str, team: str, minute: int, own_goal: bool = False
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "team": team,
        "scorer": "Someone",
        "minute": minute,
        "own_goal": own_goal,
        "penalty": False,
    }


def test_extra_time_match_corrected_to_regulation_score():
    """Croatia 2-1 England (2018 WC semifinal) should be corrected to 1-1 —
    Mandzukic's winner came in the 109th minute (extra time)."""
    results = pd.DataFrame([_results_row("2018-07-11", "Croatia", "England", 2, 1)])
    goalscorers = pd.DataFrame(
        [
            _goal_row("2018-07-11", "Croatia", "England", "England", 5),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 68),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 109),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 1
    assert row["away_score"] == 1


def test_match_without_et_goals_is_unchanged():
    """A match with no goal past minute 90 must be returned untouched."""
    results = pd.DataFrame([_results_row("2019-06-01", "France", "Germany", 2, 0)])
    goalscorers = pd.DataFrame(
        [
            _goal_row("2019-06-01", "France", "Germany", "France", 10),
            _goal_row("2019-06-01", "France", "Germany", "France", 80),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 2
    assert row["away_score"] == 0


def test_match_without_goalscorer_coverage_is_unchanged():
    """A match absent from goalscorers.csv must be left exactly as-is —
    its extra-time status cannot be verified."""
    results = pd.DataFrame([_results_row("1955-03-01", "Uruguay", "Peru", 3, 1)])
    goalscorers = pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "team",
            "scorer",
            "minute",
            "own_goal",
            "penalty",
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 3
    assert row["away_score"] == 1


def test_own_goal_in_regulation_time_credited_correctly():
    """An own goal within regulation time counts toward the benefiting team
    (the `team` column already reflects this) — must not need flipping.
    A later extra-time goal is added so the match enters the correction path."""
    results = pd.DataFrame([_results_row("1917-10-06", "Argentina", "Chile", 1, 0)])
    goalscorers = pd.DataFrame(
        [
            _goal_row("1917-10-06", "Argentina", "Chile", "Argentina", 76, own_goal=True),
            _goal_row("1917-10-06", "Argentina", "Chile", "Chile", 95),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 1
    assert row["away_score"] == 0


def test_multiple_matches_only_et_ones_corrected():
    """With several matches in the frame, only the one with an ET goal changes."""
    results = pd.DataFrame(
        [
            _results_row("2018-07-11", "Croatia", "England", 2, 1),
            _results_row("2018-07-10", "France", "Belgium", 1, 0),
        ]
    )
    goalscorers = pd.DataFrame(
        [
            _goal_row("2018-07-11", "Croatia", "England", "England", 5),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 68),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 109),
            _goal_row("2018-07-10", "France", "Belgium", "France", 51),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    croatia_row = out[out["home_team"] == "Croatia"].iloc[0]
    france_row = out[out["home_team"] == "France"].iloc[0]
    assert (croatia_row["home_score"], croatia_row["away_score"]) == (1, 1)
    assert (france_row["home_score"], france_row["away_score"]) == (1, 0)
