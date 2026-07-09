from __future__ import annotations

import pandas as pd

from data.ingest.wc2026 import inject_completed_wc2026_matches


class _StubRegistry:
    """Minimal stand-in for SquadRegistry — avoids network/file IO in tests."""

    def get_features(self, team: str, year: int, tournament: str) -> dict:
        return {"top5_share": 0.5, "avg_caps_norm": 0.3, "intl_goals_per_cap": 0.1}


def _base_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "home_team": ["Argentina", "France"],
            "away_team": ["Brazil", "Germany"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )


def test_group_stage_match_tagged_not_knockout():
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
            "home_score": [2],
            "away_score": [0],
        }
    )
    out = inject_completed_wc2026_matches(
        results,
        completed,
        _StubRegistry(),  # ty: ignore[invalid-argument-type]
    )
    injected = out[out["home_team"] == "Spain"].iloc[0]
    assert injected["is_knockout"] == False  # noqa: E712


def test_knockout_stage_match_tagged_knockout():
    """South Africa vs Canada, 2026-06-28, is the first Round of 32 match —
    the day after the group stage's final matchday (2026-06-27)."""
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-28")],
            "home_team": ["South Africa"],
            "away_team": ["Canada"],
            "home_score": [0],
            "away_score": [1],
        }
    )
    out = inject_completed_wc2026_matches(
        results,
        completed,
        _StubRegistry(),  # ty: ignore[invalid-argument-type]
    )
    injected = out[out["home_team"] == "South Africa"].iloc[0]
    assert injected["is_knockout"] == True  # noqa: E712


def test_same_day_group_finale_not_tagged_knockout_when_stage_given():
    """Group J's final matchday kicked off 2026-06-28 02:00, hours before
    Round of 32's opener later that same calendar day — a pure date cutoff
    misclassifies it as knockout. The `stage` label disambiguates it."""
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-28 02:00:00")],
            "home_team": ["Algeria"],
            "away_team": ["Austria"],
            "home_score": [1],
            "away_score": [1],
            "stage": ["Group J - Matchday 17"],
        }
    )
    out = inject_completed_wc2026_matches(
        results,
        completed,
        _StubRegistry(),  # ty: ignore[invalid-argument-type]
    )
    injected = out[out["home_team"] == "Algeria"].iloc[0]
    assert injected["is_knockout"] == False  # noqa: E712


def test_ambiguous_generic_group_label_falls_back_to_date():
    """The Wikipedia-scrape fallback always writes the bare stage 'Group'
    with no matchday detail — too ambiguous to trust, so this still falls
    back to the date cutoff rather than blanket-treating it as group stage."""
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-28")],
            "home_team": ["South Africa"],
            "away_team": ["Canada"],
            "home_score": [0],
            "away_score": [1],
            "stage": ["Group"],
        }
    )
    out = inject_completed_wc2026_matches(
        results,
        completed,
        _StubRegistry(),  # ty: ignore[invalid-argument-type]
    )
    injected = out[out["home_team"] == "South Africa"].iloc[0]
    assert injected["is_knockout"] == True  # noqa: E712


def test_injected_matches_are_appended_and_sorted_by_date():
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
            "home_score": [2],
            "away_score": [0],
        }
    )
    out = inject_completed_wc2026_matches(
        results,
        completed,
        _StubRegistry(),  # ty: ignore[invalid-argument-type]
    )
    assert len(out) == 3
    assert out["date"].is_monotonic_increasing


def test_empty_completed_returns_results_unchanged():
    results = _base_results()
    completed = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])
    out = inject_completed_wc2026_matches(
        results,
        completed,
        _StubRegistry(),  # ty: ignore[invalid-argument-type]
    )
    assert len(out) == len(results)
