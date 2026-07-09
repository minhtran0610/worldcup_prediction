from __future__ import annotations

import pandas as pd

import data.ingest.cache as cache_module
from data.ingest.results import load_results


def _write_results_csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_results_applies_extra_time_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)

    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "neutral": "True",
                "country": "Russia",
            }
        ],
    )

    goalscorers_csv = tmp_path / "goalscorers.csv"
    pd.DataFrame(
        [
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "team": "England",
                "scorer": "Trippier",
                "minute": 5,
                "own_goal": False,
                "penalty": False,
            },
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "team": "Croatia",
                "scorer": "Perisic",
                "minute": 68,
                "own_goal": False,
                "penalty": False,
            },
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "team": "Croatia",
                "scorer": "Mandzukic",
                "minute": 109,
                "own_goal": False,
                "penalty": False,
            },
        ]
    ).to_csv(goalscorers_csv, index=False)

    out = load_results(
        csv_path=results_csv, force_refresh=True, goalscorers_csv_path=goalscorers_csv
    )

    row = out.iloc[0]
    assert row["home_score"] == 1
    assert row["away_score"] == 1


def test_load_results_skips_correction_when_goalscorers_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)

    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "neutral": "True",
                "country": "Russia",
            }
        ],
    )

    out = load_results(
        csv_path=results_csv,
        force_refresh=True,
        goalscorers_csv_path=tmp_path / "does_not_exist.csv",
    )
    row = out.iloc[0]
    assert row["home_score"] == 2
    assert row["away_score"] == 1
