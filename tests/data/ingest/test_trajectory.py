from __future__ import annotations

import pandas as pd

import data.ingest.trajectory as trajectory
from data.ingest.llm_form import FormAnalysis


def test_get_team_trajectory_uses_cache_and_skips_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", tmp_path / "team_trajectory.json")

    calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        calls.append((team, opponent, match_date))
        return ["Spain were dominant."], ["https://example.com/a"]

    def fake_analyse(team, texts, urls=None, model=None):
        return FormAnalysis(team=team, form_score=0.5, confidence=0.8, n_articles=len(texts))

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_form", fake_analyse)

    matches = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
        }
    )

    first = trajectory.get_team_trajectory("Spain", matches)
    assert len(first) == 1
    assert first[0].form_score == 0.5
    assert len(calls) == 1

    second = trajectory.get_team_trajectory("Spain", matches)
    assert len(second) == 1
    assert len(calls) == 1  # not re-fetched — served from cache


def test_get_team_trajectory_chronological_order(tmp_path, monkeypatch):
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", tmp_path / "team_trajectory.json")

    def fake_fetch(team, opponent, match_date, api_key=None):
        return [f"Report vs {opponent}"], ["https://example.com/x"]

    def fake_analyse(team, texts, urls=None, model=None):
        return FormAnalysis(team=team, form_score=0.1, confidence=0.5, performance_context=texts[0])

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_form", fake_analyse)

    matches = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-24"), pd.Timestamp("2026-06-13")],
            "home_team": ["Spain", "Spain"],
            "away_team": ["Uruguay", "Cape Verde"],
        }
    )

    out = trajectory.get_team_trajectory("Spain", matches)
    assert [a.performance_context for a in out] == [
        "Report vs Cape Verde",
        "Report vs Uruguay",
    ]
