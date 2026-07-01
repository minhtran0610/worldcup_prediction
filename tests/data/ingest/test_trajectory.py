from __future__ import annotations

import json

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


def test_get_team_trajectory_recovers_from_malformed_cache_entry(tmp_path, monkeypatch):
    """Test that a malformed cache entry is recovered gracefully by re-fetching.

    A malformed cache entry (e.g. missing the required 'team' key) should not
    cause the function to raise. Instead, it should be treated as a cache miss,
    re-fetched, and the cache should be updated.
    """
    cache_path = tmp_path / "team_trajectory.json"
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", cache_path)

    calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        calls.append((team, opponent, match_date))
        return ["Good form report"], ["https://example.com/report"]

    def fake_analyse(team, texts, urls=None, model=None):
        return FormAnalysis(team=team, form_score=0.7, confidence=0.9, n_articles=len(texts))

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_form", fake_analyse)

    # Seed cache with one malformed entry: missing the required 'team' key.
    # This will cause FormAnalysis.from_dict() to raise TypeError.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "Spain|Croatia|2026-06-15": {
                    "form_score": 0.5,
                    "confidence": 0.8,
                    "n_articles": 1,
                    # Missing 'team' key — will cause TypeError when deserializing
                }
            },
            f,
        )

    matches = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
        }
    )

    # Call should not raise despite malformed cache entry
    result = trajectory.get_team_trajectory("Spain", matches)

    # Should have recovered from malformed entry by re-fetching
    assert len(result) == 1
    assert result[0].team == "Spain"
    assert result[0].form_score == 0.7
    assert result[0].confidence == 0.9
    # Should have made one fetch call for the re-fetched entry
    assert len(calls) == 1
    assert calls[0] == ("Spain", "Croatia", pd.Timestamp("2026-06-15"))
