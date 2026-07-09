from __future__ import annotations

import json

import pandas as pd

import data.ingest.trajectory as trajectory
from data.ingest.llm_form import FormAnalysis


def _patch_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(trajectory, "_ARTICLE_CACHE_PATH", tmp_path / "team_match_articles.json")
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", tmp_path / "team_trajectory.json")


def _matches(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """rows: (date, opponent, home_score, away_score) — home_team is always 'Spain'."""
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(r[0]) for r in rows],
            "home_team": ["Spain"] * len(rows),
            "away_team": [r[1] for r in rows],
            "home_score": [r[2] for r in rows],
            "away_score": [r[3] for r in rows],
        }
    )


def test_get_team_trajectory_empty_matches_returns_neutral_without_calls(tmp_path, monkeypatch):
    _patch_caches(monkeypatch, tmp_path)

    calls = []
    monkeypatch.setattr(
        trajectory, "fetch_team_match_report", lambda *a, **k: calls.append(1) or ([], [])
    )
    monkeypatch.setattr(
        trajectory,
        "analyse_team_trajectory",
        lambda *a, **k: calls.append(1) or FormAnalysis.neutral("Spain"),
    )

    result = trajectory.get_team_trajectory("Spain", pd.DataFrame(columns=["date"]))
    assert result.team == "Spain"
    assert result.confidence == 0.0
    assert calls == []


def test_get_team_trajectory_combines_all_matches_into_one_call(tmp_path, monkeypatch):
    _patch_caches(monkeypatch, tmp_path)

    def fake_fetch(team, opponent, match_date, api_key=None):
        return [f"Report vs {opponent}"], [f"https://example.com/{opponent}"]

    analyse_calls = []

    def fake_analyse(team, match_blocks, urls=None, model=None):
        analyse_calls.append(match_blocks)
        return FormAnalysis(team=team, form_score=0.4, confidence=0.7, n_articles=len(urls or []))

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_trajectory", fake_analyse)

    # Deliberately out of chronological order in the input.
    matches = _matches(
        [
            ("2026-06-24", "Uruguay", 2, 0),
            ("2026-06-13", "Cape Verde", 3, 0),
        ]
    )

    result = trajectory.get_team_trajectory("Spain", matches)

    assert result.form_score == 0.4
    assert len(analyse_calls) == 1
    blocks = analyse_calls[0]
    assert len(blocks) == 2
    # Chronological order regardless of input row order.
    assert "vs Cape Verde" in blocks[0] and "Match 1 of 2" in blocks[0]
    assert "vs Uruguay" in blocks[1] and "Match 2 of 2" in blocks[1]
    # Result/scoreline is stated from Spain's perspective.
    assert "won 3-0" in blocks[0]
    assert "won 2-0" in blocks[1]


def test_get_team_trajectory_tier2_cache_hit_skips_fetch_and_llm(tmp_path, monkeypatch):
    _patch_caches(monkeypatch, tmp_path)

    calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        calls.append("fetch")
        return ["report"], ["https://example.com/x"]

    def fake_analyse(team, match_blocks, urls=None, model=None):
        calls.append("analyse")
        return FormAnalysis(team=team, form_score=0.2, confidence=0.6)

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_trajectory", fake_analyse)

    matches = _matches([("2026-06-15", "Croatia", 1, 0)])

    first = trajectory.get_team_trajectory("Spain", matches)
    assert first.form_score == 0.2
    assert calls == ["fetch", "analyse"]

    calls.clear()
    second = trajectory.get_team_trajectory("Spain", matches)
    assert second.form_score == 0.2
    assert calls == []  # tier-2 cache hit — no fetch, no LLM call


def test_get_team_trajectory_new_match_reuses_tier1_cache_for_old_matches(tmp_path, monkeypatch):
    _patch_caches(monkeypatch, tmp_path)

    fetch_calls = []
    analyse_calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        fetch_calls.append(opponent)
        return [f"report vs {opponent}"], [f"https://example.com/{opponent}"]

    def fake_analyse(team, match_blocks, urls=None, model=None):
        analyse_calls.append(len(match_blocks))
        return FormAnalysis(team=team, form_score=0.1, confidence=0.5)

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_trajectory", fake_analyse)

    two_matches = _matches([("2026-06-13", "Cape Verde", 3, 0), ("2026-06-19", "Japan", 1, 1)])
    trajectory.get_team_trajectory("Spain", two_matches)
    assert fetch_calls == ["Cape Verde", "Japan"]
    assert analyse_calls == [2]

    fetch_calls.clear()
    three_matches = _matches(
        [
            ("2026-06-13", "Cape Verde", 3, 0),
            ("2026-06-19", "Japan", 1, 1),
            ("2026-06-24", "Uruguay", 2, 0),
        ]
    )
    trajectory.get_team_trajectory("Spain", three_matches)
    # Only the new match needed a fresh Guardian fetch.
    assert fetch_calls == ["Uruguay"]
    # But the combined LLM call re-ran (new tier-2 key) across all 3 matches.
    assert analyse_calls == [2, 3]


def test_get_team_trajectory_no_articles_skips_llm_call(tmp_path, monkeypatch):
    _patch_caches(monkeypatch, tmp_path)

    monkeypatch.setattr(trajectory, "fetch_team_match_report", lambda *a, **k: ([], []))

    analyse_calls = []
    monkeypatch.setattr(
        trajectory,
        "analyse_team_trajectory",
        lambda *a, **k: analyse_calls.append(1) or FormAnalysis(team="Spain"),
    )

    matches = _matches([("2026-06-15", "Croatia", 1, 0)])
    result = trajectory.get_team_trajectory("Spain", matches)

    assert analyse_calls == []
    assert result.confidence == 0.0
    assert result.form_score == 0.0

    # Cached — a second call also makes no fetch/LLM calls.
    result2 = trajectory.get_team_trajectory("Spain", matches)
    assert result2.confidence == 0.0


def test_get_team_trajectory_does_not_cache_a_failed_analysis(tmp_path, monkeypatch):
    _patch_caches(monkeypatch, tmp_path)

    fetch_calls = []
    analyse_calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        fetch_calls.append(opponent)
        return ["report"], ["https://example.com/x"]

    # First call fails (e.g. Ollama returned unparseable JSON); second call
    # succeeds — simulating a transient failure that should be retried, not
    # permanently frozen into the cache.
    def fake_analyse(team, match_blocks, urls=None, model=None):
        analyse_calls.append(1)
        if len(analyse_calls) == 1:
            return FormAnalysis.neutral(team, "Expecting value: line 1 column 1 (char 0)")
        return FormAnalysis(team=team, form_score=0.5, confidence=0.8)

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_trajectory", fake_analyse)

    matches = _matches([("2026-06-15", "Croatia", 1, 0)])

    first = trajectory.get_team_trajectory("Spain", matches)
    assert first.error is not None
    assert first.confidence == 0.0

    second = trajectory.get_team_trajectory("Spain", matches)
    assert second.error is None
    assert second.form_score == 0.5
    assert second.confidence == 0.8
    # The failed first attempt was not cached — the second call recomputed
    # (re-ran the LLM call), but did not need to re-fetch the article text.
    assert analyse_calls == [1, 1]
    assert fetch_calls == ["Croatia"]


def test_get_team_trajectory_recovers_from_malformed_tier2_cache_entry(tmp_path, monkeypatch):
    cache_path = tmp_path / "team_trajectory.json"
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", cache_path)
    monkeypatch.setattr(trajectory, "_ARTICLE_CACHE_PATH", tmp_path / "team_match_articles.json")

    calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        calls.append("fetch")
        return ["Good form report"], ["https://example.com/report"]

    def fake_analyse(team, match_blocks, urls=None, model=None):
        calls.append("analyse")
        return FormAnalysis(team=team, form_score=0.7, confidence=0.9)

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_trajectory", fake_analyse)

    matches = _matches([("2026-06-15", "Croatia", 1, 0)])
    key = trajectory._trajectory_key("Spain", [pd.Timestamp("2026-06-15")])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        # Missing the required 'team' key — FormAnalysis.from_dict() raises.
        json.dump({key: {"form_score": 0.5, "confidence": 0.8}}, f)

    result = trajectory.get_team_trajectory("Spain", matches)

    assert result.team == "Spain"
    assert result.form_score == 0.7
    assert result.confidence == 0.9
    assert calls == ["fetch", "analyse"]
