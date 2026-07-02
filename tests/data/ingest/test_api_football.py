from __future__ import annotations

from data.ingest.api_football import _extract_fulltime_score


def test_extract_fulltime_score_uses_fulltime_not_goals():
    """A match decided in extra time: `goals` reflects the AET score, but
    score.fulltime is the clean 90-minute score we want."""
    fx = {
        "goals": {"home": 2, "away": 1},
        "score": {
            "halftime": {"home": 0, "away": 0},
            "fulltime": {"home": 1, "away": 1},
            "extratime": {"home": 2, "away": 1},
            "penalty": {"home": None, "away": None},
        },
    }
    home, away = _extract_fulltime_score(fx)
    assert home == 1
    assert away == 1


def test_extract_fulltime_score_not_yet_played_returns_none():
    fx = {
        "goals": {"home": None, "away": None},
        "score": {"fulltime": {"home": None, "away": None}},
    }
    home, away = _extract_fulltime_score(fx)
    assert home is None
    assert away is None


def test_extract_fulltime_score_missing_score_block_returns_none():
    fx = {"goals": {"home": 1, "away": 0}}
    home, away = _extract_fulltime_score(fx)
    assert home is None
    assert away is None
