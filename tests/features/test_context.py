from __future__ import annotations

import pandas as pd
import pytest

from features.context import (
    FRIENDLY_SAMPLE_WEIGHT,
    RECENCY_HALF_LIFE_DAYS,
    WC2026_BOOST,
    compute_sample_weight,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_most_recent_match_has_recency_factor_one():
    """The row at the latest date in the frame has days_ago=0 -> recency=1.0."""
    df = _df(
        [
            {"date": "2020-01-01", "tournament": "UEFA Euro qualification"},
            {"date": "2026-06-15", "tournament": "FIFA World Cup"},
        ]
    )
    weights = compute_sample_weight(df)
    # Row 1 is both the latest date AND a WC2026 match.
    assert weights.iloc[1] == pytest.approx(WC2026_BOOST, rel=1e-6)


def test_friendly_gets_base_weight_times_recency():
    df = _df([{"date": "2026-06-01", "tournament": "Friendly"}])
    weights = compute_sample_weight(df)
    # Sole row is also the latest date -> recency=1.0.
    assert weights.iloc[0] == pytest.approx(FRIENDLY_SAMPLE_WEIGHT, rel=1e-6)


def test_older_match_has_decayed_recency_factor():
    df = _df(
        [
            {"date": "2000-01-01", "tournament": "UEFA Euro qualification"},
            {"date": "2026-01-01", "tournament": "UEFA Euro qualification"},
        ]
    )
    weights = compute_sample_weight(df)
    days_ago = (pd.Timestamp("2026-01-01") - pd.Timestamp("2000-01-01")).days
    expected_recency = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    assert weights.iloc[0] == pytest.approx(expected_recency, rel=1e-6)
    assert weights.iloc[0] < weights.iloc[1]


def test_wc2026_match_gets_boost_on_top_of_recency():
    df = _df(
        [
            {"date": "2026-06-01", "tournament": "FIFA World Cup"},
            {"date": "2026-06-10", "tournament": "FIFA World Cup"},
        ]
    )
    weights = compute_sample_weight(df)
    expected_recency = 0.5 ** (9 / RECENCY_HALF_LIFE_DAYS)
    assert weights.iloc[0] == pytest.approx(expected_recency * WC2026_BOOST, rel=1e-6)


def test_pre_2026_world_cup_match_gets_no_boost():
    """A 2022 World Cup match is NOT a WC2026 match and must not get the boost."""
    df = _df(
        [
            {"date": "2022-12-18", "tournament": "FIFA World Cup"},
            {"date": "2026-01-01", "tournament": "Friendly"},
        ]
    )
    weights = compute_sample_weight(df)
    days_ago = (pd.Timestamp("2026-01-01") - pd.Timestamp("2022-12-18")).days
    expected_recency = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    assert weights.iloc[0] == pytest.approx(expected_recency, rel=1e-6)
