"""Tests for features/form.py — get_form_sequence and get_form_aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.form import (
    FORM_FEATURE_DIM,
    FORM_N,
    get_form_aggregates,
    get_form_sequence,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_results_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal results DataFrame from a list of dicts.

    Required keys per row: date, home_team, away_team, home_score, away_score,
    elo_home_pre, elo_away_pre.
    """
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _base_row(
    date: str,
    home: str,
    away: str,
    hs: int,
    as_: int,
    elo_h: float = 1500.0,
    elo_a: float = 1500.0,
) -> dict:
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "elo_home_pre": elo_h,
        "elo_away_pre": elo_a,
    }


# ---------------------------------------------------------------------------
# 1. Date-safety tests (NON-NEGOTIABLE)
# ---------------------------------------------------------------------------


def test_form_sequence_excludes_match_on_as_of_date():
    """A match whose date == as_of_date must NOT appear in the returned sequence."""
    as_of = pd.Timestamp("2020-06-15")
    results = _make_results_df(
        [
            _base_row("2020-06-14", "A", "B", 3, 0),  # before → included
            _base_row("2020-06-15", "A", "C", 2, 1),  # ON as_of_date → excluded
            _base_row("2020-06-16", "A", "D", 1, 0),  # after → excluded
        ]
    )
    seq = get_form_sequence(results, "A", as_of, n=FORM_N)
    # Only one match qualifies; gf=3, ga=0, pts=3 for that match
    # The rest of the rows should be zeros (padding)
    non_zero_rows = seq[seq.any(axis=1)]
    assert len(non_zero_rows) == 1
    assert non_zero_rows[0, 0] == pytest.approx(3.0)  # gf
    assert non_zero_rows[0, 1] == pytest.approx(0.0)  # ga


def test_form_aggregates_excludes_match_on_as_of_date():
    """get_form_aggregates must exclude a match whose date == as_of_date."""
    as_of = pd.Timestamp("2020-06-15")
    results = _make_results_df(
        [
            _base_row("2020-06-14", "A", "B", 2, 0),  # before → included
            _base_row("2020-06-15", "A", "C", 5, 5),  # ON as_of_date → excluded
        ]
    )
    aggs = get_form_aggregates(results, "A", as_of, n=5)
    # Only the first match is visible; gf should be 2, not distorted by the 5-5 match
    assert aggs["form_gf"] == pytest.approx(2.0)
    assert aggs["form_ga"] == pytest.approx(0.0)
    assert aggs["form_n_played"] == 1


def test_form_sequence_excludes_future_match():
    """A match after as_of_date is always excluded."""
    as_of = pd.Timestamp("2020-06-10")
    results = _make_results_df(
        [
            _base_row("2020-06-11", "X", "Y", 4, 4),  # future
        ]
    )
    seq = get_form_sequence(results, "X", as_of, n=FORM_N)
    assert (seq == 0.0).all()


# ---------------------------------------------------------------------------
# 2. Padding tests
# ---------------------------------------------------------------------------


def test_form_sequence_zero_pads_short_history():
    """With only 3 matches and n=10, the first 7 rows must be all zeros."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "T", "B", 1, 0),
            _base_row("2020-02-01", "T", "C", 2, 1),
            _base_row("2020-03-01", "D", "T", 0, 3),
        ]
    )
    as_of = pd.Timestamp("2020-06-01")
    seq = get_form_sequence(results, "T", as_of, n=10)
    assert seq.shape == (10, FORM_FEATURE_DIM)
    # First 7 rows are padding zeros
    assert (seq[:7] == 0.0).all()
    # Last 3 rows are non-zero (goals were scored in each match)
    assert seq[7:].any()


def test_form_sequence_exactly_n_matches_no_padding():
    """With exactly n matches, no padding should be added."""
    n = 5
    results = _make_results_df(
        [_base_row(f"2020-0{i + 1}-01", "T", f"O{i}", 1, 0) for i in range(n)]
    )
    as_of = pd.Timestamp("2020-06-01")
    seq = get_form_sequence(results, "T", as_of, n=n)
    assert seq.shape == (n, FORM_FEATURE_DIM)
    # No zero rows (every match has gf=1)
    assert seq[:, 0].sum() == pytest.approx(float(n))


# ---------------------------------------------------------------------------
# 3. Shape tests
# ---------------------------------------------------------------------------


def test_form_sequence_shape_default_n():
    """get_form_sequence always returns (FORM_N, FORM_FEATURE_DIM) regardless of history."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "A", "B", 2, 2),
        ]
    )
    seq = get_form_sequence(results, "A", pd.Timestamp("2021-01-01"))
    assert seq.shape == (FORM_N, FORM_FEATURE_DIM)
    assert seq.dtype == np.float32


def test_form_sequence_shape_custom_n():
    """Shape respects custom n argument."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "A", "B", 1, 0),
            _base_row("2020-02-01", "A", "C", 2, 1),
        ]
    )
    for n in [3, 7, 15]:
        seq = get_form_sequence(results, "A", pd.Timestamp("2021-01-01"), n=n)
        assert seq.shape == (n, FORM_FEATURE_DIM)


# ---------------------------------------------------------------------------
# 4. Perspective correctness
# ---------------------------------------------------------------------------


def test_form_sequence_home_perspective():
    """When the team is home: gf=home_score, ga=away_score, is_home=1.0."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "HomeTeam", "OppTeam", 3, 1, elo_h=1600.0, elo_a=1400.0),
        ]
    )
    seq = get_form_sequence(results, "HomeTeam", pd.Timestamp("2020-06-01"), n=1)
    assert seq.shape == (1, FORM_FEATURE_DIM)
    gf, ga, pts, opp_elo_norm, is_home = seq[0]
    assert gf == pytest.approx(3.0)
    assert ga == pytest.approx(1.0)
    assert is_home == pytest.approx(1.0)


def test_form_sequence_away_perspective():
    """When the team is away: gf=away_score, ga=home_score, is_home=0.0."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "OppTeam", "AwayTeam", 1, 2, elo_h=1400.0, elo_a=1600.0),
        ]
    )
    seq = get_form_sequence(results, "AwayTeam", pd.Timestamp("2020-06-01"), n=1)
    gf, ga, pts, opp_elo_norm, is_home = seq[0]
    assert gf == pytest.approx(2.0)
    assert ga == pytest.approx(1.0)
    assert is_home == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Points encoding
# ---------------------------------------------------------------------------


def test_form_sequence_points_win():
    """A win gives pts=3.0."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "T", "B", 2, 0),
        ]
    )
    seq = get_form_sequence(results, "T", pd.Timestamp("2021-01-01"), n=1)
    assert seq[0, 2] == pytest.approx(3.0)


def test_form_sequence_points_draw():
    """A draw gives pts=1.0."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "T", "B", 1, 1),
        ]
    )
    seq = get_form_sequence(results, "T", pd.Timestamp("2021-01-01"), n=1)
    assert seq[0, 2] == pytest.approx(1.0)


def test_form_sequence_points_loss():
    """A loss gives pts=0.0."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "T", "B", 0, 2),
        ]
    )
    seq = get_form_sequence(results, "T", pd.Timestamp("2021-01-01"), n=1)
    assert seq[0, 2] == pytest.approx(0.0)


def test_form_aggregates_points_encoding():
    """Win=3, draw=1, loss=0 average correctly over three matches."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "T", "B", 2, 0),  # win → 3
            _base_row("2020-02-01", "T", "C", 1, 1),  # draw → 1
            _base_row("2020-03-01", "T", "D", 0, 1),  # loss → 0
        ]
    )
    aggs = get_form_aggregates(results, "T", pd.Timestamp("2021-01-01"), n=5)
    assert aggs["form_pts"] == pytest.approx((3.0 + 1.0 + 0.0) / 3.0)


# ---------------------------------------------------------------------------
# 6. No history edge cases
# ---------------------------------------------------------------------------


def test_form_sequence_no_history_returns_zeros():
    """Team with no matches returns an all-zeros array of correct shape."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "OtherA", "OtherB", 1, 0),
        ]
    )
    seq = get_form_sequence(results, "Ghost", pd.Timestamp("2021-01-01"))
    assert seq.shape == (FORM_N, FORM_FEATURE_DIM)
    assert (seq == 0.0).all()


def test_form_aggregates_no_history_returns_zeros():
    """Team with no matches returns all-zero aggregates and n_played=0."""
    results = _make_results_df(
        [
            _base_row("2020-01-01", "X", "Y", 2, 1),
        ]
    )
    aggs = get_form_aggregates(results, "Ghost", pd.Timestamp("2021-01-01"))
    assert aggs["form_gf"] == pytest.approx(0.0)
    assert aggs["form_ga"] == pytest.approx(0.0)
    assert aggs["form_pts"] == pytest.approx(0.0)
    assert aggs["form_opp_elo"] == pytest.approx(0.0)
    assert aggs["form_n_played"] == 0


def test_form_sequence_empty_results_returns_zeros():
    """Empty DataFrame returns an all-zeros array."""
    empty = pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "elo_home_pre",
            "elo_away_pre",
        ]
    )
    empty["date"] = pd.to_datetime(empty["date"])
    seq = get_form_sequence(empty, "T", pd.Timestamp("2020-01-01"))
    assert seq.shape == (FORM_N, FORM_FEATURE_DIM)
    assert (seq == 0.0).all()


# ---------------------------------------------------------------------------
# 7. opp_elo_norm correctness
# ---------------------------------------------------------------------------


def test_form_sequence_opp_elo_norm_when_home():
    """When team is home, opp_elo_norm = elo_away_pre / 1500.0."""
    elo_away = 1800.0
    results = _make_results_df(
        [
            _base_row("2020-01-01", "HomeTeam", "OppTeam", 1, 0, elo_h=1500.0, elo_a=elo_away),
        ]
    )
    seq = get_form_sequence(results, "HomeTeam", pd.Timestamp("2021-01-01"), n=1)
    assert seq[0, 3] == pytest.approx(elo_away / 1500.0)


def test_form_sequence_opp_elo_norm_when_away():
    """When team is away, opp_elo_norm = elo_home_pre / 1500.0."""
    elo_home = 1200.0
    results = _make_results_df(
        [
            _base_row("2020-01-01", "OppTeam", "AwayTeam", 0, 1, elo_h=elo_home, elo_a=1500.0),
        ]
    )
    seq = get_form_sequence(results, "AwayTeam", pd.Timestamp("2021-01-01"), n=1)
    assert seq[0, 3] == pytest.approx(elo_home / 1500.0)


# ---------------------------------------------------------------------------
# 8. Chronological ordering (oldest-first in returned sequence)
# ---------------------------------------------------------------------------


def test_form_sequence_chronological_order():
    """Returned rows are oldest-first chronologically."""
    results = _make_results_df(
        [
            _base_row("2020-03-01", "T", "C", 3, 0),  # most recent
            _base_row("2020-01-01", "T", "A", 1, 0),  # oldest
            _base_row("2020-02-01", "T", "B", 2, 0),  # middle
        ]
    )
    as_of = pd.Timestamp("2020-06-01")
    seq = get_form_sequence(results, "T", as_of, n=3)
    # gf column (index 0) should be 1, 2, 3 (chronological)
    assert seq[0, 0] == pytest.approx(1.0)
    assert seq[1, 0] == pytest.approx(2.0)
    assert seq[2, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 9. get_form_aggregates n limit
# ---------------------------------------------------------------------------


def test_form_aggregates_respects_n_limit():
    """get_form_aggregates only considers the last n matches."""
    # 6 matches total; n=3 means only the 3 most recent count
    results = _make_results_df(
        [
            _base_row(f"2020-0{i + 1}-01", "T", f"O{i}", 0, 5)
            for i in range(3)  # old losses
        ]
        + [
            _base_row(f"2020-0{i + 4}-01", "T", f"P{i}", 3, 0)
            for i in range(3)  # recent wins
        ]
    )
    as_of = pd.Timestamp("2020-09-01")
    aggs = get_form_aggregates(results, "T", as_of, n=3)
    # Only the 3 wins should count
    assert aggs["form_pts"] == pytest.approx(3.0)
    assert aggs["form_gf"] == pytest.approx(3.0)
    assert aggs["form_n_played"] == 3
