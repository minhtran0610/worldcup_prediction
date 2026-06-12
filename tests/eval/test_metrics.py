"""Tests for eval.metrics: remove_margin, rps_1x2, nll_scoreline, brier_score,
aggregate_rps, aggregate_nll.
"""

import numpy as np
import pandas as pd
import pytest

from eval.metrics import (
    aggregate_nll,
    aggregate_rps,
    brier_score,
    nll_scoreline,
    remove_margin,
    rps_1x2,
)
from models.grid import build_grid

# ---------------------------------------------------------------------------
# remove_margin
# ---------------------------------------------------------------------------


def test_remove_margin_sums_to_one():
    result = remove_margin({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


def test_remove_margin_fair_odds():
    """50/50 market: each side at 2.0 → true prob = 0.5 each."""
    result = remove_margin({"h": 2.0, "a": 2.0})
    assert result["h"] == pytest.approx(0.5, abs=1e-6)
    assert result["a"] == pytest.approx(0.5, abs=1e-6)


def test_remove_margin_keys_preserved():
    odds = {"home": 2.1, "draw": 3.4, "away": 3.6}
    result = remove_margin(odds)
    assert set(result.keys()) == set(odds.keys())


def test_remove_margin_3way_sums_to_one():
    """Standard 3-way overround market should also normalise to 1."""
    result = remove_margin({"home": 2.1, "draw": 3.4, "away": 3.6})
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# rps_1x2
# ---------------------------------------------------------------------------


def test_rps_perfect_home_prediction():
    """Predict home with certainty; home wins → RPS = 0."""
    score = rps_1x2(np.array([1.0, 0.0, 0.0]), outcome_idx=0)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_rps_worst_home_prediction():
    """Predict away with certainty; home wins → maximum RPS.

    CDF_pred  = [0, 0] (all mass on away)
    CDF_actual = [1, 1] (home win)
    RPS = ((0-1)^2 + (0-1)^2) / 2 = 1.0
    """
    score = rps_1x2(np.array([0.0, 0.0, 1.0]), outcome_idx=0)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_rps_uniform_prediction_home_win():
    """Uniform [1/3, 1/3, 1/3]; home wins (outcome_idx=0).

    CDF_pred  = [1/3, 2/3]
    CDF_actual = [1, 1]
    RPS = ((1/3-1)^2 + (2/3-1)^2) / 2
        = ((−2/3)^2 + (−1/3)^2) / 2
        = (4/9 + 1/9) / 2
        = 5/18
    """
    expected = 5.0 / 18.0
    score = rps_1x2(np.array([1 / 3, 1 / 3, 1 / 3]), outcome_idx=0)
    assert score == pytest.approx(expected, abs=1e-6)


def test_rps_uniform_prediction_draw():
    """Uniform [1/3, 1/3, 1/3]; draw (outcome_idx=1).

    CDF_pred  = [1/3, 2/3]
    CDF_actual = [0, 1]
    RPS = ((1/3-0)^2 + (2/3-1)^2) / 2
        = (1/9 + 1/9) / 2
        = 1/9
    """
    expected = 1.0 / 9.0
    score = rps_1x2(np.array([1 / 3, 1 / 3, 1 / 3]), outcome_idx=1)
    assert score == pytest.approx(expected, abs=1e-6)


def test_rps_symmetry():
    """For uniform probs, home win and away win should give the same RPS (5/18)
    because the distribution is symmetric around the draw.
    """
    probs = np.array([1 / 3, 1 / 3, 1 / 3])
    assert rps_1x2(probs, outcome_idx=0) == pytest.approx(rps_1x2(probs, outcome_idx=2), abs=1e-6)


def test_rps_perfect_draw_prediction():
    """Predict draw with certainty; draw occurs → RPS = 0."""
    score = rps_1x2(np.array([0.0, 1.0, 0.0]), outcome_idx=1)
    assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# nll_scoreline
# ---------------------------------------------------------------------------


def test_nll_at_correct_cell():
    """NLL should equal -log(grid[i, j]) for the actual cell."""
    grid = build_grid(1.5, 1.2, 0.1)
    home_goals, away_goals = 1, 0
    expected = -np.log(grid[home_goals, away_goals])
    result = nll_scoreline(grid, home_goals, away_goals)
    assert result == pytest.approx(expected, abs=1e-6)


def test_nll_clipping():
    """A grid cell at exactly 0.0 should return a finite NLL (not inf) due to clipping."""
    grid = np.zeros((8, 8), dtype=float)
    grid[0, 0] = 1.0  # only cell with mass; all others are 0.0
    result = nll_scoreline(grid, 1, 0)  # target cell is 0.0
    assert np.isfinite(result)
    # Clipped at 1e-10: NLL = -log(1e-10) ≈ 23.025
    assert result == pytest.approx(-np.log(1e-10), abs=1e-4)


def test_nll_positive():
    """NLL must always be positive (it is -log(p) and p <= 1, so -log(p) >= 0)."""
    grid = build_grid(1.4, 1.1, 0.05)
    result = nll_scoreline(grid, 2, 1)
    assert result > 0.0


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


def test_brier_perfect():
    assert brier_score(1.0, True) == pytest.approx(0.0, abs=1e-6)


def test_brier_worst():
    assert brier_score(0.0, True) == pytest.approx(1.0, abs=1e-6)


def test_brier_half_probability_correct():
    """P=0.5, outcome=True → BS = (0.5 - 1)^2 = 0.25."""
    assert brier_score(0.5, True) == pytest.approx(0.25, abs=1e-6)


def test_brier_half_probability_incorrect():
    """P=0.5, outcome=False → BS = (0.5 - 0)^2 = 0.25."""
    assert brier_score(0.5, False) == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# aggregate_rps
# ---------------------------------------------------------------------------


def test_aggregate_rps_mean():
    """aggregate_rps should equal the mean of individual rps_1x2 calls."""
    rows = [
        {"prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2, "outcome_idx": 0},
        {"prob_home": 0.2, "prob_draw": 0.4, "prob_away": 0.4, "outcome_idx": 2},
        {"prob_home": 0.3, "prob_draw": 0.4, "prob_away": 0.3, "outcome_idx": 1},
    ]
    df = pd.DataFrame(rows)

    individual = [
        rps_1x2(np.array([r["prob_home"], r["prob_draw"], r["prob_away"]]), int(r["outcome_idx"]))
        for r in rows
    ]
    expected_mean = float(np.mean(individual))

    result = aggregate_rps(df)
    assert result == pytest.approx(expected_mean, abs=1e-6)


def test_aggregate_rps_missing_column_raises():
    """Missing prob_home should raise ValueError."""
    df = pd.DataFrame(
        {
            "prob_draw": [0.3],
            "prob_away": [0.3],
            "outcome_idx": [0],
        }
    )
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_rps(df)


def test_aggregate_rps_single_perfect_home():
    """Single-row df with perfect home prediction and home win → RPS = 0."""
    df = pd.DataFrame([{"prob_home": 1.0, "prob_draw": 0.0, "prob_away": 0.0, "outcome_idx": 0}])
    assert aggregate_rps(df) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# aggregate_nll
# ---------------------------------------------------------------------------


def test_aggregate_nll_mean():
    """aggregate_nll should equal mean of individual nll_scoreline calls."""
    g1 = build_grid(1.5, 1.2, 0.0)
    g2 = build_grid(1.0, 0.8, 0.05)
    grids = [g1, g2]

    df = pd.DataFrame(
        {
            "home_goals": [1, 0],
            "away_goals": [0, 1],
        }
    )

    expected = float(np.mean([nll_scoreline(g1, 1, 0), nll_scoreline(g2, 0, 1)]))
    result = aggregate_nll(df, grids)
    assert result == pytest.approx(expected, abs=1e-6)


def test_aggregate_nll_missing_column_raises():
    df = pd.DataFrame({"home_goals": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        aggregate_nll(df, [build_grid(1.0, 1.0, 0.0)])


def test_aggregate_nll_length_mismatch_raises():
    df = pd.DataFrame({"home_goals": [1, 0], "away_goals": [0, 1]})
    with pytest.raises(ValueError):
        aggregate_nll(df, [build_grid(1.0, 1.0, 0.0)])  # only 1 grid, 2 rows
