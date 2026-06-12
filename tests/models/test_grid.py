"""Tests for models.grid — score grid construction and market derivations."""

import numpy as np
import pytest

from models.grid import MAX_GOALS, build_grid, derive_markets, expected_goals, top_scorelines

# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def test_grid_sums_to_one():
    grid = build_grid(1.5, 1.2, 0.1)
    assert grid.sum() == pytest.approx(1.0, abs=1e-6)


def test_grid_sums_to_one_zero_rho():
    grid = build_grid(1.5, 1.2, 0.0)
    assert grid.sum() == pytest.approx(1.0, abs=1e-6)


def test_grid_shape():
    grid = build_grid(1.5, 1.2, 0.1)
    assert grid.shape == (MAX_GOALS + 1, MAX_GOALS + 1)


def test_grid_all_nonnegative():
    grid = build_grid(1.5, 1.2, 0.1)
    assert np.all(grid >= 0.0)


def test_grid_small_lambda():
    """Very small lambdas should still produce a valid, finite grid."""
    grid = build_grid(0.05, 0.05, 0.0)
    assert grid.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.isfinite(grid))
    assert np.all(grid >= 0.0)


def test_tau_correction_zero_zero():
    """With rho > 0, the (0,0) cell should be reduced by the τ factor
    relative to the pure independent Poisson prediction.

    For (0,0): τ = 1 - λ_h * λ_a * ρ.  When ρ > 0 and λs > 0, τ < 1,
    so P(0,0) with correction < P(0,0) without correction.
    We compare grids built with rho=0.2 vs rho=0.0.
    """
    lambda_h, lambda_a = 1.5, 1.2
    rho_positive = 0.2
    rho_zero = 0.0

    grid_corrected = build_grid(lambda_h, lambda_a, rho_positive)
    grid_independent = build_grid(lambda_h, lambda_a, rho_zero)

    # τ < 1 for (0,0) means the numerator P(0,0) is smaller.
    # After renormalisation both grids sum to 1, but the ratio of the
    # (0,0) cell to the rest shifts downward for the corrected grid.
    assert grid_corrected[0, 0] < grid_independent[0, 0]


# ---------------------------------------------------------------------------
# Market derivations — sums
# ---------------------------------------------------------------------------


def test_markets_1x2_sum():
    grid = build_grid(1.5, 1.2, 0.1)
    m = derive_markets(grid)
    assert m["home_win"] + m["draw"] + m["away_win"] == pytest.approx(1.0, abs=1e-6)


def test_markets_over_under_sum_0_5():
    grid = build_grid(1.5, 1.2, 0.1)
    m = derive_markets(grid)
    assert m["over_0_5"] + m["under_0_5"] == pytest.approx(1.0, abs=1e-6)


def test_markets_over_under_sum_1_5():
    grid = build_grid(1.5, 1.2, 0.1)
    m = derive_markets(grid)
    assert m["over_1_5"] + m["under_1_5"] == pytest.approx(1.0, abs=1e-6)


def test_markets_over_under_sum_2_5():
    grid = build_grid(1.5, 1.2, 0.1)
    m = derive_markets(grid)
    assert m["over_2_5"] + m["under_2_5"] == pytest.approx(1.0, abs=1e-6)


def test_markets_over_under_sum_3_5():
    grid = build_grid(1.5, 1.2, 0.1)
    m = derive_markets(grid)
    assert m["over_3_5"] + m["under_3_5"] == pytest.approx(1.0, abs=1e-6)


def test_markets_btts_sum():
    grid = build_grid(1.5, 1.2, 0.1)
    m = derive_markets(grid)
    assert m["btts_yes"] + m["btts_no"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Expected goals
# ---------------------------------------------------------------------------


def test_expected_goals_positive():
    grid = build_grid(1.5, 1.2, 0.0)
    exp_home, exp_away = expected_goals(grid)
    assert exp_home > 0.0
    assert exp_away > 0.0


def test_expected_goals_ordering():
    """Higher λ_home => higher expected home goals."""
    grid = build_grid(2.0, 1.0, 0.0)
    exp_home, exp_away = expected_goals(grid)
    assert exp_home > exp_away


# ---------------------------------------------------------------------------
# Top scorelines
# ---------------------------------------------------------------------------


def test_top_scorelines_length():
    grid = build_grid(1.5, 1.2, 0.1)
    results = top_scorelines(grid, n=5)
    assert len(results) == 5


def test_top_scorelines_sorted():
    """Returned probabilities must be descending."""
    grid = build_grid(1.5, 1.2, 0.1)
    results = top_scorelines(grid, n=8)
    probs = [p for _, _, p in results]
    assert probs == sorted(probs, reverse=True)


def test_top_scorelines_sum_leq_one():
    grid = build_grid(1.5, 1.2, 0.1)
    results = top_scorelines(grid, n=5)
    total = sum(p for _, _, p in results)
    assert total <= 1.0 + 1e-6
