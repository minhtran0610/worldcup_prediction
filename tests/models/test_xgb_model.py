"""Tests for models/xgb_model.py — XGBModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.xgb_model import XGBModel

# ---------------------------------------------------------------------------
# Shared synthetic data helper
# ---------------------------------------------------------------------------


def make_results(n: int = 200) -> pd.DataFrame:
    """Synthetic results DataFrame with all enriched columns needed by XGBModel."""
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    dates = pd.date_range("2018-01-01", periods=n, freq="7D")
    home_teams = [teams[i % 8] for i in range(n)]
    away_teams = [teams[(i + 3) % 8] for i in range(n)]
    return pd.DataFrame(
        {
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_score": rng.integers(0, 4, size=n),
            "away_score": rng.integers(0, 4, size=n),
            "tournament": ["Friendly"] * n,
            "neutral": [True] * n,
            "elo_home_pre": rng.uniform(1400, 1600, size=n),
            "elo_away_pre": rng.uniform(1400, 1600, size=n),
            "elo_diff": rng.uniform(-200, 200, size=n),
            "rest_days_home": rng.uniform(7, 60, size=n),
            "rest_days_away": rng.uniform(7, 60, size=n),
            "is_knockout": [False] * n,
            "is_host_home": [False] * n,
            "is_host_away": [False] * n,
            "sample_weight": [1.0] * n,
        }
    )


@pytest.fixture(scope="module")
def fitted_model() -> XGBModel:
    """Train a single XGBModel instance shared across all tests in this module."""
    results = make_results(200)
    model = XGBModel()
    model.fit(results)
    return model


@pytest.fixture(scope="module")
def test_matches() -> pd.DataFrame:
    """A small batch of matches for predict_batch tests."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=10, freq="14D"),
            "home_team": [f"T{i % 8}" for i in range(10)],
            "away_team": [f"T{(i + 3) % 8}" for i in range(10)],
            "home_score": rng.integers(0, 4, size=10),
            "away_score": rng.integers(0, 4, size=10),
            "tournament": ["Friendly"] * 10,
            "neutral": [True] * 10,
            "elo_home_pre": rng.uniform(1400, 1600, size=10),
            "elo_away_pre": rng.uniform(1400, 1600, size=10),
            "elo_diff": rng.uniform(-200, 200, size=10),
            "rest_days_home": rng.uniform(7, 60, size=10),
            "rest_days_away": rng.uniform(7, 60, size=10),
            "is_knockout": [False] * 10,
            "is_host_home": [False] * 10,
            "is_host_away": [False] * 10,
            "sample_weight": [1.0] * 10,
        }
    )


# ---------------------------------------------------------------------------
# 1. predict_batch output columns
# ---------------------------------------------------------------------------


def test_predict_batch_has_required_columns(fitted_model, test_matches):
    """predict_batch output must contain all required prediction columns."""
    result = fitted_model.predict_batch(test_matches)
    for col in ("prob_home", "prob_draw", "prob_away", "expected_home", "expected_away", "grid"):
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# 2. Market coherence
# ---------------------------------------------------------------------------


def test_predict_batch_market_sums_to_one(fitted_model, test_matches):
    """prob_home + prob_draw + prob_away must sum to ~1.0 for every row."""
    result = fitted_model.predict_batch(test_matches)
    totals = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    for i, total in enumerate(totals):
        assert total == pytest.approx(1.0, abs=1e-6), f"Row {i}: market sum = {total}"


# ---------------------------------------------------------------------------
# 3. Grid shape
# ---------------------------------------------------------------------------


def test_predict_batch_grid_shape_is_8x8(fitted_model, test_matches):
    """Every grid in predict_batch output must be an (8, 8) ndarray."""
    result = fitted_model.predict_batch(test_matches)
    for i, grid in enumerate(result["grid"]):
        assert isinstance(grid, np.ndarray), f"Row {i}: grid is not ndarray"
        assert grid.shape == (8, 8), f"Row {i}: grid.shape = {grid.shape}"


# ---------------------------------------------------------------------------
# 4. Grid sums to 1
# ---------------------------------------------------------------------------


def test_predict_batch_grid_sums_to_one(fitted_model, test_matches):
    """Every grid in predict_batch output must sum to ~1.0."""
    result = fitted_model.predict_batch(test_matches)
    for i, grid in enumerate(result["grid"]):
        assert grid.sum() == pytest.approx(1.0, abs=1e-6), f"Row {i}: grid sum = {grid.sum()}"


# ---------------------------------------------------------------------------
# 5. Non-negative lambdas from predict_lambda
# ---------------------------------------------------------------------------


def test_predict_lambda_positive_lambdas(fitted_model):
    """predict_lambda must return lh > 0 and la > 0."""
    lh, la, rho = fitted_model.predict_lambda("T0", "T3")
    assert lh > 0.0, f"lambda_home={lh} is not positive"
    assert la > 0.0, f"lambda_away={la} is not positive"


# ---------------------------------------------------------------------------
# 6. rho fixed at 0 from predict_lambda
# ---------------------------------------------------------------------------


def test_predict_lambda_rho_is_zero(fitted_model):
    """XGBModel always returns rho=0.0 (independent Poisson)."""
    _, _, rho = fitted_model.predict_lambda("T0", "T3")
    assert rho == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. Interface compatibility — input columns preserved
# ---------------------------------------------------------------------------


def test_predict_batch_preserves_input_columns(fitted_model, test_matches):
    """predict_batch must return all input columns in addition to prediction columns."""
    result = fitted_model.predict_batch(test_matches)
    for col in test_matches.columns:
        assert col in result.columns, f"Input column lost: {col}"


def test_predict_batch_row_count_unchanged(fitted_model, test_matches):
    """predict_batch must return the same number of rows as the input."""
    result = fitted_model.predict_batch(test_matches)
    assert len(result) == len(test_matches)


# ---------------------------------------------------------------------------
# 8. Unknown team fallback — must not raise
# ---------------------------------------------------------------------------


def test_predict_lambda_unknown_team_does_not_raise(fitted_model):
    """predict_lambda with an unseen team name must not raise an exception."""
    try:
        lh, la, rho = fitted_model.predict_lambda("Unknown_Team_XYZ", "T0")
    except Exception as e:
        pytest.fail(f"predict_lambda raised for unknown team: {e}")
    assert lh > 0.0
    assert la > 0.0


def test_predict_batch_unknown_team_does_not_raise(fitted_model):
    """predict_batch with an unseen team must not raise."""
    match = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01")],
            "home_team": ["Unknown_Team_XYZ"],
            "away_team": ["T1"],
            "home_score": [0],
            "away_score": [0],
            "tournament": ["Friendly"],
            "neutral": [True],
            "elo_home_pre": [1500.0],
            "elo_away_pre": [1500.0],
            "elo_diff": [0.0],
            "rest_days_home": [30.0],
            "rest_days_away": [30.0],
            "is_knockout": [False],
            "is_host_home": [False],
            "is_host_away": [False],
            "sample_weight": [1.0],
        }
    )
    try:
        result = fitted_model.predict_batch(match)
    except Exception as e:
        pytest.fail(f"predict_batch raised for unknown team: {e}")
    assert len(result) == 1
    total = result["prob_home"].iloc[0] + result["prob_draw"].iloc[0] + result["prob_away"].iloc[0]
    assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Unfitted model raises RuntimeError
# ---------------------------------------------------------------------------


def test_predict_batch_unfitted_raises():
    """Calling predict_batch before fit() must raise RuntimeError."""
    model = XGBModel()
    dummy = make_results(5)
    with pytest.raises(RuntimeError):
        model.predict_batch(dummy)


# ---------------------------------------------------------------------------
# 10. predict_lambda before fit returns mean fallback (no raise)
# ---------------------------------------------------------------------------


def test_predict_lambda_before_fit_returns_means():
    """predict_lambda before fit() returns the mean fallback, not an error."""
    model = XGBModel()
    lh, la, rho = model.predict_lambda("T0", "T1")
    # Should return _mean_home and _mean_away defaults
    assert lh > 0.0
    assert la > 0.0
    assert rho == pytest.approx(0.0)
