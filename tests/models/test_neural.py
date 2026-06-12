"""Tests for models/neural.py — ScoreGridNet and NeuralModel."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from features.form import FORM_FEATURE_DIM
from models.neural import FORM_N, NeuralModel, ScoreGridNet

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------


def make_results(n: int = 200) -> pd.DataFrame:
    """Synthetic results DataFrame with all enriched columns needed by NeuralModel."""
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


def _small_model() -> NeuralModel:
    """Construct a NeuralModel with minimal training budget for fast tests."""
    return NeuralModel(n_epochs=3, patience=999, device="cpu")


@pytest.fixture(scope="module")
def fitted_neural_model() -> NeuralModel:
    """Train a single NeuralModel instance shared across tests in this module."""
    results = make_results(200)
    model = _small_model()
    model.fit(results)
    return model


@pytest.fixture(scope="module")
def test_matches() -> pd.DataFrame:
    """Small batch of matches for predict_batch tests."""
    rng = np.random.default_rng(42)
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
# ScoreGridNet forward-pass tests (architecture-level, no training)
# ---------------------------------------------------------------------------


def _make_random_batch(batch_size: int, n_teams: int = 9):
    """Build random tensors for a ScoreGridNet forward pass."""
    torch.manual_seed(0)
    home_idx = torch.randint(1, n_teams, (batch_size,))
    away_idx = torch.randint(1, n_teams, (batch_size,))
    home_seq = torch.randn(batch_size, FORM_N, FORM_FEATURE_DIM)
    away_seq = torch.randn(batch_size, FORM_N, FORM_FEATURE_DIM)
    context = torch.randn(batch_size, 7)
    return home_idx, away_idx, home_seq, away_seq, context


def test_scoregridnet_forward_output_shapes():
    """With batch_size=4, each output must have shape (4,)."""
    net = ScoreGridNet(n_teams=9)
    net.eval()
    batch_size = 4
    with torch.no_grad():
        lh, la, rho = net(*_make_random_batch(batch_size))
    assert lh.shape == (batch_size,), f"lambda_home shape {lh.shape}"
    assert la.shape == (batch_size,), f"lambda_away shape {la.shape}"
    assert rho.shape == (batch_size,), f"rho shape {rho.shape}"


def test_scoregridnet_forward_positive_lambdas():
    """softplus output: all lambda_home and lambda_away values must be > 0."""
    net = ScoreGridNet(n_teams=9)
    net.eval()
    with torch.no_grad():
        lh, la, _ = net(*_make_random_batch(batch_size=16))
    assert (lh > 0).all().item(), "lambda_home has non-positive values"
    assert (la > 0).all().item(), "lambda_away has non-positive values"


def test_scoregridnet_forward_rho_bounded():
    """rho = tanh * 0.2, so all values must be in (-0.2, 0.2)."""
    net = ScoreGridNet(n_teams=9)
    net.eval()
    with torch.no_grad():
        _, _, rho = net(*_make_random_batch(batch_size=32))
    rho_np = rho.numpy()
    assert (rho_np > -0.2).all(), f"rho min={rho_np.min():.6f} violates lower bound"
    assert (rho_np < 0.2).all(), f"rho max={rho_np.max():.6f} violates upper bound"


def test_scoregridnet_forward_batch_size_one():
    """Forward pass with batch_size=1 must still produce shape (1,) tensors."""
    net = ScoreGridNet(n_teams=9)
    net.eval()
    with torch.no_grad():
        lh, la, rho = net(*_make_random_batch(batch_size=1))
    assert lh.shape == (1,)
    assert la.shape == (1,)
    assert rho.shape == (1,)


# ---------------------------------------------------------------------------
# NeuralModel.predict_batch output column tests
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {
    "prob_home",
    "prob_draw",
    "prob_away",
    "expected_home",
    "expected_away",
    "grid",
}


def test_neural_predict_batch_has_required_columns(fitted_neural_model, test_matches):
    """predict_batch output must include all required market columns."""
    result = fitted_neural_model.predict_batch(test_matches)
    for col in _REQUIRED_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_neural_predict_batch_also_has_lambda_rho(fitted_neural_model, test_matches):
    """NeuralModel also exposes lambda_home, lambda_away, rho in its output."""
    result = fitted_neural_model.predict_batch(test_matches)
    for col in ("lambda_home", "lambda_away", "rho"):
        assert col in result.columns, f"Missing lambda/rho column: {col}"


# ---------------------------------------------------------------------------
# Market coherence
# ---------------------------------------------------------------------------


def test_neural_predict_batch_market_sums_to_one(fitted_neural_model, test_matches):
    """prob_home + prob_draw + prob_away must sum to ~1.0 for every row."""
    result = fitted_neural_model.predict_batch(test_matches)
    totals = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    for i, total in enumerate(totals):
        assert total == pytest.approx(1.0, abs=1e-6), f"Row {i}: market sum = {total}"


# ---------------------------------------------------------------------------
# Grid shape and sum
# ---------------------------------------------------------------------------


def test_neural_predict_batch_grid_shape_is_8x8(fitted_neural_model, test_matches):
    """Every grid must be an (8, 8) ndarray."""
    result = fitted_neural_model.predict_batch(test_matches)
    for i, grid in enumerate(result["grid"]):
        assert isinstance(grid, np.ndarray), f"Row {i}: grid not ndarray"
        assert grid.shape == (8, 8), f"Row {i}: grid.shape = {grid.shape}"


def test_neural_predict_batch_grid_sums_to_one(fitted_neural_model, test_matches):
    """Every grid must sum to ~1.0."""
    result = fitted_neural_model.predict_batch(test_matches)
    for i, grid in enumerate(result["grid"]):
        assert grid.sum() == pytest.approx(1.0, abs=1e-6), f"Row {i}: grid sum = {grid.sum()}"


# ---------------------------------------------------------------------------
# Unknown team (UNK fallback)
# ---------------------------------------------------------------------------


def test_neural_predict_batch_unknown_team_does_not_raise(fitted_neural_model):
    """predict_batch with a team not in the training vocab must not raise."""
    match = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-06-01")],
            "home_team": ["Completely_Unknown_FC"],
            "away_team": ["T0"],
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
        result = fitted_neural_model.predict_batch(match)
    except Exception as e:
        pytest.fail(f"predict_batch raised for unknown team: {e}")
    assert len(result) == 1
    total = result["prob_home"].iloc[0] + result["prob_draw"].iloc[0] + result["prob_away"].iloc[0]
    assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Unfitted model raises RuntimeError
# ---------------------------------------------------------------------------


def test_neural_predict_batch_unfitted_raises():
    """predict_batch before fit() must raise RuntimeError."""
    model = _small_model()
    dummy = make_results(5)
    with pytest.raises(RuntimeError):
        model.predict_batch(dummy)


# ---------------------------------------------------------------------------
# Input columns preserved
# ---------------------------------------------------------------------------


def test_neural_predict_batch_preserves_input_columns(fitted_neural_model, test_matches):
    """predict_batch must return all input columns plus predictions."""
    result = fitted_neural_model.predict_batch(test_matches)
    for col in test_matches.columns:
        assert col in result.columns, f"Input column lost: {col}"


def test_neural_predict_batch_row_count_unchanged(fitted_neural_model, test_matches):
    """predict_batch must return the same number of rows as input."""
    result = fitted_neural_model.predict_batch(test_matches)
    assert len(result) == len(test_matches)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_neural_save_load_round_trip(fitted_neural_model, test_matches):
    """save then load must produce identical predictions (within 1e-5)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.pt"
        fitted_neural_model.save(str(ckpt_path))

        loaded = NeuralModel.load(str(ckpt_path))
        # Inject training results so form look-ups work after load
        loaded._train_results = fitted_neural_model._train_results

        original_preds = fitted_neural_model.predict_batch(test_matches)
        loaded_preds = loaded.predict_batch(test_matches)

        for col in ("prob_home", "prob_draw", "prob_away"):
            orig_vals = original_preds[col].to_numpy()
            load_vals = loaded_preds[col].to_numpy()
            np.testing.assert_allclose(
                orig_vals, load_vals, atol=1e-5, err_msg=f"Mismatch in {col}"
            )


def test_neural_save_load_produces_valid_markets(fitted_neural_model, test_matches):
    """Loaded model's predictions must still have market sums ≈ 1.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.pt"
        fitted_neural_model.save(str(ckpt_path))

        loaded = NeuralModel.load(str(ckpt_path))
        loaded._train_results = fitted_neural_model._train_results

        result = loaded.predict_batch(test_matches)
        totals = result["prob_home"] + result["prob_draw"] + result["prob_away"]
        for i, total in enumerate(totals):
            assert total == pytest.approx(1.0, abs=1e-6), f"Row {i}: sum={total}"


# ---------------------------------------------------------------------------
# Optional val_results argument to fit()
# ---------------------------------------------------------------------------


def test_neural_fit_with_explicit_val_results():
    """fit() must succeed when val_results is provided explicitly."""
    results = make_results(100)
    train = results.iloc[:80].reset_index(drop=True)
    val = results.iloc[80:].reset_index(drop=True)
    model = _small_model()
    model.fit(train, val_results=val)
    batch = val.head(5).copy()
    result = model.predict_batch(batch)
    assert len(result) == 5
    totals = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    for i, total in enumerate(totals):
        assert total == pytest.approx(1.0, abs=1e-6), f"Row {i}: sum={total}"
