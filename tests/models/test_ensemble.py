"""Tests for models.ensemble.EnsembleModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.dixon_coles import DixonColesModel
from models.ensemble import EnsembleModel

# ---------------------------------------------------------------------------
# Stub models
# ---------------------------------------------------------------------------


class _ConstantModel:
    """Returns a fixed (8,8) grid for every row."""

    def __init__(self, grid: np.ndarray) -> None:
        self._grid = grid.copy()

    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        out = matches.copy()
        out["grid"] = [self._grid.copy() for _ in range(len(matches))]
        return out


class _RaisingModel:
    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("deliberate failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_grid() -> np.ndarray:
    """Return a valid (8,8) grid with uniform probability 1/64 everywhere."""
    g = np.ones((8, 8), dtype=np.float64) / 64.0
    return g


def _peaked_grid() -> np.ndarray:
    """Return a valid (8,8) grid with all mass at (1,0)."""
    g = np.zeros((8, 8), dtype=np.float64)
    g[1, 0] = 1.0
    return g


def make_match_df(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="7D"),
            "home_team": [f"T{i % 4}" for i in range(n)],
            "away_team": [f"T{(i + 2) % 4}" for i in range(n)],
            "home_score": rng.integers(0, 3, size=n),
            "away_score": rng.integers(0, 3, size=n),
            "tournament": ["Friendly"] * n,
            "neutral": [True] * n,
            "elo_home_pre": [1500.0] * n,
            "elo_away_pre": [1500.0] * n,
            "elo_diff": [0.0] * n,
            "rest_days_home": [14.0] * n,
            "rest_days_away": [14.0] * n,
            "is_knockout": [False] * n,
            "is_host_home": [False] * n,
            "is_host_away": [False] * n,
            "sample_weight": [1.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Construction — valid cases
# ---------------------------------------------------------------------------


def test_equal_weights_default_two_models():
    """With 2 models and no weights arg, each gets weight 0.5."""
    g = _uniform_grid()
    m1 = _ConstantModel(g)
    m2 = _ConstantModel(g)
    ens = EnsembleModel([m1, m2])
    assert len(ens.weights) == 2
    assert ens.weights[0] == pytest.approx(0.5, abs=1e-10)
    assert ens.weights[1] == pytest.approx(0.5, abs=1e-10)


def test_equal_weights_default_single_model():
    """Single-model ensemble gets weight 1.0."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    assert ens.weights[0] == pytest.approx(1.0, abs=1e-10)


def test_equal_weights_default_four_models():
    """With 4 models, each weight should be 0.25."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)] * 4)
    for w in ens.weights:
        assert w == pytest.approx(0.25, abs=1e-10)


# ---------------------------------------------------------------------------
# Construction — error cases
# ---------------------------------------------------------------------------


def test_empty_models_raises():
    """`EnsembleModel([])` should raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        EnsembleModel([])


def test_mismatched_weights_raises():
    """Weights of wrong length should raise ValueError."""
    g = _uniform_grid()
    m = _ConstantModel(g)
    with pytest.raises(ValueError, match=r"len\(weights\)"):
        EnsembleModel([m, m], weights=[0.5, 0.3, 0.2])


def test_weights_not_summing_to_one_raises():
    """Weights that do not sum to 1.0 should raise ValueError."""
    g = _uniform_grid()
    m = _ConstantModel(g)
    with pytest.raises(ValueError, match="sum to 1.0"):
        EnsembleModel([m, m], weights=[0.4, 0.4])


# ---------------------------------------------------------------------------
# Grid averaging
# ---------------------------------------------------------------------------


def test_grid_averaging_equal_weights():
    """With 2 models returning known grids A and B, the result should be 0.5*A + 0.5*B."""
    g_a = _uniform_grid()
    g_b = _peaked_grid()
    expected = 0.5 * g_a + 0.5 * g_b

    ens = EnsembleModel([_ConstantModel(g_a), _ConstantModel(g_b)])
    matches = make_match_df(n=3)
    result = ens.predict_batch(matches)

    for row_grid in result["grid"]:
        np.testing.assert_allclose(row_grid, expected, atol=1e-10)


def test_grid_averaging_unequal_weights():
    """Explicit weights are applied correctly when averaging grids."""
    g_a = _uniform_grid()
    g_b = _peaked_grid()
    w_a, w_b = 0.3, 0.7
    expected = w_a * g_a + w_b * g_b

    ens = EnsembleModel([_ConstantModel(g_a), _ConstantModel(g_b)], weights=[w_a, w_b])
    matches = make_match_df(n=2)
    result = ens.predict_batch(matches)

    for row_grid in result["grid"]:
        np.testing.assert_allclose(row_grid, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Grid shape and sum
# ---------------------------------------------------------------------------


def test_grid_shape_is_8x8():
    """Output grid column contains (8,8) arrays."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    matches = make_match_df(n=4)
    result = ens.predict_batch(matches)
    for row_grid in result["grid"]:
        assert row_grid.shape == (8, 8)


def test_grid_sums_to_one():
    """Each grid in the output must sum to ~1.0."""
    g_a = _uniform_grid()
    g_b = _peaked_grid()
    ens = EnsembleModel([_ConstantModel(g_a), _ConstantModel(g_b)])
    matches = make_match_df(n=5)
    result = ens.predict_batch(matches)
    for row_grid in result["grid"]:
        assert row_grid.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Market coherence
# ---------------------------------------------------------------------------


def test_market_coherence_1x2_sums_to_one():
    """prob_home + prob_draw + prob_away must equal 1.0 for every row."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    matches = make_match_df(n=6)
    result = ens.predict_batch(matches)
    total = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    for val in total:
        assert val == pytest.approx(1.0, abs=1e-6)


def test_market_coherence_two_models():
    """Market coherence holds when averaging two different grids."""
    g_a = _uniform_grid()
    g_b = _peaked_grid()
    ens = EnsembleModel([_ConstantModel(g_a), _ConstantModel(g_b)])
    matches = make_match_df(n=5)
    result = ens.predict_batch(matches)
    total = result["prob_home"] + result["prob_draw"] + result["prob_away"]
    for val in total:
        assert val == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Output columns
# ---------------------------------------------------------------------------


def test_output_columns_present():
    """All required output columns must be present."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    matches = make_match_df(n=3)
    result = ens.predict_batch(matches)
    for col in ("prob_home", "prob_draw", "prob_away", "expected_home", "expected_away", "grid"):
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Input columns preserved
# ---------------------------------------------------------------------------


def test_input_columns_preserved():
    """predict_batch output retains all input columns."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    matches = make_match_df(n=4)
    result = ens.predict_batch(matches)
    for col in matches.columns:
        assert col in result.columns, f"Input column lost: {col}"


# ---------------------------------------------------------------------------
# Single-model ensemble identity
# ---------------------------------------------------------------------------


def test_single_model_ensemble_identity():
    """Ensemble of one model with weight=1.0 gives identical grid to the constituent."""
    g = _uniform_grid()
    model = _ConstantModel(g)
    ens = EnsembleModel([model], weights=[1.0])
    matches = make_match_df(n=3)

    direct = model.predict_batch(matches)
    ensemble_out = ens.predict_batch(matches)

    for direct_grid, ens_grid in zip(direct["grid"], ensemble_out["grid"], strict=False):
        np.testing.assert_allclose(ens_grid, direct_grid, atol=1e-10)


# ---------------------------------------------------------------------------
# RPS-weighted construction
# ---------------------------------------------------------------------------


def test_from_rps_scores_weights_formula():
    """Verify weight_i = (1/rps_i) / sum(1/rps_j) numerically."""
    rps_scores = [0.25, 0.50, 0.10]
    inv = [1.0 / r for r in rps_scores]
    total_inv = sum(inv)
    expected_weights = [v / total_inv for v in inv]

    g = _uniform_grid()
    models = [_ConstantModel(g) for _ in range(3)]
    ens = EnsembleModel.from_rps_scores(models, rps_scores)

    for actual_w, exp_w in zip(ens.weights, expected_weights, strict=False):
        assert actual_w == pytest.approx(exp_w, abs=1e-10)


def test_from_rps_scores_weights_sum_to_one():
    """Weights produced by from_rps_scores must sum to 1.0."""
    rps_scores = [0.3, 0.2, 0.5, 0.4]
    g = _uniform_grid()
    models = [_ConstantModel(g) for _ in range(4)]
    ens = EnsembleModel.from_rps_scores(models, rps_scores)
    assert sum(ens.weights) == pytest.approx(1.0, abs=1e-10)


def test_from_rps_scores_lower_rps_gets_higher_weight():
    """A model with lower RPS (better) should receive a higher weight."""
    rps_scores = [0.10, 0.40]  # first model is better
    g = _uniform_grid()
    models = [_ConstantModel(g), _ConstantModel(g)]
    ens = EnsembleModel.from_rps_scores(models, rps_scores)
    assert ens.weights[0] > ens.weights[1]


def test_from_rps_scores_mismatched_lengths_raises():
    """from_rps_scores raises ValueError when models and rps_scores differ in length."""
    g = _uniform_grid()
    models = [_ConstantModel(g), _ConstantModel(g)]
    with pytest.raises(ValueError):
        EnsembleModel.from_rps_scores(models, [0.2])


def test_from_rps_scores_zero_rps_raises():
    """from_rps_scores raises ValueError when any RPS score is zero."""
    g = _uniform_grid()
    models = [_ConstantModel(g), _ConstantModel(g)]
    with pytest.raises(ValueError, match="strictly positive"):
        EnsembleModel.from_rps_scores(models, [0.3, 0.0])


def test_from_rps_scores_negative_rps_raises():
    """from_rps_scores raises ValueError when any RPS score is negative."""
    g = _uniform_grid()
    models = [_ConstantModel(g), _ConstantModel(g)]
    with pytest.raises(ValueError, match="strictly positive"):
        EnsembleModel.from_rps_scores(models, [0.3, -0.1])


# ---------------------------------------------------------------------------
# Constituent failure propagation
# ---------------------------------------------------------------------------


def test_constituent_failure_propagates():
    """If a constituent raises, the ensemble re-raises with a message naming the model."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g), _RaisingModel()], weights=[0.5, 0.5])
    matches = make_match_df(n=2)
    with pytest.raises(RuntimeError, match="_RaisingModel"):
        ens.predict_batch(matches)


def test_constituent_failure_message_includes_index():
    """RuntimeError message includes the constituent model index."""
    ens = EnsembleModel([_RaisingModel()], weights=[1.0])
    matches = make_match_df(n=1)
    with pytest.raises(RuntimeError, match="#0"):
        ens.predict_batch(matches)


# ---------------------------------------------------------------------------
# Expected goals sanity
# ---------------------------------------------------------------------------


def test_expected_goals_nonnegative():
    """expected_home and expected_away should be >= 0 for any valid grid."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    matches = make_match_df(n=5)
    result = ens.predict_batch(matches)
    assert (result["expected_home"] >= 0).all()
    assert (result["expected_away"] >= 0).all()


def test_expected_goals_uniform_grid():
    """For a uniform 1/64 grid, E[goals] = 3.5 for both home and away."""
    g = _uniform_grid()
    ens = EnsembleModel([_ConstantModel(g)])
    matches = make_match_df(n=2)
    result = ens.predict_batch(matches)
    for exp_h, exp_a in zip(result["expected_home"], result["expected_away"], strict=False):
        assert exp_h == pytest.approx(3.5, abs=1e-6)
        assert exp_a == pytest.approx(3.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Integration test: DixonColesModel ensemble
# ---------------------------------------------------------------------------


def _make_dc_results(n_matches: int = 60, seed: int = 7) -> pd.DataFrame:
    """Synthetic results DataFrame suitable for fitting DixonColesModel."""
    rng = np.random.default_rng(seed)
    teams = ["T0", "T1", "T2", "T3"]
    n = n_matches
    dates = pd.date_range("2021-01-01", periods=n, freq="7D")

    home_teams = []
    away_teams = []
    for _ in range(n):
        pair = rng.choice(len(teams), size=2, replace=False)
        home_teams.append(teams[pair[0]])
        away_teams.append(teams[pair[1]])

    return pd.DataFrame(
        {
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_score": rng.poisson(1.3, n).astype(int),
            "away_score": rng.poisson(1.1, n).astype(int),
            "neutral": rng.choice([True, False], size=n),
        }
    )


def test_dc_ensemble_of_two_identical_models_equals_dc_output():
    """Ensemble of [dc, dc] with equal weights should yield the same grid as dc alone."""
    results = _make_dc_results()
    dc = DixonColesModel().fit(results)
    ens = EnsembleModel([dc, dc])

    matches = pd.DataFrame(
        {
            "home_team": ["T0", "T1"],
            "away_team": ["T2", "T3"],
            "neutral": [True, False],
        }
    )

    dc_out = dc.predict_batch(matches)
    ens_out = ens.predict_batch(matches)

    for dc_grid, ens_grid in zip(dc_out["grid"], ens_out["grid"], strict=False):
        np.testing.assert_allclose(ens_grid, dc_grid, atol=1e-10)
