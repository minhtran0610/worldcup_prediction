"""Tests for models.dixon_coles.DixonColesModel."""

import numpy as np
import pandas as pd
import pytest

from models.dixon_coles import DixonColesModel

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------


def make_synthetic_results(n_matches: int = 200, n_teams: int = 6, seed: int = 42) -> pd.DataFrame:
    """Return a small DataFrame with required columns for fitting DixonColesModel.

    Uses Poisson(1.3) for each team's goals and random pairings spanning 5 years.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    dates = pd.date_range("2018-01-01", periods=n_matches, freq="7D")

    home_teams = []
    away_teams = []
    for _ in range(n_matches):
        pair = rng.choice(n_teams, size=2, replace=False)
        home_teams.append(teams[pair[0]])
        away_teams.append(teams[pair[1]])

    home_scores = rng.poisson(lam=1.3, size=n_matches).astype(int)
    away_scores = rng.poisson(lam=1.3, size=n_matches).astype(int)
    neutral = rng.choice([True, False], size=n_matches)

    return pd.DataFrame(
        {
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_score": home_scores,
            "away_score": away_scores,
            "neutral": neutral,
        }
    )


@pytest.fixture(scope="module")
def fitted_model() -> DixonColesModel:
    df = make_synthetic_results()
    return DixonColesModel().fit(df)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_results()


# ---------------------------------------------------------------------------
# Fit interface
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    df = make_synthetic_results(n_matches=50, seed=0)
    model = DixonColesModel()
    result = model.fit(df)
    assert result is model


# ---------------------------------------------------------------------------
# Predict lambda
# ---------------------------------------------------------------------------


def test_predict_lambda_positive(fitted_model):
    lh, la, rho = fitted_model.predict_lambda("Team0", "Team1", neutral=True)
    assert lh > 0.0
    assert la > 0.0


# ---------------------------------------------------------------------------
# Predict grid
# ---------------------------------------------------------------------------


def test_predict_grid_shape(fitted_model):
    grid, _ = fitted_model.predict("Team0", "Team1", neutral=True)
    assert grid.shape == (8, 8)


def test_predict_grid_sums_to_one(fitted_model):
    grid, _ = fitted_model.predict("Team0", "Team1", neutral=True)
    assert grid.sum() == pytest.approx(1.0, abs=1e-6)


def test_predict_markets_1x2_sum(fitted_model):
    _, markets = fitted_model.predict("Team0", "Team1", neutral=True)
    total = markets["home_win"] + markets["draw"] + markets["away_win"]
    assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Home advantage
# ---------------------------------------------------------------------------


def test_neutral_vs_home_advantage(fitted_model):
    """neutral=False applies the γ multiplier to λ_home; neutral=True does not.

    We verify:
    1. λ_home differs between neutral=True and neutral=False (γ != 1).
    2. λ_away is identical regardless of neutrality (γ only affects λ_home).
    We do NOT assert the direction (γ > 1) because on random synthetic data
    the optimiser may converge to γ < 1.
    """
    lh_neutral, la_neutral, _ = fitted_model.predict_lambda("Team0", "Team1", neutral=True)
    lh_home, la_home, _ = fitted_model.predict_lambda("Team0", "Team1", neutral=False)

    # γ is applied to λ_home when not neutral → the two must differ
    assert lh_home != pytest.approx(lh_neutral, abs=1e-9)

    # λ_away is not affected by neutral flag
    assert la_neutral == pytest.approx(la_home, abs=1e-9)


def test_neutral_vs_home_advantage_direction():
    """On data where home teams genuinely win more (realistic signal),
    the fitted γ should be > 1, making λ_home higher on non-neutral ground.
    """
    rng = np.random.default_rng(99)
    teams = [f"T{i}" for i in range(6)]
    n = 300
    dates = pd.date_range("2018-01-01", periods=n, freq="7D")

    home_teams = []
    away_teams = []
    for _ in range(n):
        pair = rng.choice(len(teams), size=2, replace=False)
        home_teams.append(teams[pair[0]])
        away_teams.append(teams[pair[1]])

    # Home teams score more (Poisson 1.8 vs 1.0) on non-neutral ground
    neutral_flags = rng.choice([True, False], size=n)
    home_goals = np.where(
        neutral_flags,
        rng.poisson(1.3, n),
        rng.poisson(1.8, n),
    ).astype(int)
    away_goals = rng.poisson(1.0, n).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_score": home_goals,
            "away_score": away_goals,
            "neutral": neutral_flags,
        }
    )

    model = DixonColesModel().fit(df)
    lh_neutral, _, _ = model.predict_lambda("T0", "T1", neutral=True)
    lh_home, _, _ = model.predict_lambda("T0", "T1", neutral=False)
    # With a realistic home advantage signal, γ > 1 → λ_home larger when not neutral
    assert lh_home > lh_neutral


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_team_raises(fitted_model):
    with pytest.raises(ValueError, match="Team not in fitted parameters"):
        fitted_model.predict("UnknownTeam", "Team1")


def test_unknown_away_team_raises(fitted_model):
    with pytest.raises(ValueError, match="Team not in fitted parameters"):
        fitted_model.predict("Team0", "NoSuchTeam")


# ---------------------------------------------------------------------------
# Parameter validity
# ---------------------------------------------------------------------------


def test_rho_in_valid_range(fitted_model):
    assert -0.99 < fitted_model._rho < 0.99


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


def test_predict_batch_shape(fitted_model):
    matches = pd.DataFrame(
        {
            "home_team": ["Team0", "Team1", "Team2"],
            "away_team": ["Team1", "Team2", "Team3"],
            "neutral": [True, False, True],
        }
    )
    result = fitted_model.predict_batch(matches)
    assert len(result) == len(matches)


def test_predict_batch_columns(fitted_model):
    matches = pd.DataFrame(
        {
            "home_team": ["Team0", "Team1"],
            "away_team": ["Team1", "Team2"],
            "neutral": [True, True],
        }
    )
    result = fitted_model.predict_batch(matches)
    for col in ("lambda_home", "lambda_away", "rho", "prob_home", "prob_draw", "prob_away", "grid"):
        assert col in result.columns
