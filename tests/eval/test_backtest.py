"""Tests for eval.backtest: generate_folds and compute_value_bets.

The walk-forward leakage test (test_generate_folds_no_leakage) is
non-negotiable: random-shuffle leakage is the #1 evaluation bug in
hobby football models.
"""

import numpy as np
import pandas as pd
import pytest

from eval.backtest import MIN_EDGE, MIN_TRAIN_MONTHS, compute_value_bets, generate_folds

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------


def make_dated_results(n_years: int = 8) -> pd.DataFrame:
    """Return results spanning n_years with one match per week."""
    dates = pd.date_range("2015-01-01", periods=n_years * 52, freq="7D")
    n = len(dates)
    teams = [f"Team{i}" for i in range(8)]
    rng = np.random.default_rng(42)
    home_teams = [teams[i % len(teams)] for i in range(n)]
    away_teams = [teams[(i + 1) % len(teams)] for i in range(n)]
    home_scores = rng.integers(0, 4, size=n).tolist()
    away_scores = rng.integers(0, 4, size=n).tolist()
    return pd.DataFrame(
        {
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_score": home_scores,
            "away_score": away_scores,
        }
    )


# ---------------------------------------------------------------------------
# generate_folds — leakage (NON-NEGOTIABLE)
# ---------------------------------------------------------------------------


def test_generate_folds_no_leakage():
    """No date that appears in the training set may appear in the validation set.

    This is the primary safeguard against data leakage via random shuffling or
    incorrect fold boundaries. Any overlap is a hard failure.
    """
    results = make_dated_results(n_years=8)
    folds = generate_folds(results)

    assert len(folds) > 0, "Need at least one fold to test"

    for val_start, val_end, train_end in folds:
        # The contract: train_end == val_start (no gap and no overlap)
        assert train_end <= val_start, (
            f"train_end ({train_end}) bleeds into val_start ({val_start})"
        )

        train = results[results["date"] < train_end]
        val = results[(results["date"] >= val_start) & (results["date"] < val_end)]

        train_dates = set(train["date"].dt.date)
        val_dates = set(val["date"].dt.date)
        overlap = train_dates & val_dates
        assert len(overlap) == 0, f"Date overlap found between train and val fold: {overlap}"


# ---------------------------------------------------------------------------
# generate_folds — structural correctness
# ---------------------------------------------------------------------------


def test_generate_folds_chronological():
    """val_start must increase monotonically across folds."""
    results = make_dated_results(n_years=8)
    folds = generate_folds(results)

    val_starts = [val_start for val_start, _, _ in folds]
    for i in range(1, len(val_starts)):
        assert val_starts[i] > val_starts[i - 1], (
            f"val_start is not monotonically increasing at fold {i}"
        )


def test_generate_folds_min_train():
    """The first validation window must start at least MIN_TRAIN_MONTHS after
    the first match date, so there is a meaningful training set.
    """
    results = make_dated_results(n_years=8)
    folds = generate_folds(results)

    assert len(folds) > 0
    first_val_start, _, _ = folds[0]
    first_date = pd.to_datetime(results["date"]).min()
    min_start = first_date + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    assert first_val_start >= min_start, (
        f"first val_start {first_val_start} is before min required {min_start}"
    )


def test_generate_folds_nonempty():
    """8 years of weekly data must produce at least one fold."""
    results = make_dated_results(n_years=8)
    folds = generate_folds(results)
    assert len(folds) >= 1


def test_generate_folds_val_end_within_data():
    """Every val_end must not exceed the last date in the dataset
    (generate_folds stops before that).
    """
    results = make_dated_results(n_years=8)
    last_date = pd.to_datetime(results["date"]).max()
    folds = generate_folds(results)

    for _val_start, val_end, _train_end in folds:
        assert val_end <= last_date + pd.Timedelta(days=1), (
            f"val_end {val_end} exceeds last_date {last_date}"
        )


def test_generate_folds_train_val_boundary():
    """For every fold: train_end == val_start (no gap, no overlap at boundary)."""
    results = make_dated_results(n_years=8)
    folds = generate_folds(results)

    for val_start, _val_end, train_end in folds:
        assert train_end == val_start, (
            f"Expected train_end == val_start, got {train_end} != {val_start}"
        )


# ---------------------------------------------------------------------------
# compute_value_bets — edge filtering
# ---------------------------------------------------------------------------


def _make_predictions_no_odds() -> pd.DataFrame:
    """Predictions without odds columns."""
    return pd.DataFrame(
        {
            "prob_home": [0.6, 0.3, 0.4],
            "prob_draw": [0.2, 0.4, 0.3],
            "prob_away": [0.2, 0.3, 0.3],
            "home_score": [2, 0, 1],
            "away_score": [0, 2, 1],
        }
    )


def _make_predictions_with_odds(
    *,
    prob_home: float = 0.55,
    prob_draw: float = 0.25,
    prob_away: float = 0.20,
    odds_home: float = 1.80,
    odds_draw: float = 3.60,
    odds_away: float = 4.50,
    home_score: int = 2,
    away_score: int = 0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prob_home": [prob_home],
            "prob_draw": [prob_draw],
            "prob_away": [prob_away],
            "odds_home": [odds_home],
            "odds_draw": [odds_draw],
            "odds_away": [odds_away],
            "home_score": [home_score],
            "away_score": [away_score],
        }
    )


def test_compute_value_bets_filters_edge():
    """Rows whose best edge is below MIN_EDGE must be excluded."""
    # Build a row where model == market (edge = 0) — should be excluded
    df = pd.DataFrame(
        {
            "prob_home": [0.5],
            "prob_draw": [0.25],
            "prob_away": [0.25],
            "odds_home": [2.0],
            "odds_draw": [4.0],
            "odds_away": [4.0],
            "home_score": [1],
            "away_score": [1],
        }
    )
    result = compute_value_bets(df, min_edge=MIN_EDGE)
    assert len(result) == 0


def test_compute_value_bets_no_odds_returns_empty_or_nan():
    """Without odds, best_edge is NaN for all rows → NaN >= MIN_EDGE is False
    → the function returns an empty DataFrame.
    """
    df = _make_predictions_no_odds()
    result = compute_value_bets(df, min_edge=MIN_EDGE)
    assert len(result) == 0


def test_compute_value_bets_outcome_won_home_correct():
    """outcome_won=True when the predicted outcome (home) actually occurred."""
    df = _make_predictions_with_odds(
        prob_home=0.70,  # large edge on home
        prob_draw=0.15,
        prob_away=0.15,
        odds_home=1.80,
        odds_draw=4.00,
        odds_away=5.00,
        home_score=2,
        away_score=0,  # home wins
    )
    result = compute_value_bets(df, min_edge=MIN_EDGE)
    assert len(result) == 1
    assert result.iloc[0]["outcome_won"]


def test_compute_value_bets_outcome_won_away_correct():
    """outcome_won=False when the predicted outcome (home) did NOT occur."""
    df = _make_predictions_with_odds(
        prob_home=0.70,
        prob_draw=0.15,
        prob_away=0.15,
        odds_home=1.80,
        odds_draw=4.00,
        odds_away=5.00,
        home_score=0,
        away_score=2,  # away wins; predicted home → lost
    )
    result = compute_value_bets(df, min_edge=MIN_EDGE)
    assert len(result) == 1
    assert not result.iloc[0]["outcome_won"]


def test_compute_value_bets_draw_outcome_won():
    """outcome_won=True when predicted draw and draw actually occurred."""
    # Construct a row where draw has the highest edge
    df = pd.DataFrame(
        {
            "prob_home": [0.20],
            "prob_draw": [0.65],  # model strongly favours draw
            "prob_away": [0.15],
            "odds_home": [3.50],
            "odds_draw": [2.50],  # market under-prices draw
            "odds_away": [5.00],
            "home_score": [1],
            "away_score": [1],  # draw occurred
        }
    )
    result = compute_value_bets(df, min_edge=MIN_EDGE)
    assert len(result) == 1
    assert result.iloc[0]["best_edge_outcome"] == "draw"
    assert result.iloc[0]["outcome_won"]


def test_compute_value_bets_with_odds_kelly_stake():
    """Verify kelly_stake = KELLY_FRACTION * edge / decimal_odds.

    KELLY_FRACTION = 0.25.
    """
    from eval.backtest import KELLY_FRACTION

    # Carefully set up: market prob for home = 1/1.80 ≈ 0.5556 (after normalisation).
    # We need edge = prob_home - margin_home > MIN_EDGE.
    # Use very generous model prob and market odds that give a clear edge.
    # Simple 2-way-ish market: home=1.80, draw=3.60, away=4.50
    # implied: 1/1.80 + 1/3.60 + 1/4.50 = 0.5556 + 0.2778 + 0.2222 = 1.0556
    # margin_home = (1/1.80) / 1.0556 = 0.5556 / 1.0556 ≈ 0.5264
    # edge_home = 0.70 - 0.5264 ≈ 0.1736 > MIN_EDGE (0.02) ✓
    # best_decimal_odds = 1.80
    # kelly_stake = 0.25 * 0.1736 / 1.80 ≈ 0.02411

    prob_home = 0.70
    odds_home = 1.80
    odds_draw = 3.60
    odds_away = 4.50

    implied_sum = 1 / odds_home + 1 / odds_draw + 1 / odds_away
    margin_home = (1 / odds_home) / implied_sum
    expected_edge = prob_home - margin_home
    expected_kelly = KELLY_FRACTION * expected_edge / odds_home

    df = pd.DataFrame(
        {
            "prob_home": [prob_home],
            "prob_draw": [0.20],
            "prob_away": [0.10],
            "odds_home": [odds_home],
            "odds_draw": [odds_draw],
            "odds_away": [odds_away],
            "home_score": [2],
            "away_score": [0],
        }
    )
    result = compute_value_bets(df, min_edge=MIN_EDGE)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["best_edge_outcome"] == "home"
    assert row["best_edge"] == pytest.approx(expected_edge, abs=1e-6)
    assert row["kelly_stake"] == pytest.approx(expected_kelly, abs=1e-6)


# ---------------------------------------------------------------------------
# select_value_bet — favorite-longshot-aware gating
# ---------------------------------------------------------------------------


def test_select_value_bet_rejects_longshot_below_probability_floor():
    """Model 9% vs market 6% clears the flat absolute-edge threshold but must
    be rejected by the probability floor — this is the DR Congo/England case."""
    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.09,
        prob_draw=0.20,
        prob_away=0.71,
        market_home=0.06,
        market_draw=0.19,
        market_away=0.75,
    )
    assert result["best_outcome"] == "home"
    assert result["is_value"] is False


def test_select_value_bet_rejects_edge_below_relative_threshold():
    """A 2pp edge on a mid-probability outcome (42% vs 39%, ~7.7% relative)
    clears the absolute and floor gates but must fail the relative-edge gate."""
    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.42,
        prob_draw=0.30,
        prob_away=0.28,
        market_home=0.39,
        market_draw=0.32,
        market_away=0.29,
    )
    assert result["best_outcome"] == "home"
    assert result["best_edge"] == pytest.approx(0.03, abs=1e-9)
    assert result["is_value"] is False


def test_select_value_bet_accepts_bet_clearing_all_three_gates():
    """A well-supported edge on a mid-range favorite must still be flagged."""
    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.55,
        prob_draw=0.25,
        prob_away=0.20,
        market_home=0.40,
        market_draw=0.30,
        market_away=0.30,
    )
    assert result["best_outcome"] == "home"
    assert result["is_value"] is True


def test_select_value_bet_handles_nan_market_probs():
    """NaN market probabilities (no odds available) must never be flagged
    as value, and must not raise."""
    import math

    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.5,
        prob_draw=0.3,
        prob_away=0.2,
        market_home=math.nan,
        market_draw=math.nan,
        market_away=math.nan,
    )
    assert result["is_value"] is False
