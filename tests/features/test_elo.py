"""Tests for features.elo: Elo rating computation and helpers."""

import pandas as pd
import pytest

from features.elo import (
    DEFAULT_K,
    INITIAL_RATING,
    compute_elo_ratings,
    get_competition_k,
    get_current_ratings,
)

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------


def make_results(**kwargs) -> pd.DataFrame:
    """Return a small DataFrame with the columns expected by elo functions."""
    defaults = dict(
        date=[pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")],
        home_team=["Brazil", "Germany"],
        away_team=["Argentina", "France"],
        home_score=[2, 1],
        away_score=[1, 1],
        tournament=["FIFA World Cup", "Friendly"],
        neutral=[True, False],
        country=["Qatar", "Germany"],
    )
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def make_single_win_result(*, home_wins: bool = True) -> pd.DataFrame:
    """Single match where home team wins (home_wins=True) or loses."""
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "home_team": ["Brazil"],
            "away_team": ["Argentina"],
            "home_score": [2 if home_wins else 0],
            "away_score": [0 if home_wins else 2],
            "tournament": ["Friendly"],
            "neutral": [True],
            "country": ["Qatar"],
        }
    )


def make_single_draw_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "home_team": ["Brazil"],
            "away_team": ["Argentina"],
            "home_score": [1],
            "away_score": [1],
            "tournament": ["Friendly"],
            "neutral": [True],
            "country": ["Qatar"],
        }
    )


# ---------------------------------------------------------------------------
# get_competition_k
# ---------------------------------------------------------------------------


def test_competition_k_exact_match():
    assert get_competition_k("FIFA World Cup") == 60


def test_competition_k_prefix_match():
    """Qualification tournaments use the prefix key lookup."""
    assert get_competition_k("FIFA World Cup qualification (UEFA)") == 40


def test_competition_k_exact_friendly():
    assert get_competition_k("Friendly") == 20


def test_competition_k_default():
    assert get_competition_k("Unknown Tournament") == DEFAULT_K


def test_competition_k_uefa_euro():
    assert get_competition_k("UEFA Euro") == 50


# ---------------------------------------------------------------------------
# compute_elo_ratings — column presence
# ---------------------------------------------------------------------------


def test_elo_columns_added():
    df = make_results()
    out = compute_elo_ratings(df)
    assert "elo_home_pre" in out.columns
    assert "elo_away_pre" in out.columns


def test_elo_pre_rating_is_before_match():
    """elo_home_pre for row 0 must equal INITIAL_RATING — the pre-match rating,
    not the post-match rating.
    """
    df = make_results()
    out = compute_elo_ratings(df)
    assert out.iloc[0]["elo_home_pre"] == pytest.approx(INITIAL_RATING, abs=1e-6)
    assert out.iloc[0]["elo_away_pre"] == pytest.approx(INITIAL_RATING, abs=1e-6)


def test_elo_initial_rating():
    """Both teams start at INITIAL_RATING before their first ever match."""
    df = make_single_win_result()
    out = compute_elo_ratings(df)
    assert out.iloc[0]["elo_home_pre"] == pytest.approx(INITIAL_RATING, abs=1e-6)
    assert out.iloc[0]["elo_away_pre"] == pytest.approx(INITIAL_RATING, abs=1e-6)


# ---------------------------------------------------------------------------
# Rating direction after outcomes
# ---------------------------------------------------------------------------


def test_elo_winner_rating_increases():
    """After a home win the winner's rating should be higher than INITIAL_RATING."""
    df = make_single_win_result(home_wins=True)
    ratings = get_current_ratings(df)
    assert ratings["Brazil"] > INITIAL_RATING


def test_elo_loser_rating_decreases():
    """After a home win the loser's (away team's) rating should fall below INITIAL_RATING."""
    df = make_single_win_result(home_wins=True)
    ratings = get_current_ratings(df)
    assert ratings["Argentina"] < INITIAL_RATING


def test_elo_draw_moves_towards_expected():
    """Equal-rated teams draw: both are at equal strength, expected = 0.5,
    actual = 0.5, so Δ = K * (0.5 - 0.5) = 0.  Ratings remain at INITIAL_RATING.
    """
    df = make_single_draw_result()
    ratings = get_current_ratings(df)
    assert ratings["Brazil"] == pytest.approx(INITIAL_RATING, abs=1e-6)
    assert ratings["Argentina"] == pytest.approx(INITIAL_RATING, abs=1e-6)


# ---------------------------------------------------------------------------
# Neutral vs home advantage
# ---------------------------------------------------------------------------


def test_neutral_no_home_advantage():
    """For neutral=True, the effective rating for home team is not boosted.
    This is verified by checking that a draw between equal teams on a neutral
    ground leaves both ratings unchanged — which is exactly the draw test above.
    If home advantage were incorrectly applied on neutral ground the ratings
    would shift.
    """
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "home_team": ["Brazil"],
            "away_team": ["Argentina"],
            "home_score": [1],
            "away_score": [1],
            "tournament": ["Friendly"],
            "neutral": [True],
            "country": ["Qatar"],
        }
    )
    ratings = get_current_ratings(df)
    # Equal teams drawing on neutral ground: Δ = 0
    assert ratings["Brazil"] == pytest.approx(INITIAL_RATING, abs=1e-6)
    assert ratings["Argentina"] == pytest.approx(INITIAL_RATING, abs=1e-6)


def test_home_advantage_applied():
    """For neutral=False the home team's effective rating gets HOME_ADVANTAGE added.

    Concretely: equal-rated teams drawing on a non-neutral ground.
    The home team's expected score E_h > 0.5 (due to bonus), actual = 0.5,
    so Δ_home = K * (0.5 - E_h) < 0 — home team's rating falls slightly.
    This is different from the neutral=True case where Δ = 0.
    Compare the resulting ratings to prove the code path differs.
    """
    df_neutral = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "home_team": ["Brazil"],
            "away_team": ["Argentina"],
            "home_score": [1],
            "away_score": [1],
            "tournament": ["Friendly"],
            "neutral": [True],
            "country": ["Qatar"],
        }
    )
    df_home = df_neutral.copy()
    df_home["neutral"] = [False]

    ratings_neutral = get_current_ratings(df_neutral)
    ratings_home = get_current_ratings(df_home)

    # Under home advantage a draw is a slight under-performance for the home
    # team (they were favoured), so their rating should end up lower than
    # the neutral case (where it stays at INITIAL_RATING).
    assert ratings_home["Brazil"] < ratings_neutral["Brazil"]


# ---------------------------------------------------------------------------
# Inactivity decay
# ---------------------------------------------------------------------------


def test_inactivity_decay_reduces_gap():
    """A team at 1700 Elo that goes inactive for >1 year should have their
    rating decayed toward INITIAL_RATING (1500) when they next appear.

    We inject the team into two matches:
      1. A series of early matches to push their rating to ~1700.
      2. A single match 2 years later — elo_home_pre should be between 1500-1700.
    """
    # Build a run of wins for Brazil to push its rating up well above 1500.
    # 15 wins against a weaker-rated pool should suffice.
    n_warmup = 15
    warmup_dates = pd.date_range("2015-01-01", periods=n_warmup, freq="14D")
    warmup = pd.DataFrame(
        {
            "date": warmup_dates,
            "home_team": ["Brazil"] * n_warmup,
            "away_team": [f"Minnow{i}" for i in range(n_warmup)],
            "home_score": [3] * n_warmup,
            "away_score": [0] * n_warmup,
            "tournament": ["FIFA World Cup"] * n_warmup,
            "neutral": [True] * n_warmup,
            "country": ["Brazil"] * n_warmup,
        }
    )

    # After warmup, verify Brazil's rating is elevated
    ratings_after_warmup = get_current_ratings(warmup)
    brazil_high = ratings_after_warmup["Brazil"]
    assert brazil_high > INITIAL_RATING + 100, (
        f"Warmup insufficient; Brazil Elo only {brazil_high:.1f}"
    )

    # Now add a single match for Brazil ~2.5 years later
    late_date = warmup_dates[-1] + pd.DateOffset(years=2, months=6)
    late_match = pd.DataFrame(
        {
            "date": [late_date],
            "home_team": ["Brazil"],
            "away_team": ["Argentina"],
            "home_score": [1],
            "away_score": [1],
            "tournament": ["Friendly"],
            "neutral": [True],
            "country": ["Qatar"],
        }
    )

    combined = pd.concat([warmup, late_match], ignore_index=True)
    out = compute_elo_ratings(combined)

    # The elo_home_pre for the late match (last row) reflects post-decay rating
    late_pre = out.iloc[-1]["elo_home_pre"]
    assert late_pre > INITIAL_RATING, "Decay should not push rating below INITIAL_RATING"
    assert late_pre < brazil_high, "Decay should have reduced the gap"
