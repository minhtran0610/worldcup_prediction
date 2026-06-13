"""Evaluation metrics for the World Cup 2026 prediction model.

All metrics operate on the score grid P(i, j) over a 0-7 x 0-7 matrix or on
derived market probabilities. Lower values are better for all metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# RPS normalisation denominator: (num_outcomes - 1) for a 3-outcome 1X2 market
_RPS_DENOM: float = 2.0

# Floor applied before log to prevent -inf in NLL calculations
_LOG_CLIP_MIN: float = 1e-10

# Expected number of rows in the score grid (0 through 7 inclusive)
_GRID_SIZE: int = 8

# Required columns for aggregate_rps
_RPS_REQUIRED_COLS: frozenset[str] = frozenset(
    {"prob_home", "prob_draw", "prob_away", "outcome_idx"}
)

# Required columns for aggregate_nll
_NLL_REQUIRED_COLS: frozenset[str] = frozenset({"home_goals", "away_goals"})


def remove_margin(odds: dict[str, float]) -> dict[str, float]:
    """Normalise raw decimal odds to true probabilities by removing bookmaker margin.

    Formula: p_i = (1/odds_i) / sum(1/odds_j for all j)

    Args:
        odds: e.g. {"home": 2.10, "draw": 3.40, "away": 3.60}
    Returns:
        same keys -> margin-removed probability (values sum to 1.0)
    """
    implied = {k: 1.0 / v for k, v in odds.items()}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()}


def rps_1x2(probs: np.ndarray, outcome_idx: int) -> float:
    """Ranked Probability Score for a 3-outcome 1X2 market. Lower is better.

    probs: shape (3,) array [P(home_win), P(draw), P(away_win)], must sum to ~1.0
    outcome_idx: 0=home win, 1=draw, 2=away win

    Formula: RPS = (1/2) * sum_{r=0}^{1} (CDF_pred[r] - CDF_actual[r])^2
    where CDF is the cumulative distribution over the ordered outcomes H, D, A.
    """
    probs = np.asarray(probs, dtype=float)

    cum_pred_0 = probs[0]
    cum_pred_1 = probs[0] + probs[1]

    # CDF of the one-hot actual distribution at the two cutpoints
    cum_actual_0 = 1.0 if outcome_idx == 0 else 0.0
    cum_actual_1 = 1.0 if outcome_idx <= 1 else 0.0

    rps = ((cum_pred_0 - cum_actual_0) ** 2 + (cum_pred_1 - cum_actual_1) ** 2) / _RPS_DENOM
    return float(rps)


def nll_scoreline(grid: np.ndarray, home_goals: int, away_goals: int) -> float:
    """Negative log-likelihood of the actual scoreline under the predicted grid.

    Scores beyond the grid boundary (> 7) are clipped to the last index, which
    already holds the renormalized mass for "7+ goals" after grid truncation.
    Clips grid values to a minimum of 1e-10 before taking log to avoid -inf.
    Returns a positive float (lower = better model).
    """
    grid = np.asarray(grid, dtype=float)
    max_idx = grid.shape[0] - 1
    h = min(home_goals, max_idx)
    a = min(away_goals, max_idx)
    p = float(grid[h, a])
    p = max(p, _LOG_CLIP_MIN)
    return -np.log(p)


def brier_score(prob: float, outcome: bool) -> float:
    """Brier score for a single binary market prediction. Lower is better.

    BS = (prob - float(outcome))^2
    """
    return (prob - float(outcome)) ** 2


def aggregate_rps(predictions: pd.DataFrame) -> float:
    """Mean RPS over a collection of match predictions.

    predictions must have columns:
        prob_home (float), prob_draw (float), prob_away (float),
        outcome_idx (int: 0=home win, 1=draw, 2=away win)
    """
    missing = _RPS_REQUIRED_COLS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions DataFrame missing columns: {missing}")

    scores = []
    for _, row in predictions.iterrows():
        probs = np.array([row["prob_home"], row["prob_draw"], row["prob_away"]], dtype=float)
        scores.append(rps_1x2(probs, int(row["outcome_idx"])))

    return float(np.mean(scores))


def aggregate_nll(predictions: pd.DataFrame, grids: list[np.ndarray]) -> float:
    """Mean NLL over a collection of match predictions.

    predictions must have columns: home_goals (int), away_goals (int)
    grids: list of (8, 8) arrays aligned with predictions rows
    """
    missing = _NLL_REQUIRED_COLS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions DataFrame missing columns: {missing}")

    if len(predictions) != len(grids):
        raise ValueError(
            f"predictions has {len(predictions)} rows but grids has {len(grids)} entries"
        )

    scores = [
        nll_scoreline(grid, int(row["home_goals"]), int(row["away_goals"]))
        for grid, (_, row) in zip(grids, predictions.iterrows(), strict=False)
    ]
    return float(np.mean(scores))
