import math

import numpy as np

MAX_GOALS: int = 7

# Precompute factorial denominators for Poisson PMF: k! for k in 0..MAX_GOALS
_FACTORIALS: np.ndarray = np.array(
    [float(math.factorial(k)) for k in range(MAX_GOALS + 1)],
    dtype=np.float64,
)


def build_grid(lambda_h: float, lambda_a: float, rho: float) -> np.ndarray:
    """Return (MAX_GOALS+1, MAX_GOALS+1) probability matrix summing to 1.0."""
    goals = np.arange(MAX_GOALS + 1, dtype=np.float64)

    # Poisson PMF vectors: exp(-λ) * λ^k / k!
    pmf_h = np.exp(-lambda_h) * (lambda_h**goals) / _FACTORIALS
    pmf_a = np.exp(-lambda_a) * (lambda_a**goals) / _FACTORIALS

    grid = np.outer(pmf_h, pmf_a)

    # Dixon-Coles τ correction for cells where Poisson independence is wrong
    grid[0, 0] *= 1.0 - lambda_h * lambda_a * rho
    grid[1, 0] *= 1.0 + lambda_a * rho
    grid[0, 1] *= 1.0 + lambda_h * rho
    grid[1, 1] *= 1.0 - rho

    # Renormalize: corrected cells no longer guaranteed to sum to 1,
    # and truncation at MAX_GOALS also removes tail probability
    grid /= grid.sum()

    return grid


def derive_markets(grid: np.ndarray) -> dict[str, float]:
    """Derive all market probabilities as deterministic sums over the grid.

    Returns dict with keys:
        home_win, draw, away_win          — must sum to 1.0
        over_0_5, under_0_5              — must sum to 1.0
        over_1_5, under_1_5
        over_2_5, under_2_5
        over_3_5, under_3_5
        btts_yes, btts_no                — must sum to 1.0
    """
    size = grid.shape[0]
    i_idx, j_idx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    total_goals = i_idx + j_idx

    home_win = float(grid[i_idx > j_idx].sum())
    draw = float(grid[i_idx == j_idx].sum())
    away_win = float(grid[i_idx < j_idx].sum())

    over_0_5 = float(grid[total_goals >= 1].sum())
    over_1_5 = float(grid[total_goals >= 2].sum())
    over_2_5 = float(grid[total_goals >= 3].sum())
    over_3_5 = float(grid[total_goals >= 4].sum())

    btts_yes = float(grid[(i_idx >= 1) & (j_idx >= 1)].sum())

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_0_5": over_0_5,
        "under_0_5": 1.0 - over_0_5,
        "over_1_5": over_1_5,
        "under_1_5": 1.0 - over_1_5,
        "over_2_5": over_2_5,
        "under_2_5": 1.0 - over_2_5,
        "over_3_5": over_3_5,
        "under_3_5": 1.0 - over_3_5,
        "btts_yes": btts_yes,
        "btts_no": 1.0 - btts_yes,
    }


def top_scorelines(grid: np.ndarray, n: int = 5) -> list[tuple[int, int, float]]:
    """Return top-n (home_goals, away_goals, prob) sorted by probability descending."""
    flat_indices = np.argsort(grid, axis=None)[::-1][:n]
    rows, cols = np.unravel_index(flat_indices, grid.shape)
    return [(int(r), int(c), float(grid[r, c])) for r, c in zip(rows, cols, strict=False)]


def expected_goals(grid: np.ndarray) -> tuple[float, float]:
    """Return (E[home_goals], E[away_goals]) computed from the grid."""
    goals = np.arange(grid.shape[0], dtype=np.float64)
    exp_home = float(np.dot(goals, grid.sum(axis=1)))
    exp_away = float(np.dot(goals, grid.sum(axis=0)))
    return exp_home, exp_away
