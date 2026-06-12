from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from models.grid import build_grid, derive_markets

XI_DEFAULT: float = 0.0065


def _tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    def __init__(self, xi: float = XI_DEFAULT) -> None:
        self._xi = xi
        self._log_alpha: dict[str, float] = {}
        self._log_beta: dict[str, float] = {}
        self._log_gamma: float = 0.0
        self._rho: float = 0.0
        self._teams: list[str] = []
        self._reference_team: str = ""

    def fit(self, results: pd.DataFrame) -> DixonColesModel:
        """Fit α, β, γ, ρ by minimising negative weighted log-likelihood.

        Uses scipy.optimize.minimize with method='L-BFGS-B'.
        reference_team: first alphabetically — its log_alpha is fixed at 0.0.
        Returns self.
        """
        teams = sorted(set(results["home_team"].unique()) | set(results["away_team"].unique()))
        self._teams = teams
        self._reference_team = teams[0]

        n_teams = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}
        # Chronological recency weights: older matches contribute less
        latest = results["date"].max()
        days_ago = (latest - results["date"]).dt.days.to_numpy(dtype=np.float64)
        weights = np.exp(-self._xi * days_ago)

        home_goals = results["home_score"].to_numpy(dtype=np.int32)
        away_goals = results["away_score"].to_numpy(dtype=np.int32)
        home_idx = np.array([team_idx[t] for t in results["home_team"]], dtype=np.int32)
        away_idx = np.array([team_idx[t] for t in results["away_team"]], dtype=np.int32)
        neutral_flags = results["neutral"].to_numpy(dtype=bool)

        # Parameter layout:
        #   log_alpha[0..n_teams-1] (reference team's entry fixed at 0, excluded from x)
        #   log_beta[0..n_teams-1]
        #   log_gamma (scalar)
        #   rho (scalar)
        #
        # x indices for non-reference log_alpha: positions 0..n_teams-2
        # x indices for log_beta:                positions n_teams-1..2*n_teams-2
        # x index for log_gamma:                 2*n_teams-1
        # x index for rho:                       2*n_teams

        non_ref_teams = [t for t in teams if t != self._reference_team]
        non_ref_idx_in_teams = [team_idx[t] for t in non_ref_teams]
        n_non_ref = len(non_ref_teams)

        def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            log_alpha = np.zeros(n_teams, dtype=np.float64)
            for k, ti in enumerate(non_ref_idx_in_teams):
                log_alpha[ti] = x[k]
            # reference team log_alpha stays 0.0

            log_beta = x[n_non_ref : n_non_ref + n_teams]
            log_gamma = float(x[n_non_ref + n_teams])
            rho = float(x[n_non_ref + n_teams + 1])
            return log_alpha, log_beta, log_gamma, rho

        def _neg_ll(x: np.ndarray) -> float:
            log_alpha, log_beta, log_gamma, rho = _unpack(x)

            alpha = np.exp(log_alpha)
            beta = np.exp(log_beta)
            gamma = np.exp(log_gamma)

            lh = alpha[home_idx] * beta[away_idx] * np.where(neutral_flags, 1.0, gamma)
            la = alpha[away_idx] * beta[home_idx]

            tau_vals = np.array(
                [
                    max(_tau(int(hg), int(ag), float(lh[k]), float(la[k]), rho), 1e-10)
                    for k, (hg, ag) in enumerate(zip(home_goals, away_goals, strict=False))
                ],
                dtype=np.float64,
            )

            log_ll = (
                np.log(tau_vals) + poisson.logpmf(home_goals, lh) + poisson.logpmf(away_goals, la)
            )

            return -float(np.dot(weights, log_ll))

        x0 = np.zeros(n_non_ref + n_teams + 2, dtype=np.float64)
        # rho starts at 0; all log params at 0 (alpha/beta/gamma = 1)

        bounds = (
            [(None, None)] * n_non_ref  # log_alpha non-ref: unbounded
            + [(None, None)] * n_teams  # log_beta: unbounded
            + [(None, None)]  # log_gamma: unbounded
            + [(-0.99, 0.99)]  # rho: constrained
        )

        result = minimize(
            _neg_ll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        log_alpha, log_beta, log_gamma, rho = _unpack(result.x)

        self._log_alpha = {t: float(log_alpha[team_idx[t]]) for t in teams}
        self._log_beta = {t: float(log_beta[team_idx[t]]) for t in teams}
        self._log_gamma = float(log_gamma)
        self._rho = float(rho)

        return self

    def predict_lambda(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> tuple[float, float, float]:
        """Return (lambda_home, lambda_away, rho)."""
        if home_team not in self._log_alpha:
            raise ValueError(f"Team not in fitted parameters: {home_team!r}")
        if away_team not in self._log_alpha:
            raise ValueError(f"Team not in fitted parameters: {away_team!r}")

        alpha_h = np.exp(self._log_alpha[home_team])
        alpha_a = np.exp(self._log_alpha[away_team])
        beta_h = np.exp(self._log_beta[home_team])
        beta_a = np.exp(self._log_beta[away_team])
        gamma = np.exp(self._log_gamma)

        lambda_home = alpha_h * beta_a * (1.0 if neutral else gamma)
        lambda_away = alpha_a * beta_h

        return float(lambda_home), float(lambda_away), self._rho

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Return (score_grid, markets_dict).

        Calls models.grid.build_grid(lambda_h, lambda_a, rho).
        Raises ValueError if either team not in fitted parameters.
        """
        lh, la, rho = self.predict_lambda(home_team, away_team, neutral)
        grid = build_grid(lh, la, rho)
        markets = derive_markets(grid)
        return grid, markets

    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Predict for a DataFrame with columns: home_team, away_team, neutral.

        Returns DataFrame with added columns:
            lambda_home, lambda_away, rho,
            prob_home, prob_draw, prob_away,
            grid (object dtype, each cell an np.ndarray)
        """
        lambda_home_list: list[float] = []
        lambda_away_list: list[float] = []
        rho_list: list[float] = []
        prob_home_list: list[float] = []
        prob_draw_list: list[float] = []
        prob_away_list: list[float] = []
        grid_list: list[np.ndarray] = []

        for row in matches.itertuples(index=False):
            lh, la, rho = self.predict_lambda(
                home_team=row.home_team,
                away_team=row.away_team,
                neutral=bool(row.neutral),
            )
            grid = build_grid(lh, la, rho)
            markets = derive_markets(grid)

            lambda_home_list.append(lh)
            lambda_away_list.append(la)
            rho_list.append(rho)
            prob_home_list.append(markets["home_win"])
            prob_draw_list.append(markets["draw"])
            prob_away_list.append(markets["away_win"])
            grid_list.append(grid)

        out = matches.copy()
        out["lambda_home"] = lambda_home_list
        out["lambda_away"] = lambda_away_list
        out["rho"] = rho_list
        out["prob_home"] = prob_home_list
        out["prob_draw"] = prob_draw_list
        out["prob_away"] = prob_away_list
        out["grid"] = grid_list
        return out
