from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features.form import get_form_aggregates
from models.grid import build_grid, derive_markets

_XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
    "verbosity": 0,
}

_FEATURE_COLS = [
    "elo_home_pre",
    "elo_away_pre",
    "elo_diff",
    "rest_days_home",
    "rest_days_away",
    "neutral",
    "is_knockout",
    "is_host_home",
    "is_host_away",
    "form_gf_home",
    "form_ga_home",
    "form_pts_home",
    "form_opp_elo_home",
    "form_n_played_home",
    "form_gf_away",
    "form_ga_away",
    "form_pts_away",
    "form_opp_elo_away",
    "form_n_played_away",
]


def _build_form_features(
    results: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    suffix: str,
) -> dict[str, float]:
    """Compute form aggregates and rename keys with suffix (_home or _away)."""
    aggs = get_form_aggregates(results, team, as_of_date, n=5)
    return {
        f"form_gf_{suffix}": aggs["form_gf"],
        f"form_ga_{suffix}": aggs["form_ga"],
        f"form_pts_{suffix}": aggs["form_pts"],
        f"form_opp_elo_{suffix}": aggs["form_opp_elo"],
        f"form_n_played_{suffix}": float(aggs["form_n_played"]),
    }


def _build_feature_matrix(
    matches: pd.DataFrame,
    train_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix for a batch of matches.

    For each match, looks up form aggregates from train_results strictly
    before the match date.
    """
    rows: list[dict] = []
    for row in matches.itertuples(index=False):
        match_date = row.date
        home_form = _build_form_features(train_results, row.home_team, match_date, "home")
        away_form = _build_form_features(train_results, row.away_team, match_date, "away")

        feat: dict = {
            "elo_home_pre": float(row.elo_home_pre),
            "elo_away_pre": float(row.elo_away_pre),
            "elo_diff": float(row.elo_diff),
            "rest_days_home": min(
                float(row.rest_days_home) if not _is_nan(row.rest_days_home) else 365.0, 365.0
            ),
            "rest_days_away": min(
                float(row.rest_days_away) if not _is_nan(row.rest_days_away) else 365.0, 365.0
            ),
            "neutral": float(bool(row.neutral)),
            "is_knockout": float(bool(row.is_knockout)),
            "is_host_home": float(bool(row.is_host_home)),
            "is_host_away": float(bool(row.is_host_away)),
        }
        feat.update(home_form)
        feat.update(away_form)
        rows.append(feat)

    return pd.DataFrame(rows, columns=_FEATURE_COLS)


def _is_nan(val: object) -> bool:
    try:
        return bool(np.isnan(float(val)))
    except (TypeError, ValueError):
        return False


class XGBModel:
    """XGBoost model predicting (lambda_home, lambda_away) from tabular features.

    rho is fixed at 0.0 (independent Poisson; Dixon-Coles correction deferred).
    """

    def __init__(self) -> None:
        self._model_home: XGBRegressor = XGBRegressor(**_XGB_PARAMS)
        self._model_away: XGBRegressor = XGBRegressor(**_XGB_PARAMS)
        self._mean_home: float = 1.5
        self._mean_away: float = 1.1
        self._col_means: pd.Series | None = None
        self._train_results: pd.DataFrame | None = None

    def fit(self, results: pd.DataFrame) -> XGBModel:
        """Fit two XGBRegressor models — one for home goals, one for away goals.

        Builds form aggregates for every training match using only data strictly
        before that match's date (no leakage). Stores train_results for use
        in predict_batch.
        """
        self._train_results = results.copy()
        self._mean_home = float(results["home_score"].mean())
        self._mean_away = float(results["away_score"].mean())

        X = _build_feature_matrix(results, results)

        # Fill NaN with column means before fitting
        self._col_means = X.mean()
        X = X.fillna(self._col_means)

        y_home = results["home_score"].to_numpy(dtype=np.float64)
        y_away = results["away_score"].to_numpy(dtype=np.float64)

        weights: np.ndarray | None = None
        if "sample_weight" in results.columns:
            weights = results["sample_weight"].to_numpy(dtype=np.float64)

        self._model_home.fit(X, y_home, sample_weight=weights)
        self._model_away.fit(X, y_away, sample_weight=weights)

        return self

    def predict_lambda(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        home_elo: float = 1500.0,
        away_elo: float = 1500.0,
        rest_home: float = 30.0,
        rest_away: float = 30.0,
        is_knockout: bool = False,
        is_host_home: bool = False,
        is_host_away: bool = False,
    ) -> tuple[float, float, float]:
        """Return (lambda_home, lambda_away, rho=0.0).

        Uses a synthetic single-row DataFrame so the feature pipeline is
        shared with predict_batch. Form aggregates are drawn from
        self._train_results up to an effectively infinite future date so
        all history is included.
        """
        if self._train_results is None:
            return self._mean_home, self._mean_away, 0.0

        # Use a far-future date so all training history is used for form lookup
        far_future = pd.Timestamp("2099-12-31")

        single = pd.DataFrame(
            [
                {
                    "date": far_future,
                    "home_team": home_team,
                    "away_team": away_team,
                    "elo_home_pre": home_elo,
                    "elo_away_pre": away_elo,
                    "elo_diff": home_elo - away_elo,
                    "rest_days_home": min(rest_home, 365.0),
                    "rest_days_away": min(rest_away, 365.0),
                    "neutral": float(neutral),
                    "is_knockout": float(is_knockout),
                    "is_host_home": float(is_host_home),
                    "is_host_away": float(is_host_away),
                    "home_score": 0,
                    "away_score": 0,
                    "sample_weight": 1.0,
                }
            ]
        )

        X = _build_feature_matrix(single, self._train_results)
        if self._col_means is not None:
            X = X.fillna(self._col_means)

        lh = float(max(self._model_home.predict(X)[0], 0.01))
        la = float(max(self._model_away.predict(X)[0], 0.01))
        return lh, la, 0.0

    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Predict for a batch DataFrame with the same columns as training results.

        Returns DataFrame with all input columns PLUS:
            prob_home, prob_draw, prob_away,
            expected_home, expected_away,
            grid (object dtype, each cell an (8,8) np.ndarray)

        Form aggregates for each match are computed from self._train_results
        using only data strictly before the match date.
        """
        if self._train_results is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = _build_feature_matrix(matches, self._train_results)
        if self._col_means is not None:
            X = X.fillna(self._col_means)

        raw_lh = self._model_home.predict(X).astype(np.float64)
        raw_la = self._model_away.predict(X).astype(np.float64)

        # Clamp to avoid degenerate Poisson distributions
        lambda_home_arr = np.maximum(raw_lh, 0.01)
        lambda_away_arr = np.maximum(raw_la, 0.01)

        prob_home_list: list[float] = []
        prob_draw_list: list[float] = []
        prob_away_list: list[float] = []
        expected_home_list: list[float] = []
        expected_away_list: list[float] = []
        grid_list: list[np.ndarray] = []

        goals_vec = np.arange(8, dtype=np.float64)

        for lh, la in zip(lambda_home_arr, lambda_away_arr, strict=False):
            grid = build_grid(float(lh), float(la), 0.0)
            markets = derive_markets(grid)

            prob_home_list.append(markets["home_win"])
            prob_draw_list.append(markets["draw"])
            prob_away_list.append(markets["away_win"])

            exp_home = float(np.dot(goals_vec, grid.sum(axis=1)))
            exp_away = float(np.dot(goals_vec, grid.sum(axis=0)))
            expected_home_list.append(exp_home)
            expected_away_list.append(exp_away)

            grid_list.append(grid)

        out = matches.copy()
        out["prob_home"] = prob_home_list
        out["prob_draw"] = prob_draw_list
        out["prob_away"] = prob_away_list
        out["expected_home"] = expected_home_list
        out["expected_away"] = expected_away_list
        out["grid"] = grid_list
        return out
