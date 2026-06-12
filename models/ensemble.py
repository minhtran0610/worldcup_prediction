from __future__ import annotations

import numpy as np
import pandas as pd

from models.grid import derive_markets, expected_goals


class EnsembleModel:
    """Averages score grids from multiple fitted models.

    Each constituent model must implement predict_batch(matches) -> pd.DataFrame
    with a 'grid' column containing (8,8) np.ndarray objects.

    Weighting: either equal (default) or inverse-RPS (lower RPS = higher weight).
    """

    def __init__(
        self,
        models: list,
        weights: list[float] | None = None,
    ) -> None:
        if not models:
            raise ValueError("EnsembleModel requires at least one constituent model.")

        if weights is None:
            n = len(models)
            weights = [1.0 / n] * n
        else:
            if len(weights) != len(models):
                raise ValueError(
                    f"len(weights)={len(weights)} must equal len(models)={len(models)}."
                )
            total = sum(weights)
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"weights must sum to 1.0, got {total:.6f}. "
                    "Normalise before passing, or use EnsembleModel.from_rps_scores()."
                )

        self.models = models
        self.weights: list[float] = list(weights)

    @classmethod
    def from_rps_scores(
        cls,
        models: list,
        rps_scores: list[float],
    ) -> EnsembleModel:
        """Construct with weights inverse-proportional to RPS scores.

        weight_i = (1/rps_i) / sum(1/rps_j for all j)
        """
        if len(models) != len(rps_scores):
            raise ValueError(
                f"len(models)={len(models)} must equal len(rps_scores)={len(rps_scores)}."
            )
        if any(r <= 0 for r in rps_scores):
            raise ValueError("All RPS scores must be strictly positive.")

        inv = [1.0 / r for r in rps_scores]
        total = sum(inv)
        weights = [v / total for v in inv]
        return cls(models=models, weights=weights)

    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Average the score grids from all constituent models.

        For each match:
          - Run predict_batch on each constituent model
          - Weighted-average the (8,8) grids
          - Re-derive markets from the averaged grid

        Output: all input columns + prob_home, prob_draw, prob_away,
                expected_home, expected_away, grid
        """
        # Collect per-model grid arrays: list of pd.Series (each entry an (8,8) ndarray)
        grid_series: list[pd.Series] = []
        for i, model in enumerate(self.models):
            try:
                preds = model.predict_batch(matches)
            except Exception as exc:
                model_name = type(model).__name__
                raise RuntimeError(
                    f"EnsembleModel: constituent model #{i} ({model_name}) "
                    f"raised during predict_batch: {exc}"
                ) from exc
            grid_series.append(preds["grid"])

        n_rows = len(matches)

        # Weighted average of grids row by row
        avg_grids: list[np.ndarray] = []
        for row_idx in range(n_rows):
            avg = np.zeros((8, 8), dtype=np.float64)
            for w, gs in zip(self.weights, grid_series, strict=False):
                avg += w * gs.iloc[row_idx]
            avg_grids.append(avg)

        # Derive markets and expected goals from averaged grids
        prob_home_list: list[float] = []
        prob_draw_list: list[float] = []
        prob_away_list: list[float] = []
        expected_home_list: list[float] = []
        expected_away_list: list[float] = []

        for avg in avg_grids:
            markets = derive_markets(avg)
            prob_home_list.append(markets["home_win"])
            prob_draw_list.append(markets["draw"])
            prob_away_list.append(markets["away_win"])

            exp_h, exp_a = expected_goals(avg)
            expected_home_list.append(exp_h)
            expected_away_list.append(exp_a)

        out = matches.copy()
        out["prob_home"] = prob_home_list
        out["prob_draw"] = prob_draw_list
        out["prob_away"] = prob_away_list
        out["expected_home"] = expected_home_list
        out["expected_away"] = expected_away_list
        out["grid"] = avg_grids
        return out
