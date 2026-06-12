from __future__ import annotations

import math

import pandas as pd

from features.elo import ELO_SCALE, HOME_ADVANTAGE, get_current_ratings

DRAW_BASE_RATE: float = 0.26
DRAW_ELO_SCALE: float = 200.0


class EloModel:
    def __init__(self) -> None:
        self._ratings: dict[str, float] = {}

    def fit(self, results: pd.DataFrame) -> EloModel:
        """Compute Elo ratings from results. Returns self for chaining.

        Calls features.elo.compute_elo_ratings internally to build state.
        After fit, stores current ratings in self._ratings: dict[str, float].
        """
        self._ratings = get_current_ratings(results)
        return self

    def predict_1x2(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> dict[str, float]:
        """Return {"home_win": float, "draw": float, "away_win": float} summing to 1.0.

        Uses self._ratings. Raises ValueError if either team not in ratings.

        Draw probability model:
          elo_diff = home_elo - away_elo  (adjusted for home advantage if not neutral)
          draw_rate = DRAW_BASE_RATE * exp(-elo_diff^2 / (2 * DRAW_ELO_SCALE^2))
          p_win_no_draw = 1 / (1 + 10^(-(adj_elo_diff) / ELO_SCALE))
          home_win = p_win_no_draw * (1 - draw_rate)
          away_win = (1 - p_win_no_draw) * (1 - draw_rate)
          draw = draw_rate
        """
        if home_team not in self._ratings:
            raise ValueError(f"Team not in fitted ratings: {home_team!r}")
        if away_team not in self._ratings:
            raise ValueError(f"Team not in fitted ratings: {away_team!r}")

        home_elo = self._ratings[home_team]
        away_elo = self._ratings[away_team]

        advantage = 0.0 if neutral else HOME_ADVANTAGE
        adj_elo_diff = (home_elo + advantage) - away_elo

        # Draw is most likely when teams are evenly matched; suppressed for large Elo gaps
        draw_rate = DRAW_BASE_RATE * math.exp(-(adj_elo_diff**2) / (2.0 * DRAW_ELO_SCALE**2))

        p_win_no_draw = 1.0 / (1.0 + 10.0 ** (-adj_elo_diff / ELO_SCALE))

        home_win = p_win_no_draw * (1.0 - draw_rate)
        away_win = (1.0 - p_win_no_draw) * (1.0 - draw_rate)
        draw = draw_rate

        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def predict_batch(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Predict 1X2 for a DataFrame of matches.

        matches must have columns: home_team, away_team, neutral (bool)
        Returns DataFrame with added columns: prob_home, prob_draw, prob_away
        """
        prob_home: list[float] = []
        prob_draw: list[float] = []
        prob_away: list[float] = []

        for row in matches.itertuples(index=False):
            result = self.predict_1x2(
                home_team=row.home_team,
                away_team=row.away_team,
                neutral=bool(row.neutral),
            )
            prob_home.append(result["home_win"])
            prob_draw.append(result["draw"])
            prob_away.append(result["away_win"])

        out = matches.copy()
        out["prob_home"] = prob_home
        out["prob_draw"] = prob_draw
        out["prob_away"] = prob_away
        return out
