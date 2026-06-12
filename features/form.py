from __future__ import annotations

import numpy as np
import pandas as pd

FORM_N: int = 10
FORM_FEATURE_DIM: int = 5  # [gf, ga, pts, opp_elo_norm, is_home]


def get_form_sequence(
    results: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    n: int = FORM_N,
) -> np.ndarray:
    """Return (n, 5) float32 array of the team's last n matches before as_of_date.

    Each row: [goals_for, goals_against, points(3/1/0), opp_elo_pre/1500.0, float(is_home)]
    Rows are chronological oldest-first; zero-padded at the front if fewer than n matches.
    Strictly uses only matches where date < as_of_date (no leakage).
    """
    past = results[results["date"] < as_of_date]

    home_mask = past["home_team"] == team
    away_mask = past["away_team"] == team
    involved = past[home_mask | away_mask].copy()

    if involved.empty:
        return np.zeros((n, FORM_FEATURE_DIM), dtype=np.float32)

    involved = involved.sort_values("date", ascending=False).head(n)
    # Reverse to chronological oldest-first
    involved = involved.iloc[::-1].reset_index(drop=True)

    rows: list[list[float]] = []
    for row in involved.itertuples(index=False):
        if row.home_team == team:
            gf = float(row.home_score)
            ga = float(row.away_score)
            opp_elo = float(row.elo_away_pre)
            is_home = 1.0
        else:
            gf = float(row.away_score)
            ga = float(row.home_score)
            opp_elo = float(row.elo_home_pre)
            is_home = 0.0

        if gf > ga:
            pts = 3.0
        elif gf == ga:
            pts = 1.0
        else:
            pts = 0.0

        opp_elo_norm = opp_elo / 1500.0
        rows.append([gf, ga, pts, opp_elo_norm, is_home])

    seq = np.array(rows, dtype=np.float32)  # shape: (actual_n, 5)

    # Zero-pad at the front if fewer than n matches available
    actual = seq.shape[0]
    if actual < n:
        pad = np.zeros((n - actual, FORM_FEATURE_DIM), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)

    return seq


def get_form_aggregates(
    results: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp,
    n: int = 5,
) -> dict[str, float]:
    """Return flat dict of form aggregates for XGBoost tabular input.

    Keys: form_gf, form_ga, form_pts, form_opp_elo, form_n_played.
    Averages over the last n matches strictly before as_of_date.
    If no matches, all numeric values are 0.0 and form_n_played is 0.
    """
    past = results[results["date"] < as_of_date]

    home_mask = past["home_team"] == team
    away_mask = past["away_team"] == team
    involved = past[home_mask | away_mask].copy()

    if involved.empty:
        return {
            "form_gf": 0.0,
            "form_ga": 0.0,
            "form_pts": 0.0,
            "form_opp_elo": 0.0,
            "form_n_played": 0,
        }

    involved = involved.sort_values("date", ascending=False).head(n)

    gf_list: list[float] = []
    ga_list: list[float] = []
    pts_list: list[float] = []
    opp_elo_list: list[float] = []

    for row in involved.itertuples(index=False):
        if row.home_team == team:
            gf = float(row.home_score)
            ga = float(row.away_score)
            opp_elo = float(row.elo_away_pre)
        else:
            gf = float(row.away_score)
            ga = float(row.home_score)
            opp_elo = float(row.elo_home_pre)

        if gf > ga:
            pts = 3.0
        elif gf == ga:
            pts = 1.0
        else:
            pts = 0.0

        gf_list.append(gf)
        ga_list.append(ga)
        pts_list.append(pts)
        opp_elo_list.append(opp_elo)

    played = len(gf_list)
    return {
        "form_gf": float(np.mean(gf_list)),
        "form_ga": float(np.mean(ga_list)),
        "form_pts": float(np.mean(pts_list)),
        "form_opp_elo": float(np.mean(opp_elo_list)),
        "form_n_played": played,
    }
