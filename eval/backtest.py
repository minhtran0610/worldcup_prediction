from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eval.metrics import aggregate_nll, aggregate_rps, remove_margin

MIN_TRAIN_MONTHS: int = 36
VAL_DURATION_MONTHS: int = 6
STEP_MONTHS: int = 6
KELLY_FRACTION: float = 0.25
MIN_EDGE: float = 0.02


@dataclass
class FoldResult:
    fold_idx: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_matches: int
    rps: float
    nll: float
    n_value_bets: int
    flat_roi: float
    kelly_roi: float


def generate_folds(
    results: pd.DataFrame,
    min_train_months: int = MIN_TRAIN_MONTHS,
    val_duration_months: int = VAL_DURATION_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (val_start, val_end, train_end=val_start) tuples.

    train_end == val_start (strictly: train is [results_start, val_start),
    val is [val_start, val_end)).
    First val_start is min_train_months after the first match date.
    Stop when val_end would exceed the last match date.
    """
    dates = pd.to_datetime(results["date"])
    first_date = dates.min()
    last_date = dates.max()

    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    val_start = first_date + pd.DateOffset(months=min_train_months)

    while True:
        val_end = val_start + pd.DateOffset(months=val_duration_months)
        if val_end > last_date:
            break
        train_end = val_start
        folds.append((val_start, val_end, train_end))
        val_start = val_start + pd.DateOffset(months=step_months)

    return folds


def compute_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE,
) -> pd.DataFrame:
    """Filter predictions to value bets and compute staking columns.

    predictions must have columns:
        prob_home, prob_draw, prob_away           (model probabilities)
        odds_home, odds_draw, odds_away           (closing decimal odds, optional)
        home_score, away_score                    (actual result)

    Adds columns:
        margin_home, margin_draw, margin_away
        edge_home, edge_draw, edge_away
        best_edge_outcome (str: "home"/"draw"/"away")
        best_edge (float)
        kelly_stake (float)
        outcome_won (bool)
        flat_return (float)
        kelly_return (float)

    Returns only rows where best_edge >= min_edge.
    """
    df = predictions.copy()

    has_odds = all(c in df.columns for c in ("odds_home", "odds_draw", "odds_away"))

    if has_odds:
        market_probs = df.apply(
            lambda row: remove_margin(
                {
                    "home": row["odds_home"],
                    "draw": row["odds_draw"],
                    "away": row["odds_away"],
                }
            ),
            axis=1,
            result_type="expand",
        )
        df["margin_home"] = market_probs["home"]
        df["margin_draw"] = market_probs["draw"]
        df["margin_away"] = market_probs["away"]
    else:
        df["margin_home"] = np.nan
        df["margin_draw"] = np.nan
        df["margin_away"] = np.nan

    df["edge_home"] = df["prob_home"] - df["margin_home"]
    df["edge_draw"] = df["prob_draw"] - df["margin_draw"]
    df["edge_away"] = df["prob_away"] - df["margin_away"]

    edges = df[["edge_home", "edge_draw", "edge_away"]].values
    best_idx = np.argmax(edges, axis=1)
    outcome_names = np.array(["home", "draw", "away"])
    df["best_edge_outcome"] = outcome_names[best_idx]
    df["best_edge"] = edges[np.arange(len(df)), best_idx]

    df = df[df["best_edge"] >= min_edge].copy()

    if has_odds and len(df) > 0:
        odds_map = {"home": "odds_home", "draw": "odds_draw", "away": "odds_away"}
        best_decimal_odds = df.apply(
            lambda row: row[odds_map[row["best_edge_outcome"]]], axis=1
        ).astype(float)
        df["kelly_stake"] = KELLY_FRACTION * df["best_edge"] / best_decimal_odds
    else:
        df["kelly_stake"] = np.nan

    actual_outcome = df.apply(
        lambda row: (
            "home"
            if row["home_score"] > row["away_score"]
            else ("draw" if row["home_score"] == row["away_score"] else "away")
        ),
        axis=1,
    )
    df["outcome_won"] = actual_outcome == df["best_edge_outcome"]

    if has_odds and len(df) > 0:
        odds_map2 = {"home": "odds_home", "draw": "odds_draw", "away": "odds_away"}
        best_decimal_odds2 = df.apply(
            lambda row: row[odds_map2[row["best_edge_outcome"]]], axis=1
        ).astype(float)
        df["flat_return"] = np.where(
            df["outcome_won"],
            best_decimal_odds2 - 1.0,
            -1.0,
        )
        df["kelly_return"] = np.where(
            df["outcome_won"],
            df["kelly_stake"] * (best_decimal_odds2 - 1.0),
            -df["kelly_stake"],
        )
    else:
        df["flat_return"] = np.nan
        df["kelly_return"] = np.nan

    return df


def fold_metrics(
    val_df: pd.DataFrame,
    predictions: pd.DataFrame,
    grids: list[np.ndarray],
    fold_idx: int,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    train_end: pd.Timestamp,
) -> FoldResult:
    """Compute FoldResult from val predictions.

    outcome_idx is derived internally: 0 if home_score > away_score,
    1 if equal, 2 if away wins.
    """
    pred = predictions.copy()

    pred["outcome_idx"] = np.where(
        val_df["home_score"].values > val_df["away_score"].values,
        0,
        np.where(val_df["home_score"].values == val_df["away_score"].values, 1, 2),
    )

    nll_df = pd.DataFrame(
        {
            "home_goals": val_df["home_score"].values,
            "away_goals": val_df["away_score"].values,
        }
    )

    rps = aggregate_rps(pred)
    nll = aggregate_nll(nll_df, grids)

    value_df = compute_value_bets(
        pred.assign(
            home_score=val_df["home_score"].values,
            away_score=val_df["away_score"].values,
        )
    )
    n_value_bets = len(value_df)

    if n_value_bets == 0:
        flat_roi = float("nan")
        kelly_roi = float("nan")
    else:
        flat_roi = float(value_df["flat_return"].sum() / n_value_bets)
        kelly_stakes_sum = value_df["kelly_stake"].sum()
        if kelly_stakes_sum == 0 or value_df["kelly_stake"].isna().all():
            kelly_roi = float("nan")
        else:
            kelly_roi = float(value_df["kelly_return"].sum() / kelly_stakes_sum)

    return FoldResult(
        fold_idx=fold_idx,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        n_matches=len(val_df),
        rps=rps,
        nll=nll,
        n_value_bets=n_value_bets,
        flat_roi=flat_roi,
        kelly_roi=kelly_roi,
    )


def print_backtest_report(fold_results: list[FoldResult]) -> None:
    """Print a formatted summary table of all folds + aggregate stats to stdout."""
    header = (
        f"{'Fold':>4}  {'Train End':>10}  {'Val Start':>10}  {'Val End':>10}  "
        f"{'N':>5}  {'RPS':>7}  {'NLL':>7}  "
        f"{'#Bets':>5}  {'FlatROI':>8}  {'KellyROI':>9}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for fr in fold_results:
        flat_str = f"{fr.flat_roi:+.3f}" if not np.isnan(fr.flat_roi) else "     N/A"
        kelly_str = f"{fr.kelly_roi:+.3f}" if not np.isnan(fr.kelly_roi) else "      N/A"
        print(
            f"{fr.fold_idx:>4}  "
            f"{str(fr.train_end.date()):>10}  "
            f"{str(fr.val_start.date()):>10}  "
            f"{str(fr.val_end.date()):>10}  "
            f"{fr.n_matches:>5}  "
            f"{fr.rps:>7.4f}  "
            f"{fr.nll:>7.4f}  "
            f"{fr.n_value_bets:>5}  "
            f"{flat_str:>8}  "
            f"{kelly_str:>9}"
        )

    print(separator)

    if not fold_results:
        print("No folds to summarise.")
        return

    total_matches = sum(fr.n_matches for fr in fold_results)
    mean_rps = float(np.mean([fr.rps for fr in fold_results]))
    mean_nll = float(np.mean([fr.nll for fr in fold_results]))
    total_bets = sum(fr.n_value_bets for fr in fold_results)

    flat_rois = [fr.flat_roi for fr in fold_results if not np.isnan(fr.flat_roi)]
    kelly_rois = [fr.kelly_roi for fr in fold_results if not np.isnan(fr.kelly_roi)]

    flat_agg_str = f"{float(np.mean(flat_rois)):+.3f}" if flat_rois else "N/A"
    kelly_agg_str = f"{float(np.mean(kelly_rois)):+.3f}" if kelly_rois else "N/A"

    print(
        f"{'ALL':>4}  {'':>10}  {'':>10}  {'':>10}  "
        f"{total_matches:>5}  "
        f"{mean_rps:>7.4f}  "
        f"{mean_nll:>7.4f}  "
        f"{total_bets:>5}  "
        f"{flat_agg_str:>8}  "
        f"{kelly_agg_str:>9}"
    )
    print(separator)
