from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from data.ingest.injuries import fetch_wc2026_injuries
from data.ingest.llm_form import get_all_teams_form
from data.ingest.odds_live import fetch_upcoming_odds, get_api_key
from data.ingest.results import load_results
from data.ingest.wc2026 import load_wc2026_schedule
from eval.backtest import KELLY_FRACTION
from eval.metrics import remove_margin
from features.context import WC_2026_HOSTS, derive_context
from features.elo import compute_elo_ratings, extend_elo_through_matches, get_current_ratings
from features.injury import (
    apply_injury_adjustment,
    compute_injury_strength_loss,
)
from features.llm_form_feature import (
    apply_sentiment_adjustment,
    build_sentiment_report_line,
    compute_sentiment_factor,
)
from features.squad_registry import SquadRegistry
from models.grid import build_grid, derive_markets

app = typer.Typer()

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_COL_KO = 11  # "Jun 16 22:00"
_COL_MATCH = 26
_COL_PROB = 7
_COL_EDGE = 8
_COL_KELLY = 7


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def _ko_str(dt: pd.Timestamp) -> str:
    """Format kickoff as 'Jun 16 22:00' (UTC)."""
    try:
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return " " * _COL_KO


def _header_with_odds() -> str:
    return (
        f"{'Kickoff (UTC)':<{_COL_KO}} "
        f"{'Match':<{_COL_MATCH}} "
        f"{'Home%':>{_COL_PROB}} {'Draw%':>{_COL_PROB}} {'Away%':>{_COL_PROB}}  "
        f"{'MktH%':>{_COL_PROB}} {'MktD%':>{_COL_PROB}} {'MktA%':>{_COL_PROB}}  "
        f"{'Edge':>{_COL_EDGE}}  {'Kelly':>{_COL_KELLY}}"
    )


def _header_no_odds() -> str:
    return (
        f"{'Kickoff (UTC)':<{_COL_KO}} "
        f"{'Match':<{_COL_MATCH}} "
        f"{'Home%':>{_COL_PROB}} {'Draw%':>{_COL_PROB}} {'Away%':>{_COL_PROB}}"
    )


def _separator(width: int) -> str:
    return "-" * width


def _match_label(home: str, away: str) -> str:
    label = f"{home} vs {away}"
    if len(label) > _COL_MATCH:
        label = label[: _COL_MATCH - 1] + "…"
    return label


def _format_row_with_odds(
    home: str,
    away: str,
    kickoff: pd.Timestamp,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    mkt_home: float,
    mkt_draw: float,
    mkt_away: float,
    best_edge: float,
    kelly: float,
    is_value: bool,
) -> str:
    ko = _ko_str(kickoff)
    label = _match_label(home, away)
    edge_str = f"{best_edge * 100:+.1f}pp"
    kelly_str = f"{kelly * 100:.1f}%"
    value_marker = " *" if is_value else "  "
    return (
        f"{ko:<{_COL_KO}} "
        f"{label:<{_COL_MATCH}} "
        f"{_pct(prob_home):>{_COL_PROB}} {_pct(prob_draw):>{_COL_PROB}} {_pct(prob_away):>{_COL_PROB}}  "
        f"{_pct(mkt_home):>{_COL_PROB}} {_pct(mkt_draw):>{_COL_PROB}} {_pct(mkt_away):>{_COL_PROB}}  "
        f"{edge_str:>{_COL_EDGE}}{value_marker} {kelly_str:>{_COL_KELLY}}"
    )


def _format_row_no_odds(
    home: str,
    away: str,
    kickoff: pd.Timestamp,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
) -> str:
    ko = _ko_str(kickoff)
    label = _match_label(home, away)
    return (
        f"{ko:<{_COL_KO}} "
        f"{label:<{_COL_MATCH}} "
        f"{_pct(prob_home):>{_COL_PROB}} {_pct(prob_draw):>{_COL_PROB}} {_pct(prob_away):>{_COL_PROB}}"
    )


# ---------------------------------------------------------------------------
# Match building: synthesise the context columns that models expect
# ---------------------------------------------------------------------------


def _build_upcoming_match_df(
    schedule: pd.DataFrame,
    elo_ratings: dict[str, float],
    squad_registry: SquadRegistry | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of upcoming WC matches with all columns models need.

    Uses the last known Elo ratings for each team (computed over all training
    data).  All WC matches are neutral-venue, group-stage by default.
    Rest days default to 7 (a reasonable inter-match rest for WC group stage).
    Squad features are populated from squad_registry if provided.
    """
    DEFAULT_ELO = 1500.0
    DEFAULT_REST = 7.0

    rows: list[dict] = []
    for row in schedule.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        elo_h = elo_ratings.get(home, DEFAULT_ELO)
        elo_a = elo_ratings.get(away, DEFAULT_ELO)

        # Squad quality features for WC 2026
        if squad_registry is not None:
            feats_h = squad_registry.get_features(home, 2026, "FIFA World Cup")
            feats_a = squad_registry.get_features(away, 2026, "FIFA World Cup")
        else:
            feats_h = {"top5_share": 0.0, "avg_caps_norm": 0.0}
            feats_a = {"top5_share": 0.0, "avg_caps_norm": 0.0}

        rows.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                # Placeholder scores for columns that models iterate over.
                # These are never used for RPS/NLL since we are only predicting.
                "home_score": 0,
                "away_score": 0,
                "tournament": "FIFA World Cup",
                "neutral": True,
                "country": "United States",
                "elo_home_pre": elo_h,
                "elo_away_pre": elo_a,
                "elo_diff": elo_h - elo_a,
                "rest_days_home": DEFAULT_REST,
                "rest_days_away": DEFAULT_REST,
                "is_knockout": False,
                "is_host_home": home in WC_2026_HOSTS,
                "is_host_away": away in WC_2026_HOSTS,
                "sample_weight": 1.0,
                "squad_top5_home": feats_h["top5_share"],
                "squad_top5_away": feats_a["top5_share"],
                "squad_caps_home": feats_h["avg_caps_norm"],
                "squad_caps_away": feats_a["avg_caps_norm"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Odds matching
# ---------------------------------------------------------------------------

# Odds-API team name (lowercased) -> our training-data name (lowercased)
_ODDS_NAME_MAP: dict[str, str] = {
    "usa": "united states",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "korea republic": "south korea",
    "ir iran": "iran",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "dr congo": "dr congo",
    "democratic republic of congo": "dr congo",
    "czechia": "czech republic",
    "cape verde islands": "cape verde",
    "curacao": "curaçao",
}


def _normalise_odds_name(name: str) -> str:
    lower = name.lower()
    return _ODDS_NAME_MAP.get(lower, lower)


def _build_odds_index(live_odds: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """Build a lookup dict keyed by (normalised_home, normalised_away)."""
    index: dict[tuple[str, str], dict[str, float]] = {}
    for entry in live_odds:
        key = (_normalise_odds_name(entry["home_team"]), _normalise_odds_name(entry["away_team"]))
        index[key] = entry
    return index


def _find_odds(
    home: str,
    away: str,
    odds_index: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float] | None:
    """Look up odds for a match.  Falls back to reversed team order."""
    key = (_normalise_odds_name(home), _normalise_odds_name(away))
    if key in odds_index:
        return odds_index[key]
    # Sometimes the API lists the match with teams swapped
    rev_key = (_normalise_odds_name(away), _normalise_odds_name(home))
    if rev_key in odds_index:
        entry = odds_index[rev_key]
        # Swap home/away prices so they align with our team ordering
        return {
            "odds_home": entry["odds_away"],
            "odds_draw": entry["odds_draw"],
            "odds_away": entry["odds_home"],
        }
    return None


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _fit_model(model_name: str, results: pd.DataFrame, checkpoint: Path | None):
    """Fit the requested model on all results and return the fitted object."""
    if model_name == "dc":
        from models.dixon_coles import DixonColesModel

        m = DixonColesModel()
        m.fit(results)
        return m

    if model_name == "xgb":
        from models.xgb_model import XGBModel

        m = XGBModel()
        m.fit(results)
        return m

    if model_name == "neural":
        from models.neural import NeuralModel

        if checkpoint is not None and Path(checkpoint).exists():
            try:
                m = NeuralModel.load(str(checkpoint))
                m._train_results = results
                return m
            except Exception as exc:
                typer.echo(
                    f"Warning: could not load checkpoint {checkpoint}: {exc} — fitting from scratch.",
                    err=True,
                )
        m = NeuralModel()
        m.fit(results)
        return m

    if model_name == "ensemble":
        from models.dixon_coles import DixonColesModel
        from models.ensemble import EnsembleModel
        from models.xgb_model import XGBModel

        typer.echo("Fitting DixonColes...", err=True)
        dc = DixonColesModel()
        dc.fit(results)

        typer.echo("Fitting XGBoost...", err=True)
        xgb = XGBModel()
        xgb.fit(results)

        constituents: list = [dc, xgb]
        weights: list[float] = [0.5, 0.5]

        if checkpoint is not None and Path(checkpoint).exists():
            from models.neural import NeuralModel

            typer.echo(f"Loading neural checkpoint from {checkpoint}...", err=True)
            try:
                neural = NeuralModel.load(str(checkpoint))
                neural._train_results = results
                constituents.append(neural)
                # Equal weight across all three
                n = len(constituents)
                weights = [1.0 / n] * n
                typer.echo("Neural model included in ensemble.", err=True)
            except Exception as exc:
                typer.echo(
                    f"Warning: could not load neural checkpoint: {exc} — using DC+XGB ensemble.",
                    err=True,
                )

        return EnsembleModel(models=constituents, weights=weights)

    raise ValueError(f"Unknown model: {model_name!r}. Use 'dc', 'xgb', 'neural', or 'ensemble'.")


# ---------------------------------------------------------------------------
# Odds snapshot (persists pre-match market odds for later validation)
# ---------------------------------------------------------------------------

_ODDS_SNAPSHOT_PATH: Path = Path("data/raw/wc2026_odds_snapshot.json")


def _save_odds_snapshot(live_odds: list[dict]) -> None:
    """Merge new live odds into the on-disk snapshot (upsert by team pair).

    Keyed by (normalised_home, normalised_away) so repeated runs before the
    same match update rather than duplicate.  Silently swallows write errors.
    """
    existing: list[dict] = []
    if _ODDS_SNAPSHOT_PATH.exists():
        try:
            with _ODDS_SNAPSHOT_PATH.open() as f:
                existing = json.load(f)
        except Exception:
            existing = []

    index: dict[tuple[str, str], dict] = {
        (_normalise_odds_name(e["home_team"]), _normalise_odds_name(e["away_team"])): e
        for e in existing
    }
    for entry in live_odds:
        key = (_normalise_odds_name(entry["home_team"]), _normalise_odds_name(entry["away_team"]))
        index[key] = entry  # upsert

    try:
        _ODDS_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _ODDS_SNAPSHOT_PATH.open("w") as f:
            json.dump(list(index.values()), f, indent=2, default=str)
        typer.echo(
            f"[wc2026] Odds snapshot updated ({len(index)} entries) → {_ODDS_SNAPSHOT_PATH}",
            err=True,
        )
    except Exception as exc:
        typer.echo(f"[wc2026] WARNING: could not save odds snapshot — {exc}", err=True)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    model: str = typer.Option("neural", help="Model: 'dc', 'xgb', 'neural', 'ensemble'"),
    checkpoint: Path | None = typer.Option(
        None, help="Neural checkpoint path (for neural/ensemble)"
    ),
    min_edge: float = typer.Option(0.02, help="Minimum edge to flag as value bet"),
    show_all: bool = typer.Option(False, help="Show all matches, not just value bets"),
    injuries: bool = typer.Option(True, help="Apply injury/suspension λ adjustment"),
    injury_k: float = typer.Option(0.5, help="Injury dampening coefficient K (0–1)"),
    llm_form: bool = typer.Option(
        False,
        "--llm-form/--no-llm-form",
        help="Apply LLM narrative form sentiment adjustment (~2 min)",
    ),
    llm_model: str = typer.Option("qwen3.5:9b", help="Ollama model for LLM form analysis"),
) -> None:
    """Predict all upcoming WC 2026 matches and compare vs live odds.

    Workflow:
      1. Load results, compute Elo, derive context
      2. Fit the selected model on all historical data
      3. Load the WC 2026 schedule
      4. For each upcoming match, predict 1X2 probabilities
      5. Fetch live odds from the-odds-api.com (if API key set)
      6. Display a table: match | model probs | market probs | edge | kelly

    If no API key: show model probabilities only (no edge column).
    If schedule scrape fails: print a message and exit gracefully.
    """
    # ------------------------------------------------------------------
    # 1. Load and prepare training data
    # ------------------------------------------------------------------
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)

    results = compute_elo_ratings(results)

    typer.echo("Building squad registry...", err=True)
    registry = SquadRegistry.build()
    results = derive_context(results, squad_registry=registry)

    # ------------------------------------------------------------------
    # 2. Fetch WC 2026 schedule and inject completed results into context
    # ------------------------------------------------------------------
    typer.echo("Fetching WC 2026 schedule (live)...", err=True)
    schedule = load_wc2026_schedule(force_refresh=True)

    if not schedule.empty:
        completed = (
            schedule[schedule["is_completed"]].dropna(subset=["home_score", "away_score"]).copy()
        )
        if not completed.empty:
            completed["home_score"] = completed["home_score"].astype(int)
            completed["away_score"] = completed["away_score"].astype(int)
            # Wikipedia (1×3) match-table parser doesn't extract dates — fall back to
            # WC 2026 start date so the Elo forward pass and form sequences work correctly.
            completed["date"] = completed["date"].fillna(pd.Timestamp("2026-06-11"))
            completed = completed.sort_values("date").reset_index(drop=True)
            # neutral and tournament must be set before extend_elo_through_matches
            # (used for home-advantage and K-factor in the Elo forward pass)
            completed["neutral"] = True
            completed["tournament"] = "FIFA World Cup"
            completed = extend_elo_through_matches(results, completed)
            completed["country"] = "United States"
            completed["is_knockout"] = True
            completed["is_host_home"] = completed["home_team"].isin(WC_2026_HOSTS)
            completed["is_host_away"] = completed["away_team"].isin(WC_2026_HOSTS)
            completed["rest_days_home"] = 7.0
            completed["rest_days_away"] = 7.0
            completed["sample_weight"] = 1.0
            for col in ["squad_top5_home", "squad_top5_away", "squad_caps_home", "squad_caps_away"]:
                completed[col] = 0.0
            for i, wc_row in completed.iterrows():
                fh = registry.get_features(wc_row["home_team"], 2026, "FIFA World Cup")
                fa = registry.get_features(wc_row["away_team"], 2026, "FIFA World Cup")
                completed.at[i, "squad_top5_home"] = fh["top5_share"]
                completed.at[i, "squad_top5_away"] = fa["top5_share"]
                completed.at[i, "squad_caps_home"] = fh["avg_caps_norm"]
                completed.at[i, "squad_caps_away"] = fa["avg_caps_norm"]
            results = (
                pd.concat([results, completed], ignore_index=True)
                .sort_values("date")
                .reset_index(drop=True)
            )
            typer.echo(
                f"Injected {len(completed)} completed WC 2026 match(es) into training context.",
                err=True,
            )

    n_matches = len(results)
    typer.echo(f"Total context matches (incl. live WC 2026): {n_matches}.", err=True)

    # ------------------------------------------------------------------
    # 3. Fit model
    # ------------------------------------------------------------------
    typer.echo(f"Fitting model '{model}'...", err=True)
    try:
        fitted_model = _fit_model(model, results, checkpoint)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Error fitting model: {exc}", err=False)
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # 4. Filter schedule to upcoming matches only
    # ------------------------------------------------------------------
    if schedule.empty:
        typer.echo(
            "Could not load WC 2026 schedule (scrape failed or returned empty). "
            "Check your internet connection or run with --force-refresh.",
            err=False,
        )
        raise typer.Exit(1)

    if "is_completed" in schedule.columns:
        upcoming = schedule[~schedule["is_completed"]].reset_index(drop=True)
    else:
        upcoming = schedule.reset_index(drop=True)

    if upcoming.empty:
        typer.echo("No upcoming WC 2026 matches found in schedule.", err=False)
        raise typer.Exit(0)

    typer.echo(f"Found {len(upcoming)} upcoming match(es).", err=True)

    # ------------------------------------------------------------------
    # 4. Build synthetic match DataFrame for prediction
    # ------------------------------------------------------------------
    elo_ratings = get_current_ratings(results)
    match_df = _build_upcoming_match_df(upcoming, elo_ratings, registry)

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    try:
        pred_batch = fitted_model.predict_batch(match_df)
    except Exception as exc:
        typer.echo(f"Error during prediction: {exc}", err=False)
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # 6. Fetch live odds
    # ------------------------------------------------------------------
    has_api_key = bool(get_api_key())
    live_odds: list[dict] = []

    if has_api_key:
        typer.echo("Fetching live odds from the-odds-api.com...", err=True)
        live_odds = fetch_upcoming_odds()
        if live_odds:
            _save_odds_snapshot(live_odds)
    else:
        typer.echo(
            "THE_ODDS_API_KEY not set — showing model probabilities only (no market comparison).",
            err=True,
        )

    odds_index = _build_odds_index(live_odds)

    # ------------------------------------------------------------------
    # 7. Display table
    # ------------------------------------------------------------------
    typer.echo("")
    typer.echo("WC 2026 Upcoming Predictions")

    if has_api_key and live_odds:
        header = _header_with_odds()
        typer.echo(_separator(len(header)))
        typer.echo(header)
        typer.echo(_separator(len(header)))
    else:
        header = _header_no_odds()
        typer.echo(_separator(len(header)))
        typer.echo(header)
        typer.echo(_separator(len(header)))

    value_bets_shown = 0
    all_rows_shown = 0

    # Sort upcoming chronologically; keep pred_batch aligned by merging on position
    sort_order = upcoming["date"].argsort(kind="stable").values
    upcoming = upcoming.iloc[sort_order].reset_index(drop=True)
    pred_batch = pred_batch.iloc[sort_order].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7b. Apply injury / suspension adjustments (post-hoc λ scaling)
    # ------------------------------------------------------------------
    injury_data: dict[str, list[str]] = {}
    injury_losses: dict[int, tuple[float, float]] = {}  # i -> (loss_home, loss_away)

    if injuries:
        typer.echo("Fetching injury/suspension data...", err=True)
        injury_data = fetch_wc2026_injuries()
        squad_df = registry._cache.get("wc2026", pd.DataFrame())

        n_adjusted = 0
        for i, uprow in enumerate(upcoming.itertuples(index=False)):
            home, away = uprow.home_team, uprow.away_team
            absent_home = injury_data.get(home, [])
            absent_away = injury_data.get(away, [])

            loss_h = compute_injury_strength_loss(home, absent_home, squad_df)
            loss_a = compute_injury_strength_loss(away, absent_away, squad_df)
            injury_losses[i] = (loss_h, loss_a)

            if loss_h == 0.0 and loss_a == 0.0:
                continue

            lh = float(pred_batch["lambda_home"].iloc[i])
            la = float(pred_batch["lambda_away"].iloc[i])
            rho = float(pred_batch["rho"].iloc[i])
            lh_adj, la_adj, rho_adj = apply_injury_adjustment(
                lh, la, rho, loss_h, loss_a, k=injury_k
            )

            grid_adj = build_grid(lh_adj, la_adj, rho_adj)
            markets = derive_markets(grid_adj)
            goals = np.arange(grid_adj.shape[0], dtype=np.float64)

            pred_batch.at[i, "lambda_home"] = lh_adj
            pred_batch.at[i, "lambda_away"] = la_adj
            pred_batch.at[i, "prob_home"] = markets["home_win"]
            pred_batch.at[i, "prob_draw"] = markets["draw"]
            pred_batch.at[i, "prob_away"] = markets["away_win"]
            pred_batch.at[i, "expected_home"] = float(np.dot(goals, grid_adj.sum(axis=1)))
            pred_batch.at[i, "expected_away"] = float(np.dot(goals, grid_adj.sum(axis=0)))
            pred_batch.at[i, "grid"] = grid_adj
            n_adjusted += 1

        if n_adjusted:
            typer.echo(f"Injury adjustment applied to {n_adjusted} match(es).", err=True)
        else:
            typer.echo("No injuries/suspensions found for upcoming matches.", err=True)

    # ------------------------------------------------------------------
    # 7c. Apply LLM narrative form adjustment (post-hoc sentiment scaling)
    # ------------------------------------------------------------------
    if llm_form:
        unique_teams = sorted(
            {t for row in upcoming.itertuples(index=False) for t in (row.home_team, row.away_team)}
        )
        typer.echo(
            f"Running LLM form analysis ({llm_model}) for {len(unique_teams)} team(s)"
            " — this may take a few minutes...",
            err=True,
        )
        try:
            form_analyses = get_all_teams_form(unique_teams, model=llm_model)
        except ConnectionError as exc:
            typer.echo(f"[llm-form] Ollama unreachable: {exc} — skipping.", err=True)
            form_analyses = {}
        except Exception as exc:
            typer.echo(f"[llm-form] Error: {exc} — skipping.", err=True)
            form_analyses = {}

        if form_analyses:
            typer.echo("\n[LLM Form Analysis]", err=True)
            seen_teams: set[str] = set()
            for uprow in upcoming.itertuples(index=False):
                for team in (uprow.home_team, uprow.away_team):
                    if team in seen_teams:
                        continue
                    seen_teams.add(team)
                    fa = form_analyses.get(team)
                    if fa:
                        factor = compute_sentiment_factor(fa.form_score, fa.confidence)
                        typer.echo(
                            build_sentiment_report_line(
                                team, fa.form_score, fa.confidence, fa.key_absences, factor
                            ),
                            err=True,
                        )

            n_sentiment = 0
            for i, uprow in enumerate(upcoming.itertuples(index=False)):
                home, away = uprow.home_team, uprow.away_team
                fa_h = form_analyses.get(home)
                fa_a = form_analyses.get(away)
                score_h = fa_h.form_score if fa_h else 0.0
                conf_h = fa_h.confidence if fa_h else 0.0
                score_a = fa_a.form_score if fa_a else 0.0
                conf_a = fa_a.confidence if fa_a else 0.0

                lh = float(pred_batch["lambda_home"].iloc[i])
                la = float(pred_batch["lambda_away"].iloc[i])
                rho = float(pred_batch["rho"].iloc[i])
                lh_adj, la_adj, _ = apply_sentiment_adjustment(
                    lh, la, rho, score_h, score_a, conf_h, conf_a
                )

                if lh_adj == lh and la_adj == la:
                    continue

                grid_adj = build_grid(lh_adj, la_adj, rho)
                markets = derive_markets(grid_adj)
                goals = np.arange(grid_adj.shape[0], dtype=np.float64)

                pred_batch.at[i, "lambda_home"] = lh_adj
                pred_batch.at[i, "lambda_away"] = la_adj
                pred_batch.at[i, "prob_home"] = markets["home_win"]
                pred_batch.at[i, "prob_draw"] = markets["draw"]
                pred_batch.at[i, "prob_away"] = markets["away_win"]
                pred_batch.at[i, "expected_home"] = float(np.dot(goals, grid_adj.sum(axis=1)))
                pred_batch.at[i, "expected_away"] = float(np.dot(goals, grid_adj.sum(axis=0)))
                pred_batch.at[i, "grid"] = grid_adj
                n_sentiment += 1

            if n_sentiment:
                typer.echo(f"Sentiment adjustment applied to {n_sentiment} match(es).", err=True)
            else:
                typer.echo("No sentiment signal above confidence threshold.", err=True)

    for i, row in enumerate(upcoming.itertuples(index=False)):
        home = row.home_team
        away = row.away_team
        kickoff = row.date
        prob_home = float(pred_batch["prob_home"].iloc[i])
        prob_draw = float(pred_batch["prob_draw"].iloc[i])
        prob_away = float(pred_batch["prob_away"].iloc[i])

        if has_api_key and live_odds:
            odds_entry = _find_odds(home, away, odds_index)

            if odds_entry is None:
                # No odds found for this match — show model probs, mark as no market
                if show_all:
                    line = _format_row_no_odds(home, away, kickoff, prob_home, prob_draw, prob_away)
                    typer.echo(line + "  (no market odds)")
                    all_rows_shown += 1
                continue

            raw_odds_home = float(odds_entry["odds_home"])
            raw_odds_draw = float(odds_entry.get("odds_draw", 0.0))
            raw_odds_away = float(odds_entry["odds_away"])

            # Handle missing draw odds (two-outcome markets) gracefully
            if raw_odds_draw <= 0.0 or not np.isfinite(raw_odds_draw):
                if show_all:
                    line = _format_row_no_odds(home, away, kickoff, prob_home, prob_draw, prob_away)
                    typer.echo(line + "  (no draw odds)")
                    all_rows_shown += 1
                continue

            raw_odds_dict = {
                "home": raw_odds_home,
                "draw": raw_odds_draw,
                "away": raw_odds_away,
            }
            market_probs = remove_margin(raw_odds_dict)

            edge_home = prob_home - market_probs["home"]
            edge_draw = prob_draw - market_probs["draw"]
            edge_away = prob_away - market_probs["away"]

            edges = {
                "home": (edge_home, raw_odds_home),
                "draw": (edge_draw, raw_odds_draw),
                "away": (edge_away, raw_odds_away),
            }
            best_outcome = max(edges, key=lambda k: edges[k][0])
            best_edge, best_decimal_odds = edges[best_outcome]

            is_value = best_edge >= min_edge

            if best_decimal_odds > 1.0 and is_value:
                kelly = KELLY_FRACTION * best_edge / best_decimal_odds
            else:
                kelly = 0.0

            if not show_all and not is_value:
                continue

            line = _format_row_with_odds(
                home=home,
                away=away,
                kickoff=kickoff,
                prob_home=prob_home,
                prob_draw=prob_draw,
                prob_away=prob_away,
                mkt_home=market_probs["home"],
                mkt_draw=market_probs["draw"],
                mkt_away=market_probs["away"],
                best_edge=best_edge,
                kelly=kelly,
                is_value=is_value,
            )
            typer.echo(line)
            all_rows_shown += 1
            if is_value:
                value_bets_shown += 1

        else:
            # No API key — show model probs only
            line = _format_row_no_odds(home, away, kickoff, prob_home, prob_draw, prob_away)
            typer.echo(line)
            all_rows_shown += 1

    typer.echo(_separator(len(header)))

    # Summary footer
    if has_api_key and live_odds:
        if not show_all:
            typer.echo(
                f"\n{value_bets_shown} value bet(s) found (edge >= {min_edge * 100:.0f}pp). "
                f"Use --show-all to see all matches.",
                err=False,
            )
        else:
            typer.echo(
                f"\n{value_bets_shown} value bet(s) found (edge >= {min_edge * 100:.0f}pp).",
                err=False,
            )
        typer.echo(
            "* = value bet flagged.  Kelly is fractional (1/4 Kelly).  "
            "Market odds are margin-removed.",
            err=False,
        )
    else:
        typer.echo(
            "\nSet THE_ODDS_API_KEY to enable market comparison and value-bet detection.",
            err=False,
        )
