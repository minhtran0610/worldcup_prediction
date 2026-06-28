from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from data.ingest.injuries import fetch_wc2026_injuries
from data.ingest.llm_form import get_all_matches_form
from data.ingest.odds_live import fetch_upcoming_odds, get_api_key
from data.ingest.results import drop_wc2026, load_results
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
_COL_BET = 5  # "Home" / "Draw" / "Away"
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
        f"{'Edge':>{_COL_EDGE}}  {'Bet':<{_COL_BET}}  {'Kelly':>{_COL_KELLY}}"
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
    best_outcome: str,
    kelly: float,
    is_value: bool,
) -> str:
    ko = _ko_str(kickoff)
    label = _match_label(home, away)
    edge_str = f"{best_edge * 100:+.1f}pp"
    bet_str = best_outcome.capitalize() if is_value else ""
    kelly_str = f"{kelly * 100:.1f}%" if is_value else ""
    value_marker = " *" if is_value else "  "
    return (
        f"{ko:<{_COL_KO}} "
        f"{label:<{_COL_MATCH}} "
        f"{_pct(prob_home):>{_COL_PROB}} {_pct(prob_draw):>{_COL_PROB}} {_pct(prob_away):>{_COL_PROB}}  "
        f"{_pct(mkt_home):>{_COL_PROB}} {_pct(mkt_draw):>{_COL_PROB}} {_pct(mkt_away):>{_COL_PROB}}  "
        f"{edge_str:>{_COL_EDGE}}{value_marker}{bet_str:<{_COL_BET}}  {kelly_str:>{_COL_KELLY}}"
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
# Telegram formatter
# ---------------------------------------------------------------------------


def _format_telegram_match(
    home: str,
    away: str,
    kickoff: pd.Timestamp,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    lambda_home: float,
    lambda_away: float,
    market_detail: dict,
    market_probs: dict | None,
    best_edge: float | None,
    best_outcome: str | None,
    best_decimal_odds: float | None,
    kelly: float | None,
    is_value: bool,
    fa_home,
    fa_away,
    min_edge: float,
) -> str:
    parts: list[str] = []
    parts.append(f"⚽ <b>{home} vs {away}</b>  ·  {_ko_str(kickoff)} UTC")

    # Form
    if fa_home is not None or fa_away is not None:
        parts.append("")
        parts.append("<b>📰 Form</b>")
        for team, fa in ((home, fa_home), (away, fa_away)):
            if fa is None:
                parts.append(f"⬜ <b>{team}</b>  no data")
                continue
            emoji = "🟢" if fa.form_score > 0.15 else ("🔴" if fa.form_score < -0.15 else "🟡")
            parts.append(f"{emoji} <b>{team}</b>  {fa.form_score:+.2f} ({fa.confidence:.0%} conf)")
            if fa.key_absences:
                parts.append(f"    ⚠️ Out: {', '.join(fa.key_absences[:3])}")
            ctx = fa.performance_context or (
                f"({fa.n_articles} article(s) — no narrative extracted)"
                if fa.n_articles > 0
                else "(no recent articles found)"
            )
            parts.append(f"    {ctx}")

    # Probabilities
    parts.append("")
    parts.append("<b>📊 Model vs Market</b>" if market_probs else "<b>📊 Probabilities</b>")
    for key, label, our_p in (
        ("home", f"Home ({home})", prob_home),
        ("draw", "Draw", prob_draw),
        ("away", f"Away ({away})", prob_away),
    ):
        if market_probs:
            mkt_p = market_probs[key]
            edge = our_p - mkt_p
            flag = "  ✅ VALUE" if (key == best_outcome and is_value) else ""
            parts.append(f"  {label:<22} {our_p:.1%}  mkt {mkt_p:.1%}  ({edge:+.1%}){flag}")
        else:
            parts.append(f"  {label:<22} {our_p:.1%}")
    parts.append(f"  xG: {lambda_home:.2f} – {lambda_away:.2f}")

    # Value bet
    parts.append("")
    if is_value and best_outcome and kelly is not None:
        names = {"home": home, "draw": "Draw", "away": away}
        odds_str = (
            f" @ {best_decimal_odds:.2f}" if best_decimal_odds and best_decimal_odds > 1 else ""
        )
        parts.append(
            f"💡 <b>Bet: {names[best_outcome]}{odds_str}</b>"
            f"  ·  Edge {best_edge:+.1%}"
            f"  ·  ¼Kelly {kelly:.1%}"
        )
    elif market_probs:
        parts.append(f"No value bet (threshold {min_edge:.0%})")

    # Other markets
    m = market_detail
    parts.append("")
    parts.append("<b>📈 Other Markets</b>")
    parts.append(f"  Over 2.5: {m['over_2_5']:.0%}   BTTS: {m['btts_yes']:.0%}")
    ah_m05, ah_p05 = m["ah_home_m0_5"], m["ah_home_p0_5"]
    parts.append(f"  AH {home[:14]:<14}  −0.5: {ah_m05:.0%}   +0.5: {ah_p05:.0%}")
    parts.append(f"  AH {away[:14]:<14}  −0.5: {1 - ah_p05:.0%}   +0.5: {1 - ah_m05:.0%}")

    return "\n".join(parts)


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
            feats_h = {"top5_share": 0.0, "avg_caps_norm": 0.0, "intl_goals_per_cap": 0.0}
            feats_a = {"top5_share": 0.0, "avg_caps_norm": 0.0, "intl_goals_per_cap": 0.0}

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
                "squad_goals_home": feats_h["intl_goals_per_cap"],
                "squad_goals_away": feats_a["intl_goals_per_cap"],
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
) -> dict | None:
    """Look up odds for a match.  Falls back to reversed team order."""
    key = (_normalise_odds_name(home), _normalise_odds_name(away))
    if key in odds_index:
        return odds_index[key]
    # Sometimes the API lists the match with teams swapped — return full entry
    # with 1X2 and spreads swapped so all callers see home=home, away=away.
    rev_key = (_normalise_odds_name(away), _normalise_odds_name(home))
    if rev_key in odds_index:
        entry = odds_index[rev_key]
        raw_spreads = entry.get("spreads")
        spreads_list: list = raw_spreads if isinstance(raw_spreads, list) else []
        swapped_spreads = [
            {
                "home_handicap": -s["away_handicap"],
                "home_odds": s["away_odds"],
                "away_handicap": -s["home_handicap"],
                "away_odds": s["home_odds"],
            }
            for s in spreads_list
        ]
        return {
            **entry,
            "home_team": home,
            "away_team": away,
            "odds_home": entry["odds_away"],
            "odds_away": entry["odds_home"],
            "spreads": swapped_spreads,
            # totals (O/U goals) are symmetric — no swap needed
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
# Completed-match injection helper
# ---------------------------------------------------------------------------


def _inject_completed_wc_matches(
    results: pd.DataFrame,
    completed: pd.DataFrame,
    registry: SquadRegistry,
) -> pd.DataFrame:
    """Inject all completed WC matches into training context and return updated results."""
    to_inject = completed.copy()
    if to_inject.empty:
        return results

    to_inject["date"] = to_inject["date"].fillna(pd.Timestamp("2026-06-11"))
    to_inject = to_inject.sort_values("date").reset_index(drop=True)
    to_inject["neutral"] = True
    to_inject["tournament"] = "FIFA World Cup"
    to_inject = extend_elo_through_matches(results, to_inject)
    to_inject["country"] = "United States"
    to_inject["is_knockout"] = True
    to_inject["is_host_home"] = to_inject["home_team"].isin(WC_2026_HOSTS)
    to_inject["is_host_away"] = to_inject["away_team"].isin(WC_2026_HOSTS)
    to_inject["rest_days_home"] = 7.0
    to_inject["rest_days_away"] = 7.0
    to_inject["sample_weight"] = 1.0
    for col in [
        "squad_top5_home",
        "squad_top5_away",
        "squad_caps_home",
        "squad_caps_away",
        "squad_goals_home",
        "squad_goals_away",
    ]:
        to_inject[col] = 0.0
    for i, wc_row in to_inject.iterrows():
        fh = registry.get_features(wc_row["home_team"], 2026, "FIFA World Cup")
        fa = registry.get_features(wc_row["away_team"], 2026, "FIFA World Cup")
        to_inject.at[i, "squad_top5_home"] = fh["top5_share"]
        to_inject.at[i, "squad_top5_away"] = fa["top5_share"]
        to_inject.at[i, "squad_caps_home"] = fh["avg_caps_norm"]
        to_inject.at[i, "squad_caps_away"] = fa["avg_caps_norm"]
        to_inject.at[i, "squad_goals_home"] = fh["intl_goals_per_cap"]
        to_inject.at[i, "squad_goals_away"] = fa["intl_goals_per_cap"]

    return (
        pd.concat([results, to_inject], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Odds snapshot (persists pre-match market odds for later validation)
# ---------------------------------------------------------------------------

_ODDS_SNAPSHOT_PATH: Path = Path("data/cache/wc2026_odds_snapshot.json")


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


_FORM_LOG_PATH: Path = Path("data/cache/llm_form_log.jsonl")


def _log_form_reads(form_analyses: dict) -> None:
    """Append each team's pre-match LLM form read to an on-disk JSONL log.

    One record per team per run, stamped with the run time. This is the live
    validation trail: narrative form can't be backtested (no historical RSS), so
    we accumulate the LLM's *pre-match* reads here and later compare them against
    actual results to judge whether the signal earns its weight. Best-effort —
    write failures are swallowed so they never break a prediction run.
    """
    ts = datetime.now(tz=UTC).isoformat()
    try:
        _FORM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _FORM_LOG_PATH.open("a") as f:
            for team, fa in form_analyses.items():
                rec = {
                    "logged_at": ts,
                    "team": team,
                    "form_score": fa.form_score,
                    "confidence": fa.confidence,
                    "key_absences": fa.key_absences,
                    "performance_context": fa.performance_context,
                    "n_articles": fa.n_articles,
                }
                f.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:
        typer.echo(f"[wc2026] WARNING: could not write form log — {exc}", err=True)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    model: str = typer.Option("neural", help="Model: 'dc', 'xgb', 'neural', 'ensemble'"),
    checkpoint: Path | None = typer.Option(
        Path("checkpoints/neural.pt"),
        help="Neural checkpoint path (for neural/ensemble); loaded if present, else fits fresh",
    ),
    min_edge: float = typer.Option(0.02, help="Minimum edge to flag as value bet"),
    show_all: bool = typer.Option(False, help="Show all matches, not just value bets"),
    injuries: bool = typer.Option(True, help="Apply injury/suspension λ adjustment"),
    injury_k: float = typer.Option(0.5, help="Injury dampening coefficient K (0–1)"),
    llm_form: bool = typer.Option(
        True,
        "--llm-form/--no-llm-form",
        help="Apply LLM narrative tournament-form adjustment (~2 min). On by default.",
    ),
    llm_model: str = typer.Option("qwen3.5:9b", help="Ollama model for LLM form analysis"),
    next_only: bool = typer.Option(
        False,
        "--next/--all",
        help="Predict only the next matchday's matches (fastest; LLM reads freshest articles).",
    ),
    force_odds: bool = typer.Option(
        False,
        "--force-odds",
        help="Bypass the 15-minute odds cache and fetch fresh odds from the API (costs credits).",
    ),
    telegram: bool = typer.Option(
        False,
        "--telegram",
        help="Output human-readable Telegram HTML instead of ASCII tables (used by the scheduler bot).",
    ),
) -> None:
    """Predict all upcoming WC 2026 matches and compare vs live odds.

    Workflow:
      1. Load results, compute Elo, derive context
      2. Fit the selected model on all historical data
      3. Load the WC 2026 schedule; inject completed matches as training context
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

    # Drop any WC 2026 matches that have leaked into the Kaggle cache — the live
    # schedule injected below is the single source of truth for them, so keeping
    # the cached copies would double-count in the Elo forward pass and form
    # sequences.
    n_before = len(results)
    results = drop_wc2026(results)
    n_dropped = n_before - len(results)
    if n_dropped:
        typer.echo(
            f"Dropped {n_dropped} cached WC 2026 match(es); using live schedule as source.",
            err=True,
        )

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
            n_before_injection = len(results)
            results = _inject_completed_wc_matches(results, completed, registry)
            typer.echo(
                f"Injected {len(results) - n_before_injection} completed WC 2026 match(es)"
                " into training context.",
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
    # 4b. Optionally filter to next matchday only
    # ------------------------------------------------------------------
    if next_only:
        next_ko = upcoming["date"].dropna().min()
        # Keep matches kicking off within 30 min of the earliest kickoff — captures
        # simultaneous group-stage pairs; excludes later same-day matches.
        upcoming = upcoming[upcoming["date"] <= next_ko + pd.Timedelta(minutes=30)].reset_index(
            drop=True
        )
        typer.echo(
            f"--next: {len(upcoming)} match(es) kicking off at {_ko_str(next_ko)} UTC.", err=True
        )

    # ------------------------------------------------------------------
    # 4c. Build synthetic match DataFrame for prediction
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
        live_odds = fetch_upcoming_odds(force_refresh=force_odds)
        if live_odds:
            _save_odds_snapshot(live_odds)
    else:
        typer.echo(
            "THE_ODDS_API_KEY not set — showing model probabilities only (no market comparison).",
            err=True,
        )

    odds_index = _build_odds_index(live_odds)

    # ------------------------------------------------------------------
    # 7. Sort and apply post-hoc adjustments
    # All processing happens here (stderr only). stdout is flushed at end.
    # ------------------------------------------------------------------
    sort_order = upcoming["date"].argsort(kind="stable").values
    upcoming = upcoming.iloc[sort_order].reset_index(drop=True)
    pred_batch = pred_batch.iloc[sort_order].reset_index(drop=True)

    # 7b. Injury / suspension adjustments
    injury_data: dict[str, list[str]] = {}

    if injuries:
        typer.echo("Fetching injury/suspension data...", err=True)
        predict_teams = list(
            {t for row in upcoming.itertuples(index=False) for t in (row.home_team, row.away_team)}
        )
        injury_data = fetch_wc2026_injuries(teams=predict_teams)
        squad_df = registry._cache.get("wc2026", pd.DataFrame())

        n_adjusted = 0
        for i, uprow in enumerate(upcoming.itertuples(index=False)):
            home, away = uprow.home_team, uprow.away_team
            absent_home = injury_data.get(home, [])
            absent_away = injury_data.get(away, [])

            loss_h = compute_injury_strength_loss(home, absent_home, squad_df)
            loss_a = compute_injury_strength_loss(away, absent_away, squad_df)

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

    # 7c. LLM narrative form adjustment
    # form_notes collected here; printed to stdout in the display block below.
    form_notes: list[str] = []
    form_analyses: dict = {}

    if llm_form:
        match_pairs = [(row.home_team, row.away_team) for row in upcoming.itertuples(index=False)]
        typer.echo(
            f"Running LLM form analysis ({llm_model}) for {len(match_pairs)} match(es)"
            " — this may take a few minutes...",
            err=True,
        )
        try:
            form_analyses = get_all_matches_form(match_pairs, model=llm_model)
        except ConnectionError as exc:
            typer.echo(f"[llm-form] Ollama unreachable: {exc} — skipping.", err=True)
        except Exception as exc:
            typer.echo(f"[llm-form] Error: {exc} — skipping.", err=True)

        if form_analyses:
            _log_form_reads(form_analyses)
            seen_teams: set[str] = set()
            for uprow in upcoming.itertuples(index=False):
                for team in (uprow.home_team, uprow.away_team):
                    if team in seen_teams:
                        continue
                    seen_teams.add(team)
                    fa = form_analyses.get(team)
                    if fa:
                        factor = compute_sentiment_factor(fa.form_score, fa.confidence)
                        report = build_sentiment_report_line(
                            team, fa.form_score, fa.confidence, fa.key_absences, factor
                        )
                        typer.echo(report, err=True)
                        form_notes.append(report)
                        ctx = fa.performance_context or (
                            f"({fa.n_articles} article(s) read — no narrative extracted)"
                            if fa.n_articles > 0
                            else "(no recent articles found)"
                        )
                        form_notes.append(f"    {ctx}")

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

    # ------------------------------------------------------------------
    # 8. Build output rows and market detail blocks (no printing yet)
    # ------------------------------------------------------------------
    output_lines: list[str] = []
    market_detail_lines: list[str] = []
    value_bets_shown = 0
    all_rows_shown = 0

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
                if show_all:
                    line = _format_row_no_odds(home, away, kickoff, prob_home, prob_draw, prob_away)
                    output_lines.append(line + "  (no market odds)")
                    all_rows_shown += 1
                continue

            raw_odds_home = float(odds_entry["odds_home"])
            raw_odds_draw = float(odds_entry.get("odds_draw", 0.0))
            raw_odds_away = float(odds_entry["odds_away"])

            if raw_odds_draw <= 0.0 or not np.isfinite(raw_odds_draw):
                if show_all:
                    line = _format_row_no_odds(home, away, kickoff, prob_home, prob_draw, prob_away)
                    output_lines.append(line + "  (no draw odds)")
                    all_rows_shown += 1
                continue

            raw_odds_dict = {
                "home": raw_odds_home,
                "draw": raw_odds_draw,
                "away": raw_odds_away,
            }
            market_probs = remove_margin(raw_odds_dict)

            edges = {
                "home": (prob_home - market_probs["home"], raw_odds_home),
                "draw": (prob_draw - market_probs["draw"], raw_odds_draw),
                "away": (prob_away - market_probs["away"], raw_odds_away),
            }
            best_outcome = max(edges, key=lambda k: edges[k][0])
            best_edge, best_decimal_odds = edges[best_outcome]

            is_value = best_edge >= min_edge

            kelly = (
                KELLY_FRACTION * best_edge / best_decimal_odds
                if is_value and best_decimal_odds > 1.0
                else 0.0
            )

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
                best_outcome=best_outcome,
                kelly=kelly,
                is_value=is_value,
            )
            output_lines.append(line)
            all_rows_shown += 1
            if is_value:
                value_bets_shown += 1

        else:
            line = _format_row_no_odds(home, away, kickoff, prob_home, prob_draw, prob_away)
            output_lines.append(line)
            all_rows_shown += 1

        # Build per-match market detail block (O/U, BTTS, AH — model probabilities only)
        grid = pred_batch["grid"].iloc[i]
        m = derive_markets(grid)
        label = _match_label(home, away)
        ko = _ko_str(kickoff)

        market_detail_lines.append(f"{label}  ·  {ko} UTC")
        market_detail_lines.append(
            f"  O/U  0.5 {m['over_0_5'] * 100:.0f}%↑"
            f"  1.5 {m['over_1_5'] * 100:.0f}%↑"
            f"  2.5 {m['over_2_5'] * 100:.0f}%↑"
            f"  3.5 {m['over_3_5'] * 100:.0f}%↑"
            f"  4.5 {m['over_4_5'] * 100:.0f}%↑"
            f"   BTTS {m['btts_yes'] * 100:.0f}%"
        )
        market_detail_lines.append(
            f"  AH {home[:14]:<14}"
            f"  -1.5:{m['ah_home_m1_5'] * 100:>3.0f}%"
            f"  -0.5:{m['ah_home_m0_5'] * 100:>3.0f}%"
            f"  +0.5:{m['ah_home_p0_5'] * 100:>3.0f}%"
            f"  +1.5:{m['ah_home_p1_5'] * 100:>3.0f}%"
        )
        market_detail_lines.append(
            f"  AH {away[:14]:<14}"
            f"  -1.5:{(1 - m['ah_home_p1_5']) * 100:>3.0f}%"
            f"  -0.5:{m['away_win'] * 100:>3.0f}%"
            f"  +0.5:{(1 - m['ah_home_m0_5']) * 100:>3.0f}%"
            f"  +1.5:{(1 - m['ah_home_m1_5']) * 100:>3.0f}%"
        )
        market_detail_lines.append("")

    # ------------------------------------------------------------------
    # 8b. Build Telegram blocks (only when --telegram flag is set)
    # ------------------------------------------------------------------
    telegram_blocks: list[str] = []

    if telegram:
        for i, row in enumerate(upcoming.itertuples(index=False)):
            home = row.home_team
            away = row.away_team
            kickoff = row.date
            prob_home = float(pred_batch["prob_home"].iloc[i])
            prob_draw = float(pred_batch["prob_draw"].iloc[i])
            prob_away = float(pred_batch["prob_away"].iloc[i])
            lambda_home = float(pred_batch["lambda_home"].iloc[i])
            lambda_away = float(pred_batch["lambda_away"].iloc[i])
            grid = pred_batch["grid"].iloc[i]
            m = derive_markets(grid)

            _market_probs = None
            _best_edge: float | None = None
            _best_outcome: str | None = None
            _best_decimal_odds: float | None = None
            _kelly: float | None = None
            _is_value = False

            if has_api_key and live_odds:
                odds_entry = _find_odds(home, away, odds_index)
                if odds_entry is not None:
                    raw_h = float(odds_entry["odds_home"])
                    raw_d = float(odds_entry.get("odds_draw", 0.0))
                    raw_a = float(odds_entry["odds_away"])
                    if raw_d > 0.0 and np.isfinite(raw_d):
                        raw_odds_dict = {"home": raw_h, "draw": raw_d, "away": raw_a}
                        _market_probs = remove_margin(raw_odds_dict)
                        edges = {
                            "home": (prob_home - _market_probs["home"], raw_h),
                            "draw": (prob_draw - _market_probs["draw"], raw_d),
                            "away": (prob_away - _market_probs["away"], raw_a),
                        }
                        _best_outcome = max(edges, key=lambda k: edges[k][0])
                        _best_edge, _best_decimal_odds = edges[_best_outcome]
                        _is_value = _best_edge >= min_edge
                        _kelly = (
                            KELLY_FRACTION * _best_edge / _best_decimal_odds
                            if _is_value and _best_decimal_odds > 1.0
                            else 0.0
                        )

            block = _format_telegram_match(
                home=home,
                away=away,
                kickoff=kickoff,
                prob_home=prob_home,
                prob_draw=prob_draw,
                prob_away=prob_away,
                lambda_home=lambda_home,
                lambda_away=lambda_away,
                market_detail=m,
                market_probs=_market_probs,
                best_edge=_best_edge,
                best_outcome=_best_outcome,
                best_decimal_odds=_best_decimal_odds,
                kelly=_kelly,
                is_value=_is_value,
                fa_home=form_analyses.get(home),
                fa_away=form_analyses.get(away),
                min_edge=min_edge,
            )
            telegram_blocks.append(block)

    # ------------------------------------------------------------------
    # 9. Flush all stdout at once
    # ------------------------------------------------------------------
    if telegram:
        typer.echo(("\n\n" + "—" * 32 + "\n\n").join(telegram_blocks))
        return

    header = _header_with_odds() if (has_api_key and live_odds) else _header_no_odds()
    sep = _separator(len(header))

    typer.echo("\nWC 2026 Upcoming Predictions")

    if form_notes:
        typer.echo("\nForm")
        typer.echo("-" * 60)
        for note in form_notes:
            typer.echo(note)

    typer.echo("")
    typer.echo(sep)
    typer.echo(header)
    typer.echo(sep)
    for line in output_lines:
        typer.echo(line)
    typer.echo(sep)

    if has_api_key and live_odds:
        suffix = "" if show_all else "  Use --show-all to see all matches."
        typer.echo(
            f"\n{value_bets_shown} value bet(s) found (edge >= {min_edge * 100:.0f}pp).{suffix}"
        )
        typer.echo(
            "* = value bet.  Bet column shows which outcome.  Kelly is 1/4 Kelly.  Market odds margin-removed."
        )
    else:
        typer.echo("\nSet THE_ODDS_API_KEY to enable market comparison and value-bet detection.")

    if market_detail_lines:
        typer.echo("\nMarkets  (model probabilities, no margin)")
        typer.echo("-" * 72)
        for line in market_detail_lines:
            typer.echo(line)
