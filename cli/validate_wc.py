"""Retroactive validation of model predictions against completed WC 2026 matches.

Usage:
    python -m cli.validate_wc [OPTIONS]

Splits completed WC matches into:
  - Form (before --form-cutoff): injected into training context
  - Eval (from --eval-from):     predicted and scored vs actual results

Metrics: RPS per match + aggregate, NLL, 1X2 accuracy, and value-bet ROI
if a pre-match odds snapshot exists at data/raw/wc2026_odds_snapshot.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

# Re-use helpers from the prediction CLI to stay DRY
from cli.wc2026 import (
    _build_upcoming_match_df,
    _fit_model,
    _normalise_odds_name,
)
from data.ingest.results import load_results
from data.ingest.wc2026 import load_wc2026_schedule
from eval.metrics import nll_scoreline, remove_margin, rps_1x2
from features.context import WC_2026_HOSTS, derive_context
from features.elo import compute_elo_ratings, extend_elo_through_matches, get_current_ratings
from features.squad_registry import SquadRegistry

app = typer.Typer()

ODDS_SNAPSHOT_PATH = Path("data/cache/wc2026_odds_snapshot.json")

# ── Column widths ────────────────────────────────────────────────────────────
_W_MATCH = 30
_W_PCT = 6
_W_SCORE = 7
_W_OUT = 5
_W_RPS = 6
_W_NLL = 6
_W_EDGE = 8
_W_WON = 4


def _outcome_str(home: int, away: int) -> str:
    if home > away:
        return "Home"
    if home == away:
        return "Draw"
    return "Away"


def _outcome_idx(home: int, away: int) -> int:
    if home > away:
        return 0
    if home == away:
        return 1
    return 2


def _inject_wc_form_matches(
    results: pd.DataFrame, form_matches: pd.DataFrame, registry: SquadRegistry
) -> pd.DataFrame:
    """Inject completed WC form matches into training context."""
    if form_matches.empty:
        return results

    fm = form_matches.copy()
    fm["home_score"] = fm["home_score"].astype(int)
    fm["away_score"] = fm["away_score"].astype(int)
    fm["date"] = fm["date"].fillna(pd.Timestamp("2026-06-11"))
    fm = fm.sort_values("date").reset_index(drop=True)
    fm["neutral"] = True
    fm["tournament"] = "FIFA World Cup"
    fm = extend_elo_through_matches(results, fm)
    fm["country"] = "United States"
    fm["is_knockout"] = False
    fm["is_host_home"] = fm["home_team"].isin(WC_2026_HOSTS)
    fm["is_host_away"] = fm["away_team"].isin(WC_2026_HOSTS)
    fm["rest_days_home"] = 7.0
    fm["rest_days_away"] = 7.0
    fm["sample_weight"] = 1.0
    for col in ["squad_top5_home", "squad_top5_away", "squad_caps_home", "squad_caps_away"]:
        fm[col] = 0.0
    for i, row in fm.iterrows():
        fh = registry.get_features(row["home_team"], 2026, "FIFA World Cup")
        fa = registry.get_features(row["away_team"], 2026, "FIFA World Cup")
        fm.at[i, "squad_top5_home"] = fh["top5_share"]
        fm.at[i, "squad_top5_away"] = fa["top5_share"]
        fm.at[i, "squad_caps_home"] = fh["avg_caps_norm"]
        fm.at[i, "squad_caps_away"] = fa["avg_caps_norm"]

    return pd.concat([results, fm], ignore_index=True).sort_values("date").reset_index(drop=True)


def _load_odds_snapshot() -> dict[tuple[str, str], dict]:
    """Load pre-match odds snapshot keyed by (norm_home, norm_away)."""
    if not ODDS_SNAPSHOT_PATH.exists():
        return {}
    try:
        with ODDS_SNAPSHOT_PATH.open() as f:
            raw: list[dict] = json.load(f)
        index: dict[tuple[str, str], dict] = {}
        for entry in raw:
            key = (
                _normalise_odds_name(entry["home_team"]),
                _normalise_odds_name(entry["away_team"]),
            )
            index[key] = entry
        return index
    except Exception as exc:
        typer.echo(f"[validate] Could not load odds snapshot: {exc}", err=True)
        return {}


def _find_snapshot_odds(home: str, away: str, index: dict[tuple[str, str], dict]) -> dict | None:
    key = (_normalise_odds_name(home), _normalise_odds_name(away))
    if key in index:
        return index[key]
    rev = (_normalise_odds_name(away), _normalise_odds_name(home))
    if rev in index:
        e = index[rev]
        return {
            **e,
            "odds_home": e["odds_away"],
            "odds_away": e["odds_home"],
            "home_team": home,
            "away_team": away,
        }
    return None


# ── Main command ─────────────────────────────────────────────────────────────


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    model: str = typer.Option("neural", help="Model: dc / xgb / neural / ensemble"),
    checkpoint: Path | None = typer.Option(None, help="Neural checkpoint path"),
    form_cutoff: str = typer.Option(
        "2026-06-18",
        help="Inject WC matches completed BEFORE this date as form context",
    ),
    eval_from: str | None = typer.Option(
        None,
        help="Evaluate matches from this date (default: same as --form-cutoff)",
    ),
    eval_to: str | None = typer.Option(
        None,
        help="Evaluate matches up to and including this date (default: all completed)",
    ),
    min_edge: float = typer.Option(0.02, help="Min edge (pp) to flag value bets"),
    force_refresh: bool = typer.Option(False, help="Force re-fetch WC schedule"),
) -> None:
    """Validate model on completed WC 2026 matches using earlier results as form."""

    cutoff_dt = pd.Timestamp(form_cutoff)
    eval_from_dt = pd.Timestamp(eval_from) if eval_from else cutoff_dt
    eval_to_dt = pd.Timestamp(eval_to) if eval_to else pd.Timestamp("2999-12-31")

    # ── 1. Training data ─────────────────────────────────────────────────────
    typer.echo("Loading historical results...", err=True)
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1)

    results = compute_elo_ratings(results)
    typer.echo("Building squad registry...", err=True)
    registry = SquadRegistry.build()
    results = derive_context(results, squad_registry=registry)

    # ── 2. WC schedule ───────────────────────────────────────────────────────
    typer.echo("Fetching WC 2026 schedule...", err=True)
    schedule = load_wc2026_schedule(force_refresh=force_refresh)

    if schedule.empty:
        typer.echo("Could not load WC 2026 schedule.")
        raise typer.Exit(1)

    completed = schedule[schedule["is_completed"]].dropna(subset=["home_score", "away_score"])

    form_matches = completed[completed["date"] < cutoff_dt].reset_index(drop=True)
    eval_matches = completed[
        (completed["date"] >= eval_from_dt) & (completed["date"] <= eval_to_dt)
    ].reset_index(drop=True)

    if eval_matches.empty:
        typer.echo(
            f"No completed eval matches found between {eval_from_dt.date()} "
            f"and {eval_to_dt.date()}."
        )
        raise typer.Exit(0)

    typer.echo(
        f"Form: {len(form_matches)} matches before {cutoff_dt.date()} | "
        f"Eval: {len(eval_matches)} matches ({eval_from_dt.date()} → {eval_to_dt.date()})",
        err=True,
    )

    # ── 3. Inject form into training context ─────────────────────────────────
    if not form_matches.empty:
        results = _inject_wc_form_matches(results, form_matches, registry)
        typer.echo(f"Injected {len(form_matches)} form matches into training context.", err=True)

    # ── 4. Fit model ─────────────────────────────────────────────────────────
    typer.echo(f"Fitting model '{model}'...", err=True)
    try:
        fitted = _fit_model(model, results, checkpoint)
    except Exception as exc:
        typer.echo(f"Error fitting model: {exc}")
        raise typer.Exit(1)

    # ── 5. Predict eval matches ───────────────────────────────────────────────
    elo_ratings = get_current_ratings(results)
    match_df = _build_upcoming_match_df(eval_matches, elo_ratings, registry)

    try:
        preds = fitted.predict_batch(match_df)
    except Exception as exc:
        typer.echo(f"Error during prediction: {exc}")
        raise typer.Exit(1)

    # ── 6. Load pre-match odds snapshot (optional) ────────────────────────────
    odds_index = _load_odds_snapshot()
    has_odds = bool(odds_index)
    if has_odds:
        typer.echo(
            f"Loaded odds snapshot ({len(odds_index)} entries) for market comparison.",
            err=True,
        )
    else:
        typer.echo(
            "No odds snapshot found — market comparison skipped. "
            "Run 'python -m cli.wc2026' before the next matchday to cache odds.",
            err=True,
        )

    # ── 7. Build per-match results table ─────────────────────────────────────
    rows: list[dict] = []
    for i, ev_row in enumerate(eval_matches.itertuples(index=False)):
        home = ev_row.home_team
        away = ev_row.away_team
        h_score = int(ev_row.home_score)
        a_score = int(ev_row.away_score)
        actual_out = _outcome_str(h_score, a_score)
        oidx = _outcome_idx(h_score, a_score)

        ph = float(preds["prob_home"].iloc[i])
        pd_ = float(preds["prob_draw"].iloc[i])
        pa = float(preds["prob_away"].iloc[i])
        probs = np.array([ph, pd_, pa])

        rps = rps_1x2(probs, oidx)
        grid = preds["grid"].iloc[i]
        nll = nll_scoreline(grid, h_score, a_score)

        # Market odds (from snapshot)
        odds_entry = _find_snapshot_odds(home, away, odds_index) if has_odds else None
        edge = float("nan")
        best_outcome = ""
        best_decimal = float("nan")
        mkt_home = mkt_draw = mkt_away = float("nan")
        if odds_entry and np.isfinite(odds_entry.get("odds_draw", float("nan"))):
            raw = {
                "home": float(odds_entry["odds_home"]),
                "draw": float(odds_entry["odds_draw"]),
                "away": float(odds_entry["odds_away"]),
            }
            mkt = remove_margin(raw)
            mkt_home, mkt_draw, mkt_away = mkt["home"], mkt["draw"], mkt["away"]
            edges = {
                "home": (ph - mkt_home, raw["home"]),
                "draw": (pd_ - mkt_draw, raw["draw"]),
                "away": (pa - mkt_away, raw["away"]),
            }
            best_outcome = max(edges, key=lambda k: edges[k][0])
            edge = edges[best_outcome][0]
            best_decimal = edges[best_outcome][1]

        rows.append(
            {
                "home": home,
                "away": away,
                "date": ev_row.date,
                "h_score": h_score,
                "a_score": a_score,
                "actual_out": actual_out,
                "outcome_idx": oidx,
                "prob_home": ph,
                "prob_draw": pd_,
                "prob_away": pa,
                "rps": rps,
                "nll": nll,
                "mkt_home": mkt_home,
                "mkt_draw": mkt_draw,
                "mkt_away": mkt_away,
                "best_edge_outcome": best_outcome,
                "best_edge": edge,
                "best_decimal": best_decimal,
            }
        )

    df = pd.DataFrame(rows)

    # ── 8. Display ────────────────────────────────────────────────────────────
    typer.echo("")

    title = f"WC 2026 Matchday Validation — form through {cutoff_dt.date()}, model: {model}"
    typer.echo(title)

    has_mkt = has_odds and df["best_edge"].notna().any()

    # Header
    hdr = (
        f"{'Match':<{_W_MATCH}} "
        f"{'H%':>{_W_PCT}} {'D%':>{_W_PCT}} {'A%':>{_W_PCT}}  "
        f"{'Score':>{_W_SCORE}}  {'Out':{_W_OUT}}  {'RPS':>{_W_RPS}}  {'NLL':>{_W_NLL}}"
    )
    if has_mkt:
        hdr += (
            f"  {'MktH%':>{_W_PCT}} {'MktD%':>{_W_PCT}} {'MktA%':>{_W_PCT}}  "
            f"{'Edge':>{_W_EDGE}}  {'Val?':>{_W_WON}}"
        )
    sep = "─" * len(hdr)
    typer.echo(sep)
    typer.echo(hdr)
    typer.echo(sep)

    for r in rows:
        label = f"{r['home']} vs {r['away']}"
        if len(label) > _W_MATCH:
            label = label[: _W_MATCH - 1] + "…"
        score_str = f"{r['h_score']}-{r['a_score']}"
        line = (
            f"{label:<{_W_MATCH}} "
            f"{r['prob_home'] * 100:>{_W_PCT}.1f} "
            f"{r['prob_draw'] * 100:>{_W_PCT}.1f} "
            f"{r['prob_away'] * 100:>{_W_PCT}.1f}  "
            f"{score_str:>{_W_SCORE}}  "
            f"{r['actual_out']:<{_W_OUT}}  "
            f"{r['rps']:>{_W_RPS}.3f}  "
            f"{r['nll']:>{_W_NLL}.3f}"
        )
        if has_mkt and np.isfinite(r["best_edge"]):
            is_value = r["best_edge"] >= min_edge
            val_str = "  * " if is_value else "    "
            line += (
                f"  {r['mkt_home'] * 100:>{_W_PCT}.1f} "
                f"{r['mkt_draw'] * 100:>{_W_PCT}.1f} "
                f"{r['mkt_away'] * 100:>{_W_PCT}.1f}  "
                f"{r['best_edge'] * 100:>+{_W_EDGE}.1f}pp"
                f"{val_str}"
            )
        elif has_mkt:
            line += "  " + " " * (_W_PCT * 3 + 6 + _W_EDGE + _W_WON + 4)
        typer.echo(line)

    typer.echo(sep)

    # ── 9. Aggregate metrics ──────────────────────────────────────────────────
    n = len(df)
    mean_rps = float(df["rps"].mean())
    mean_nll = float(df["nll"].mean())

    # 1X2 accuracy: did the highest-prob outcome win?
    predicted_out = df[["prob_home", "prob_draw", "prob_away"]].values.argmax(axis=1)
    correct = int((predicted_out == df["outcome_idx"].values).sum())

    out_counts = df["actual_out"].value_counts()
    home_n = out_counts.get("Home", 0)
    draw_n = out_counts.get("Draw", 0)
    away_n = out_counts.get("Away", 0)

    typer.echo("")
    typer.echo(f"Aggregate  (N={n})")
    typer.echo(f"  Mean RPS :  {mean_rps:.4f}  (uniform baseline: 0.1667)")
    typer.echo(f"  Mean NLL :  {mean_nll:.4f}")
    typer.echo(f"  1X2 correct:  {correct}/{n}  ({correct / n * 100:.1f}%)")
    typer.echo(f"  Outcomes actual: {home_n}H  {draw_n}D  {away_n}A")

    if has_mkt:
        value_rows = df[df["best_edge"].notna() & (df["best_edge"] >= min_edge)]
        typer.echo("")
        typer.echo(f"Value bets flagged (edge ≥ {min_edge * 100:.0f}pp):  {len(value_rows)}/{n}")

        if not value_rows.empty:
            typer.echo("")
            typer.echo(
                f"  {'Match':<{_W_MATCH}} {'Bet on':<6} {'Edge':>8}  {'Won?':>4}  {'FlatRet':>8}"
            )
            typer.echo("  " + "─" * 60)
            flat_returns: list[float] = []
            for _, vr in value_rows.iterrows():
                won = vr["actual_out"].lower() == vr["best_edge_outcome"]
                flat_ret = vr["best_decimal"] - 1.0 if won else -1.0
                flat_returns.append(flat_ret)
                label = f"{vr['home']} vs {vr['away']}"
                if len(label) > _W_MATCH:
                    label = label[: _W_MATCH - 1] + "…"
                typer.echo(
                    f"  {label:<{_W_MATCH}} "
                    f"{vr['best_edge_outcome']:<6} "
                    f"{vr['best_edge'] * 100:>+8.1f}pp  "
                    f"{'Yes' if won else 'No ':>4}  "
                    f"{flat_ret:>+8.2f}"
                )
            flat_roi = float(np.mean(flat_returns))
            won_n = sum(1 for r in flat_returns if r > 0)
            typer.echo("  " + "─" * 60)
            typer.echo(
                f"  Win rate: {won_n}/{len(flat_returns)}  "
                f"Flat ROI per bet: {flat_roi:+.3f}  "
                f"({'profit' if flat_roi > 0 else 'loss'})"
            )


if __name__ == "__main__":
    app()
