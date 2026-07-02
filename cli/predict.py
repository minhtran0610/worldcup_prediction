from __future__ import annotations

from pathlib import Path

import typer

from data.ingest.results import load_results
from eval.backtest import (
    KELLY_FRACTION,
    MIN_EDGE,
    MIN_MARKET_PROB,
    MIN_RELATIVE_EDGE,
    select_value_bet,
)
from eval.metrics import remove_margin
from features.context import derive_context
from features.elo import compute_elo_ratings
from models.dixon_coles import DixonColesModel
from models.elo_model import EloModel
from models.grid import expected_goals, top_scorelines

app = typer.Typer()

_STAGE_LABELS: dict[str, str] = {
    "group": "group",
    "knockout": "knockout",
}


def _context_line(home: str, away: str, neutral: bool, stage: str) -> str:
    venue = "neutral" if neutral else "home/away"
    if stage in _STAGE_LABELS and stage != "group":
        return f"{home} vs {away}  ({venue}, {_STAGE_LABELS[stage]})"
    return f"{home} vs {away}  ({venue})"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


@app.command()
def main(
    home: str = typer.Option(..., help="Home team name"),
    away: str = typer.Option(..., help="Away team name"),
    neutral: bool = typer.Option(True, help="Neutral venue (default True for WC)"),
    stage: str = typer.Option("group", help="Match stage: group or knockout"),
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV (if not cached)"),
    odds_home: float | None = typer.Option(None, help="Closing decimal odds for home win"),
    odds_draw: float | None = typer.Option(None, help="Closing decimal odds for draw"),
    odds_away: float | None = typer.Option(None, help="Closing decimal odds for away win"),
    model: str = typer.Option("neural", help="Model: 'neural', 'dc', or 'elo'"),
    checkpoint: Path = typer.Option(
        Path("checkpoints/neural.pt"), help="Neural model checkpoint path"
    ),
) -> None:
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)

    results = compute_elo_ratings(results)
    results = derive_context(results)

    n_matches = len(results)
    typer.echo(f"Fitting model on {n_matches} matches...", err=True)

    if model == "dc":
        dc = DixonColesModel()
        dc.fit(results)
        try:
            grid, markets = dc.predict(home_team=home, away_team=away, neutral=neutral)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=False)
            raise typer.Exit(1)

        exp_home, exp_away = expected_goals(grid)
        scores = top_scorelines(grid, n=5)

        typer.echo("")
        typer.echo(_context_line(home, away, neutral, stage))
        typer.echo("")
        typer.echo(f"  Expected goals:  {exp_home:.2f} - {exp_away:.2f}")
        typer.echo(
            f"  1X2:       Home {_format_pct(markets['home_win'])}"
            f"   Draw {_format_pct(markets['draw'])}"
            f"   Away {_format_pct(markets['away_win'])}"
        )
        typer.echo(
            f"  Over 2.5:  {_format_pct(markets['over_2_5'])}"
            f"   BTTS: {_format_pct(markets['btts_yes'])}"
        )

        score_parts = " | ".join(f"{h}-{a} {_format_pct(p)}" for h, a, p in scores)
        typer.echo(f"  Top scores:  {score_parts}")

        has_odds = odds_home is not None and odds_draw is not None and odds_away is not None

        if has_odds:
            raw_odds: dict[str, float] = {
                "home": odds_home,  # type: ignore[arg-type]
                "draw": odds_draw,  # type: ignore[arg-type]
                "away": odds_away,  # type: ignore[arg-type]
            }
            market_probs = remove_margin(raw_odds)

            edge_home = markets["home_win"] - market_probs["home"]
            edge_draw = markets["draw"] - market_probs["draw"]
            edge_away = markets["away_win"] - market_probs["away"]

            selection = select_value_bet(
                markets["home_win"],
                markets["draw"],
                markets["away_win"],
                market_probs["home"],
                market_probs["draw"],
                market_probs["away"],
                min_edge=MIN_EDGE,
                min_market_prob=MIN_MARKET_PROB,
                min_relative_edge=MIN_RELATIVE_EDGE,
            )
            best_outcome = selection["best_outcome"]
            best_edge = selection["best_edge"]
            is_value = selection["is_value"]
            best_decimal_odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}[
                best_outcome
            ]

            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )

            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100

            if is_value:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")

    elif model == "elo":
        elo = EloModel()
        elo.fit(results)
        try:
            probs = elo.predict_1x2(home_team=home, away_team=away, neutral=neutral)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=False)
            raise typer.Exit(1)

        typer.echo("")
        typer.echo(_context_line(home, away, neutral, stage))
        typer.echo("")
        typer.echo(
            f"  1X2:       Home {_format_pct(probs['home_win'])}"
            f"   Draw {_format_pct(probs['draw'])}"
            f"   Away {_format_pct(probs['away_win'])}"
        )

        has_odds = odds_home is not None and odds_draw is not None and odds_away is not None

        if has_odds:
            raw_odds = {
                "home": odds_home,  # type: ignore[arg-type]
                "draw": odds_draw,  # type: ignore[arg-type]
                "away": odds_away,  # type: ignore[arg-type]
            }
            market_probs = remove_margin(raw_odds)

            edge_home = probs["home_win"] - market_probs["home"]
            edge_draw = probs["draw"] - market_probs["draw"]
            edge_away = probs["away_win"] - market_probs["away"]

            selection = select_value_bet(
                probs["home_win"],
                probs["draw"],
                probs["away_win"],
                market_probs["home"],
                market_probs["draw"],
                market_probs["away"],
                min_edge=MIN_EDGE,
                min_market_prob=MIN_MARKET_PROB,
                min_relative_edge=MIN_RELATIVE_EDGE,
            )
            best_outcome = selection["best_outcome"]
            best_edge = selection["best_edge"]
            is_value = selection["is_value"]
            best_decimal_odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}[
                best_outcome
            ]

            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )

            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100

            if is_value:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")

    elif model == "neural":
        from features.squad_registry import SquadRegistry
        from models.neural import NeuralModel

        registry = SquadRegistry.build()
        results = derive_context(results, squad_registry=registry)

        if checkpoint.exists():
            m = NeuralModel.load(str(checkpoint))
            m._train_results = results
        else:
            typer.echo("No checkpoint found — fitting neural model from scratch...", err=True)
            m = NeuralModel()
            m.fit(results)

        import pandas as pd

        row = pd.DataFrame(
            [
                {
                    "home_team": home,
                    "away_team": away,
                    "date": pd.Timestamp.now(),
                    "neutral": neutral,
                    "rest_days_home": 7,
                    "rest_days_away": 7,
                    "elo_home_pre": results.loc[results["home_team"] == home, "elo_home_pre"].iloc[
                        -1
                    ]
                    if len(results[results["home_team"] == home]) > 0
                    else 1500.0,
                    "elo_away_pre": results.loc[results["away_team"] == away, "elo_away_pre"].iloc[
                        -1
                    ]
                    if len(results[results["away_team"] == away]) > 0
                    else 1500.0,
                    "is_knockout": stage == "knockout",
                    "is_host_home": False,
                    "is_host_away": False,
                    "squad_top5_home": 0.0,
                    "squad_top5_away": 0.0,
                    "squad_caps_home": 0.0,
                    "squad_caps_away": 0.0,
                    "home_score": 0,
                    "away_score": 0,
                    "tournament": "FIFA World Cup",
                }
            ]
        )
        feats_h = registry.get_features(home, 2026, "FIFA World Cup")
        feats_a = registry.get_features(away, 2026, "FIFA World Cup")
        row["squad_top5_home"] = feats_h["top5_share"]
        row["squad_top5_away"] = feats_a["top5_share"]
        row["squad_caps_home"] = feats_h["avg_caps_norm"]
        row["squad_caps_away"] = feats_a["avg_caps_norm"]

        pred = m.predict_batch(row)
        grid = pred["grid"].iloc[0]
        markets = {
            "home_win": float(pred["prob_home"].iloc[0]),
            "draw": float(pred["prob_draw"].iloc[0]),
            "away_win": float(pred["prob_away"].iloc[0]),
            "over_2_5": float(pred["prob_over_2_5"].iloc[0]),
            "btts_yes": float(pred["prob_btts"].iloc[0]),
        }
        exp_home, exp_away = expected_goals(grid)
        scores = top_scorelines(grid, n=5)

        typer.echo("")
        typer.echo(_context_line(home, away, neutral, stage))
        typer.echo("")
        typer.echo(f"  Expected goals:  {exp_home:.2f} - {exp_away:.2f}")
        typer.echo(
            f"  1X2:       Home {_format_pct(markets['home_win'])}"
            f"   Draw {_format_pct(markets['draw'])}"
            f"   Away {_format_pct(markets['away_win'])}"
        )
        typer.echo(
            f"  Over 2.5:  {_format_pct(markets['over_2_5'])}"
            f"   BTTS: {_format_pct(markets['btts_yes'])}"
        )
        score_parts = " | ".join(f"{h}-{a} {_format_pct(p)}" for h, a, p in scores)
        typer.echo(f"  Top scores:  {score_parts}")

        has_odds = odds_home is not None and odds_draw is not None and odds_away is not None
        if has_odds:
            raw_odds = {
                "home": odds_home,  # type: ignore[arg-type]
                "draw": odds_draw,  # type: ignore[arg-type]
                "away": odds_away,  # type: ignore[arg-type]
            }
            market_probs = remove_margin(raw_odds)
            edge_home = markets["home_win"] - market_probs["home"]
            edge_draw = markets["draw"] - market_probs["draw"]
            edge_away = markets["away_win"] - market_probs["away"]
            selection = select_value_bet(
                markets["home_win"],
                markets["draw"],
                markets["away_win"],
                market_probs["home"],
                market_probs["draw"],
                market_probs["away"],
                min_edge=MIN_EDGE,
                min_market_prob=MIN_MARKET_PROB,
                min_relative_edge=MIN_RELATIVE_EDGE,
            )
            best_outcome = selection["best_outcome"]
            best_edge = selection["best_edge"]
            is_value = selection["is_value"]
            best_decimal_odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}[
                best_outcome
            ]
            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )
            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100
            if is_value:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")

    else:
        typer.echo(f"Error: unknown model {model!r}. Use 'neural', 'dc', or 'elo'.", err=False)
        raise typer.Exit(1)
