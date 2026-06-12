from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from data.ingest.results import load_results
from features.context import derive_context
from features.elo import compute_elo_ratings
from models.dixon_coles import DixonColesModel
from models.elo_model import EloModel

app = typer.Typer()


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    output: Path = typer.Option(Path("predictions.csv"), help="Output CSV path"),
    model: str = typer.Option("dc", help="Model: 'dc' or 'elo'"),
    tournament: str | None = typer.Option(None, help="Filter by tournament name"),
    since: str | None = typer.Option(None, help="Only matches on or after this date (YYYY-MM-DD)"),
) -> None:
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)

    results = compute_elo_ratings(results)
    results = derive_context(results)

    filter_mask = pd.Series([True] * len(results), index=results.index)

    if tournament is not None:
        filter_mask &= results["tournament"] == tournament

    if since is not None:
        since_ts = pd.Timestamp(since)
        filter_mask &= results["date"] >= since_ts

    train_df = results[~filter_mask].reset_index(drop=True)
    predict_df = results[filter_mask].reset_index(drop=True)

    if predict_df.empty:
        typer.echo("No matches match the specified filters.")
        raise typer.Exit(1)

    if model == "dc":
        dc = DixonColesModel()
        dc.fit(train_df)
        try:
            pred_batch = dc.predict_batch(predict_df)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=False)
            raise typer.Exit(1)

        from models.grid import derive_markets  # noqa: PLC0415

        over_2_5_vals: list[float] = []
        btts_yes_vals: list[float] = []
        for g in pred_batch["grid"]:
            m = derive_markets(g)
            over_2_5_vals.append(m["over_2_5"])
            btts_yes_vals.append(m["btts_yes"])

        out_df = pd.DataFrame(
            {
                "date": predict_df["date"].dt.date,
                "home_team": predict_df["home_team"],
                "away_team": predict_df["away_team"],
                "prob_home": pred_batch["prob_home"],
                "prob_draw": pred_batch["prob_draw"],
                "prob_away": pred_batch["prob_away"],
                "over_2_5": over_2_5_vals,
                "btts_yes": btts_yes_vals,
            }
        )

    elif model == "elo":
        elo = EloModel()
        elo.fit(train_df)
        try:
            pred_batch = elo.predict_batch(predict_df)
        except ValueError as exc:
            typer.echo(f"Error: {exc}", err=False)
            raise typer.Exit(1)

        out_df = pd.DataFrame(
            {
                "date": predict_df["date"].dt.date,
                "home_team": predict_df["home_team"],
                "away_team": predict_df["away_team"],
                "prob_home": pred_batch["prob_home"],
                "prob_draw": pred_batch["prob_draw"],
                "prob_away": pred_batch["prob_away"],
            }
        )

    else:
        typer.echo(f"Error: unknown model {model!r}. Use 'dc' or 'elo'.", err=False)
        raise typer.Exit(1)

    out_df.to_csv(output, index=False)
    typer.echo(f"Exported {len(out_df)} predictions to {output}")
