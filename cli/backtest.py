from __future__ import annotations

from pathlib import Path

import typer

from data.ingest.results import load_results
from eval.backtest import FoldResult, fold_metrics, generate_folds, print_backtest_report
from features.context import derive_context
from features.elo import compute_elo_ratings
from models.dixon_coles import DixonColesModel
from models.elo_model import EloModel

app = typer.Typer()


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    model: str = typer.Option("dc", help="Model: 'dc' or 'elo'"),
    min_train_months: int = typer.Option(36, help="Minimum training months before first fold"),
    val_months: int = typer.Option(6, help="Validation window length in months"),
    tournament_filter: str | None = typer.Option(None, help="Filter to a specific tournament name"),
) -> None:
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)

    results = compute_elo_ratings(results)
    results = derive_context(results)

    if tournament_filter is not None:
        results = results[results["tournament"] == tournament_filter].reset_index(drop=True)

    folds = generate_folds(
        results, min_train_months=min_train_months, val_duration_months=val_months
    )

    if not folds:
        typer.echo("No folds generated — not enough data for the requested configuration.")
        raise typer.Exit(1)

    n_folds = len(folds)
    fold_results: list[FoldResult] = []

    for fold_idx, (val_start, val_end, train_end) in enumerate(folds, start=1):
        train_df = results[results["date"] < train_end].reset_index(drop=True)
        val_df = results[(results["date"] >= val_start) & (results["date"] < val_end)].reset_index(
            drop=True
        )

        if val_df.empty:
            continue

        n_train = len(train_df)
        n_val = len(val_df)
        typer.echo(
            f"Fold {fold_idx}/{n_folds}: fitting on {n_train} matches, evaluating on {n_val}...",
            err=True,
        )

        if model == "dc":
            dc = DixonColesModel()
            dc.fit(train_df)
            pred_batch = dc.predict_batch(val_df)
            predictions = pred_batch[["prob_home", "prob_draw", "prob_away"]].copy()
            grids = list(pred_batch["grid"])

        elif model == "elo":
            elo = EloModel()
            elo.fit(train_df)
            pred_batch = elo.predict_batch(val_df)
            predictions = pred_batch[["prob_home", "prob_draw", "prob_away"]].copy()
            grids = []

        else:
            typer.echo(f"Error: unknown model {model!r}. Use 'dc' or 'elo'.", err=False)
            raise typer.Exit(1)

        fr = fold_metrics(
            val_df=val_df,
            predictions=predictions,
            grids=grids,
            fold_idx=fold_idx,
            val_start=val_start,
            val_end=val_end,
            train_end=train_end,
        )
        fold_results.append(fr)

    print_backtest_report(fold_results)
