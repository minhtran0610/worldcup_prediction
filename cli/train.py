from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from data.ingest.results import load_results
from eval.metrics import aggregate_rps
from features.context import derive_context
from features.elo import compute_elo_ratings
from features.squad_registry import SquadRegistry
from models.neural import NeuralModel

app = typer.Typer()


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    checkpoint: Path = typer.Option(
        Path("data/raw/neural_checkpoint.pt"), help="Where to save checkpoint"
    ),
    n_epochs: int = typer.Option(150, help="Maximum training epochs"),
    lr: float = typer.Option(3e-4, help="Adam learning rate"),
    val_months: int = typer.Option(6, help="Hold-out validation months at end of data"),
    tournament_filter: str | None = typer.Option(None, help="Filter to specific tournament"),
) -> None:
    # 1. Load and enrich results
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)

    results = compute_elo_ratings(results)

    typer.echo("Building squad registry...", err=True)
    registry = SquadRegistry.build()
    results = derive_context(results, squad_registry=registry)

    if tournament_filter is not None:
        results = results[results["tournament"] == tournament_filter].reset_index(drop=True)

    if results.empty:
        typer.echo("No data after filtering — check tournament_filter.", err=False)
        raise typer.Exit(1)

    # 2. Chronological train/val split
    cutoff: pd.Timestamp = results["date"].max() - pd.DateOffset(months=val_months)
    train_df = results[results["date"] < cutoff].reset_index(drop=True)
    val_df = results[results["date"] >= cutoff].reset_index(drop=True)

    if train_df.empty:
        typer.echo(
            f"Training set is empty after holding out {val_months} months of validation data.",
            err=False,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Train: {len(train_df)} matches  "
        f"({train_df['date'].min().date()} – {train_df['date'].max().date()})",
        err=True,
    )
    typer.echo(
        f"Val:   {len(val_df)} matches  "
        f"({val_df['date'].min().date()} – {val_df['date'].max().date()})"
        if not val_df.empty
        else "Val:   0 matches",
        err=True,
    )

    # 3. Train the neural model
    model = NeuralModel(n_epochs=n_epochs, lr=lr, checkpoint_path=str(checkpoint))
    model.fit(
        train_results=train_df,
        val_results=val_df if not val_df.empty else None,
    )

    # 4. Report val RPS
    if not val_df.empty:
        pred_batch = model.predict_batch(val_df)

        # Derive outcome_idx: 0=home win, 1=draw, 2=away win
        outcome_idx = np.where(
            val_df["home_score"].values > val_df["away_score"].values,
            0,
            np.where(val_df["home_score"].values == val_df["away_score"].values, 1, 2),
        )
        rps_df = pred_batch[["prob_home", "prob_draw", "prob_away"]].copy()
        rps_df["outcome_idx"] = outcome_idx

        val_rps = aggregate_rps(rps_df)
        typer.echo(f"Val RPS: {val_rps:.4f}")
    else:
        typer.echo("No validation data; skipping RPS report.")

    # 5. Save checkpoint
    model.save(checkpoint)
    typer.echo(f"Checkpoint saved to {checkpoint}")
