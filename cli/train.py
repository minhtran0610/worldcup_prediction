from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from data.ingest.results import drop_wc2026, is_wc2026_match, load_results
from data.ingest.wc2026 import inject_completed_wc2026_matches, load_wc2026_schedule
from eval.metrics import aggregate_rps
from features.context import compute_sample_weight, derive_context
from features.elo import compute_elo_ratings
from features.squad_registry import SquadRegistry
from models.neural import NeuralModel

app = typer.Typer()


@app.command()
def main(
    csv_path: Path | None = typer.Option(None, help="Path to Kaggle results CSV"),
    checkpoint: Path = typer.Option(Path("checkpoints/neural.pt"), help="Where to save checkpoint"),
    n_epochs: int = typer.Option(150, help="Maximum training epochs"),
    lr: float = typer.Option(3e-4, help="Adam learning rate"),
    val_months: int = typer.Option(6, help="Hold-out validation months at end of data"),
    tournament_filter: str | None = typer.Option(None, help="Filter to specific tournament"),
    no_holdout: bool = typer.Option(
        False,
        "--no-holdout/--holdout",
        help="Production refit: after the holdout run reports Val RPS, retrain on ALL "
        "data for the best epoch count (no held-out validation).",
    ),
) -> None:
    # 1. Load and enrich results
    try:
        results = load_results(csv_path=csv_path)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=False)
        raise typer.Exit(1)

    # Drop the unreliable Kaggle-sourced 2026 rows first — verified completed
    # WC2026 matches are re-injected below from the live schedule instead,
    # then heavily boosted via sample_weight (see compute_sample_weight).
    n_before = len(results)
    results = drop_wc2026(results)
    n_dropped = n_before - len(results)
    if n_dropped:
        typer.echo(f"Excluded {n_dropped} WC 2026 match(es) from training data.", err=True)

    results = compute_elo_ratings(results)

    typer.echo("Building squad registry...", err=True)
    registry = SquadRegistry.build()
    results = derive_context(results, squad_registry=registry)

    typer.echo("Fetching WC 2026 schedule (live) to inject completed matches...", err=True)
    schedule = load_wc2026_schedule(force_refresh=True)
    if not schedule.empty:
        completed = (
            schedule[schedule["is_completed"]].dropna(subset=["home_score", "away_score"]).copy()
        )
        if not completed.empty:
            completed["home_score"] = completed["home_score"].astype(int)
            completed["away_score"] = completed["away_score"].astype(int)
            n_before = len(results)
            results = inject_completed_wc2026_matches(results, completed, registry)
            results["sample_weight"] = compute_sample_weight(results)
            typer.echo(
                f"Injected {len(results) - n_before} completed WC 2026 match(es) into training "
                "data; sample_weight recomputed over the full merged frame.",
                err=True,
            )

    if tournament_filter is not None:
        results = results[results["tournament"] == tournament_filter].reset_index(drop=True)

    if results.empty:
        typer.echo("No data after filtering — check tournament_filter.", err=False)
        raise typer.Exit(1)

    # 2. Chronological train/val split — WC2026 rows always stay in train.
    # Held out here, they'd never inform early stopping at all (see
    # compute_sample_weight — WC2026_BOOST is currently 1.0, a no-op, but the
    # rows still deserve to be learned from); there are only ~96 of them,
    # too few to be a meaningful val window on their own anyway.
    cutoff: pd.Timestamp = results["date"].max() - pd.DateOffset(months=val_months)
    is_wc26 = is_wc2026_match(results)
    train_df = results[(results["date"] < cutoff) | is_wc26].reset_index(drop=True)
    val_df = results[(results["date"] >= cutoff) & (~is_wc26)].reset_index(drop=True)

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

    # 3. Train the neural model (holdout run — honest early-stopping + Val RPS)
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
    if no_holdout:
        # Production refit: retrain a fresh model on ALL data (train + val) for the
        # epoch count the holdout run selected, with early stopping off.
        best_epoch = model.best_epoch_ or n_epochs
        typer.echo(
            f"Final refit on all {len(results)} matches for {best_epoch} epochs (no holdout)...",
            err=True,
        )
        final = NeuralModel(n_epochs=best_epoch, lr=lr, checkpoint_path=str(checkpoint))
        final.fit(train_results=results, early_stopping=False)
        final.save(checkpoint)
        typer.echo(f"Production checkpoint (all data) saved to {checkpoint}")
    else:
        model.save(checkpoint)
        typer.echo(f"Checkpoint saved to {checkpoint}")
