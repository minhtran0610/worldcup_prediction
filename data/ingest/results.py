from pathlib import Path

import pandas as pd

from data.ingest.cache import load_cache, save_cache

RESULTS_CACHE_KEY: str = "results"

# WC 2026 matches now appear in the Kaggle results dataset. We keep them out of
# the trained weights (the net learns general football, not this one tournament)
# and let the live schedule be the single source of truth for WC 2026 form at
# predict time — otherwise cached WC rows + live-schedule injection double-count.
WC2026_START: pd.Timestamp = pd.Timestamp("2026-06-01")


def is_wc2026_match(df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting WC 2026 tournament matches in a results-style frame."""
    return (df["tournament"] == "FIFA World Cup") & (df["date"] >= WC2026_START)


def drop_wc2026(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with WC 2026 matches removed (index reset)."""
    return df[~is_wc2026_match(df)].reset_index(drop=True)


KEEP_COLUMNS: list[str] = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
    "country",
]


def load_results(
    csv_path: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if not force_refresh:
        cached = load_cache(RESULTS_CACHE_KEY)
        if cached is not None:
            return cached

    if csv_path is None:
        raise FileNotFoundError(
            "No cached results found. Provide csv_path= pointing to the Kaggle results CSV."
        )

    df = pd.read_csv(csv_path)

    # Parse date to timezone-naive datetime
    df["date"] = pd.to_datetime(df["date"], utc=False)
    df["date"] = df["date"].dt.tz_localize(None)

    # Normalise neutral: CSV may contain string "True"/"False" or actual bools
    df["neutral"] = (
        df["neutral"]
        .map(lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true")
        .astype(bool)
    )

    # Drop rows with missing scores before casting to int
    df = df.dropna(subset=["home_score", "away_score"])

    df = df[KEEP_COLUMNS].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    save_cache(RESULTS_CACHE_KEY, df)
    return df
