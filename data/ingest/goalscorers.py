"""Build a player → career international goals lookup from the Kaggle goalscorers CSV.

Excludes own goals — those reflect the opposing team's advantage, not the scorer's
ability, and would wrongly penalise the team that conceded them.

Used by features/squad_registry.py to compute squad-level goal-scoring pedigree.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from data.ingest.cache import load_cache, save_cache

_GOALSCORERS_CSV: Path = Path("data/raw/goalscorers.csv")
_CACHE_KEY: str = "player_goals"


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s).strip().lower())


def load_player_goals(
    csv_path: Path | str | None = None,
    force_refresh: bool = False,
) -> dict[str, int]:
    """Return {nfc_player_name: career_international_goals} (own goals excluded).

    Aggregates all non-OG goals per player across all international matches in the
    Kaggle dataset.  Names are NFC-normalised and lowercased so matching against
    squad-registry names is consistent.

    Caches to data/raw/player_goals.parquet on first load.  Pass
    force_refresh=True to rebuild from the CSV.
    """
    if not force_refresh:
        cached = load_cache(_CACHE_KEY)
        if cached is not None:
            return {row.name: int(row.goals) for row in cached.itertuples(index=False)}

    path = Path(csv_path) if csv_path else _GOALSCORERS_CSV
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    df = df[df["own_goal"] == False].copy()  # noqa: E712  (pandas bool column)
    df["name_norm"] = df["scorer"].apply(_nfc)
    agg = df.groupby("name_norm").size().reset_index(name="goals")
    agg = agg.rename(columns={"name_norm": "name"})

    save_cache(_CACHE_KEY, agg)

    return {row.name: int(row.goals) for row in agg.itertuples(index=False)}
