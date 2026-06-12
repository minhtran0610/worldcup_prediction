from pathlib import Path

import pandas as pd

# Resolved from this file's location: data/ingest/cache.py -> data/raw/
CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "raw"


def load_cache(name: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        print(f"[cache] HIT {name}")
        return pd.read_parquet(path, engine="pyarrow")
    print(f"[cache] MISS {name}")
    return None


def save_cache(name: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    print(f"[cache] SAVED {name} ({len(df)} rows)")
