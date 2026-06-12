import logging

import pandas as pd

from data.ingest.cache import load_cache, save_cache

logger = logging.getLogger(__name__)

PLAYER_STATS_CACHE_KEY: str = "player_stats_2024_25"
SEASON: str = "2024-25"

_BIG5 = "Big 5 European Leagues Combined"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns to single-level by joining levels with '_'.

    Joins all non-empty parts of each MultiIndex tuple, then lowercases the result.
    This handles soccerdata's convention where the first level is already snake_cased
    but the second level retains original HTML casing (e.g. 'Min', 'Gls', 'xG').
    """
    if df.columns.nlevels == 1:
        return df
    df = df.copy()
    df.columns = [
        "_".join(str(part) for part in col if str(part) not in {"", "nan"}).strip("_").lower()
        for col in df.columns
    ]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name whose lower-stripped form matches any candidate."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _fetch_standard(fbref) -> pd.DataFrame:
    raw = fbref.read_player_season_stats(stat_type="standard")
    df = _flatten_columns(raw.reset_index())

    player_col = _find_col(df, ["player", "player_name"])
    team_col = _find_col(df, ["team"])
    league_col = _find_col(df, ["league"])

    if player_col is None:
        raise RuntimeError(
            f"FBref standard stats missing expected 'player' column. "
            f"Actual columns: {list(df.columns)}"
        )

    renames: dict[str, str] = {player_col: "player_name"}
    if team_col and team_col != "player_name":
        renames[team_col] = "club"
    if league_col and league_col not in renames:
        renames[league_col] = "league"
    df = df.rename(columns=renames)

    minutes_col = _find_col(df, ["playing_time_min", "unnamed:_playing_time_min"])
    goals_col = _find_col(df, ["performance_gls", "unnamed:_performance_gls"])
    assists_col = _find_col(df, ["performance_ast", "unnamed:_performance_ast"])
    if minutes_col and minutes_col not in {"player_name", "club", "league"}:
        df = df.rename(columns={minutes_col: "minutes"})
    if goals_col and goals_col not in {"player_name", "club", "league"}:
        df = df.rename(columns={goals_col: "goals"})
    if assists_col and assists_col not in {"player_name", "club", "league"}:
        df = df.rename(columns={assists_col: "assists"})

    keep = ["player_name", "club", "league"] + [
        c for c in ["minutes", "goals", "assists"] if c in df.columns
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["minutes"] = pd.to_numeric(
        df.get("minutes", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    df["goals"] = pd.to_numeric(df.get("goals", pd.Series(dtype=float)), errors="coerce").fillna(
        0.0
    )
    df["assists"] = pd.to_numeric(
        df.get("assists", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    return df


def _fetch_shooting(fbref) -> pd.DataFrame:
    raw = fbref.read_player_season_stats(stat_type="shooting")
    df = _flatten_columns(raw.reset_index())

    player_col = _find_col(df, ["player", "player_name"])
    team_col = _find_col(df, ["team"])

    if player_col is None:
        raise RuntimeError(
            f"FBref shooting stats missing expected 'player' column. "
            f"Actual columns: {list(df.columns)}"
        )

    renames: dict[str, str] = {player_col: "player_name"}
    if team_col and team_col != "player_name":
        renames[team_col] = "club"
    df = df.rename(columns=renames)

    xg_col = _find_col(df, ["expected_xg"])
    npxg_col = _find_col(df, ["expected_npxg"])
    xa_col = _find_col(df, ["expected_xa", "expected_xag"])
    if xg_col and xg_col not in {"player_name", "club"}:
        df = df.rename(columns={xg_col: "xg"})
    if npxg_col and npxg_col not in {"player_name", "club"}:
        df = df.rename(columns={npxg_col: "npxg"})
    if xa_col and xa_col not in {"player_name", "club"}:
        df = df.rename(columns={xa_col: "xa"})

    xg_cols = [c for c in ["xg", "npxg", "xa"] if c in df.columns]
    keep = ["player_name", "club"] + xg_cols
    df = df[[c for c in keep if c in df.columns]].copy()
    for col in xg_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _compute_per90(df: pd.DataFrame) -> pd.DataFrame:
    minutes = df["minutes"].clip(lower=1.0)
    nineties = minutes / 90.0
    if "xg" in df.columns:
        df["xg_per90"] = df["xg"] / nineties
    if "xa" in df.columns:
        df["xa_per90"] = df["xa"] / nineties
    if "npxg" in df.columns:
        df["npxg_per90"] = df["npxg"] / nineties
    return df


def load_player_stats(force_refresh: bool = False) -> pd.DataFrame:
    """Return per-player club stats for season 2024-25 from FBref.

    Uses soccerdata.FBref to fetch standard stats and shooting stats.
    Target columns (rename to these):
        player_name (str), club (str), league (str),
        minutes (float), goals (float), assists (float),
        xg (float), xg_per90 (float), xa (float), xa_per90 (float),
        npxg_per90 (float)       — non-penalty xG per 90

    Strategy:
    1. Check cache first
    2. Use soccerdata.FBref(leagues="Big 5 European Leagues", seasons=SEASON)
       - call .read_player_season_stats(stat_type="standard")
       - also .read_player_season_stats(stat_type="shooting") for xG columns
       - merge on player + club
    3. If soccerdata scraping fails (rate limit, network), raise RuntimeError with
       a clear message: "FBref scraping failed: {e}. Run with force_refresh=False to use cache."
    4. Cache and return.

    Note: soccerdata returns MultiIndex columns — flatten to single-level by joining levels with '_'.
    Log how many players were fetched.
    """
    if not force_refresh:
        cached = load_cache(PLAYER_STATS_CACHE_KEY)
        if cached is not None:
            return cached

    try:
        import soccerdata

        fbref = soccerdata.FBref(leagues=_BIG5, seasons=SEASON)

        standard = _fetch_standard(fbref)
        shooting = _fetch_shooting(fbref)

    except Exception as exc:
        raise RuntimeError(
            f"FBref scraping failed: {exc}. Run with force_refresh=False to use cache."
        ) from exc

    df = standard.merge(
        shooting,
        on=["player_name", "club"],
        how="left",
        suffixes=("", "_shooting"),
    )

    for col in ["xg", "npxg", "xa"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = _compute_per90(df)

    target_cols = [
        "player_name",
        "club",
        "league",
        "minutes",
        "goals",
        "assists",
        "xg",
        "xg_per90",
        "xa",
        "xa_per90",
        "npxg_per90",
    ]
    df = df[[c for c in target_cols if c in df.columns]].copy()

    df = df.drop_duplicates(subset=["player_name", "club"])
    df = df.reset_index(drop=True)

    print(f"[player_stats] fetched {len(df)} player-club rows for season {SEASON}")
    save_cache(PLAYER_STATS_CACHE_KEY, df)
    return df
