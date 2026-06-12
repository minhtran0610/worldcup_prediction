import logging
import warnings

import pandas as pd
import requests

from data.ingest.cache import load_cache, save_cache

logger = logging.getLogger(__name__)

SQUADS_CACHE_KEY: str = "wc2026_squads"

_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

_POSITION_MAP: dict[str, str] = {
    "gk": "GK",
    "goalkeeper": "GK",
    "g": "GK",
    "df": "DF",
    "defender": "DF",
    "d": "DF",
    "cb": "DF",
    "lb": "DF",
    "rb": "DF",
    "mf": "MF",
    "midfielder": "MF",
    "m": "MF",
    "cm": "MF",
    "dm": "MF",
    "am": "MF",
    "fw": "FW",
    "forward": "FW",
    "f": "FW",
    "st": "FW",
    "cf": "FW",
    "ss": "FW",
    "lw": "FW",
    "rw": "FW",
    "wf": "FW",
}


def _normalise_position(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in _POSITION_MAP:
        return _POSITION_MAP[key]
    for prefix, pos in _POSITION_MAP.items():
        if key.startswith(prefix):
            return pos
    return None


def _scrape_wikipedia() -> pd.DataFrame:
    response = requests.get(_WIKIPEDIA_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = pd.read_html(response.text)

    records: list[dict] = []

    for tbl in tables:
        tbl.columns = [
            " ".join(str(c) for c in col).strip() if isinstance(col, tuple) else str(col)
            for col in tbl.columns
        ]

        col_lower = {c: c.lower() for c in tbl.columns}

        player_col = next(
            (c for c, cl in col_lower.items() if "player" in cl or "name" in cl), None
        )
        pos_col = next((c for c, cl in col_lower.items() if "pos" in cl), None)
        club_col = next((c for c, cl in col_lower.items() if "club" in cl), None)
        club_country_col = next(
            (c for c, cl in col_lower.items() if "country" in cl or "league" in cl), None
        )
        age_col = next((c for c, cl in col_lower.items() if "age" in cl), None)
        team_col = next((c for c, cl in col_lower.items() if "nation" in cl or "team" in cl), None)

        if player_col is None or pos_col is None or club_col is None:
            continue

        team_name: str | None = None
        if team_col is not None:
            unique_teams = tbl[team_col].dropna().unique()
            if len(unique_teams) == 1:
                team_name = str(unique_teams[0])

        for _, row in tbl.iterrows():
            try:
                player_name = str(row[player_col]).strip()
                if not player_name or player_name.lower() in {"nan", "player", "name"}:
                    continue

                raw_pos = str(row[pos_col]).strip()
                pos = _normalise_position(raw_pos)
                if pos is None:
                    logger.warning(
                        "[squads] unrecognised position %r for player %r — skipping row",
                        raw_pos,
                        player_name,
                    )
                    continue

                club = str(row[club_col]).strip() if club_col else ""
                club_country = (
                    str(row[club_country_col]).strip()
                    if club_country_col and club_country_col in row.index
                    else ""
                )
                age_raw = row[age_col] if age_col else None
                try:
                    age = int(str(age_raw).split("-")[0]) if age_raw is not None else 0
                except (ValueError, TypeError):
                    age = 0

                team = (
                    str(row[team_col]).strip()
                    if team_col and not pd.isna(row.get(team_col))
                    else (team_name or "")
                )

                records.append(
                    {
                        "team": team,
                        "player_name": player_name,
                        "position": pos,
                        "club": club,
                        "club_country": club_country,
                        "age": age,
                    }
                )
            except Exception as exc:
                logger.warning("[squads] skipping row due to error: %s", exc)
                continue

    if not records:
        raise RuntimeError(
            f"Wikipedia squad scrape returned no usable rows from {_WIKIPEDIA_URL}. "
            "The page format may have changed. Inspect the tables manually and update "
            "_scrape_wikipedia() in data/ingest/squads.py."
        )

    df = pd.DataFrame(records)
    df = df[df["team"].str.strip().ne("")]
    df = df[df["player_name"].str.strip().ne("")]
    df = df.reset_index(drop=True)
    print(f"[squads] scraped {len(df)} players across {df['team'].nunique()} teams from Wikipedia")
    return df


def load_squads(force_refresh: bool = False) -> pd.DataFrame:
    """Return WC 2026 squad DataFrame.

    Columns: team (str), player_name (str), position (str: GK/DF/MF/FW),
             club (str), club_country (str), age (int)

    Strategy:
    1. Check cache first (unless force_refresh)
    2. Try scraping from Wikipedia using requests + pandas.read_html
       URL: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
       Parse each team's table; extract columns.
       Normalise position strings to GK/DF/MF/FW.
    3. If scraping fails, raise RuntimeError with message explaining what to do.
    4. Cache and return.

    Log unrecognised position strings as warnings (don't crash).
    """
    if not force_refresh:
        cached = load_cache(SQUADS_CACHE_KEY)
        if cached is not None:
            return cached

    try:
        df = _scrape_wikipedia()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Wikipedia squad scrape failed: {exc}. "
            f"URL attempted: {_WIKIPEDIA_URL}. "
            "Check network connectivity and that the Wikipedia page exists. "
            "Alternatively, place a manually prepared parquet at "
            "data/raw/wc2026_squads.parquet and re-run without force_refresh."
        ) from exc

    save_cache(SQUADS_CACHE_KEY, df)
    return df
