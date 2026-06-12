from __future__ import annotations

import re
from io import StringIO

import pandas as pd

from data.ingest.cache import load_cache, save_cache

WC2026_WIKI_URL: str = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"

# Maps Wikipedia team name -> Kaggle "international_football_results.csv" team name
# where the two differ.  Keys are exactly as they appear in Wikipedia table cells.
TEAM_NAME_MAP: dict[str, str] = {
    # North / Central America
    "United States": "United States",  # Same in Kaggle
    "USA": "United States",
    "US": "United States",
    # Asia
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
    "DPR Korea": "North Korea",
    # Africa
    "DR Congo": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    # Misc
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Czechia": "Czech Republic",
    "North Macedonia": "North Macedonia",
}

_EMPTY_COLUMNS: list[str] = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "stage",
    "venue",
    "is_completed",
]


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    return pd.DataFrame(columns=_EMPTY_COLUMNS).astype(
        {
            "date": "datetime64[ns]",
            "home_team": "object",
            "away_team": "object",
            "home_score": "object",  # int | None via object dtype
            "away_score": "object",
            "stage": "object",
            "venue": "object",
            "is_completed": "bool",
        }
    )


def _normalise_team(name: str) -> str:
    """Strip footnote markers and apply TEAM_NAME_MAP."""
    name = re.sub(r"\[.*?\]", "", str(name)).strip()
    return TEAM_NAME_MAP.get(name, name)


def _parse_score(raw: str) -> tuple[int | None, int | None]:
    """Parse a score string like '2–1' or '2-1' into (home, away).

    Returns (None, None) if the string is not a completed-match score.
    Handles em-dash (–) and hyphen (-).
    """
    raw = str(raw).strip()
    # Remove anything in parentheses (e.g. AET, penalties)
    raw = re.sub(r"\(.*?\)", "", raw).strip()
    # Try matching 'N–M' or 'N-M'
    m = re.match(r"^(\d+)\s*[–\-]\s*(\d+)$", raw)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _scrape_wikipedia() -> pd.DataFrame:
    """Fetch the WC 2026 Wikipedia page and extract the group stage schedule.

    Wikipedia's WC pages use a consistent pattern: each group has a table
    with columns roughly like:
        Date | Home team | Score | Away team | Venue | ...

    We scan all tables returned by pd.read_html for ones that contain
    both a score-like column and recognisable team columns.

    Returns a DataFrame with _EMPTY_COLUMNS schema, or raises on failure.
    """
    import requests  # local import to avoid hard dependency at module load

    resp = requests.get(
        WC2026_WIKI_URL,
        headers={"User-Agent": "Mozilla/5.0 (worldcup-prediction research bot)"},
        timeout=30,
    )
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text), flavor="html5lib")

    rows: list[dict] = []

    # Heuristic: WC group-stage match tables on Wikipedia have these features:
    #   - A column whose values look like "N–M" (score) or "–" (upcoming)
    #   - Two team-name columns adjacent to the score column
    #   - A date column
    #
    # The exact column names vary across Wikipedia article versions.
    # We try multiple known column-name conventions.

    SCORE_ALIASES = {"Score", "score", "Result", "result", "Res.", "FT"}
    HOME_ALIASES = {"Home team", "Home", "Team 1", "home_team"}
    AWAY_ALIASES = {"Away team", "Away", "Team 2", "away_team"}
    DATE_ALIASES = {"Date", "date", "Match date"}
    VENUE_ALIASES = {"Venue", "venue", "Stadium", "Ground"}
    STAGE_ALIASES = {"Group", "Stage", "Round", "Group stage"}

    for tbl in tables:
        # Flatten multi-level columns if present
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [" ".join(str(c) for c in col).strip() for col in tbl.columns]

        score_col = next((c for c in tbl.columns if c in SCORE_ALIASES), None)
        home_col = next((c for c in tbl.columns if c in HOME_ALIASES), None)
        away_col = next((c for c in tbl.columns if c in AWAY_ALIASES), None)
        date_col = next((c for c in tbl.columns if c in DATE_ALIASES), None)
        venue_col = next((c for c in tbl.columns if c in VENUE_ALIASES), None)
        stage_col = next((c for c in tbl.columns if c in STAGE_ALIASES), None)

        # Skip tables that don't look like match schedules
        if not (score_col and home_col and away_col):
            continue

        for _, row in tbl.iterrows():
            home_raw = str(row.get(home_col, "")).strip()
            away_raw = str(row.get(away_col, "")).strip()
            score_raw = str(row.get(score_col, "")).strip()

            # Skip header-like rows or empty rows
            if not home_raw or not away_raw or home_raw.lower() in {"nan", "home team", "home"}:
                continue

            home = _normalise_team(home_raw)
            away = _normalise_team(away_raw)

            home_score, away_score = _parse_score(score_raw)
            is_completed = home_score is not None

            # Date parsing: try multiple formats
            date_raw = str(row.get(date_col, "")) if date_col else ""
            date_raw = re.sub(r"\[.*?\]", "", date_raw).strip()
            parsed_date: pd.Timestamp | None = None
            for fmt in ("%d %B %Y", "%B %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y"):
                try:
                    candidate = pd.to_datetime(date_raw, format=fmt)
                    if candidate is not pd.NaT:
                        parsed_date = pd.Timestamp(candidate)
                    break
                except Exception:
                    continue
            if parsed_date is None:
                try:
                    candidate = pd.to_datetime(date_raw, dayfirst=True)
                    parsed_date = pd.Timestamp(candidate) if candidate is not pd.NaT else None
                except Exception:
                    pass

            venue = str(row.get(venue_col, "")).strip() if venue_col else ""
            venue = re.sub(r"\[.*?\]", "", venue).strip()

            stage = str(row.get(stage_col, "")).strip() if stage_col else "Group"
            stage = re.sub(r"\[.*?\]", "", stage).strip()
            if not stage or stage.lower() == "nan":
                stage = "Group"

            rows.append(
                {
                    "date": parsed_date,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "stage": stage,
                    "venue": venue,
                    "is_completed": is_completed,
                }
            )

    if not rows:
        # Fallback: try a second parsing strategy targeting tables where the
        # score is embedded in a single "match" column or unnamed columns.
        rows = _scrape_fallback(tables)

    if not rows:
        raise ValueError(
            "No group stage match rows found on the Wikipedia page. "
            "The page structure may have changed."
        )

    df = pd.DataFrame(rows, columns=_EMPTY_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df["is_completed"] = df["is_completed"].astype(bool)
    return df


def _scrape_fallback(tables: list[pd.DataFrame]) -> list[dict]:
    """Secondary parsing strategy: scan for tables that have score-like values
    in any column, positionally identify team columns, and extract what we can.

    This handles Wikipedia layouts where column names don't match our aliases
    (e.g. translated pages, article rewrites, or tables with no header row).
    """
    rows: list[dict] = []

    for tbl in tables:
        # Flatten multi-level columns
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [" ".join(str(c) for c in col).strip() for col in tbl.columns]

        # Find a column where ≥30% of non-null values look like scores
        score_col_idx: int | None = None
        for ci, _col in enumerate(tbl.columns):
            series = tbl.iloc[:, ci].astype(str)
            score_hits = series.str.match(r"^\d+\s*[–\-]\s*\d+$").sum()
            if score_hits >= max(1, int(0.3 * len(tbl))):
                score_col_idx = ci
                break

        if score_col_idx is None:
            continue

        # Assume home team is in the column immediately before, away team after
        if score_col_idx < 1 or score_col_idx >= len(tbl.columns) - 1:
            continue

        home_col_idx = score_col_idx - 1
        away_col_idx = score_col_idx + 1

        for _, row_data in tbl.iterrows():
            vals = row_data.tolist()
            home_raw = str(vals[home_col_idx]).strip()
            score_raw = str(vals[score_col_idx]).strip()
            away_raw = str(vals[away_col_idx]).strip()

            if not home_raw or not away_raw or home_raw.lower() == "nan":
                continue

            home_score, away_score = _parse_score(score_raw)
            is_completed = home_score is not None

            rows.append(
                {
                    "date": pd.NaT,
                    "home_team": _normalise_team(home_raw),
                    "away_team": _normalise_team(away_raw),
                    "home_score": home_score,
                    "away_score": away_score,
                    "stage": "Group",
                    "venue": "",
                    "is_completed": is_completed,
                }
            )

    return rows


def load_wc2026_schedule(force_refresh: bool = False) -> pd.DataFrame:
    """Load WC 2026 group stage schedule from cache or Wikipedia.

    Returns DataFrame with columns:
        date (pd.Timestamp, UTC-naive)
        home_team (str) — mapped to training-data name
        away_team (str) — mapped to training-data name
        home_score (int | None) — None if match not yet played
        away_score (int | None) — None if match not yet played
        stage (str) — "Group A", "Group B", ..., or "Round of 32"
        venue (str)
        is_completed (bool)

    Caches to data/raw/wc2026_schedule.parquet.
    Falls back to returning an empty DataFrame (with correct columns) if
    scraping fails, so the rest of the pipeline still works.
    """
    cache_name = "wc2026_schedule"

    if not force_refresh:
        cached = load_cache(cache_name)
        if cached is not None:
            return cached

    try:
        df = _scrape_wikipedia()
    except Exception as exc:
        print(f"[wc2026] WARNING: scraping failed — {exc}")
        print("[wc2026] Returning empty DataFrame; pipeline can continue without schedule.")
        return _empty_df()

    try:
        save_cache(cache_name, df)
    except Exception as exc:
        print(f"[wc2026] WARNING: could not save cache — {exc}")

    return df
