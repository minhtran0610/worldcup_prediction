from __future__ import annotations

import os
import sys

import pandas as pd
import requests

_BASE_URL: str = "https://v3.football.api-sports.io"
_WC_LEAGUE_ID: int = 1
_WC_SEASON: int = 2026
_TIMEOUT_SECONDS: int = 20

# Statuses that mean the match is fully complete (not live, not abandoned)
_FINISHED_STATUSES: frozenset[str] = frozenset({"FT", "AET", "PEN"})

# API-Football team name -> Kaggle training-data name (where they differ)
_TEAM_NAME_MAP: dict[str, str] = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "IR Iran": "Iran",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DPR Korea": "North Korea",
    "Czechia": "Czech Republic",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Democratic Republic of Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "USA": "United States",
    "Cape Verde Islands": "Cape Verde",
    "Curacao": "Curaçao",
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


def get_api_key() -> str | None:
    """Return API_FOOTBALL_KEY from environment, or None if not set."""
    return os.environ.get("API_FOOTBALL_KEY")


def _normalise_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def _parse_date(raw: str) -> pd.Timestamp | None:
    """Parse an ISO 8601 date string from the API into a UTC-naive Timestamp."""
    try:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts  # ty: ignore[invalid-return-type]
    except Exception:
        return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS).astype(
        {
            "date": "datetime64[ns]",
            "home_team": "object",
            "away_team": "object",
            "home_score": "object",
            "away_score": "object",
            "stage": "object",
            "venue": "object",
            "is_completed": "bool",
        }
    )


def _extract_fulltime_score(fx: dict) -> tuple[int | None, int | None]:
    """Return the 90-minute regulation score from an API-Football fixture object.

    Uses score.fulltime rather than the top-level `goals` field — `goals`
    reflects the match's final result including extra time for matches that
    went there, while score.fulltime is the clean 90-minute score that 1X2
    betting markets settle on.
    """
    fulltime: dict = (fx.get("score") or {}).get("fulltime") or {}
    raw_home = fulltime.get("home")
    raw_away = fulltime.get("away")
    home_score = int(raw_home) if raw_home is not None else None
    away_score = int(raw_away) if raw_away is not None else None
    return home_score, away_score


def fetch_wc2026_fixtures(api_key: str | None = None) -> pd.DataFrame:
    """Fetch all WC 2026 fixtures from API-Football (league=1, season=2026).

    Returns a DataFrame matching the load_wc2026_schedule() schema:
        date         (pd.Timestamp, UTC-naive) — real kickoff time
        home_team    (str) — mapped to Kaggle training-data name
        away_team    (str)
        home_score   (int | None) — None if match not yet played
        away_score   (int | None)
        stage        (str) — e.g. "Group Stage - 1", "Round of 16"
        venue        (str)
        is_completed (bool)

    Returns an empty DataFrame (with correct schema) on any failure so the
    pipeline can continue gracefully.
    """
    if api_key is None:
        api_key = get_api_key()

    if not api_key:
        print(
            "[api_football] API_FOOTBALL_KEY not set — skipping API-Football fetch.",
            file=sys.stderr,
        )
        return _empty_df()

    url = f"{_BASE_URL}/fixtures"
    headers = {"x-apisports-key": api_key}
    params: dict[str, int] = {"league": _WC_LEAGUE_ID, "season": _WC_SEASON}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"[api_football] HTTP {status} fetching fixtures: {exc}", file=sys.stderr)
        return _empty_df()
    except requests.exceptions.RequestException as exc:
        print(f"[api_football] Request error: {exc}", file=sys.stderr)
        return _empty_df()

    try:
        data: dict = resp.json()
    except ValueError as exc:
        print(f"[api_football] JSON parse error: {exc}", file=sys.stderr)
        return _empty_df()

    # Log quota so the user can track daily usage
    remaining = resp.headers.get("x-ratelimit-requests-remaining")
    used = resp.headers.get("x-ratelimit-requests-used")
    if remaining is not None or used is not None:
        print(
            f"[api_football] Daily quota — used: {used}, remaining: {remaining}",
            file=sys.stderr,
        )

    fixtures: list[dict] = data.get("response", [])
    if not fixtures:
        errors = data.get("errors", {})
        print(
            f"[api_football] No fixtures in response. errors={errors}",
            file=sys.stderr,
        )
        return _empty_df()

    rows: list[dict] = []
    for fx in fixtures:
        fixture_info: dict = fx.get("fixture", {})
        teams: dict = fx.get("teams", {})
        league_info: dict = fx.get("league", {})

        status_short: str = fixture_info.get("status", {}).get("short", "")
        is_completed: bool = status_short in _FINISHED_STATUSES

        dt = _parse_date(fixture_info.get("date", ""))
        home = _normalise_team(teams.get("home", {}).get("name", ""))
        away = _normalise_team(teams.get("away", {}).get("name", ""))

        home_score, away_score = _extract_fulltime_score(fx)

        stage: str = league_info.get("round", "Group Stage")
        venue: str = fixture_info.get("venue", {}).get("name", "") or ""

        rows.append(
            {
                "date": dt,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "stage": stage,
                "venue": venue,
                "is_completed": is_completed,
            }
        )

    df = pd.DataFrame(rows)
    df["is_completed"] = df["is_completed"].astype(bool)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    n_completed = int(df["is_completed"].sum())
    print(
        f"[api_football] Fetched {len(df)} WC 2026 fixtures ({n_completed} completed).",
        file=sys.stderr,
    )
    return df
