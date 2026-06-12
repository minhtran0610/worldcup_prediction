from __future__ import annotations

import os
import sys

import requests

THE_ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
SPORT_KEY: str = "soccer_fifa_world_cup"
MARKETS: str = "h2h"  # head-to-head = 1X2
REGIONS: str = "eu"  # European bookmakers

_REQUEST_TIMEOUT_SECONDS: int = 15


def get_api_key() -> str | None:
    """Return THE_ODDS_API_KEY from environment, or None if not set."""
    return os.environ.get("THE_ODDS_API_KEY")


def _best_odds_from_event(event: dict) -> dict[str, float | str] | None:
    """Extract best (max) decimal odds for home, draw, and away across all bookmakers.

    Returns {"home_team": str, "away_team": str, "odds_home": float,
             "odds_draw": float, "odds_away": float} or None if no h2h market found.
    """
    home_team: str = event.get("home_team", "")
    away_team: str = event.get("away_team", "")

    best_home: float = 0.0
    best_draw: float = 0.0
    best_away: float = 0.0

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name: str = outcome.get("name", "")
                price: float = float(outcome.get("price", 0.0))
                if name == "Draw":
                    if price > best_draw:
                        best_draw = price
                elif name == home_team:
                    if price > best_home:
                        best_home = price
                elif name == away_team and price > best_away:
                    best_away = price

    # Only return if at least some odds were found
    if best_home <= 0.0 or best_away <= 0.0:
        return None

    # Draw may be 0 for two-outcome events; treat as unavailable but still return
    return {
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": event.get("commence_time", ""),
        "odds_home": best_home,
        "odds_draw": best_draw if best_draw > 0.0 else float("nan"),
        "odds_away": best_away,
    }


def fetch_upcoming_odds(
    api_key: str | None = None,
    odds_format: str = "decimal",
) -> list[dict]:
    """Fetch upcoming WC 2026 match odds from the-odds-api.com.

    Returns list of match dicts, each with:
        home_team (str)
        away_team (str)
        commence_time (str, ISO 8601 UTC)
        odds_home (float)   — best available decimal odds
        odds_draw (float)
        odds_away (float)

    Returns empty list if API key is not set or request fails.
    Prints a warning to stderr if key is missing.
    """
    if api_key is None:
        api_key = get_api_key()

    if not api_key:
        print(
            "[odds_live] WARNING: THE_ODDS_API_KEY is not set — skipping live odds fetch.",
            file=sys.stderr,
        )
        return []

    url = f"{THE_ODDS_API_BASE}/sports/{SPORT_KEY}/odds/"
    params: dict[str, str] = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": odds_format,
    }

    try:
        response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(
            f"[odds_live] HTTP error {status} fetching odds: {exc}",
            file=sys.stderr,
        )
        return []
    except requests.exceptions.RequestException as exc:
        print(f"[odds_live] Request error fetching odds: {exc}", file=sys.stderr)
        return []

    try:
        events: list[dict] = response.json()
    except ValueError as exc:
        print(f"[odds_live] Failed to parse JSON response: {exc}", file=sys.stderr)
        return []

    if not isinstance(events, list):
        print(
            f"[odds_live] Unexpected response type: {type(events).__name__} (expected list)",
            file=sys.stderr,
        )
        return []

    results: list[dict] = []
    for event in events:
        parsed = _best_odds_from_event(event)
        if parsed is not None:
            results.append(parsed)

    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    if remaining is not None or used is not None:
        print(
            f"[odds_live] API credits — used: {used}, remaining: {remaining}",
            file=sys.stderr,
        )

    print(f"[odds_live] Fetched {len(results)} upcoming matches with odds.", file=sys.stderr)
    return results


def _normalise_odds_name(name: str) -> str:
    """Normalise a team name for fuzzy matching across data sources.

    Removes punctuation, replaces '&' with 'and', collapses whitespace.
    e.g. "Bosnia & Herzegovina" and "Bosnia and Herzegovina" both become
    "bosnia and herzegovina".
    """
    import re as _re

    name = name.lower()
    name = name.replace("&", "and")
    name = _re.sub(r"[^\w\s]", "", name)  # strip punctuation
    name = _re.sub(r"\s+", " ", name).strip()
    return name


def fetch_odds_for_teams(
    home_team: str,
    away_team: str,
    api_key: str | None = None,
) -> dict[str, float] | None:
    """Return {"odds_home": float, "odds_draw": float, "odds_away": float} for a specific match.

    Searches upcoming matches for the given team pair using normalised name matching
    (handles differences like 'Bosnia & Herzegovina' vs 'Bosnia and Herzegovina').
    Returns None if match not found or API key not set.
    """
    matches = fetch_upcoming_odds(api_key=api_key)

    home_norm = _normalise_odds_name(home_team)
    away_norm = _normalise_odds_name(away_team)

    for match in matches:
        if (
            _normalise_odds_name(match.get("home_team", "")) == home_norm
            and _normalise_odds_name(match.get("away_team", "")) == away_norm
        ):
            return {
                "odds_home": match["odds_home"],
                "odds_draw": match["odds_draw"],
                "odds_away": match["odds_away"],
            }

    return None
