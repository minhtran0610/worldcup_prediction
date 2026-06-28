from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

_CACHE_PATH = Path("data/cache/odds_live_cache.json")
_CACHE_TTL_SECONDS = 15 * 60  # 15 minutes — covers re-runs; use force_refresh for closing line

THE_ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
SPORT_KEY: str = "soccer_fifa_world_cup"
MARKETS: str = "h2h"
REGIONS: str = "eu"

_REQUEST_TIMEOUT_SECONDS: int = 15


def get_api_key() -> str | None:
    return os.environ.get("THE_ODDS_API_KEY")


def _best_h2h(event: dict) -> tuple[float, float, float]:
    """Return (best_home, best_draw, best_away) across all bookmakers."""
    best_home = best_draw = best_away = 0.0
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = float(outcome.get("price", 0.0))
                if name == "Draw":
                    best_draw = max(best_draw, price)
                elif name == home_team:
                    best_home = max(best_home, price)
                elif name == away_team:
                    best_away = max(best_away, price)
    return best_home, best_draw, best_away


def _parse_event(event: dict) -> dict | None:
    """Parse a single API event. Returns None if 1X2 odds are missing."""
    best_home, best_draw, best_away = _best_h2h(event)
    if best_home <= 0.0 or best_away <= 0.0:
        return None
    return {
        "home_team": event.get("home_team", ""),
        "away_team": event.get("away_team", ""),
        "commence_time": event.get("commence_time", ""),
        "odds_home": best_home,
        "odds_draw": best_draw if best_draw > 0.0 else float("nan"),
        "odds_away": best_away,
    }


def _load_odds_cache() -> list[dict] | None:
    if not _CACHE_PATH.exists():
        return None
    age = time.time() - _CACHE_PATH.stat().st_mtime
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        data = json.loads(_CACHE_PATH.read_text())
        remaining = int(_CACHE_TTL_SECONDS - age) // 60
        print(
            f"[odds_live] Cache HIT (expires in ~{remaining}m) — skipping API call.",
            file=sys.stderr,
        )
        return data
    except Exception:
        return None


def _save_odds_cache(data: list[dict]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
    except Exception as exc:
        print(f"[odds_live] WARNING: could not write odds cache — {exc}", file=sys.stderr)


def fetch_upcoming_odds(
    api_key: str | None = None,
    odds_format: str = "decimal",
    force_refresh: bool = False,
) -> list[dict]:
    """Fetch upcoming WC 2026 1X2 odds (best across EU bookmakers).

    Each returned dict: home_team, away_team, commence_time,
                        odds_home, odds_draw, odds_away
    Returns [] on missing key or request failure.
    """
    if api_key is None:
        api_key = get_api_key()
    if not api_key:
        print(
            "[odds_live] WARNING: THE_ODDS_API_KEY is not set — skipping live odds fetch.",
            file=sys.stderr,
        )
        return []

    if not force_refresh:
        cached = _load_odds_cache()
        if cached is not None:
            return cached

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
        print(f"[odds_live] HTTP error {status}: {exc}", file=sys.stderr)
        return []
    except requests.exceptions.RequestException as exc:
        print(f"[odds_live] Request error: {exc}", file=sys.stderr)
        return []

    try:
        events: list[dict] = response.json()
    except ValueError as exc:
        print(f"[odds_live] Failed to parse JSON: {exc}", file=sys.stderr)
        return []

    if not isinstance(events, list):
        print(f"[odds_live] Unexpected response type: {type(events).__name__}", file=sys.stderr)
        return []

    results = [parsed for e in events if (parsed := _parse_event(e)) is not None]
    _save_odds_cache(results)

    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    if remaining is not None or used is not None:
        print(f"[odds_live] API credits — used: {used}, remaining: {remaining}", file=sys.stderr)
    print(f"[odds_live] Fetched {len(results)} matches with 1X2 odds.", file=sys.stderr)
    return results


def fetch_odds_for_teams(
    home_team: str,
    away_team: str,
    api_key: str | None = None,
) -> dict | None:
    """Return odds dict for a specific match, or None if not found."""
    matches = fetch_upcoming_odds(api_key=api_key)
    home_norm = _normalise_odds_name(home_team)
    away_norm = _normalise_odds_name(away_team)
    for match in matches:
        if (
            _normalise_odds_name(match.get("home_team", "")) == home_norm
            and _normalise_odds_name(match.get("away_team", "")) == away_norm
        ):
            return match
    return None


def _normalise_odds_name(name: str) -> str:
    import re as _re

    name = name.lower().replace("&", "and")
    name = _re.sub(r"[^\w\s]", "", name)
    return _re.sub(r"\s+", " ", name).strip()
