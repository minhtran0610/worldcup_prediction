"""Fetch injury and suspension data for WC 2026 upcoming matches.

Primary source:  API-Football /injuries?league=1&season=2026 (requires API_FOOTBALL_KEY)
Fallback source: Transfermarkt national-team injury pages (HTML scrape, no auth needed)

Returns
-------
dict[str, list[str]]
    Canonical team name (Kaggle training-data style) → list of absent player names.
    Only includes players with type "Injury" or "Suspension" (i.e. confirmed OUT or doubtful).
    Empty dict on total failure — callers treat missing data as zero injury loss.

Cache
-----
data/raw/wc2026_injuries_YYYY-MM-DD.json  (one file per calendar day, UTC)
Pass force_refresh=True to bypass the cache and re-fetch.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from data.ingest.api_football import _TEAM_NAME_MAP, get_api_key

_BASE_URL = "https://v3.football.api-sports.io"
_WC_LEAGUE_ID = 1
_WC_SEASON = 2026
_TIMEOUT = 20
_CACHE_DIR = Path("data/cache")
_MANUAL_ABSENCES_PATH = Path("data/cache/manual_absences.json")


def _today_cache_path() -> Path:
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    return _CACHE_DIR / f"wc2026_injuries_{today}.json"


def _normalise_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


# ── API-Football ─────────────────────────────────────────────────────────────


def _fetch_api_football() -> dict[str, list[str]] | None:
    """Return {team: [absent_player_names]} from API-Football, or None on failure."""
    api_key = get_api_key()
    if not api_key:
        return None

    url = f"{_BASE_URL}/injuries"
    headers = {"x-apisports-key": api_key}
    params: dict = {"league": _WC_LEAGUE_ID, "season": _WC_SEASON}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[injuries] API-Football request failed: {exc}", file=sys.stderr)
        return None

    remaining = resp.headers.get("x-ratelimit-requests-remaining")
    used = resp.headers.get("x-ratelimit-requests-used")
    if remaining is not None or used is not None:
        print(f"[injuries] API quota — used: {used}, remaining: {remaining}", file=sys.stderr)

    try:
        data = resp.json()
    except ValueError as exc:
        print(f"[injuries] JSON parse error: {exc}", file=sys.stderr)
        return None

    records: list[dict] = data.get("response", [])
    if not records:
        errors = data.get("errors", {})
        print(f"[injuries] No injury records returned. errors={errors}", file=sys.stderr)
        return None

    # Filter to upcoming fixtures only (fixture date >= today UTC)
    now = datetime.now(tz=UTC)
    result: dict[str, list[str]] = {}
    for rec in records:
        fixture_info = rec.get("fixture", {})
        raw_date = fixture_info.get("date", "")
        try:
            fixture_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            if fixture_dt < now:
                continue  # skip past fixtures
        except (ValueError, AttributeError):
            pass  # include if date unparseable

        team_name = _normalise_team(rec.get("team", {}).get("name", ""))
        player_name = rec.get("player", {}).get("name", "")
        injury_type = rec.get("type", "")  # "Injury" or "Suspension"

        if not team_name or not player_name:
            continue
        if injury_type not in ("Injury", "Suspension"):
            continue

        result.setdefault(team_name, [])
        if player_name not in result[team_name]:
            result[team_name].append(player_name)

    n_teams = len(result)
    n_players = sum(len(v) for v in result.values())
    print(
        f"[injuries] API-Football: {n_players} absent player(s) across {n_teams} team(s).",
        file=sys.stderr,
    )
    return result if result else None


# ── Transfermarkt fallback ────────────────────────────────────────────────────

# Canonical team name → (transfermarkt_slug, tm_team_id)
# Covers all 48 WC 2026 nations. IDs verified from transfermarkt.us URLs.
_TM_TEAMS: dict[str, tuple[str, int]] = {
    # Group A
    "Mexico": ("mexiko", 3437),
    "South Africa": ("sudafrika", 3624),
    "South Korea": ("südkorea", 3470),
    "Czech Republic": ("tschechien", 3376),
    # Group B
    "Canada": ("kanada", 3580),
    "Bosnia and Herzegovina": ("bosnien-herzegowina", 4861),
    "Qatar": ("katar", 14614),
    "Switzerland": ("schweiz", 3384),
    # Group C
    "Brazil": ("brasilien", 3439),
    "Morocco": ("marokko", 3622),
    "Haiti": ("haiti", 3588),
    "Scotland": ("schottland", 3380),
    # Group D
    "United States": ("vereinigte-staaten", 3410),
    "Paraguay": ("paraguay", 3447),
    "Australia": ("australien", 4351),
    "Turkey": ("turkei", 3381),
    # Group E
    "Germany": ("deutschland", 3262),
    "Curaçao": ("curacao", 32364),
    "Ivory Coast": ("elfenbeinküste", 3613),
    "Ecuador": ("ecuador", 3457),
    # Group F
    "Netherlands": ("niederlande", 3379),
    "Japan": ("japan", 3471),
    "Sweden": ("schweden", 3557),
    "Tunisia": ("tunesien", 3670),
    # Group G
    "Belgium": ("belgien", 3382),
    "Egypt": ("agypten", 3598),
    "Iran": ("iran", 3508),
    "New Zealand": ("neuseeland", 4512),
    # Group H
    "Spain": ("spanien", 3375),
    "Cape Verde": ("kap-verde", 4311),
    "Saudi Arabia": ("saudi-arabien", 3512),
    "Uruguay": ("uruguay", 3448),
    # Group I
    "France": ("frankreich", 3377),
    "Senegal": ("senegal", 3499),
    "Iraq": ("irak", 3505),
    "Norway": ("norwegen", 3388),
    # Group J
    "Argentina": ("argentinien", 3409),
    "Algeria": ("algerien", 3607),
    "Austria": ("osterreich", 3383),
    "Jordan": ("jordanien", 3509),
    # Group K
    "Portugal": ("portugal", 3374),
    "Colombia": ("kolumbien", 3816),
    "DR Congo": ("dr-kongo", 3621),
    "Uzbekistan": ("usbekistan", 3527),
    # Group L
    "England": ("england", 3),
    "Croatia": ("kroatien", 3556),
    "Ghana": ("ghana", 3609),
    "Panama": ("panama", 3591),
}

_TM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_TM_BASE = "https://www.transfermarkt.us"
_TM_SLEEP = 3.0  # seconds between requests — be polite


def _fetch_transfermarkt_team(team: str) -> list[str]:
    """Scrape Transfermarkt injury/suspension page for one national team.

    Returns list of absent player names (empty on any failure).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    entry = _TM_TEAMS.get(team)
    if entry is None:
        return []

    slug, tm_id = entry
    url = f"{_TM_BASE}/{slug}/sperrenundverletzungen/verein/{tm_id}"

    try:
        resp = requests.get(url, headers=_TM_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            # Page doesn't exist for this team — not a runtime error, skip silently
            return []
        print(f"[injuries] TM request failed for {team!r}: {exc}", file=sys.stderr)
        return []
    except requests.exceptions.RequestException as exc:
        print(f"[injuries] TM request failed for {team!r}: {exc}", file=sys.stderr)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # The injury table uses class="items"; first table on the page
    table = soup.find("table", class_="items")
    if table is None:
        return []

    names: list[str] = []
    for tr in table.find_all("tr", class_=["odd", "even"]):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        # Player name is in the second td (first is photo), inside an <a> tag
        name_td = cells[1]
        anchor = name_td.find("a")
        name = anchor.get_text(strip=True) if anchor else name_td.get_text(strip=True)
        if name:
            names.append(name)

    return names


def _fetch_transfermarkt(teams: list[str]) -> dict[str, list[str]]:
    """Scrape Transfermarkt for all given teams. Sleeps between requests."""
    result: dict[str, list[str]] = {}
    for i, team in enumerate(teams):
        if i > 0:
            time.sleep(_TM_SLEEP)
        absent = _fetch_transfermarkt_team(team)
        if absent:
            result[team] = absent
            print(f"[injuries] TM {team}: {len(absent)} absent player(s).", file=sys.stderr)
    return result


# ── Manual overrides ─────────────────────────────────────────────────────────


def _load_manual_absences() -> dict[str, list[str]]:
    """Load data/cache/manual_absences.json if it exists.

    Format: {"Paraguay": ["Miguel Almiron"], "England": ["Harry Kane"]}
    Entries here are merged on top of scraped data — useful when TM/API-Football
    haven't updated yet (e.g. red-card suspension just issued).
    """
    if not _MANUAL_ABSENCES_PATH.exists():
        return {}
    try:
        with _MANUAL_ABSENCES_PATH.open() as f:
            data = json.load(f)
        n = sum(len(v) for v in data.values())
        print(
            f"[injuries] Manual absences: {n} player(s) across {len(data)} team(s).",
            file=sys.stderr,
        )
        return data
    except Exception as exc:
        print(f"[injuries] WARNING: could not read manual_absences.json: {exc}", file=sys.stderr)
        return {}


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_wc2026_injuries(
    force_refresh: bool = False,
    teams: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return upcoming WC 2026 injury/suspension data as {team: [player_names]}.

    Parameters
    ----------
    force_refresh:
        Bypass the daily cache and re-fetch from source.
    teams:
        If provided, only return (and scrape) these teams. Useful for targeted
        pre-match fetches. Ignored when using API-Football (returns all teams).

    Source priority
    ---------------
    1. Daily cache  (skipped if force_refresh=True)
    2. API-Football /injuries  (requires API_FOOTBALL_KEY env var)
    3. Transfermarkt scrape   (best-effort; only for teams in _TM_TEAMS)

    Returns empty dict on complete failure — callers treat this as no injuries known.
    """
    cache_path = _today_cache_path()

    if not force_refresh and cache_path.exists():
        try:
            with cache_path.open() as f:
                data: dict[str, list[str]] = json.load(f)
            n = sum(len(v) for v in data.values())
            print(
                f"[injuries] Cache HIT ({cache_path.name}): "
                f"{n} absent player(s) across {len(data)} team(s).",
                file=sys.stderr,
            )
            for team, players in _load_manual_absences().items():
                existing = data.setdefault(team, [])
                for p in players:
                    if p not in existing:
                        existing.append(p)
            if teams:
                return {t: data[t] for t in teams if t in data}
            return data
        except Exception as exc:
            print(f"[injuries] Cache read failed: {exc} — re-fetching.", file=sys.stderr)

    # Primary: API-Football
    result = _fetch_api_football()

    # Fallback: Transfermarkt (used when API key absent or API returned nothing)
    if result is None:
        print("[injuries] Falling back to Transfermarkt scrape...", file=sys.stderr)
        target_teams = teams if teams else list(_TM_TEAMS.keys())
        result = _fetch_transfermarkt(target_teams)

    if result is None:
        result = {}

    # Save to daily cache
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"[injuries] Cached to {cache_path.name}.", file=sys.stderr)
    except Exception as exc:
        print(f"[injuries] WARNING: could not write cache: {exc}", file=sys.stderr)

    # Merge manual overrides on top (TM/API may not update immediately after red cards)
    for team, players in _load_manual_absences().items():
        existing = result.setdefault(team, [])
        for p in players:
            if p not in existing:
                existing.append(p)

    if teams:
        return {t: result[t] for t in teams if t in result}
    return result
