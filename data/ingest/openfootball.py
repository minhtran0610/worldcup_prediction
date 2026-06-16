from __future__ import annotations

import re
import sys

import pandas as pd
import requests

_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
_TIMEOUT_SECONDS = 20

# openfootball name -> Kaggle training-data name (only where they differ)
_TEAM_NAME_MAP: dict[str, str] = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}

# Knockout placeholder patterns: "1A", "2B", "W73", "L101", "3A/B/C/D/F", etc.
_PLACEHOLDER_RE = re.compile(r"^(\d[A-Z]|[WL]\d+|\d[A-Z](\/[A-Z])+)$")

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*UTC([+-]\d+)$")


def _parse_kickoff_utc(date_str: str, time_str: str) -> pd.Timestamp | None:
    """Combine date ("2026-06-11") and time ("13:00 UTC-6") into a UTC-naive Timestamp."""
    try:
        m = _TIME_RE.match(time_str.strip())
        if m:
            hour, minute, offset_h = int(m.group(1)), int(m.group(2)), int(m.group(3))
            local = pd.Timestamp(f"{date_str} {hour:02d}:{minute:02d}:00")
            return local - pd.Timedelta(hours=offset_h)  # ty: ignore[invalid-return-type]
        # No time available — fall back to midnight on the given date
        return pd.Timestamp(date_str)  # ty: ignore[invalid-return-type]
    except Exception:
        return None


_EMPTY_COLUMNS = [
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


def _normalise_team(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name)


def _is_placeholder(name: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(name))


def fetch_wc2026_fixtures() -> pd.DataFrame:
    """Fetch WC 2026 fixtures from openfootball/worldcup.json (GitHub raw, no auth).

    Returns DataFrame matching load_wc2026_schedule() schema:
        date         (pd.Timestamp, UTC-naive, date-precision)
        home_team    (str) — mapped to Kaggle training-data name
        away_team    (str)
        home_score   (int | None) — None if not yet played
        away_score   (int | None)
        stage        (str) — e.g. "Group A - Matchday 1", "Round of 16"
        venue        (str)
        is_completed (bool)

    Skips knockout-round entries whose team names are still TBD placeholders
    (e.g. "W73", "1A", "3A/B/C/D/F").

    Returns an empty DataFrame on any failure.
    """
    try:
        resp = requests.get(_URL, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[openfootball] Request failed: {exc}", file=sys.stderr)
        return _empty_df()

    try:
        data: dict = resp.json()
    except ValueError as exc:
        print(f"[openfootball] JSON parse error: {exc}", file=sys.stderr)
        return _empty_df()

    raw_matches: list[dict] = data.get("matches", [])
    if not raw_matches:
        print("[openfootball] No matches in response.", file=sys.stderr)
        return _empty_df()

    rows: list[dict] = []
    for m in raw_matches:
        home_raw: str = m.get("team1", "")
        away_raw: str = m.get("team2", "")

        # Skip TBD knockout-bracket placeholders
        if _is_placeholder(home_raw) or _is_placeholder(away_raw):
            continue

        home = _normalise_team(home_raw)
        away = _normalise_team(away_raw)

        # Parse date + time into a UTC-naive Timestamp
        # date: "2026-06-11", time: "13:00 UTC-6"
        raw_date: str = m.get("date", "")
        raw_time: str = m.get("time", "")
        dt = _parse_kickoff_utc(raw_date, raw_time)

        # Full-time score: list [home_goals, away_goals] or absent / null
        ft = m.get("score", {}).get("ft")
        if isinstance(ft, (list, tuple)) and len(ft) == 2:
            home_score: int | None = int(ft[0])
            away_score: int | None = int(ft[1])
            is_completed = True
        else:
            home_score = None
            away_score = None
            is_completed = False

        group: str = m.get("group", "")
        round_name: str = m.get("round", "")
        stage = f"{group} - {round_name}" if group else round_name
        venue: str = (
            m.get("ground", {}).get("name", "")
            if isinstance(m.get("ground"), dict)
            else str(m.get("ground", ""))
        )

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

    if not rows:
        print("[openfootball] No usable match rows after filtering.", file=sys.stderr)
        return _empty_df()

    df = pd.DataFrame(rows)
    df["is_completed"] = df["is_completed"].astype(bool)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    n_completed = int(df["is_completed"].sum())
    print(
        f"[openfootball] Fetched {len(df)} WC 2026 fixtures ({n_completed} completed).",
        file=sys.stderr,
    )
    return df
