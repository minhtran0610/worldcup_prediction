"""Build and cache a per-team WC2026 "trajectory" — one FormAnalysis per
completed match, derived from Guardian archive search rather than the
always-fresh RSS feed llm_form.py otherwise uses.

This data is static once a match is over, so results are cached to disk
permanently (unlike the always-fresh pre-match narrative layer) — matches
already covered are never re-fetched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from data.ingest.llm_form import (
    DEFAULT_MODEL,
    FormAnalysis,
    analyse_team_form,
    fetch_team_match_report,
)

_TRAJECTORY_CACHE_PATH: Path = Path("data/cache/team_trajectory.json")


def _cache_key(team: str, opponent: str, match_date: pd.Timestamp) -> str:
    return f"{team}|{opponent}|{match_date.date().isoformat()}"


def _load_cache() -> dict[str, dict]:
    if not _TRAJECTORY_CACHE_PATH.exists():
        return {}
    try:
        with _TRAJECTORY_CACHE_PATH.open() as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    try:
        _TRAJECTORY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TRAJECTORY_CACHE_PATH.open("w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        print(f"[trajectory] WARNING: could not save cache — {exc}", file=sys.stderr)


def get_team_trajectory(
    team: str,
    team_matches: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[FormAnalysis]:
    """Return one FormAnalysis per completed match in team_matches, in
    chronological order, using a persistent on-disk cache.

    team_matches must have columns: date, home_team, away_team (already
    filtered to matches involving `team`). Matches already present in the
    cache are returned without re-fetching or re-running the LLM. New
    matches are fetched via Guardian archive search, analysed, and the
    cache is updated on disk before returning.
    """
    cache = _load_cache()
    results: list[FormAnalysis] = []
    cache_dirty = False

    for row in team_matches.sort_values("date").itertuples(index=False):
        opponent = row.away_team if row.home_team == team else row.home_team
        match_date = row.date
        key = _cache_key(team, opponent, match_date)

        if key in cache:
            try:
                results.append(FormAnalysis.from_dict(cache[key]))
                continue
            except Exception:
                # Cache entry is malformed (e.g. missing required 'team' key,
                # wrong field types). Degrade gracefully by treating as cache
                # miss and re-fetching.
                pass

        texts, urls = fetch_team_match_report(team, opponent, match_date, api_key=api_key)
        analysis = analyse_team_form(team, texts, urls=urls, model=model)
        cache[key] = analysis.to_dict()
        cache_dirty = True
        results.append(analysis)

    if cache_dirty:
        _save_cache(cache)

    return results
