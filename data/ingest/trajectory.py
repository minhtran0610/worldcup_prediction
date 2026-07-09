"""Build and cache a per-team WC2026 "trajectory" — one combined FormAnalysis
across ALL of a team's completed matches, derived from Guardian archive
search rather than the always-fresh RSS feed llm_form.py otherwise uses.

The matches are fed to a single LLM call as an explicitly ordered,
chronologically-labeled sequence (see llm_form.analyse_team_trajectory), so
the model can reason about how form has changed across matches rather than
us reconstructing a trend from independently-scored, order-blind averages.

Two-tier cache, since this data is static once a match is over:
  - tier 1 (data/cache/team_match_articles.json): raw per-match article
    text/urls, keyed team|opponent|date. Avoids re-hitting the Guardian API
    for matches already fetched when a new match completes.
  - tier 2 (data/cache/team_trajectory.json): the combined trajectory
    FormAnalysis, keyed team|sorted,iso,match,dates (the team's full
    completed-match-date set). A hit skips both Guardian fetching and the
    LLM call entirely — nothing to do until a new match changes the key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from data.ingest.llm_form import (
    DEFAULT_MODEL,
    FormAnalysis,
    analyse_team_trajectory,
    fetch_team_match_report,
)

_ARTICLE_CACHE_PATH: Path = Path("data/cache/team_match_articles.json")
_TRAJECTORY_CACHE_PATH: Path = Path("data/cache/team_trajectory.json")


def _match_key(team: str, opponent: str, match_date: pd.Timestamp) -> str:
    return f"{team}|{opponent}|{match_date.date().isoformat()}"


def _trajectory_key(team: str, match_dates: list[pd.Timestamp]) -> str:
    dates_str = ",".join(sorted(d.date().isoformat() for d in match_dates))
    return f"{team}|{dates_str}"


def _load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_cache(path: Path, cache: dict, label: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        print(f"[trajectory] WARNING: could not save {label} cache — {exc}", file=sys.stderr)


def _describe_result(team: str, home_team: str, home_score: int, away_score: int) -> str:
    team_score = home_score if team == home_team else away_score
    opp_score = away_score if team == home_team else home_score
    if team_score == opp_score:
        word = "drew"
    elif team_score > opp_score:
        word = "won"
    else:
        word = "lost"
    return f"{word} {team_score}-{opp_score}"


def _get_or_fetch_match_articles(
    team: str,
    opponent: str,
    match_date: pd.Timestamp,
    cache: dict,
    api_key: str | None,
) -> tuple[list[str], list[str], bool]:
    """Return (texts, urls, cache_was_dirty) for one match's articles.

    Checks/populates the tier-1 article `cache` dict in place (caller
    persists it). Logs one line either way, so --next output shows per-match
    fetch activity instead of a single opaque summary at the end.
    """
    key = _match_key(team, opponent, match_date)
    label = f"{team} vs {opponent} ({match_date.date()})"

    if key in cache:
        entry = cache[key]
        texts, urls = entry.get("texts", []), entry.get("urls", [])
        print(
            f"[trajectory] {label}: articles cache hit ({len(texts)} article(s))",
            file=sys.stderr,
        )
        return texts, urls, False

    print(f"[trajectory] {label}: fetching Guardian archive report...", file=sys.stderr)
    texts, urls = fetch_team_match_report(team, opponent, match_date, api_key=api_key)
    cache[key] = {"texts": texts, "urls": urls}
    print(f"[trajectory] {label}: {len(texts)} article(s) fetched", file=sys.stderr)
    return texts, urls, True


def get_team_trajectory(
    team: str,
    team_matches: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> FormAnalysis:
    """Return one combined, trajectory-aware FormAnalysis across all
    completed matches in team_matches, using a two-tier persistent cache.

    team_matches must have columns: date, home_team, away_team, home_score,
    away_score (already filtered to matches involving `team`). Returns
    FormAnalysis.neutral(team) immediately (no fetch/LLM call) if
    team_matches is empty.

    The tier-2 cache (this team's full match-date set) is checked first — a
    hit returns the cached combined analysis directly, with no Guardian or
    LLM calls at all. On a miss, each match's article text is fetched via
    the tier-1 per-match cache (so matches already covered by an earlier,
    smaller match-set are not re-fetched from Guardian), the texts are
    combined into one chronologically-labeled prompt, and a single LLM call
    produces the trajectory-level FormAnalysis — which is then cached under
    the new tier-2 key. If no article was found for any match, the LLM call
    is skipped entirely and a neutral FormAnalysis is cached instead.

    A failed LLM call (Ollama down, unparseable JSON, etc.) is NOT cached —
    only a clean result (including the legitimate "no articles found" case)
    is persisted to tier-2, so a transient failure gets retried on the next
    run instead of permanently freezing that team's trajectory at "no
    signal." Tier-1 article text is still cached either way, so the retry
    doesn't re-hit Guardian.
    """
    if team_matches.empty:
        return FormAnalysis.neutral(team)

    matches = team_matches.sort_values("date").reset_index(drop=True)
    match_dates = list(matches["date"])
    traj_key = _trajectory_key(team, match_dates)

    traj_cache = _load_json_cache(_TRAJECTORY_CACHE_PATH)
    if traj_key in traj_cache:
        try:
            analysis = FormAnalysis.from_dict(traj_cache[traj_key])
            print(
                f"[trajectory] {team}: cache hit for {len(matches)}-match trajectory "
                f"({match_dates[0].date()}..{match_dates[-1].date()}) — "
                f"form {analysis.form_score:+.2f} conf {analysis.confidence:.2f}",
                file=sys.stderr,
            )
            return analysis
        except Exception:
            print(
                f"[trajectory] {team}: trajectory cache entry malformed — recomputing.",
                file=sys.stderr,
            )

    article_cache = _load_json_cache(_ARTICLE_CACHE_PATH)
    article_cache_dirty = False

    n = len(matches)
    match_blocks: list[str] = []
    all_urls: list[str] = []
    has_any_article = False

    for i, row in enumerate(matches.itertuples(index=False), start=1):
        opponent = row.away_team if row.home_team == team else row.home_team
        match_date = row.date

        texts, urls, dirty = _get_or_fetch_match_articles(
            team, opponent, match_date, article_cache, api_key
        )
        article_cache_dirty = article_cache_dirty or dirty
        if texts:
            has_any_article = True

        result = _describe_result(team, row.home_team, int(row.home_score), int(row.away_score))
        header = f"=== Match {i} of {n} — vs {opponent} ({match_date.date()}), {team} {result} ==="
        body = "\n\n".join(texts) if texts else "(no articles found for this match)"
        match_blocks.append(f"{header}\n{body}")
        all_urls.extend(urls)

    if article_cache_dirty:
        _save_json_cache(_ARTICLE_CACHE_PATH, article_cache, "article")

    if has_any_article:
        analysis = analyse_team_trajectory(team, match_blocks, urls=all_urls, model=model)
        print(
            f"[trajectory] {team}: combined analysis across {n} match(es) -> "
            f"form {analysis.form_score:+.2f} conf {analysis.confidence:.2f}",
            file=sys.stderr,
        )
    else:
        print(
            f"[trajectory] {team}: no articles found across {n} completed match(es) — no signal.",
            file=sys.stderr,
        )
        analysis = FormAnalysis.neutral(team)

    if analysis.error is None:
        traj_cache[traj_key] = analysis.to_dict()
        _save_json_cache(_TRAJECTORY_CACHE_PATH, traj_cache, "trajectory")
    else:
        # A transient failure (Ollama down, bad JSON, etc.) — don't memorialise
        # it as this team's permanent trajectory read. Article text is still
        # tier-1 cached above, so the retry on the next run is cheap (no
        # re-fetch from Guardian, just a fresh LLM call).
        print(
            f"[trajectory] {team}: not caching — analysis failed ({analysis.error}); "
            "will retry next run.",
            file=sys.stderr,
        )

    return analysis
