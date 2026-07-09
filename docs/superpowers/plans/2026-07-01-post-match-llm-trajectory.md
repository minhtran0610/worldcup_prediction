# Post-Match LLM Trajectory Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the LLM narrative-form layer a genuine multi-match "tournament trajectory" signal per team (across all of a team's completed WC2026 matches), applied as an additional, coarser λ adjustment alongside the existing always-fresh pre-match sentiment read.

**Architecture:** The existing pipeline (`data/ingest/llm_form.py`) only reads currently-live RSS feed items — there's no archive/search, so match-1 and match-2 reports have already rotated out by the time group stage ends. A new Guardian Content API client (`data/ingest/guardian_api.py`) adds real archive search with date filtering, used to retroactively fetch one report per team per completed WC2026 match. Each report runs through the *existing* extraction prompt (`analyse_team_form`, unchanged — Guardian articles are the same kind of clean journalism text the pipeline already trusts). Results are cached permanently to disk (this data is static once a match is over) and aggregated into a confidence-weighted λ multiplier, applied after the existing injury and pre-match sentiment adjustments.

**Tech Stack:** requests, pandas, pytest, monkeypatch-based HTTP mocking (no live network calls in tests).

## Global Constraints

- **Prerequisite: register a free Guardian Content API developer key** at https://bonobo.capi.gutools.co.uk/register/developer and set `GUARDIAN_API_KEY` in the environment. Every task below that touches `data/ingest/guardian_api.py` degrades gracefully (returns empty results, logs to stderr) when the key is absent, but the feature has no effect without it.
- This is a genuinely new, coarser signal layered on top of the existing pre-match sentiment adjustment, not a replacement — both journalism-derived, so `K_TRAJECTORY` is deliberately smaller than `K_SENTIMENT` (0.15 vs 0.30) and clamped to a tighter range, to avoid double-counting the same underlying "team is hot/cold" narrative twice at full strength.
- The trajectory cache (`data/cache/team_trajectory.json`) is permanent — matches already covered are never re-fetched or re-analysed. `data/cache/` is already gitignored.
- No test makes a real HTTP call; all Guardian API and LLM interactions are monkeypatched.

---

### Task 1: Guardian Content API client

**Files:**
- Create: `data/ingest/guardian_api.py`
- Create: `tests/data/ingest/test_guardian_api.py`
- (Create `tests/data/__init__.py` / `tests/data/ingest/__init__.py` if not already present from another plan)

**Interfaces:**
- Produces: `get_api_key() -> str | None`, `search_articles(query: str, from_date: pd.Timestamp, to_date: pd.Timestamp, api_key: str | None = None, page_size: int = 5) -> list[dict]` — each dict has keys `title`, `url`, `published`, `body_html` (raw HTML, not yet cleaned).

- [ ] **Step 1: Create test package directories (skip if they already exist)**

```bash
mkdir -p tests/data/ingest
touch tests/data/__init__.py tests/data/ingest/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/ingest/test_guardian_api.py`:

```python
from __future__ import annotations

import pandas as pd

import data.ingest.guardian_api as guardian_api


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_data


def test_search_articles_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    result = guardian_api.search_articles(
        "Spain Croatia", pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-17")
    )
    assert result == []


def test_search_articles_parses_response(monkeypatch):
    monkeypatch.setenv("GUARDIAN_API_KEY", "fake-key")

    def fake_get(url, params=None, timeout=None):
        assert params["q"] == "Spain Croatia"
        assert params["from-date"] == "2026-06-15"
        assert params["to-date"] == "2026-06-17"
        return _FakeResponse(
            {
                "response": {
                    "results": [
                        {
                            "webTitle": "Spain thrash Croatia",
                            "webUrl": "https://theguardian.com/x",
                            "webPublicationDate": "2026-06-15T22:00:00Z",
                            "fields": {"body": "<p>Spain were dominant.</p>"},
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(guardian_api.requests, "get", fake_get)

    result = guardian_api.search_articles(
        "Spain Croatia", pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-17")
    )
    assert len(result) == 1
    assert result[0]["title"] == "Spain thrash Croatia"
    assert result[0]["body_html"] == "<p>Spain were dominant.</p>"


def test_search_articles_returns_empty_on_request_failure(monkeypatch):
    monkeypatch.setenv("GUARDIAN_API_KEY", "fake-key")

    def fake_get(url, params=None, timeout=None):
        raise guardian_api.requests.exceptions.RequestException("boom")

    monkeypatch.setattr(guardian_api.requests, "get", fake_get)

    result = guardian_api.search_articles(
        "Spain Croatia", pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-17")
    )
    assert result == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_guardian_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.ingest.guardian_api'`

- [ ] **Step 4: Write the implementation**

Create `data/ingest/guardian_api.py`:

```python
"""Guardian Content API client — full-text article search with date filtering.

Used to retroactively recover post-match analysis for WC2026 matches whose
reports have already rotated out of the RSS feeds llm_form.py otherwise
relies on (RSS exposes only currently-live items, no archive/search).

Requires a free Guardian Content API developer key:
https://bonobo.capi.gutools.co.uk/register/developer
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import requests

_BASE_URL: str = "https://content.guardianapis.com/search"
_TIMEOUT_SECONDS: int = 20


def get_api_key() -> str | None:
    """Return GUARDIAN_API_KEY from environment, or None if not set."""
    return os.environ.get("GUARDIAN_API_KEY")


def search_articles(
    query: str,
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    api_key: str | None = None,
    page_size: int = 5,
) -> list[dict]:
    """Full-text search the Guardian Content API within a date range.

    Returns a list of {title, url, published, body_html} dicts (body_html is
    the raw article body HTML — cleaning is the caller's responsibility, so
    HTML-stripping logic stays in one place). Returns [] on any failure
    (missing key, network error, empty results) so callers can degrade
    gracefully.
    """
    if api_key is None:
        api_key = get_api_key()
    if not api_key:
        print("[guardian_api] GUARDIAN_API_KEY not set — skipping search.", file=sys.stderr)
        return []

    params = {
        "q": query,
        "from-date": from_date.strftime("%Y-%m-%d"),
        "to-date": to_date.strftime("%Y-%m-%d"),
        "show-fields": "body",
        "page-size": page_size,
        "api-key": api_key,
    }

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[guardian_api] Request failed: {exc}", file=sys.stderr)
        return []

    try:
        data: dict = resp.json()
    except ValueError as exc:
        print(f"[guardian_api] JSON parse error: {exc}", file=sys.stderr)
        return []

    results = data.get("response", {}).get("results", [])
    articles: list[dict] = []
    for r in results:
        body_html = r.get("fields", {}).get("body", "")
        articles.append(
            {
                "title": r.get("webTitle", ""),
                "url": r.get("webUrl", ""),
                "published": r.get("webPublicationDate", ""),
                "body_html": body_html,
            }
        )
    return articles
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_guardian_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add data/ingest/guardian_api.py tests/data/__init__.py tests/data/ingest/__init__.py tests/data/ingest/test_guardian_api.py
git commit -m "feat(data): add Guardian Content API client for archive search"
```

---

### Task 2: `fetch_team_match_report` in `llm_form.py`

**Files:**
- Modify: `data/ingest/llm_form.py`
- Test: `tests/data/ingest/test_llm_form.py` (new file)

**Interfaces:**
- Consumes: `guardian_api.search_articles` (Task 1), imported at module level as `from data.ingest import guardian_api` (not a local import — needed so tests can monkeypatch `llm_form.guardian_api.search_articles` directly).
- Produces: `clean_html(text: str) -> str` (renamed from the existing private `_clean_html`), `fetch_team_match_report(team: str, opponent: str, match_date: pd.Timestamp, api_key: str | None = None) -> tuple[list[str], list[str]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/ingest/test_llm_form.py`:

```python
from __future__ import annotations

import pandas as pd

import data.ingest.llm_form as llm_form


def test_fetch_team_match_report_filters_to_team_title_matches(monkeypatch):
    def fake_search(query, from_date, to_date, api_key=None):
        assert query == "Spain Croatia"
        return [
            {
                "title": "Spain thrash Croatia in group stage rout",
                "url": "https://theguardian.com/spain-croatia",
                "published": "2026-06-15T22:00:00Z",
                "body_html": "<p>Spain were utterly dominant throughout.</p>",
            },
            {
                "title": "Transfer news: unrelated gossip column",
                "url": "https://theguardian.com/unrelated",
                "published": "2026-06-15T10:00:00Z",
                "body_html": "<p>Nothing to do with this match.</p>",
            },
        ]

    monkeypatch.setattr(llm_form.guardian_api, "search_articles", fake_search)

    texts, urls = llm_form.fetch_team_match_report("Spain", "Croatia", pd.Timestamp("2026-06-15"))
    assert len(texts) == 1
    assert "dominant" in texts[0]
    assert urls == ["https://theguardian.com/spain-croatia"]


def test_fetch_team_match_report_returns_empty_when_no_results(monkeypatch):
    def fake_search(query, from_date, to_date, api_key=None):
        return []

    monkeypatch.setattr(llm_form.guardian_api, "search_articles", fake_search)

    texts, urls = llm_form.fetch_team_match_report("Spain", "Croatia", pd.Timestamp("2026-06-15"))
    assert texts == []
    assert urls == []


def test_clean_html_strips_tags_and_entities():
    assert llm_form.clean_html("<p>Spain &amp; Croatia</p>") == "Spain & Croatia"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_llm_form.py -v`
Expected: FAIL with `AttributeError: module 'data.ingest.llm_form' has no attribute 'guardian_api'`

- [ ] **Step 3: Modify `data/ingest/llm_form.py`**

Add these imports near the top, alongside the existing stdlib imports:

```python
import pandas as pd

from data.ingest import guardian_api
```

Rename `_clean_html` to `clean_html` (drop the leading underscore — it's now used outside this module) at its definition:

```python
def clean_html(text: str) -> str:
    """Strip tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
```

Update its three call sites within this file (in `_fetch_article_text` and `fetch_team_news`) from `_clean_html(...)` to `clean_html(...)`.

Add this function after `fetch_team_news`:

```python
def fetch_team_match_report(
    team: str,
    opponent: str,
    match_date: pd.Timestamp,
    api_key: str | None = None,
) -> tuple[list[str], list[str]]:
    """Fetch a post-match report for one specific completed match via the
    Guardian Content API's archive search.

    fetch_team_news (RSS-based) can't reach matches whose reports have
    already rotated out of the live feed — this function exists for exactly
    that case: retroactively recovering earlier WC2026 match reports.

    Searches a 2-day window starting on match_date. Filters results to those
    whose title mentions the team (via the existing _SEARCH_TERMS keyword
    map) to reduce false positives from the Guardian's broader query match.
    Returns ([], []) on no results or API failure.
    """
    query = f"{team} {opponent}"
    to_date = match_date + pd.Timedelta(days=2)
    articles = guardian_api.search_articles(query, match_date, to_date, api_key=api_key)

    search_terms = [t.lower() for t in _SEARCH_TERMS.get(team, [team])]
    texts: list[str] = []
    urls: list[str] = []
    for a in articles:
        title_lower = a["title"].lower()
        if not any(term in title_lower for term in search_terms):
            continue
        body = clean_html(a["body_html"])[:_MAX_COMBINED_CHARS]
        if body:
            texts.append(body)
            urls.append(a["url"])
    return texts, urls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_llm_form.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests — confirms the `_clean_html` → `clean_html` rename didn't break anything else)

- [ ] **Step 6: Commit**

```bash
git add data/ingest/llm_form.py tests/data/ingest/test_llm_form.py
git commit -m "feat(data): add Guardian-backed fetch_team_match_report to llm_form"
```

---

### Task 3: Trajectory orchestration and persistent cache

**Files:**
- Create: `data/ingest/trajectory.py`
- Test: `tests/data/ingest/test_trajectory.py`

**Interfaces:**
- Consumes: `FormAnalysis`, `DEFAULT_MODEL`, `analyse_team_form`, `fetch_team_match_report` from `data.ingest.llm_form` (Task 2), all imported by name at module level (for monkeypatching in tests).
- Produces: `get_team_trajectory(team: str, team_matches: pd.DataFrame, model: str = DEFAULT_MODEL, api_key: str | None = None) -> list[FormAnalysis]` — `team_matches` needs columns `date, home_team, away_team`, already filtered to matches involving `team`. Returns one `FormAnalysis` per match, chronological order.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/ingest/test_trajectory.py`:

```python
from __future__ import annotations

import pandas as pd

import data.ingest.trajectory as trajectory
from data.ingest.llm_form import FormAnalysis


def test_get_team_trajectory_uses_cache_and_skips_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", tmp_path / "team_trajectory.json")

    calls = []

    def fake_fetch(team, opponent, match_date, api_key=None):
        calls.append((team, opponent, match_date))
        return ["Spain were dominant."], ["https://example.com/a"]

    def fake_analyse(team, texts, urls=None, model=None):
        return FormAnalysis(team=team, form_score=0.5, confidence=0.8, n_articles=len(texts))

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_form", fake_analyse)

    matches = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
        }
    )

    first = trajectory.get_team_trajectory("Spain", matches)
    assert len(first) == 1
    assert first[0].form_score == 0.5
    assert len(calls) == 1

    second = trajectory.get_team_trajectory("Spain", matches)
    assert len(second) == 1
    assert len(calls) == 1  # not re-fetched — served from cache


def test_get_team_trajectory_chronological_order(tmp_path, monkeypatch):
    monkeypatch.setattr(trajectory, "_TRAJECTORY_CACHE_PATH", tmp_path / "team_trajectory.json")

    def fake_fetch(team, opponent, match_date, api_key=None):
        return [f"Report vs {opponent}"], ["https://example.com/x"]

    def fake_analyse(team, texts, urls=None, model=None):
        return FormAnalysis(team=team, form_score=0.1, confidence=0.5, performance_context=texts[0])

    monkeypatch.setattr(trajectory, "fetch_team_match_report", fake_fetch)
    monkeypatch.setattr(trajectory, "analyse_team_form", fake_analyse)

    matches = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-24"), pd.Timestamp("2026-06-13")],
            "home_team": ["Spain", "Spain"],
            "away_team": ["Uruguay", "Cape Verde"],
        }
    )

    out = trajectory.get_team_trajectory("Spain", matches)
    assert [a.performance_context for a in out] == [
        "Report vs Cape Verde",
        "Report vs Uruguay",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_trajectory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.ingest.trajectory'`

- [ ] **Step 3: Write the implementation**

Create `data/ingest/trajectory.py`:

```python
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

from data.ingest.llm_form import DEFAULT_MODEL, FormAnalysis, analyse_team_form, fetch_team_match_report

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
            results.append(FormAnalysis.from_dict(cache[key]))
            continue

        texts, urls = fetch_team_match_report(team, opponent, match_date, api_key=api_key)
        analysis = analyse_team_form(team, texts, urls=urls, model=model)
        cache[key] = analysis.to_dict()
        cache_dirty = True
        results.append(analysis)

    if cache_dirty:
        _save_cache(cache)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_trajectory.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add data/ingest/trajectory.py tests/data/ingest/test_trajectory.py
git commit -m "feat(data): add per-team WC2026 trajectory orchestration with disk cache"
```

---

### Task 4: Aggregate the trajectory into a λ adjustment

**Files:**
- Modify: `features/llm_form_feature.py`
- Test: `tests/features/test_llm_form_feature.py` (new file)

**Interfaces:**
- Produces: `K_TRAJECTORY: float = 0.15`, `TRAJECTORY_MIN_CONFIDENCE: float = 0.25`, `compute_trajectory_factor(analyses: list, k: float = K_TRAJECTORY, min_confidence: float = TRAJECTORY_MIN_CONFIDENCE) -> float`, `apply_trajectory_adjustment(lambda_home, lambda_away, rho, trajectory_factor_home, trajectory_factor_away) -> tuple[float, float, float]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_llm_form_feature.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest

from features.llm_form_feature import (
    K_TRAJECTORY,
    apply_trajectory_adjustment,
    compute_trajectory_factor,
)


@dataclass
class _FakeAnalysis:
    form_score: float
    confidence: float


def test_compute_trajectory_factor_no_usable_entries_returns_one():
    analyses = [_FakeAnalysis(form_score=0.9, confidence=0.1)]  # below default min_confidence
    assert compute_trajectory_factor(analyses) == pytest.approx(1.0)


def test_compute_trajectory_factor_confidence_weighted_average():
    analyses = [
        _FakeAnalysis(form_score=1.0, confidence=0.5),
        _FakeAnalysis(form_score=0.0, confidence=0.5),
    ]
    expected = 1.0 + K_TRAJECTORY * 0.5  # weighted avg form_score = 0.5
    assert compute_trajectory_factor(analyses) == pytest.approx(expected, rel=1e-6)


def test_compute_trajectory_factor_clamped_to_range():
    analyses = [_FakeAnalysis(form_score=1.0, confidence=1.0)] * 5
    factor = compute_trajectory_factor(analyses)
    assert 0.85 <= factor <= 1.15


def test_apply_trajectory_adjustment_scales_lambdas_and_passes_rho_through():
    lh_adj, la_adj, rho_adj = apply_trajectory_adjustment(2.0, 1.0, 0.05, 1.10, 0.90)
    assert lh_adj == pytest.approx(2.2)
    assert la_adj == pytest.approx(0.9)
    assert rho_adj == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features/test_llm_form_feature.py -v`
Expected: FAIL with `ImportError: cannot import name 'K_TRAJECTORY'`

- [ ] **Step 3: Add to `features/llm_form_feature.py`**

Add these constants near the existing `K_SENTIMENT` / `MIN_CONFIDENCE`:

```python
K_TRAJECTORY: float = 0.15
TRAJECTORY_MIN_CONFIDENCE: float = 0.25
_TRAJECTORY_FACTOR_MIN: float = 0.85
_TRAJECTORY_FACTOR_MAX: float = 1.15
```

Add these functions at the end of the file:

```python
def compute_trajectory_factor(
    analyses: list,
    k: float = K_TRAJECTORY,
    min_confidence: float = TRAJECTORY_MIN_CONFIDENCE,
) -> float:
    """Return a λ multiplier summarising a team's WC2026 match-by-match
    trajectory (see data.ingest.trajectory.get_team_trajectory).

    Confidence-weighted average of form_score across entries with
    confidence >= min_confidence; entries below threshold are dropped
    entirely (not zero-weighted) to avoid diluting real signal with noise.
    Returns 1.0 (no adjustment) if no entries clear the confidence bar.

    Clamped to a tighter range than compute_sentiment_factor's [0.70, 1.30]
    — this is a secondary signal correlated with the same journalism-derived
    narrative as the pre-match sentiment read, so its individual
    contribution is kept modest to avoid double-counting the same
    underlying "team is hot/cold" signal twice at full strength.
    """
    usable = [a for a in analyses if a.confidence >= min_confidence]
    if not usable:
        return 1.0

    total_weight = sum(a.confidence for a in usable)
    weighted_score = sum(a.form_score * a.confidence for a in usable) / total_weight

    factor = 1.0 + k * weighted_score
    return max(_TRAJECTORY_FACTOR_MIN, min(_TRAJECTORY_FACTOR_MAX, factor))


def apply_trajectory_adjustment(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    trajectory_factor_home: float,
    trajectory_factor_away: float,
) -> tuple[float, float, float]:
    """Return (lambda_home_adj, lambda_away_adj, rho) after trajectory scaling.

    rho passed through unchanged, matching apply_injury_adjustment and
    apply_sentiment_adjustment. Apply AFTER apply_sentiment_adjustment —
    the trajectory signal is coarser and should be layered on top of the
    fresher pre-match read, not compete with it for primacy.
    """
    lh_adj = max(lambda_home * trajectory_factor_home, 0.01)
    la_adj = max(lambda_away * trajectory_factor_away, 0.01)
    return lh_adj, la_adj, rho
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features/test_llm_form_feature.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add features/llm_form_feature.py tests/features/test_llm_form_feature.py
git commit -m "feat(features): aggregate WC2026 trajectory into a lambda adjustment"
```

---

### Task 5: Wire into `cli/wc2026.py`

**Files:**
- Modify: `cli/wc2026.py`

**Interfaces:**
- Consumes: `get_team_trajectory` (Task 3), `apply_trajectory_adjustment`, `compute_trajectory_factor` (Task 4), `get_api_key` from `data.ingest.guardian_api` (Task 1).

No dedicated test — `cli/wc2026.py` has no existing test file (consistent with the rest of the CLI layer). Verified via the full test suite (regression check) plus a manual run.

- [ ] **Step 1: Add imports**

Add alongside the existing imports in `cli/wc2026.py`:

```python
from data.ingest.guardian_api import get_api_key as get_guardian_api_key
from data.ingest.trajectory import get_team_trajectory
from features.llm_form_feature import (
    apply_sentiment_adjustment,
    apply_trajectory_adjustment,
    build_sentiment_report_line,
    compute_sentiment_factor,
    compute_trajectory_factor,
)
```

(this replaces the existing `from features.llm_form_feature import (apply_sentiment_adjustment, build_sentiment_report_line, compute_sentiment_factor)` import block — add the two new names to it rather than creating a duplicate import line).

- [ ] **Step 2: Add the CLI flag**

In `main()`'s signature, alongside the existing `llm_form` option, add:

```python
    trajectory: bool = typer.Option(
        True,
        "--trajectory/--no-trajectory",
        help="Apply WC2026 match-by-match trajectory lambda adjustment (requires GUARDIAN_API_KEY).",
    ),
```

- [ ] **Step 3: Add the trajectory adjustment block**

Insert this new block immediately after the existing "7c. LLM narrative form adjustment" block (after its closing `if form_analyses:` section ends, right before the "8. Build output rows..." comment):

```python
    # 7d. WC2026 match-by-match trajectory adjustment (requires GUARDIAN_API_KEY)
    if trajectory and get_guardian_api_key():
        typer.echo("Computing WC2026 trajectory adjustment...", err=True)
        teams_in_play = list(
            {t for row in upcoming.itertuples(index=False) for t in (row.home_team, row.away_team)}
        )
        trajectory_factors: dict[str, float] = {}
        for team in teams_in_play:
            team_matches = (
                completed[(completed["home_team"] == team) | (completed["away_team"] == team)]
                if not schedule.empty
                else completed.iloc[0:0]
            )
            if team_matches.empty:
                trajectory_factors[team] = 1.0
                continue
            analyses = get_team_trajectory(team, team_matches)
            trajectory_factors[team] = compute_trajectory_factor(analyses)

        n_trajectory = 0
        for i, uprow in enumerate(upcoming.itertuples(index=False)):
            home, away = uprow.home_team, uprow.away_team
            fh = trajectory_factors.get(home, 1.0)
            fa = trajectory_factors.get(away, 1.0)
            if fh == 1.0 and fa == 1.0:
                continue

            lh = float(pred_batch["lambda_home"].iloc[i])
            la = float(pred_batch["lambda_away"].iloc[i])
            rho = float(pred_batch["rho"].iloc[i])
            lh_adj, la_adj, _ = apply_trajectory_adjustment(lh, la, rho, fh, fa)

            grid_adj = build_grid(lh_adj, la_adj, rho)
            markets = derive_markets(grid_adj)
            goals = np.arange(grid_adj.shape[0], dtype=np.float64)

            pred_batch.at[i, "lambda_home"] = lh_adj
            pred_batch.at[i, "lambda_away"] = la_adj
            pred_batch.at[i, "prob_home"] = markets["home_win"]
            pred_batch.at[i, "prob_draw"] = markets["draw"]
            pred_batch.at[i, "prob_away"] = markets["away_win"]
            pred_batch.at[i, "expected_home"] = float(np.dot(goals, grid_adj.sum(axis=1)))
            pred_batch.at[i, "expected_away"] = float(np.dot(goals, grid_adj.sum(axis=0)))
            pred_batch.at[i, "grid"] = grid_adj
            n_trajectory += 1

        if n_trajectory:
            typer.echo(f"Trajectory adjustment applied to {n_trajectory} match(es).", err=True)
    elif trajectory:
        typer.echo("GUARDIAN_API_KEY not set — skipping trajectory adjustment.", err=True)
```

Note: this block reuses the `completed` DataFrame already built earlier in `main()` (in the "2. Fetch WC 2026 schedule and inject completed results into context" section) — no new schedule fetch needed.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 5: Manual smoke test — without a Guardian key**

Run: `wc2026 --show-all` (with `GUARDIAN_API_KEY` unset)
Expected: stderr includes `GUARDIAN_API_KEY not set — skipping trajectory adjustment.`; output otherwise unchanged from before this plan.

- [ ] **Step 6: Manual smoke test — with a Guardian key**

Register a free key at https://bonobo.capi.gutools.co.uk/register/developer, then run:

```bash
GUARDIAN_API_KEY=<your-key> wc2026 --show-all
```

Expected: stderr includes "Computing WC2026 trajectory adjustment..." followed by a "Trajectory adjustment applied to N match(es)" line (N may be 0 on the first run for teams with no completed matches yet, or if no Guardian articles are found — check `data/cache/team_trajectory.json` gets created and populated).

- [ ] **Step 7: Commit**

```bash
git add cli/wc2026.py
git commit -m "feat(cli): wire WC2026 trajectory adjustment into wc2026 predict"
```
