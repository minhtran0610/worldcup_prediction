# Extra-Time Score Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct historical training-data match scores that were recorded with extra-time-inclusive goals instead of the 90-minute regulation score that 1X2 markets settle on, and fix the same bug class in the dormant API-Football live-data path.

**Architecture:** A pure function reconstructs the regulation-time score from per-goal minute data (`goalscorers.csv`) and is applied once during CSV→parquet ingestion in `data/ingest/results.py::load_results`. A parallel one-line fix in `data/ingest/api_football.py` switches from a field that includes extra time to the API's clean regulation-time field.

**Tech Stack:** pandas, pytest.

## Global Constraints

- Matches with no `goalscorers.csv` coverage are left unmodified — their extra-time status cannot be verified, and this is an accepted limitation, not a bug to work around.
- Own-goal rows in `goalscorers.csv` already credit the benefiting team in the `team` column (verified against Argentina 1–0 Chile, 1917) — no own-goal special-casing is needed anywhere in this plan.
- `_ET_MINUTE_THRESHOLD = 90` is the cutoff between regulation and extra time throughout this plan.

---

### Task 1: `correct_extra_time_scores` — the core reconstruction function

**Files:**
- Create: `data/ingest/extra_time.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/ingest/__init__.py`
- Test: `tests/data/ingest/test_extra_time.py`

**Interfaces:**
- Produces: `correct_extra_time_scores(results: pd.DataFrame, goalscorers: pd.DataFrame) -> pd.DataFrame` — `results` needs columns `date, home_team, away_team, home_score, away_score` (plus any others, passed through unchanged); `goalscorers` needs columns `date, home_team, away_team, team, minute`. Returns a copy of `results` with `home_score`/`away_score` corrected for matches with a detected extra-time goal.
- Produces: `GOALSCORERS_CSV_DEFAULT: Path` — default path constant `Path("data/raw/goalscorers.csv")`, reused by Task 2.

- [ ] **Step 1: Create test package directories**

```bash
mkdir -p tests/data/ingest
touch tests/data/__init__.py tests/data/ingest/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/data/ingest/test_extra_time.py`:

```python
from __future__ import annotations

import pandas as pd

from data.ingest.extra_time import correct_extra_time_scores


def _results_row(date: str, home: str, away: str, hs: int, as_: int) -> dict:
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": "FIFA World Cup",
        "neutral": True,
        "country": "Russia",
    }


def _goal_row(date: str, home: str, away: str, team: str, minute: int, own_goal: bool = False) -> dict:
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "team": team,
        "scorer": "Someone",
        "minute": minute,
        "own_goal": own_goal,
        "penalty": False,
    }


def test_extra_time_match_corrected_to_regulation_score():
    """Croatia 2-1 England (2018 WC semifinal) should be corrected to 1-1 —
    Mandzukic's winner came in the 109th minute (extra time)."""
    results = pd.DataFrame([_results_row("2018-07-11", "Croatia", "England", 2, 1)])
    goalscorers = pd.DataFrame(
        [
            _goal_row("2018-07-11", "Croatia", "England", "England", 5),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 68),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 109),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 1
    assert row["away_score"] == 1


def test_match_without_et_goals_is_unchanged():
    """A match with no goal past minute 90 must be returned untouched."""
    results = pd.DataFrame([_results_row("2019-06-01", "France", "Germany", 2, 0)])
    goalscorers = pd.DataFrame(
        [
            _goal_row("2019-06-01", "France", "Germany", "France", 10),
            _goal_row("2019-06-01", "France", "Germany", "France", 80),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 2
    assert row["away_score"] == 0


def test_match_without_goalscorer_coverage_is_unchanged():
    """A match absent from goalscorers.csv must be left exactly as-is —
    its extra-time status cannot be verified."""
    results = pd.DataFrame([_results_row("1955-03-01", "Uruguay", "Peru", 3, 1)])
    goalscorers = pd.DataFrame(
        columns=["date", "home_team", "away_team", "team", "scorer", "minute", "own_goal", "penalty"]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 3
    assert row["away_score"] == 1


def test_own_goal_in_regulation_time_credited_correctly():
    """An own goal within regulation time counts toward the benefiting team
    (the `team` column already reflects this) — must not need flipping.
    A later extra-time goal is added so the match enters the correction path."""
    results = pd.DataFrame([_results_row("1917-10-06", "Argentina", "Chile", 1, 0)])
    goalscorers = pd.DataFrame(
        [
            _goal_row("1917-10-06", "Argentina", "Chile", "Argentina", 76, own_goal=True),
            _goal_row("1917-10-06", "Argentina", "Chile", "Chile", 95),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    row = out.iloc[0]
    assert row["home_score"] == 1
    assert row["away_score"] == 0


def test_multiple_matches_only_et_ones_corrected():
    """With several matches in the frame, only the one with an ET goal changes."""
    results = pd.DataFrame(
        [
            _results_row("2018-07-11", "Croatia", "England", 2, 1),
            _results_row("2018-07-10", "France", "Belgium", 1, 0),
        ]
    )
    goalscorers = pd.DataFrame(
        [
            _goal_row("2018-07-11", "Croatia", "England", "England", 5),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 68),
            _goal_row("2018-07-11", "Croatia", "England", "Croatia", 109),
            _goal_row("2018-07-10", "France", "Belgium", "France", 51),
        ]
    )
    out = correct_extra_time_scores(results, goalscorers)
    croatia_row = out[out["home_team"] == "Croatia"].iloc[0]
    france_row = out[out["home_team"] == "France"].iloc[0]
    assert (croatia_row["home_score"], croatia_row["away_score"]) == (1, 1)
    assert (france_row["home_score"], france_row["away_score"]) == (1, 0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_extra_time.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.ingest.extra_time'`

- [ ] **Step 4: Write the implementation**

Create `data/ingest/extra_time.py`:

```python
"""Reconstruct 90-minute regulation scores for matches whose stored score
includes extra time.

1X2 betting markets settle on the regulation-time result, but the Kaggle
results CSV stores the full-time score INCLUDING extra time for some
historical knockout matches (verified: Croatia 2-1 England, 2018 WC
semifinal, was 1-1 after 90 minutes — Mandzukic's winner came in the 109th
minute).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

GOALSCORERS_CSV_DEFAULT: Path = Path("data/raw/goalscorers.csv")

_ET_MINUTE_THRESHOLD: int = 90


def correct_extra_time_scores(results: pd.DataFrame, goalscorers: pd.DataFrame) -> pd.DataFrame:
    """Return results with home_score/away_score corrected to the 90-minute
    regulation score for matches that went to extra time.

    goalscorers must have columns: date, home_team, away_team, team, minute.
    A match is corrected only if goalscorers has coverage for it (matched on
    date/home_team/away_team) AND at least one goal has minute > 90 — i.e. it
    went to extra time. Matches without goalscorer coverage, or with no goal
    past minute 90, are returned unchanged.

    own_goal rows in goalscorers.csv already credit the benefiting team in
    the `team` column (verified: Argentina 1-0 Chile, 1917 — the sole goal
    row is team=Argentina, own_goal=True, scored by a Chilean player), so no
    special-casing for own goals is needed.
    """
    gs = goalscorers.copy()
    gs["date"] = pd.to_datetime(gs["date"])

    out = results.copy()
    out["date"] = pd.to_datetime(out["date"])

    if gs.empty:
        return out

    max_minute = gs.groupby(["date", "home_team", "away_team"])["minute"].max()
    et_keys = max_minute[max_minute > _ET_MINUTE_THRESHOLD].index

    if len(et_keys) == 0:
        return out

    reg_goals = gs[gs["minute"] <= _ET_MINUTE_THRESHOLD]
    reg_counts = (
        reg_goals.groupby(["date", "home_team", "away_team", "team"]).size().rename("n").reset_index()
    )

    n_corrected = 0
    for date, home, away in et_keys:
        mask = (out["date"] == date) & (out["home_team"] == home) & (out["away_team"] == away)
        if not mask.any():
            continue

        home_rows = reg_counts[
            (reg_counts["date"] == date)
            & (reg_counts["home_team"] == home)
            & (reg_counts["away_team"] == away)
            & (reg_counts["team"] == home)
        ]
        away_rows = reg_counts[
            (reg_counts["date"] == date)
            & (reg_counts["home_team"] == home)
            & (reg_counts["away_team"] == away)
            & (reg_counts["team"] == away)
        ]
        new_home_score = int(home_rows["n"].iloc[0]) if len(home_rows) else 0
        new_away_score = int(away_rows["n"].iloc[0]) if len(away_rows) else 0

        out.loc[mask, "home_score"] = new_home_score
        out.loc[mask, "away_score"] = new_away_score
        n_corrected += 1

    print(
        f"[extra_time] Corrected {n_corrected} match(es) to 90-minute regulation score.",
        file=sys.stderr,
    )
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_extra_time.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add data/ingest/extra_time.py tests/data/__init__.py tests/data/ingest/__init__.py tests/data/ingest/test_extra_time.py
git commit -m "feat(data): reconstruct 90-minute regulation scores for extra-time matches"
```

---

### Task 2: Wire the correction into `load_results`

**Files:**
- Modify: `data/ingest/results.py`
- Test: `tests/data/ingest/test_results.py`

**Interfaces:**
- Consumes: `correct_extra_time_scores`, `GOALSCORERS_CSV_DEFAULT` from `data.ingest.extra_time` (Task 1).
- Produces: `load_results(csv_path=None, force_refresh=False, goalscorers_csv_path: Path | None = None) -> pd.DataFrame` — new optional `goalscorers_csv_path` parameter, defaults to `GOALSCORERS_CSV_DEFAULT`.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/ingest/test_results.py`:

```python
from __future__ import annotations

import pandas as pd

import data.ingest.cache as cache_module
from data.ingest.results import load_results


def _write_results_csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_results_applies_extra_time_correction(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)

    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "neutral": "True",
                "country": "Russia",
            }
        ],
    )

    goalscorers_csv = tmp_path / "goalscorers.csv"
    pd.DataFrame(
        [
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "team": "England",
                "scorer": "Trippier",
                "minute": 5,
                "own_goal": False,
                "penalty": False,
            },
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "team": "Croatia",
                "scorer": "Perisic",
                "minute": 68,
                "own_goal": False,
                "penalty": False,
            },
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "team": "Croatia",
                "scorer": "Mandzukic",
                "minute": 109,
                "own_goal": False,
                "penalty": False,
            },
        ]
    ).to_csv(goalscorers_csv, index=False)

    out = load_results(csv_path=results_csv, force_refresh=True, goalscorers_csv_path=goalscorers_csv)

    row = out.iloc[0]
    assert row["home_score"] == 1
    assert row["away_score"] == 1


def test_load_results_skips_correction_when_goalscorers_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)

    results_csv = tmp_path / "results.csv"
    _write_results_csv(
        results_csv,
        [
            {
                "date": "2018-07-11",
                "home_team": "Croatia",
                "away_team": "England",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "neutral": "True",
                "country": "Russia",
            }
        ],
    )

    out = load_results(
        csv_path=results_csv,
        force_refresh=True,
        goalscorers_csv_path=tmp_path / "does_not_exist.csv",
    )
    row = out.iloc[0]
    assert row["home_score"] == 2
    assert row["away_score"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_results.py -v`
Expected: FAIL with `TypeError: load_results() got an unexpected keyword argument 'goalscorers_csv_path'`

- [ ] **Step 3: Modify `data/ingest/results.py`**

Add imports at the top (after existing `from pathlib import Path` / `import pandas as pd`):

```python
import sys
from pathlib import Path

import pandas as pd

from data.ingest.cache import load_cache, save_cache
from data.ingest.extra_time import GOALSCORERS_CSV_DEFAULT, correct_extra_time_scores
```

Change the `load_results` signature and body (the section between `df["away_score"] = df["away_score"].astype(int)` and `df = df.sort_values(...)`):

```python
def load_results(
    csv_path: Path | None = None,
    force_refresh: bool = False,
    goalscorers_csv_path: Path | None = None,
) -> pd.DataFrame:
    if not force_refresh:
        cached = load_cache(RESULTS_CACHE_KEY)
        if cached is not None:
            return cached

    if csv_path is None:
        raise FileNotFoundError(
            "No cached results found. Provide csv_path= pointing to the Kaggle results CSV."
        )

    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date"], utc=False)
    df["date"] = df["date"].dt.tz_localize(None)

    df["neutral"] = (
        df["neutral"]
        .map(lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true")
        .astype(bool)
    )

    df = df.dropna(subset=["home_score", "away_score"])

    df = df[KEEP_COLUMNS].copy()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    gs_path = Path(goalscorers_csv_path) if goalscorers_csv_path else GOALSCORERS_CSV_DEFAULT
    if gs_path.exists():
        goalscorers = pd.read_csv(gs_path)
        df = correct_extra_time_scores(df, goalscorers)
    else:
        print(
            f"[results] goalscorers CSV not found at {gs_path} — skipping extra-time correction.",
            file=sys.stderr,
        )

    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    save_cache(RESULTS_CACHE_KEY, df)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_results.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full test suite to check nothing else broke**

Run: `pytest -v`
Expected: PASS (all tests, including the previously-existing suite)

- [ ] **Step 6: Commit**

```bash
git add data/ingest/results.py tests/data/ingest/test_results.py
git commit -m "feat(data): apply extra-time score correction during results ingestion"
```

---

### Task 3: Fix the dormant API-Football extra-time bug

**Files:**
- Modify: `data/ingest/api_football.py:152-165`
- Test: `tests/data/ingest/test_api_football.py`

**Interfaces:**
- Produces: `_extract_fulltime_score(fx: dict) -> tuple[int | None, int | None]`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/ingest/test_api_football.py`:

```python
from __future__ import annotations

from data.ingest.api_football import _extract_fulltime_score


def test_extract_fulltime_score_uses_fulltime_not_goals():
    """A match decided in extra time: `goals` reflects the AET score, but
    score.fulltime is the clean 90-minute score we want."""
    fx = {
        "goals": {"home": 2, "away": 1},
        "score": {
            "halftime": {"home": 0, "away": 0},
            "fulltime": {"home": 1, "away": 1},
            "extratime": {"home": 2, "away": 1},
            "penalty": {"home": None, "away": None},
        },
    }
    home, away = _extract_fulltime_score(fx)
    assert home == 1
    assert away == 1


def test_extract_fulltime_score_not_yet_played_returns_none():
    fx = {
        "goals": {"home": None, "away": None},
        "score": {"fulltime": {"home": None, "away": None}},
    }
    home, away = _extract_fulltime_score(fx)
    assert home is None
    assert away is None


def test_extract_fulltime_score_missing_score_block_returns_none():
    fx = {"goals": {"home": 1, "away": 0}}
    home, away = _extract_fulltime_score(fx)
    assert home is None
    assert away is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_api_football.py -v`
Expected: FAIL with `ImportError: cannot import name '_extract_fulltime_score'`

- [ ] **Step 3: Modify `data/ingest/api_football.py`**

Add this function above `fetch_wc2026_fixtures` (before line 83):

```python
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
```

Inside `fetch_wc2026_fixtures`, replace these lines:

```python
        goals: dict = fx.get("goals", {})
        league_info: dict = fx.get("league", {})

        status_short: str = fixture_info.get("status", {}).get("short", "")
        is_completed: bool = status_short in _FINISHED_STATUSES

        dt = _parse_date(fixture_info.get("date", ""))
        home = _normalise_team(teams.get("home", {}).get("name", ""))
        away = _normalise_team(teams.get("away", {}).get("name", ""))

        raw_home_score = goals.get("home")
        raw_away_score = goals.get("away")
        home_score: int | None = int(raw_home_score) if raw_home_score is not None else None
        away_score: int | None = int(raw_away_score) if raw_away_score is not None else None
```

with:

```python
        league_info: dict = fx.get("league", {})

        status_short: str = fixture_info.get("status", {}).get("short", "")
        is_completed: bool = status_short in _FINISHED_STATUSES

        dt = _parse_date(fixture_info.get("date", ""))
        home = _normalise_team(teams.get("home", {}).get("name", ""))
        away = _normalise_team(teams.get("away", {}).get("name", ""))

        home_score, away_score = _extract_fulltime_score(fx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_api_football.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add data/ingest/api_football.py tests/data/ingest/test_api_football.py
git commit -m "fix(data): use score.fulltime instead of extra-time-inclusive goals field"
```

---

### Task 4: Regenerate the cached corpus and verify

**Files:** none (operational step)

- [ ] **Step 1: Force-refresh the cached results parquet**

`data/raw/` doesn't keep a persisted copy of the source CSV (only the cached `results.parquet` and the other Kaggle files like `goalscorers.csv`/`shootouts.csv`) — re-download it via the `kaggle` CLI, which is already configured in this environment:

```bash
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p /tmp/kaggle_results --unzip
python3 -c "
from pathlib import Path
from data.ingest.results import load_results
df = load_results(csv_path=Path('/tmp/kaggle_results/results.csv'), force_refresh=True)
print(len(df), 'rows loaded')
"
```

Expected stderr output includes a line like `[extra_time] Corrected 177 match(es) to 90-minute regulation score.` (exact count may differ slightly if the Kaggle CSV has been updated since this plan was written).

- [ ] **Step 2: Verify the Croatia vs England fix directly**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/raw/results.parquet')
row = df[(df['home_team']=='Croatia') & (df['away_team']=='England') & (df['date']=='2018-07-11')]
print(row[['date','home_team','away_team','home_score','away_score']])
"
```

Expected: `home_score=1, away_score=1` (previously `2, 1`).

- [ ] **Step 3: Note the operational impact for the user**

This regenerates `data/raw/results.parquet`, which is consumed by every model (`train`, `backtest`, `wc2026`, `predict`). Any previously-trained checkpoint (`checkpoints/neural.pt`) was trained on the old, uncorrected data — it should be retrained (`train` command) to benefit from this fix. No code change in this step; flag this to the user in your task handoff.
