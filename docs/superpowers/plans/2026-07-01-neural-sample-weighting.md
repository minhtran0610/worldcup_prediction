# Neural Sample-Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the neural model's *training* weight completed WC2026 matches (group stage and knockouts played so far) heavily, via recency + tournament sample-weighting, instead of excluding them from gradient training entirely as it does today.

**Architecture:** Extend the existing (currently neural-unused) `sample_weight` column with a recency-decay term and a WC2026 boost; wire per-sample weighting into `NeuralModel`'s loss; extract the existing predict-time WC2026-injection logic out of `cli/wc2026.py` into a shared, reusable function (fixing a latent `is_knockout` mistagging bug in the process); and use that shared function in `cli/train.py` to pull live WC2026 results into the actual training set, reversing the current `drop_wc2026()`-only behavior for training purposes.

**Tech Stack:** pandas, numpy, PyTorch, pytest.

## Global Constraints

- Recency decay must be *gentle* (multi-year half-life) — this model trains once over ~150 years of history, unlike Dixon-Coles' much steeper per-refit `xi` decay (~106-day half-life). A DC-style decay here would erase almost all historical signal.
- WC2026 training rows must come from the live schedule source (`data/ingest/wc2026.py::load_wc2026_schedule`), never from the Kaggle CSV / `results.parquet` — those rows are unreliable/incomplete for 2026 (this is exactly why `drop_wc2026()` exists).
- Scope of the WC2026 boost: all completed WC2026 matches to date — group stage (`is_knockout=False`) AND any completed knockout rounds (`is_knockout=True`), not just the 72 group-stage games.
- WC2026 rows must never be excluded from *training* by the chronological validation holdout — that would defeat the point of boosting them.
- Validation-set NLL stays unweighted (uniform weights) — early stopping should reflect an honest, comparable held-out metric, not one skewed by the same boost being optimized for.
- This plan should land after the extra-time-score-correction plan (`docs/superpowers/plans/2026-07-01-extra-time-score-correction.md`), so retraining in Task 4 uses corrected historical scores. If that plan hasn't landed yet, this plan's Task 4 verification step still works, but the resulting checkpoint should be retrained again once the correction lands.

---

### Task 1: `compute_sample_weight` — recency + WC2026 boost

**Files:**
- Modify: `features/context.py`
- Test: `tests/features/test_context.py` (new file)

**Interfaces:**
- Consumes: `is_wc2026_match` from `data.ingest.results` (already exists).
- Produces: `RECENCY_HALF_LIFE_DAYS: float`, `WC2026_BOOST: float` (module constants); `compute_sample_weight(results: pd.DataFrame) -> pd.Series` — requires `date` (datetime64) and `tournament` columns.

- [ ] **Step 1: Write the failing tests**

Create `tests/features/test_context.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from features.context import (
    FRIENDLY_SAMPLE_WEIGHT,
    RECENCY_HALF_LIFE_DAYS,
    WC2026_BOOST,
    compute_sample_weight,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_most_recent_match_has_recency_factor_one():
    """The row at the latest date in the frame has days_ago=0 -> recency=1.0."""
    df = _df(
        [
            {"date": "2020-01-01", "tournament": "UEFA Euro qualification"},
            {"date": "2026-06-15", "tournament": "FIFA World Cup"},
        ]
    )
    weights = compute_sample_weight(df)
    # Row 1 is both the latest date AND a WC2026 match.
    assert weights.iloc[1] == pytest.approx(WC2026_BOOST, rel=1e-6)


def test_friendly_gets_base_weight_times_recency():
    df = _df([{"date": "2026-06-01", "tournament": "Friendly"}])
    weights = compute_sample_weight(df)
    # Sole row is also the latest date -> recency=1.0.
    assert weights.iloc[0] == pytest.approx(FRIENDLY_SAMPLE_WEIGHT, rel=1e-6)


def test_older_match_has_decayed_recency_factor():
    df = _df(
        [
            {"date": "2000-01-01", "tournament": "UEFA Euro qualification"},
            {"date": "2026-01-01", "tournament": "UEFA Euro qualification"},
        ]
    )
    weights = compute_sample_weight(df)
    days_ago = (pd.Timestamp("2026-01-01") - pd.Timestamp("2000-01-01")).days
    expected_recency = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    assert weights.iloc[0] == pytest.approx(expected_recency, rel=1e-6)
    assert weights.iloc[0] < weights.iloc[1]


def test_wc2026_match_gets_boost_on_top_of_recency():
    df = _df(
        [
            {"date": "2026-06-01", "tournament": "FIFA World Cup"},
            {"date": "2026-06-10", "tournament": "FIFA World Cup"},
        ]
    )
    weights = compute_sample_weight(df)
    expected_recency = 0.5 ** (9 / RECENCY_HALF_LIFE_DAYS)
    assert weights.iloc[0] == pytest.approx(expected_recency * WC2026_BOOST, rel=1e-6)


def test_pre_2026_world_cup_match_gets_no_boost():
    """A 2022 World Cup match is NOT a WC2026 match and must not get the boost."""
    df = _df(
        [
            {"date": "2022-12-18", "tournament": "FIFA World Cup"},
            {"date": "2026-01-01", "tournament": "Friendly"},
        ]
    )
    weights = compute_sample_weight(df)
    days_ago = (pd.Timestamp("2026-01-01") - pd.Timestamp("2022-12-18")).days
    expected_recency = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    assert weights.iloc[0] == pytest.approx(expected_recency, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features/test_context.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_sample_weight'`

- [ ] **Step 3: Modify `features/context.py`**

Add `import numpy as np` and the `is_wc2026_match` import alongside the existing imports:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from data.ingest.results import is_wc2026_match

if TYPE_CHECKING:
    from features.squad_registry import SquadRegistry
```

Add the new constants next to the existing ones:

```python
FRIENDLY_SAMPLE_WEIGHT: float = 0.2
FRIENDLY_TOURNAMENT: str = "Friendly"
RECENCY_HALF_LIFE_DAYS: float = 365.0 * 3
WC2026_BOOST: float = 8.0
```

Add this function above `derive_context`:

```python
def compute_sample_weight(results: pd.DataFrame) -> pd.Series:
    """Return per-row training weight: tournament_base x recency x wc2026_boost.

    tournament_base: FRIENDLY_SAMPLE_WEIGHT for friendlies, 1.0 otherwise.
    recency: exponential decay with a RECENCY_HALF_LIFE_DAYS half-life,
      relative to the most recent date in `results` — a gentle decay
      (years, not days) since this trains once over ~150 years of history,
      unlike Dixon-Coles' much steeper per-refit xi decay.
    wc2026_boost: WC2026_BOOST multiplier on WC 2026 matches, on top of
      recency — current tournament form is the strongest signal for
      predicting the knockouts, and deserves more than the recency curve
      alone would give it.
    """
    base = results["tournament"].apply(
        lambda t: FRIENDLY_SAMPLE_WEIGHT if t == FRIENDLY_TOURNAMENT else 1.0
    )
    dates = results["date"]
    latest = dates.max()
    days_ago = (latest - dates).dt.days.clip(lower=0)
    recency = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    boost = np.where(is_wc2026_match(results), WC2026_BOOST, 1.0)
    return base * recency * boost
```

Replace the existing inline computation inside `derive_context`:

```python
    out["sample_weight"] = out["tournament"].apply(
        lambda t: FRIENDLY_SAMPLE_WEIGHT if t == FRIENDLY_TOURNAMENT else 1.0
    )
```

with:

```python
    out["sample_weight"] = compute_sample_weight(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features/test_context.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests, including `tests/features/test_elo.py` and `tests/features/test_form.py`, which don't touch `derive_context` directly but confirm nothing else broke)

- [ ] **Step 6: Commit**

```bash
git add features/context.py tests/features/test_context.py
git commit -m "feat(features): add recency + WC2026-boosted sample_weight"
```

---

### Task 2: Weighted loss in `NeuralModel`

**Files:**
- Modify: `models/neural.py`
- Test: `tests/models/test_neural.py`

**Interfaces:**
- Produces: `_nll_loss_batch(lambda_home, lambda_away, rho, home_scores, away_scores, weights: torch.Tensor) -> torch.Tensor` — `weights` is now a required parameter (was previously absent; the function did a plain `.mean()`).
- `NeuralModel.fit` now builds `train_weights` from the `sample_weight` column (falling back to all-ones if the column is absent) and slices it per-batch alongside `home_idx`/`away_idx`; validation loss uses all-ones weights unconditionally.

- [ ] **Step 1: Write the failing tests**

Add to `tests/models/test_neural.py` (append near the end, before any "unfitted model raises" section, or at the very end of the file):

```python
# ---------------------------------------------------------------------------
# Weighted loss
# ---------------------------------------------------------------------------


def test_nll_loss_batch_weighted_matches_manual_weighted_mean():
    from models.neural import _nll_loss_batch

    lambda_home = torch.tensor([1.0, 2.0])
    lambda_away = torch.tensor([1.0, 2.0])
    rho = torch.tensor([0.0, 0.0])
    home_scores = torch.tensor([1, 2])
    away_scores = torch.tensor([1, 2])
    weights = torch.tensor([1.0, 3.0])

    loss_0 = _nll_loss_batch(
        lambda_home[:1], lambda_away[:1], rho[:1], home_scores[:1], away_scores[:1],
        torch.tensor([1.0]),
    )
    loss_1 = _nll_loss_batch(
        lambda_home[1:], lambda_away[1:], rho[1:], home_scores[1:], away_scores[1:],
        torch.tensor([1.0]),
    )

    weighted = _nll_loss_batch(lambda_home, lambda_away, rho, home_scores, away_scores, weights)
    expected = (loss_0 * 1.0 + loss_1 * 3.0) / (1.0 + 3.0)
    assert weighted.item() == pytest.approx(expected.item(), abs=1e-5)


def test_nll_loss_batch_uniform_weights_equals_plain_mean():
    from models.neural import _nll_loss_batch

    lambda_home = torch.tensor([1.0, 2.0, 1.5])
    lambda_away = torch.tensor([1.0, 2.0, 1.5])
    rho = torch.tensor([0.0, 0.0, 0.0])
    home_scores = torch.tensor([1, 2, 0])
    away_scores = torch.tensor([1, 2, 1])

    uniform_weights = torch.ones(3)
    weighted = _nll_loss_batch(
        lambda_home, lambda_away, rho, home_scores, away_scores, uniform_weights
    )

    per_sample = [
        _nll_loss_batch(
            lambda_home[i : i + 1], lambda_away[i : i + 1], rho[i : i + 1],
            home_scores[i : i + 1], away_scores[i : i + 1], torch.tensor([1.0]),
        ).item()
        for i in range(3)
    ]
    expected = sum(per_sample) / 3
    assert weighted.item() == pytest.approx(expected, abs=1e-5)


def test_neural_fit_respects_nonuniform_sample_weight_without_crashing():
    """fit() must accept a non-uniform sample_weight column and still produce
    valid (probability-summing-to-one) predictions."""
    results = make_results(120)
    results["sample_weight"] = [1.0] * 60 + [5.0] * 60
    model = _small_model()
    model.fit(results)

    out = model.predict_batch(results.head(5))
    totals = out["prob_home"] + out["prob_draw"] + out["prob_away"]
    for total in totals:
        assert total == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_neural.py -k "nll_loss_batch or nonuniform_sample_weight" -v`
Expected: FAIL — `_nll_loss_batch() missing 1 required positional argument: 'weights'`

- [ ] **Step 3: Modify `models/neural.py`**

Change the `_nll_loss_batch` signature and final two lines:

```python
def _nll_loss_batch(
    lambda_home: torch.Tensor,
    lambda_away: torch.Tensor,
    rho: torch.Tensor,
    home_scores: torch.Tensor,
    away_scores: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean scoreline NLL over a batch, computed entirely in PyTorch
    for autograd. weights must be shape (batch,), same length as lambda_home.

    Reimplements build_grid in torch ops so gradients flow through lambda_home,
    lambda_away, and rho.
    """
```

(keep the body identical down to the `grid = grid / grid_sum` line), then replace the tail:

```python
    p = grid[batch_idx, h_idx, a_idx].clamp(min=_LOG_CLIP)
    return -torch.log(p).mean()
```

with:

```python
    p = grid[batch_idx, h_idx, a_idx].clamp(min=_LOG_CLIP)
    nll = -torch.log(p)
    return (nll * weights).sum() / weights.sum().clamp(min=_LOG_CLIP)
```

In `NeuralModel.fit`, immediately after the block that builds `train_hs`/`train_as`:

```python
        n_train = len(train_df)
        train_hs = torch.from_numpy(train_df["home_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )
        train_as = torch.from_numpy(train_df["away_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )
```

add:

```python
        if "sample_weight" in train_df.columns:
            train_weights = torch.from_numpy(
                train_df["sample_weight"].to_numpy(dtype=np.float32).copy()
            ).to(self._device)
        else:
            train_weights = torch.ones(n_train, dtype=torch.float32, device=self._device)
```

In the training batch loop, change:

```python
                home_idx = train_home_idx[idx]
                away_idx = train_away_idx[idx]
                home_seq = train_home_seq[idx]
                away_seq = train_away_seq[idx]
                ctx = train_ctx[idx]
                hs = train_hs[idx]
                as_ = train_as[idx]

                optimizer.zero_grad()
                lh, la, rho = self._net(home_idx, away_idx, home_seq, away_seq, ctx)
                loss = _nll_loss_batch(lh, la, rho, hs, as_)
                loss.backward()
                optimizer.step()
```

to:

```python
                home_idx = train_home_idx[idx]
                away_idx = train_away_idx[idx]
                home_seq = train_home_seq[idx]
                away_seq = train_away_seq[idx]
                ctx = train_ctx[idx]
                hs = train_hs[idx]
                as_ = train_as[idx]
                w = train_weights[idx]

                optimizer.zero_grad()
                lh, la, rho = self._net(home_idx, away_idx, home_seq, away_seq, ctx)
                loss = _nll_loss_batch(lh, la, rho, hs, as_, w)
                loss.backward()
                optimizer.step()
```

Right after the block that builds `val_hs`/`val_as`, add a uniform-weights tensor:

```python
        val_hs = torch.from_numpy(val_df["home_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )
        val_as = torch.from_numpy(val_df["away_score"].to_numpy(dtype=np.int64).copy()).to(
            self._device
        )
        val_weights = torch.ones(len(val_df), dtype=torch.float32, device=self._device)
```

Change the per-epoch validation call:

```python
                val_nll = _nll_loss_batch(lh_v, la_v, rho_v, val_hs, val_as).item()
```

to:

```python
                val_nll = _nll_loss_batch(lh_v, la_v, rho_v, val_hs, val_as, val_weights).item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_neural.py -v`
Expected: PASS (all tests in the file, including the 3 new ones)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add models/neural.py tests/models/test_neural.py
git commit -m "feat(neural): weight training loss by sample_weight"
```

---

### Task 3: Extract and fix `inject_completed_wc2026_matches`

**Files:**
- Modify: `data/ingest/wc2026.py`
- Modify: `cli/wc2026.py`
- Test: `tests/data/ingest/test_wc2026.py` (new file — if Task 1 of the extra-time-score-correction plan already created `tests/data/ingest/__init__.py`, reuse it; otherwise create it here)

**Interfaces:**
- Produces: `WC2026_KNOCKOUT_START: pd.Timestamp` (`2026-06-28`), `inject_completed_wc2026_matches(results: pd.DataFrame, completed: pd.DataFrame, registry: SquadRegistry) -> pd.DataFrame` in `data/ingest/wc2026.py`.
- Removes: the private `_inject_completed_wc_matches` function from `cli/wc2026.py` (moved, not duplicated).

**Note on the bug fixed here:** the original `_inject_completed_wc_matches` set `to_inject["is_knockout"] = True` unconditionally — every injected completed match, including group-stage games, was tagged as a knockout match. This was harmless today because `is_knockout` is only read from the neural context-feature builder (`_build_context_tensor`), and injected rows are currently only ever used as *form-sequence lookup context* (which doesn't read `is_knockout`), never forward-passed directly or trained on. Task 4 changes that — these rows become real training rows, where `is_knockout` **is** read as a context feature — so this mistagging would otherwise start actively corrupting the model's group-stage/knockout distinction. Fixed here by tagging via date instead of a hardcoded `True`.

- [ ] **Step 1: Write the failing tests**

Ensure `tests/data/ingest/__init__.py` exists (create if not already present from another plan):

```bash
mkdir -p tests/data/ingest
touch tests/data/__init__.py tests/data/ingest/__init__.py
```

Create `tests/data/ingest/test_wc2026.py`:

```python
from __future__ import annotations

import pandas as pd

from data.ingest.wc2026 import inject_completed_wc2026_matches


class _StubRegistry:
    """Minimal stand-in for SquadRegistry — avoids network/file IO in tests."""

    def get_features(self, team: str, year: int, tournament: str) -> dict:
        return {"top5_share": 0.5, "avg_caps_norm": 0.3, "intl_goals_per_cap": 0.1}


def _base_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "home_team": ["Argentina", "France"],
            "away_team": ["Brazil", "Germany"],
            "home_score": [2, 1],
            "away_score": [1, 1],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [True, True],
        }
    )


def test_group_stage_match_tagged_not_knockout():
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
            "home_score": [2],
            "away_score": [0],
        }
    )
    out = inject_completed_wc2026_matches(results, completed, _StubRegistry())
    injected = out[out["home_team"] == "Spain"].iloc[0]
    assert injected["is_knockout"] == False  # noqa: E712


def test_knockout_stage_match_tagged_knockout():
    """South Africa vs Canada, 2026-06-28, is the first Round of 32 match —
    the day after the group stage's final matchday (2026-06-27)."""
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-28")],
            "home_team": ["South Africa"],
            "away_team": ["Canada"],
            "home_score": [0],
            "away_score": [1],
        }
    )
    out = inject_completed_wc2026_matches(results, completed, _StubRegistry())
    injected = out[out["home_team"] == "South Africa"].iloc[0]
    assert injected["is_knockout"] == True  # noqa: E712


def test_injected_matches_are_appended_and_sorted_by_date():
    results = _base_results()
    completed = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "home_team": ["Spain"],
            "away_team": ["Croatia"],
            "home_score": [2],
            "away_score": [0],
        }
    )
    out = inject_completed_wc2026_matches(results, completed, _StubRegistry())
    assert len(out) == 3
    assert out["date"].is_monotonic_increasing


def test_empty_completed_returns_results_unchanged():
    results = _base_results()
    completed = pd.DataFrame(
        columns=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    out = inject_completed_wc2026_matches(results, completed, _StubRegistry())
    assert len(out) == len(results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/ingest/test_wc2026.py -v`
Expected: FAIL with `ImportError: cannot import name 'inject_completed_wc2026_matches'`

- [ ] **Step 3: Add the function to `data/ingest/wc2026.py`**

Add these imports at the top of the file, alongside the existing ones:

```python
from data.ingest.cache import load_cache, save_cache
from features.context import WC_2026_HOSTS
from features.elo import extend_elo_through_matches
```

Add `from typing import TYPE_CHECKING` near the top and a `TYPE_CHECKING`-guarded import (mirrors the pattern already used in `features/context.py` for the same class, avoiding a hard runtime dependency / import-cycle risk):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from features.squad_registry import SquadRegistry
```

Add this at the end of the file (after `load_wc2026_schedule`):

```python
WC2026_KNOCKOUT_START: pd.Timestamp = pd.Timestamp("2026-06-28")


def _is_knockout_stage(dates: pd.Series) -> pd.Series:
    """True for matches on/after the WC2026 knockout start date.

    Date-based rather than parsing the `stage` string: schedule sources
    disagree on stage-string format (openfootball gives "Round of 32" etc.,
    the Wikipedia scrape fallback always writes "Group"), but the WC2026
    knockout start date is fixed and known — Round of 32 began 2026-06-28
    with South Africa vs Canada, the day after the group stage's final
    matchday (2026-06-27).
    """
    return dates >= WC2026_KNOCKOUT_START


def inject_completed_wc2026_matches(
    results: pd.DataFrame,
    completed: pd.DataFrame,
    registry: SquadRegistry,
) -> pd.DataFrame:
    """Fold completed WC2026 matches into a results/context DataFrame.

    Adds Elo (carried forward from `results`'s end-state), squad-quality
    features, and context columns (is_knockout, is_host_*, rest_days, a
    sample_weight placeholder) so the merged frame matches the schema
    NeuralModel / XGBModel / derive_context-produced frames expect.

    `completed` must have: date, home_team, away_team, home_score, away_score.

    sample_weight is set to a 1.0 placeholder here — callers that need the
    recency/WC2026-boosted weight (see features.context.compute_sample_weight)
    should recompute it over the full merged frame after calling this
    function, so the boost is based on the true post-injection latest date.
    """
    to_inject = completed.copy()
    if to_inject.empty:
        return results

    to_inject["date"] = to_inject["date"].fillna(pd.Timestamp("2026-06-11"))
    to_inject = to_inject.sort_values("date").reset_index(drop=True)
    to_inject["neutral"] = True
    to_inject["tournament"] = "FIFA World Cup"
    to_inject = extend_elo_through_matches(results, to_inject)
    to_inject["country"] = "United States"
    to_inject["is_knockout"] = _is_knockout_stage(to_inject["date"])
    to_inject["is_host_home"] = to_inject["home_team"].isin(WC_2026_HOSTS)
    to_inject["is_host_away"] = to_inject["away_team"].isin(WC_2026_HOSTS)
    to_inject["rest_days_home"] = 7.0
    to_inject["rest_days_away"] = 7.0
    to_inject["sample_weight"] = 1.0
    for col in [
        "squad_top5_home",
        "squad_top5_away",
        "squad_caps_home",
        "squad_caps_away",
        "squad_goals_home",
        "squad_goals_away",
    ]:
        to_inject[col] = 0.0
    for i, wc_row in to_inject.iterrows():
        fh = registry.get_features(wc_row["home_team"], 2026, "FIFA World Cup")
        fa = registry.get_features(wc_row["away_team"], 2026, "FIFA World Cup")
        to_inject.at[i, "squad_top5_home"] = fh["top5_share"]
        to_inject.at[i, "squad_top5_away"] = fa["top5_share"]
        to_inject.at[i, "squad_caps_home"] = fh["avg_caps_norm"]
        to_inject.at[i, "squad_caps_away"] = fa["avg_caps_norm"]
        to_inject.at[i, "squad_goals_home"] = fh["intl_goals_per_cap"]
        to_inject.at[i, "squad_goals_away"] = fa["intl_goals_per_cap"]

    return (
        pd.concat([results, to_inject], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/ingest/test_wc2026.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Update `cli/wc2026.py` to use the shared function**

Change the import line:

```python
from data.ingest.wc2026 import load_wc2026_schedule
```

to:

```python
from data.ingest.wc2026 import inject_completed_wc2026_matches, load_wc2026_schedule
```

Delete the entire `_inject_completed_wc_matches` function definition (the whole block from `def _inject_completed_wc_matches(` down to its final `return (...)` statement, under the `# Completed-match injection helper` comment).

Change the call site:

```python
            results = _inject_completed_wc_matches(results, completed, registry)
```

to:

```python
            results = inject_completed_wc2026_matches(results, completed, registry)
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 7: Manual smoke test**

Run: `wc2026 --show-all`
Expected: command runs without error, prints the "Injected N completed WC 2026 match(es)" message on stderr, same as before the refactor.

- [ ] **Step 8: Commit**

```bash
git add data/ingest/wc2026.py cli/wc2026.py tests/data/__init__.py tests/data/ingest/__init__.py tests/data/ingest/test_wc2026.py
git commit -m "refactor(data): extract inject_completed_wc2026_matches, fix is_knockout mistagging"
```

---

### Task 4: Pull WC2026 results into `cli/train.py`'s training set

**Files:**
- Modify: `cli/train.py`

**Interfaces:**
- Consumes: `inject_completed_wc2026_matches`, `load_wc2026_schedule` (Task 3); `compute_sample_weight` (Task 1); `is_wc2026_match` (already exists in `data.ingest.results`).

No dedicated test file — `cli/train.py` is an orchestration script with no existing test coverage (consistent with the rest of the `cli/` layer). Verified via the full test suite (regression check) plus a manual training run.

- [ ] **Step 1: Update imports in `cli/train.py`**

Change:

```python
from data.ingest.results import drop_wc2026, load_results
from eval.metrics import aggregate_rps
from features.context import derive_context
from features.elo import compute_elo_ratings
from features.squad_registry import SquadRegistry
from models.neural import NeuralModel
```

to:

```python
from data.ingest.results import drop_wc2026, is_wc2026_match, load_results
from data.ingest.wc2026 import inject_completed_wc2026_matches, load_wc2026_schedule
from eval.metrics import aggregate_rps
from features.context import compute_sample_weight, derive_context
from features.elo import compute_elo_ratings
from features.squad_registry import SquadRegistry
from models.neural import NeuralModel
```

- [ ] **Step 2: Inject completed WC2026 matches after `derive_context`**

Find:

```python
    typer.echo("Building squad registry...", err=True)
    registry = SquadRegistry.build()
    results = derive_context(results, squad_registry=registry)

    if tournament_filter is not None:
        results = results[results["tournament"] == tournament_filter].reset_index(drop=True)
```

Replace with:

```python
    typer.echo("Building squad registry...", err=True)
    registry = SquadRegistry.build()
    results = derive_context(results, squad_registry=registry)

    typer.echo("Fetching WC 2026 schedule (live) to inject completed matches...", err=True)
    schedule = load_wc2026_schedule(force_refresh=True)
    if not schedule.empty:
        completed = (
            schedule[schedule["is_completed"]].dropna(subset=["home_score", "away_score"]).copy()
        )
        if not completed.empty:
            completed["home_score"] = completed["home_score"].astype(int)
            completed["away_score"] = completed["away_score"].astype(int)
            n_before = len(results)
            results = inject_completed_wc2026_matches(results, completed, registry)
            results["sample_weight"] = compute_sample_weight(results)
            typer.echo(
                f"Injected {len(results) - n_before} completed WC 2026 match(es) into training "
                "data; sample_weight recomputed over the full merged frame.",
                err=True,
            )

    if tournament_filter is not None:
        results = results[results["tournament"] == tournament_filter].reset_index(drop=True)
```

- [ ] **Step 3: Keep WC2026 rows in train, never in the chronological val holdout**

Find:

```python
    # 2. Chronological train/val split
    cutoff: pd.Timestamp = results["date"].max() - pd.DateOffset(months=val_months)
    train_df = results[results["date"] < cutoff].reset_index(drop=True)
    val_df = results[results["date"] >= cutoff].reset_index(drop=True)
```

Replace with:

```python
    # 2. Chronological train/val split — WC2026 rows always stay in train.
    # They're deliberately boosted via sample_weight (see compute_sample_weight)
    # so early-stopping on a genuinely held-out historical window still means
    # something; holding them out here would defeat the point of the boost,
    # and there are only ~80 of them, too few to be a meaningful val window
    # on their own anyway.
    cutoff: pd.Timestamp = results["date"].max() - pd.DateOffset(months=val_months)
    is_wc26 = is_wc2026_match(results)
    train_df = results[(results["date"] < cutoff) | is_wc26].reset_index(drop=True)
    val_df = results[(results["date"] >= cutoff) & (~is_wc26)].reset_index(drop=True)
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests — `cli/train.py` has no direct tests, this confirms no import errors or regressions elsewhere)

- [ ] **Step 5: Manual training run**

Run: `train --n-epochs 5` (small epoch count for a quick smoke test, not a real production checkpoint)

Expected stderr output includes, in order: the "Excluded N WC 2026 match(es) from training data" message (from `drop_wc2026`), the "Fetching WC 2026 schedule" message, an "Injected N completed WC 2026 match(es)... sample_weight recomputed" message, and a `Train:` row count that includes those injected matches (check the printed date range extends into 2026).

- [ ] **Step 6: Commit**

```bash
git add cli/train.py
git commit -m "feat(cli): inject completed WC2026 matches into neural training set"
```

- [ ] **Step 7: Note the operational follow-up for the user**

This plan changes what the checkpoint is trained on. Once Tasks 1–4 are merged (and ideally after the extra-time-score-correction plan has also landed), retrain the production checkpoint:

```bash
train --no-holdout
```

The `--no-holdout` production-refit path already exists in `cli/train.py` and will pick up all of this plan's changes automatically (it refits on the full `results` frame, which now includes the WC2026-boosted rows).
