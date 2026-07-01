# Value-Bet Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop flagging implausible long-shot "value" bets (e.g. model 9% vs market 6%) by replacing the flat absolute-edge threshold with a combined gate: absolute edge AND a minimum market-probability floor AND a minimum relative edge — all three must pass.

**Architecture:** One new pure function, `select_value_bet`, in `eval/backtest.py` becomes the single source of truth for "is this a value bet" — the backtest harness (`compute_value_bets`, vectorized over a DataFrame via `.apply`), the live tournament CLI (`cli/wc2026.py`, called per-match, twice — once for the ASCII table, once for the Telegram formatter), and the single-match CLI (`cli/predict.py`, which duplicates the same inline edge logic three times, once per model branch) all call it, replacing every currently-duplicated inline edge computation.

**Tech Stack:** pandas, numpy, pytest.

## Global Constraints

- The favorite-longshot bias means bookmakers price longshots ABOVE their true probability — a model claiming 9% against a market's 6% is more likely uncalibrated than value. This is why a probability floor and a relative-edge ratio are added, not just kept as a flat absolute edge.
- Default new thresholds: `MIN_MARKET_PROB = 0.08` (never flag below ~8% implied probability), `MIN_RELATIVE_EDGE = 0.30` (edge must be at least 30% of the market probability). These are starting points to be tuned later against CLV tracking, not final.
- All three gates (absolute edge, floor, relative edge) must pass for a bet to be flagged as value — this is an AND, not an OR.

---

### Task 1: `select_value_bet` — the shared gating function

**Files:**
- Modify: `eval/backtest.py`
- Test: `tests/eval/test_backtest.py`

**Interfaces:**
- Produces: `MIN_MARKET_PROB: float = 0.08`, `MIN_RELATIVE_EDGE: float = 0.30` (module-level constants).
- Produces: `select_value_bet(prob_home: float, prob_draw: float, prob_away: float, market_home: float, market_draw: float, market_away: float, min_edge: float = MIN_EDGE, min_market_prob: float = MIN_MARKET_PROB, min_relative_edge: float = MIN_RELATIVE_EDGE) -> dict` returning `{"best_outcome": str, "best_edge": float, "best_market_prob": float, "is_value": bool}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/eval/test_backtest.py` (append at the end of the file):

```python
# ---------------------------------------------------------------------------
# select_value_bet — favorite-longshot-aware gating
# ---------------------------------------------------------------------------


def test_select_value_bet_rejects_longshot_below_probability_floor():
    """Model 9% vs market 6% clears the flat absolute-edge threshold but must
    be rejected by the probability floor — this is the DR Congo/England case."""
    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.09, prob_draw=0.20, prob_away=0.71,
        market_home=0.06, market_draw=0.19, market_away=0.75,
    )
    assert result["best_outcome"] == "home"
    assert result["is_value"] is False


def test_select_value_bet_rejects_edge_below_relative_threshold():
    """A 2pp edge on a mid-probability outcome (42% vs 39%, ~7.7% relative)
    clears the absolute and floor gates but must fail the relative-edge gate."""
    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.42, prob_draw=0.30, prob_away=0.28,
        market_home=0.39, market_draw=0.32, market_away=0.29,
    )
    assert result["best_outcome"] == "home"
    assert result["best_edge"] == pytest.approx(0.03, abs=1e-9)
    assert result["is_value"] is False


def test_select_value_bet_accepts_bet_clearing_all_three_gates():
    """A well-supported edge on a mid-range favorite must still be flagged."""
    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        market_home=0.40, market_draw=0.30, market_away=0.30,
    )
    assert result["best_outcome"] == "home"
    assert result["is_value"] is True


def test_select_value_bet_handles_nan_market_probs():
    """NaN market probabilities (no odds available) must never be flagged
    as value, and must not raise."""
    import math

    from eval.backtest import select_value_bet

    result = select_value_bet(
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        market_home=math.nan, market_draw=math.nan, market_away=math.nan,
    )
    assert result["is_value"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_backtest.py -k select_value_bet -v`
Expected: FAIL with `ImportError: cannot import name 'select_value_bet'`

- [ ] **Step 3: Add the constants and function to `eval/backtest.py`**

Change the module-level constants block (currently `MIN_TRAIN_MONTHS` through `MIN_EDGE`):

```python
MIN_TRAIN_MONTHS: int = 36
VAL_DURATION_MONTHS: int = 6
STEP_MONTHS: int = 6
KELLY_FRACTION: float = 0.25
MIN_EDGE: float = 0.02
MIN_MARKET_PROB: float = 0.08
MIN_RELATIVE_EDGE: float = 0.30
```

Add this function above `compute_value_bets`:

```python
def select_value_bet(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    market_home: float,
    market_draw: float,
    market_away: float,
    min_edge: float = MIN_EDGE,
    min_market_prob: float = MIN_MARKET_PROB,
    min_relative_edge: float = MIN_RELATIVE_EDGE,
) -> dict:
    """Return the best-edge outcome for one match and whether it clears all
    value-bet gates.

    Three gates must ALL pass for is_value to be True:
      1. best_edge >= min_edge                             (absolute edge, pp)
      2. best_market_prob >= min_market_prob                (favorite-longshot floor)
      3. best_edge / best_market_prob >= min_relative_edge  (relative edge)

    The floor and relative-edge gates exist because the favorite-longshot
    bias means bookmakers already price longshots ABOVE their true
    probability — a model claiming e.g. 9% against a 6% market price is much
    more likely to be uncalibrated at the tail than to have found real value.
    A flat probability-point edge treats that case identically to a
    well-supported 42%-vs-39% edge, which is the wrong shape of test.

    NaN market probabilities (no odds available) always yield is_value=False.
    """
    edges = {
        "home": prob_home - market_home,
        "draw": prob_draw - market_draw,
        "away": prob_away - market_away,
    }
    markets = {"home": market_home, "draw": market_draw, "away": market_away}

    best_outcome = max(edges, key=lambda k: edges[k])
    best_edge = edges[best_outcome]
    best_market_prob = markets[best_outcome]

    passes_abs = best_edge >= min_edge
    passes_floor = best_market_prob >= min_market_prob
    passes_relative = best_market_prob > 0 and (best_edge / best_market_prob) >= min_relative_edge
    is_value = bool(passes_abs and passes_floor and passes_relative)

    return {
        "best_outcome": best_outcome,
        "best_edge": best_edge,
        "best_market_prob": best_market_prob,
        "is_value": is_value,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_backtest.py -k select_value_bet -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/backtest.py tests/eval/test_backtest.py
git commit -m "feat(eval): add favorite-longshot-aware select_value_bet gate"
```

---

### Task 2: Refactor `compute_value_bets` to use `select_value_bet`

**Files:**
- Modify: `eval/backtest.py`
- Modify: `tests/eval/test_backtest.py:207-238` (two existing tests need fixture updates)

**Interfaces:**
- Consumes: `select_value_bet` (Task 1).
- Produces: `compute_value_bets(predictions, min_edge=MIN_EDGE, min_market_prob=MIN_MARKET_PROB, min_relative_edge=MIN_RELATIVE_EDGE) -> pd.DataFrame` — same return shape as before, plus new columns `best_market_prob`, `is_value`. Rows are still filtered to only value bets (now via the `is_value` gate instead of a bare `best_edge >= min_edge` check).

- [ ] **Step 1: Update the two existing tests whose fixtures no longer clear the new relative-edge gate**

`test_compute_value_bets_outcome_won_home_correct` and `test_compute_value_bets_outcome_won_away_correct` currently use `prob_home=0.70` against `odds_home=1.80`, giving a market-implied home probability of ~55.2% and an edge of ~14.8pp — a ~26.7% relative edge, which is below the new 30% `MIN_RELATIVE_EDGE`. Bump `prob_home` so the tests still exercise `outcome_won` correctness while clearing the new gate.

In `tests/eval/test_backtest.py`, change both occurrences of this fixture:

```python
    df = _make_predictions_with_odds(
        prob_home=0.70,  # large edge on home
        prob_draw=0.15,
        prob_away=0.15,
        odds_home=1.80,
        odds_draw=4.00,
        odds_away=5.00,
        home_score=2,
        away_score=0,  # home wins
    )
```

to:

```python
    df = _make_predictions_with_odds(
        prob_home=0.75,  # large edge on home, clears the relative-edge gate too
        prob_draw=0.13,
        prob_away=0.12,
        odds_home=1.80,
        odds_draw=4.00,
        odds_away=5.00,
        home_score=2,
        away_score=0,  # home wins
    )
```

(and the matching block in `test_compute_value_bets_outcome_won_away_correct`, keeping `home_score=0, away_score=2` there unchanged — only `prob_home`/`prob_draw`/`prob_away` change).

- [ ] **Step 2: Run tests to verify these two now fail against the current (unrefactored) `compute_value_bets`**

Run: `pytest tests/eval/test_backtest.py -k "outcome_won_home_correct or outcome_won_away_correct" -v`
Expected: PASS still (the current implementation only checks `best_edge >= min_edge`, which the new fixture values also clear — this step confirms the fixture change itself didn't break anything before the refactor lands)

- [ ] **Step 3: Refactor `compute_value_bets` in `eval/backtest.py`**

Replace the body from `df["edge_home"] = ...` through `df = df[df["best_edge"] >= min_edge].copy()` with a call to `select_value_bet`:

```python
def compute_value_bets(
    predictions: pd.DataFrame,
    min_edge: float = MIN_EDGE,
    min_market_prob: float = MIN_MARKET_PROB,
    min_relative_edge: float = MIN_RELATIVE_EDGE,
) -> pd.DataFrame:
    """Filter predictions to value bets and compute staking columns.

    predictions must have columns:
        prob_home, prob_draw, prob_away           (model probabilities)
        odds_home, odds_draw, odds_away           (closing decimal odds, optional)
        home_score, away_score                    (actual result)

    Adds columns:
        margin_home, margin_draw, margin_away
        best_edge_outcome (str: "home"/"draw"/"away")
        best_edge (float)
        best_market_prob (float)
        kelly_stake (float)
        outcome_won (bool)
        flat_return (float)
        kelly_return (float)

    Returns only rows where select_value_bet() flags is_value=True — i.e.
    where the absolute edge, market-probability floor, and relative-edge
    gates all pass. See select_value_bet() docstring for why a flat
    probability-point edge alone is insufficient.
    """
    df = predictions.copy()

    has_odds = all(c in df.columns for c in ("odds_home", "odds_draw", "odds_away"))

    if has_odds:
        market_probs = df.apply(
            lambda row: remove_margin(
                {
                    "home": row["odds_home"],
                    "draw": row["odds_draw"],
                    "away": row["odds_away"],
                }
            ),
            axis=1,
            result_type="expand",
        )
        df["margin_home"] = market_probs["home"]
        df["margin_draw"] = market_probs["draw"]
        df["margin_away"] = market_probs["away"]
    else:
        df["margin_home"] = np.nan
        df["margin_draw"] = np.nan
        df["margin_away"] = np.nan

    selections = df.apply(
        lambda row: select_value_bet(
            row["prob_home"],
            row["prob_draw"],
            row["prob_away"],
            row["margin_home"],
            row["margin_draw"],
            row["margin_away"],
            min_edge=min_edge,
            min_market_prob=min_market_prob,
            min_relative_edge=min_relative_edge,
        ),
        axis=1,
        result_type="expand",
    )
    df["best_edge_outcome"] = selections["best_outcome"]
    df["best_edge"] = selections["best_edge"]
    df["best_market_prob"] = selections["best_market_prob"]
    df["is_value"] = selections["is_value"]

    df = df[df["is_value"]].copy()

    if has_odds and len(df) > 0:
        odds_map = {"home": "odds_home", "draw": "odds_draw", "away": "odds_away"}
        best_decimal_odds = df.apply(
            lambda row: row[odds_map[row["best_edge_outcome"]]], axis=1
        ).astype(float)
        df["kelly_stake"] = KELLY_FRACTION * df["best_edge"] / best_decimal_odds
    else:
        df["kelly_stake"] = np.nan

    actual_outcome = df.apply(
        lambda row: (
            "home"
            if row["home_score"] > row["away_score"]
            else ("draw" if row["home_score"] == row["away_score"] else "away")
        ),
        axis=1,
    )
    df["outcome_won"] = actual_outcome == df["best_edge_outcome"]

    if has_odds and len(df) > 0:
        odds_map2 = {"home": "odds_home", "draw": "odds_draw", "away": "odds_away"}
        best_decimal_odds2 = df.apply(
            lambda row: row[odds_map2[row["best_edge_outcome"]]], axis=1
        ).astype(float)
        df["flat_return"] = np.where(
            df["outcome_won"],
            best_decimal_odds2 - 1.0,
            -1.0,
        )
        df["kelly_return"] = np.where(
            df["outcome_won"],
            df["kelly_stake"] * (best_decimal_odds2 - 1.0),
            -df["kelly_stake"],
        )
    else:
        df["flat_return"] = np.nan
        df["kelly_return"] = np.nan

    return df
```

- [ ] **Step 4: Run the full backtest test suite**

Run: `pytest tests/eval/test_backtest.py -v`
Expected: PASS (all tests, including the two updated fixtures and the four `select_value_bet` tests from Task 1)

- [ ] **Step 5: Commit**

```bash
git add eval/backtest.py tests/eval/test_backtest.py
git commit -m "refactor(eval): route compute_value_bets through select_value_bet"
```

---

### Task 3: Consolidate `cli/wc2026.py`'s duplicated edge logic

**Files:**
- Modify: `cli/wc2026.py`

**Interfaces:**
- Consumes: `select_value_bet`, `MIN_MARKET_PROB`, `MIN_RELATIVE_EDGE` from `eval.backtest` (Task 1).

- [ ] **Step 1: Update the import line**

Change:

```python
from eval.backtest import KELLY_FRACTION
```

to:

```python
from eval.backtest import KELLY_FRACTION, MIN_MARKET_PROB, MIN_RELATIVE_EDGE, select_value_bet
```

- [ ] **Step 2: Add CLI options for the new thresholds**

In the `main()` function signature, alongside the existing `min_edge: float = typer.Option(0.02, help="Minimum edge to flag as value bet"),` line, add:

```python
    min_edge: float = typer.Option(0.02, help="Minimum edge to flag as value bet"),
    min_market_prob: float = typer.Option(
        MIN_MARKET_PROB, help="Minimum market-implied probability to consider for a value bet"
    ),
    min_relative_edge: float = typer.Option(
        MIN_RELATIVE_EDGE, help="Minimum edge as a fraction of market probability"
    ),
```

- [ ] **Step 3: Replace the ASCII-table-path edge block**

Find this block (inside the main match-output loop, in the `if has_api_key and live_odds:` branch):

```python
            raw_odds_dict = {
                "home": raw_odds_home,
                "draw": raw_odds_draw,
                "away": raw_odds_away,
            }
            market_probs = remove_margin(raw_odds_dict)

            edges = {
                "home": (prob_home - market_probs["home"], raw_odds_home),
                "draw": (prob_draw - market_probs["draw"], raw_odds_draw),
                "away": (prob_away - market_probs["away"], raw_odds_away),
            }
            best_outcome = max(edges, key=lambda k: edges[k][0])
            best_edge, best_decimal_odds = edges[best_outcome]

            is_value = best_edge >= min_edge

            kelly = (
                KELLY_FRACTION * best_edge / best_decimal_odds
                if is_value and best_decimal_odds > 1.0
                else 0.0
            )
```

Replace with:

```python
            raw_odds_dict = {
                "home": raw_odds_home,
                "draw": raw_odds_draw,
                "away": raw_odds_away,
            }
            market_probs = remove_margin(raw_odds_dict)

            selection = select_value_bet(
                prob_home, prob_draw, prob_away,
                market_probs["home"], market_probs["draw"], market_probs["away"],
                min_edge=min_edge,
                min_market_prob=min_market_prob,
                min_relative_edge=min_relative_edge,
            )
            best_outcome = selection["best_outcome"]
            best_edge = selection["best_edge"]
            is_value = selection["is_value"]
            best_decimal_odds = raw_odds_dict[best_outcome]

            kelly = (
                KELLY_FRACTION * best_edge / best_decimal_odds
                if is_value and best_decimal_odds > 1.0
                else 0.0
            )
```

- [ ] **Step 4: Replace the Telegram-path edge block**

Find this block (inside the `if telegram:` loop):

```python
                    if raw_d > 0.0 and np.isfinite(raw_d):
                        raw_odds_dict = {"home": raw_h, "draw": raw_d, "away": raw_a}
                        _market_probs = remove_margin(raw_odds_dict)
                        edges = {
                            "home": (prob_home - _market_probs["home"], raw_h),
                            "draw": (prob_draw - _market_probs["draw"], raw_d),
                            "away": (prob_away - _market_probs["away"], raw_a),
                        }
                        _best_outcome = max(edges, key=lambda k: edges[k][0])
                        _best_edge, _best_decimal_odds = edges[_best_outcome]
                        _is_value = _best_edge >= min_edge
                        _kelly = (
                            KELLY_FRACTION * _best_edge / _best_decimal_odds
                            if _is_value and _best_decimal_odds > 1.0
                            else 0.0
                        )
```

Replace with:

```python
                    if raw_d > 0.0 and np.isfinite(raw_d):
                        raw_odds_dict = {"home": raw_h, "draw": raw_d, "away": raw_a}
                        _market_probs = remove_margin(raw_odds_dict)
                        _selection = select_value_bet(
                            prob_home, prob_draw, prob_away,
                            _market_probs["home"], _market_probs["draw"], _market_probs["away"],
                            min_edge=min_edge,
                            min_market_prob=min_market_prob,
                            min_relative_edge=min_relative_edge,
                        )
                        _best_outcome = _selection["best_outcome"]
                        _best_edge = _selection["best_edge"]
                        _is_value = _selection["is_value"]
                        _best_decimal_odds = raw_odds_dict[_best_outcome]
                        _kelly = (
                            KELLY_FRACTION * _best_edge / _best_decimal_odds
                            if _is_value and _best_decimal_odds > 1.0
                            else 0.0
                        )
```

- [ ] **Step 5: Run the full test suite to confirm nothing else broke**

Run: `pytest -v`
Expected: PASS (all tests — `cli/wc2026.py` has no dedicated test file, consistent with the rest of the CLI layer, so this is a syntax/import sanity check via the suite plus the manual check below)

- [ ] **Step 6: Manual smoke test**

Run: `wc2026 --show-all`
Expected: command runs without error and prints the match table; if `THE_ODDS_API_KEY` is set and live odds are available, spot-check that no long-shot (<8% implied probability) row is marked with a `*` value-bet marker even if its raw edge would have cleared the old flat 2pp threshold.

- [ ] **Step 7: Commit**

```bash
git add cli/wc2026.py
git commit -m "refactor(cli): route wc2026 value-bet display through select_value_bet"
```

---

### Task 4: Consolidate `cli/predict.py`'s duplicated edge logic

**Files:**
- Modify: `cli/predict.py`

**Interfaces:**
- Consumes: `select_value_bet`, `MIN_MARKET_PROB`, `MIN_RELATIVE_EDGE` from `eval.backtest` (Task 1).

`cli/predict.py` is the single-match CLI (`predict --home X --away Y`) — the canonical usage example in this project's design brief. It has the exact same edge/best-outcome selection logic duplicated **three times**: once each in the `dc`, `elo`, and `neural` model branches. All three currently use the bare `MIN_EDGE` constant with no probability floor or relative-edge gate — without this task, `predict` would still reproduce the DR Congo/England problem even after Tasks 1–3 land.

The `dc` branch (model probs in a dict named `markets`) and the `neural` branch (also named `markets`) contain **textually identical** blocks — apply the same replacement to both. The `elo` branch uses a differently-named dict (`probs`) — apply the analogous replacement there.

No dedicated test — `cli/predict.py` has no existing test file (consistent with the rest of the CLI layer). Verified via the full test suite (regression check) plus manual runs of all three model branches.

- [ ] **Step 1: Update the import line**

Change:

```python
from eval.backtest import KELLY_FRACTION, MIN_EDGE
```

to:

```python
from eval.backtest import KELLY_FRACTION, MIN_EDGE, MIN_MARKET_PROB, MIN_RELATIVE_EDGE, select_value_bet
```

- [ ] **Step 2: Replace the edge block in the `dc` branch (and identically in the `neural` branch)**

This exact block appears twice, verbatim — once under `if model == "dc":` (around line 101) and once under `elif model == "neural":` (around line 295). Apply the same replacement both times.

Find:

```python
            edge_home = markets["home_win"] - market_probs["home"]
            edge_draw = markets["draw"] - market_probs["draw"]
            edge_away = markets["away_win"] - market_probs["away"]

            edges = {
                "home": (edge_home, odds_home),
                "draw": (edge_draw, odds_draw),
                "away": (edge_away, odds_away),
            }
            best_outcome = max(edges, key=lambda k: edges[k][0])
            best_edge, best_decimal_odds = edges[best_outcome]

            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )

            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100

            if best_edge >= MIN_EDGE:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")
```

Replace with:

```python
            edge_home = markets["home_win"] - market_probs["home"]
            edge_draw = markets["draw"] - market_probs["draw"]
            edge_away = markets["away_win"] - market_probs["away"]

            selection = select_value_bet(
                markets["home_win"], markets["draw"], markets["away_win"],
                market_probs["home"], market_probs["draw"], market_probs["away"],
                min_edge=MIN_EDGE, min_market_prob=MIN_MARKET_PROB, min_relative_edge=MIN_RELATIVE_EDGE,
            )
            best_outcome = selection["best_outcome"]
            best_edge = selection["best_edge"]
            is_value = selection["is_value"]
            best_decimal_odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}[best_outcome]

            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )

            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100

            if is_value:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")
```

- [ ] **Step 3: Replace the edge block in the `elo` branch**

Find (under `elif model == "elo":`, around line 166):

```python
            edge_home = probs["home_win"] - market_probs["home"]
            edge_draw = probs["draw"] - market_probs["draw"]
            edge_away = probs["away_win"] - market_probs["away"]

            edges = {
                "home": (edge_home, odds_home),
                "draw": (edge_draw, odds_draw),
                "away": (edge_away, odds_away),
            }
            best_outcome = max(edges, key=lambda k: edges[k][0])
            best_edge, best_decimal_odds = edges[best_outcome]

            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )

            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100

            if best_edge >= MIN_EDGE:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")
```

Replace with:

```python
            edge_home = probs["home_win"] - market_probs["home"]
            edge_draw = probs["draw"] - market_probs["draw"]
            edge_away = probs["away_win"] - market_probs["away"]

            selection = select_value_bet(
                probs["home_win"], probs["draw"], probs["away_win"],
                market_probs["home"], market_probs["draw"], market_probs["away"],
                min_edge=MIN_EDGE, min_market_prob=MIN_MARKET_PROB, min_relative_edge=MIN_RELATIVE_EDGE,
            )
            best_outcome = selection["best_outcome"]
            best_edge = selection["best_edge"]
            is_value = selection["is_value"]
            best_decimal_odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}[best_outcome]

            typer.echo("")
            typer.echo("  --- vs market (margin-removed) ---")
            typer.echo(
                f"  Market 1X2:  {_format_pct(market_probs['home'])} / "
                f"{_format_pct(market_probs['draw'])} / "
                f"{_format_pct(market_probs['away'])}"
            )
            typer.echo(
                f"  Edge:  Home {edge_home * 100:+.1f}pp"
                f"  Draw {edge_draw * 100:+.1f}pp"
                f"  Away {edge_away * 100:+.1f}pp"
            )

            label = best_outcome.capitalize()
            ev_pct = best_edge / market_probs[best_outcome] * 100

            if is_value:
                kelly_pct = KELLY_FRACTION * best_edge / best_decimal_odds * 100
                typer.echo(
                    f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%  [value]"
                )
                typer.echo(f"  Kelly (1/4):  {kelly_pct:.1f}% of bankroll")
            else:
                typer.echo(f"  Best:  {label} {best_edge * 100:+.1f}pp -> EV {ev_pct:+.1f}%")
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: PASS (all tests)

- [ ] **Step 5: Manual smoke test — reproduce and confirm the fix for the original complaint**

Run: `predict --home "DR Congo" --away England --model dc --odds-home 16.0 --odds-draw 6.5 --odds-away 1.18`
Expected: even if DR Congo's model probability comes out above the flat 2pp-edge threshold, the "Best:" line must NOT show `[value]` if DR Congo's implied probability is below the 8% floor — confirm this against the printed "Market 1X2" line's home percentage.

Also run the `elo` and `neural` branches once each (`--model elo`, `--model neural`) with the same odds to confirm all three branches behave consistently.

- [ ] **Step 6: Commit**

```bash
git add cli/predict.py
git commit -m "refactor(cli): route predict value-bet display through select_value_bet"
```
