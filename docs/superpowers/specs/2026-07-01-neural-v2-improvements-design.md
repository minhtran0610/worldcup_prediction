# Neural v2 improvements — design

Date: 2026-07-01
Status: approved by user, pending implementation plan

## Context

WC2026 group stage finished June 27 (Round of 32 knockouts began June 28, first
match South Africa vs Canada). This surfaced four issues with the current
pipeline, discussed and researched in this session:

1. The neural model has no way to weight the 72-completed-and-counting WC2026
   matches more heavily than 150 years of historical internationals, even
   though current tournament form is the strongest signal for predicting the
   knockouts.
2. The value-bet flagging logic (`min_edge`, a flat +2pp threshold) doesn't
   account for the favorite-longshot bias, and flags implausible long-shot
   "value" bets like a model saying 9% against a 6% market price.
3. The historical training corpus (`data/raw/results.parquet`, sourced from
   the Kaggle "international football results" CSV) stores extra-time-inclusive
   scores for some knockout matches instead of the 90-minute regulation score
   that 1X2 markets settle on. Verified: Croatia 2–1 England (2018 WC
   semifinal) is stored as the AET score; regulation was 1–1.
4. Reddit was floated as an additional form-sentiment source.

## Research findings (see conversation for full detail and sources)

- **Fine-tuning:** the pipeline already exposes WC2026 results to the model at
  inference time (Elo update + GRU form-sequence input via
  `_inject_completed_wc_matches` in `cli/wc2026.py`), but the network's
  *weights* never update on this data — `cli/train.py` explicitly calls
  `drop_wc2026()`, excluding all WC2026 matches from gradient training
  (deliberate, from commit `dbbf667`). Separately, `NeuralModel.fit` has no
  recency weighting at all, unlike `DixonColesModel` (exponential xi decay)
  and `XGBModel` (consumes a `sample_weight` column already computed by
  `features/context.py::derive_context`, currently just
  `{friendly: 0.2, else: 1.0}`). A full fine-tune pass on 72 samples risks
  catastrophic forgetting given the parameter count; recency + tournament
  sample-weighting reuses an existing, proven pattern at much lower risk.
- **Kaggle dataset is not a reliable WC2026 source.** Confirmed 28 WC2026 rows
  already present in `results.parquet`, which is exactly why `drop_wc2026()`
  exists — the live schedule (openfootball / API-Football, wrapped by
  `data/ingest/wc2026.py`) is the sole source of truth for 2026 matches
  throughout the codebase. Any WC2026 training data must come from that same
  live source, not the Kaggle CSV.
- **Favorite-longshot bias:** bookmakers price longshots *above* true
  probability (shorter odds than fair), the opposite of a market missing an
  underdog's value. A model claiming 9% against a 6% market price is very
  likely uncalibrated at the tail, not finding real edge. A flat
  probability-point edge threshold is the wrong shape of test — it treats a
  50%-relative overstatement (9% vs 6%) the same as a 7%-relative one (42% vs
  39%).
- **Extra-time contamination is historical-only.** The *live* 2026 pipeline is
  already correct — openfootball's JSON cleanly separates `ft`/`et`/`p` score
  keys, and `data/ingest/openfootball.py` already extracts only `ft`.
  Verified via `goalscorers.csv` (per-goal minute data): of 49,433 historical
  matches, 15,467 have goal-minute coverage; within that subset, 177 matches
  have a goal after minute 90 (went to extra time), concentrated in knockout
  stages of WC/AFCON/Euro/Asian Cup/Gold Cup/Copa América. Own-goal rows
  already credit the benefiting team (verified: Argentina 1–0 Chile, 1917 —
  the sole goal row is `team=Argentina, own_goal=True`), so reconstruction is
  a straightforward `minute <= 90` filter grouped by `team`, no own-goal
  flipping needed. The ~34k matches with no goalscorer coverage can't be
  verified and are left as-is (accepted residual limitation).
- **Reddit:** not pursued. Reddit ended self-service API app creation
  (Nov 2025, now ~1-week manual approval) and killed the unauthenticated
  `.json` scraping fallback (May 2026) — no zero-friction path exists.
  Signal quality in the literature is also mixed (tribal bias, meme/joke
  noise, brigading). Deferred indefinitely; not in scope for this round.

## Design

### A. Neural sample-weighting (recency + WC2026 boost)

**Data sourcing.** `cli/train.py` currently loads only the Kaggle CSV and
calls `drop_wc2026()`. It needs to additionally pull WC2026 match data from
the live schedule source (`data/ingest/wc2026.py::load_wc2026_schedule`),
exactly as `cli/wc2026.py` already does for predict-time context. Scope:
**all completed WC2026 matches to date** (group stage + any finished
knockout rounds, e.g. the Round of 32 matches completed as of this session),
not just the 72 group-stage games — more recent data is more signal, and
`is_knockout` is already a context feature the model consumes, so knockout
rows are tagged correctly rather than excluded.

The merge/injection logic (`_inject_completed_wc_matches` in
`cli/wc2026.py`) is reused rather than duplicated — extract it to a shared
location both `cli/train.py` and `cli/wc2026.py` call.

**Weighting.** Extend `features/context.py::derive_context`'s
`sample_weight` computation:

```
sample_weight = tournament_base_weight × recency_factor × wc2026_boost
```

- `tournament_base_weight`: unchanged (`friendly=0.2, else=1.0`).
- `recency_factor`: mild exponential decay over days-before-latest-match,
  with a long half-life (years, not Dixon-Coles' ~106-day fold-refit decay —
  this model trains once over 150 years of history, so DC's decay rate would
  erase nearly all of it). Exact half-life is a tunable hyperparameter,
  selected via held-out val RPS during implementation.
- `wc2026_boost`: a flat multiplier applied to all injected WC2026 rows
  (group stage and completed knockouts), on top of the recency factor. Exact
  value tunable via val RPS.

**Loss weighting.** `models/neural.py::_nll_loss_batch` currently returns an
unweighted `.mean()`. `NeuralModel.fit` needs a `train_weights` tensor
(built once, alongside `train_hs`/`train_as`) sliced per-batch by the same
`idx` used for `home_idx`/`away_idx`/etc., and the loss becomes a weighted
mean: `(per_sample_nll * weights).sum() / weights.sum()`. Validation loss
stays **unweighted** — early stopping should reflect an honest, comparable
held-out metric, not one skewed by the same boost being optimized for.

**Operational note:** this requires rerunning `cli/train.py` (and the
existing `--no-holdout` production refit path) to produce a new checkpoint;
no benefit until retrained.

### B. Post-match LLM trajectory signal

The existing `data/ingest/llm_form.py` extraction prompt already asks for
scoreline narrative, momentum, and performance-vs-result framing
("thumping", "lucky", "dominant") — it's largely built for this already. The
gap is retrieval: today it fetches whatever recent news the RSS feeds
surface for a team, not a report per specific completed match.

Add a retrieval path that targets each of a team's completed WC2026 matches
individually (group stage + knockouts to date), so the full tournament
trajectory is captured rather than just whatever's most recent. Since this
data is now historical/static, cache it once per team (unlike the always-fresh
pre-match buzz layer, which stays fetch-fresh). Feeds through the *same*
extraction prompt as today — this is still clean journalism text, not the
noisy crowd-sourced text that would have warranted a separate lower-trust
pipeline (that concern was specific to the Reddit option, which is out of
scope).

### C. Value-bet threshold

`eval/backtest.py::compute_value_bets` gains two additional gates alongside
the existing absolute `min_edge` check — a bet must clear all three:

- `MIN_MARKET_PROB` (proposed default 0.08): never flag a bet where the
  market-implied probability of the best-edge outcome is below this floor.
- `MIN_RELATIVE_EDGE` (proposed default 0.30): require
  `best_edge / market_prob_of_best_outcome >= MIN_RELATIVE_EDGE`, so the
  bar scales with the base rate instead of staying flat across the
  probability range.

These defaults are starting points, not final — to be validated against
CLV (closing-line-value) tracking once enough live WC2026 odds/results
accumulate (the odds-snapshot mechanism in `cli/wc2026.py` already persists
pre-match odds for this purpose), rather than hand-tuned against backtest ROI
on a handful of historical upsets.

`cli/wc2026.py` currently reimplements the edge/best-outcome/Kelly logic
inline rather than calling `compute_value_bets`, so the new thresholds would
otherwise need to be written and kept in sync twice. Consolidate into one
shared function in `eval/backtest.py` that both the backtest harness and the
live `wc2026` CLI call.

### D. Extra-time score correction

New correction step inside `data/ingest/results.py::load_results`, applied
during CSV→parquet ingestion (before caching):

1. Load `goalscorers.csv` (already parsed elsewhere by
   `data/ingest/goalscorers.py`, though that module aggregates for a
   different purpose — player pedigree — so this is a new read path over the
   same file, not a reuse of that module's output).
2. Join to `results` on `(date, home_team, away_team)`.
3. For any match where a goal row has `minute > 90`, recompute
   `home_score`/`away_score` as counts of goal rows with `minute <= 90`,
   grouped by `team` (own-goal rows already credit the benefiting team, no
   special-casing needed).
4. Matches with no goalscorer coverage are left unmodified.

Existing cached `data/raw/results.parquet` needs a `force_refresh` rebuild
from the CSV to pick this up.

**Addendum (found during planning):** `data/ingest/api_football.py` — the
*primary* WC2026 source when `API_FOOTBALL_KEY` is configured, ahead of the
openfootball fallback — has the same bug class. It reads the top-level
`goals.home`/`goals.away` field from API-Football's fixture response, which
reflects the score including extra time, instead of the `score.fulltime`
breakdown the API also provides. Not currently active (no API key set in
this environment today), but fixed alongside D as the same bug class in the
same area of code, to prevent a silent regression if the key is configured
later.

## Out of scope for this round

- Reddit as a sentiment source (deferred, see research findings above).
- A separate partial-freeze fine-tune experiment (sample-weighting only, per
  decision).
- Exact recency half-life / WC2026 boost magnitude / MIN_MARKET_PROB /
  MIN_RELATIVE_EDGE final values — implementation will pick reasonable
  starting points and these are expected to be tuned via val RPS / CLV
  tracking, not fixed by this spec.
