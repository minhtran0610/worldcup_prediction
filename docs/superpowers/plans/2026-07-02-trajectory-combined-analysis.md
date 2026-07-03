# WC2026 trajectory: combined multi-match analysis — implementation plan

Date: 2026-07-02
Status: proposed, pending approval
Supersedes: the per-match-average design shipped in
`docs/superpowers/plans/2026-07-01-post-match-llm-trajectory.md` (merged as
`365ffcd`). That version is live on `neural-v2` today; this plan replaces its
analysis/aggregation step only, not the Guardian retrieval layer underneath it.

## Why

The shipped design runs one independent LLM call per completed WC2026 match,
each producing its own `FormAnalysis(form_score, confidence)`, then combines
them with a confidence-weighted numeric average
(`features/llm_form_feature.py::compute_trajectory_factor`). That average is
order-blind: a team that lost game 1 and improved through games 2-3 scores
identically to one that started strong and faded. It also can't use any
cross-match context (opponent, scoreline) because each match is analysed in
isolation.

Discussed and agreed in this session: replace the per-match-then-average
pipeline with **one LLM call per team** that reads all of a team's completed
match reports together, explicitly ordered and labeled, and produces a single
trajectory-level `FormAnalysis` directly — letting the model itself reason
about momentum/direction instead of us reconstructing it from an average.

**Decisions made this session:**
- Keep the `FormAnalysis` schema unchanged (no new trend/direction field) —
  the trajectory reasoning shows up in `form_score` + `performance_context`.
  This means `apply_trajectory_adjustment`, `K_TRAJECTORY` clamping, and the
  `cli/wc2026.py` display/adjustment code don't need to change at all — only
  how the single `FormAnalysis` gets produced changes.
- Include **all** of a team's completed matches in the combined prompt, no
  cap at N most recent — group stage is only 3 matches, and even a finalist
  tops out at 7; full-tournament arc is more valuable than a token-budget
  saving at that scale.

## Architecture

Two-tier caching, replacing the current single-tier per-match cache:

- **Tier 1 (new): raw per-match article cache** —
  `data/cache/team_match_articles.json`, keyed `team|opponent|date` (same
  scheme the old cache used). Stores `{texts: [...], urls: [...]}` per match.
  Exists so that when a new match completes, we don't re-hit the Guardian API
  for matches already fetched — only the new match needs a fresh fetch.
- **Tier 2 (replaces `data/cache/team_trajectory.json`'s current contents):
  combined-trajectory-analysis cache** — keyed `team|sorted,iso,match,dates`
  (e.g. `Spain|2026-06-14,2026-06-19,2026-06-24`). Stores one `FormAnalysis`
  (the whole-trajectory read) per key. On a hit, skips both Guardian fetching
  and the LLM call entirely — nothing to do until a new match changes the key.

Old tier-2 entries (per-match, from the previous design) simply won't match
the new key scheme and will be ignored/orphaned — no migration needed, this
is a disk cache, not persisted data.

## Task 1 — Raw per-match article cache (tier 1)

**Files:** `data/ingest/trajectory.py`, `tests/data/ingest/test_trajectory.py`

Add `_get_or_fetch_match_articles(team, opponent, match_date, api_key) ->
tuple[list[str], list[str]]`: checks `data/cache/team_match_articles.json`
for `team|opponent|date`; on miss, calls the existing
`llm_form.fetch_team_match_report`, caches `{texts, urls}`, returns it. Mirror
the existing `_load_cache`/`_save_cache` pattern already in this file
(same graceful-degrade-on-corrupt-entry behavior).

Tests: cache hit skips `fetch_team_match_report` (assert not called); cache
miss calls it and persists; corrupt entry degrades to re-fetch (matches
existing test pattern for the old per-match cache).

## Task 2 — Re-engineered combined-trajectory prompt

**Files:** `data/ingest/llm_form.py`, `tests/data/ingest/test_llm_form.py`

Add new prompt constants (separate from `_SYSTEM_PROMPT`/`_USER_TEMPLATE`,
which stay untouched — they're still used by the pre-match single-article
sentiment path and shouldn't change):

- `_SYSTEM_PROMPT_TRAJECTORY`: same extraction rules as `_SYSTEM_PROMPT`
  (results/momentum/absences), plus explicit instruction to reason about
  *change* across matches — e.g. "You are given this team's post-match
  reports in chronological order, each labeled with the match number,
  opponent, and result. Analyse how the team's form has evolved across these
  matches — is momentum building, fading, or inconsistent? Weight more recent
  matches more heavily when they conflict with earlier ones."
- `_USER_TEMPLATE_TRAJECTORY`: same output JSON schema as `_USER_TEMPLATE`
  (no schema change per the decision above), but the `{text}` slot receives
  pre-labeled, ordered match blocks rather than a flat article dump.

Add `analyse_team_trajectory(team: str, match_blocks: list[str], urls:
list[str], model: str = DEFAULT_MODEL) -> FormAnalysis`, structurally
parallel to `analyse_team_form` (same `_call_ollama`/`_validate_raw` reuse,
same empty-input short-circuit to `FormAnalysis.neutral`), differing only in
which prompt constants it uses. `match_blocks` are pre-formatted per match,
e.g.:

```
=== Match 2 of 3 — vs Japan (2026-06-19), Spain drew 1-1 ===
<cleaned article text for this match>
```

Combine blocks with the existing `"\n\n---\n\n"` separator, capped by a new
`_MAX_TRAJECTORY_CHARS` (proposed default 16000 — double the single-match
8000 cap, since we're deliberately including every match; exact value is a
tunable, not load-bearing to the design). `n_articles` on the returned
`FormAnalysis` = total article count across all matches; `sources` = all
URLs concatenated.

Tests: verify block labeling/ordering appears in the prompt sent to Ollama
(mock `_call_ollama` / the HTTP layer and assert on the constructed prompt
text); verify truncation behavior at the char cap; verify empty
`match_blocks` short-circuits without a model call.

## Task 3 — Rewrite `get_team_trajectory` orchestration (tier 2 + return type)

**Files:** `data/ingest/trajectory.py`, `tests/data/ingest/test_trajectory.py`

**Breaking interface change:** `get_team_trajectory` currently returns
`list[FormAnalysis]` (one per match). New signature returns a single
`FormAnalysis` (the trajectory-level read):

```python
def get_team_trajectory(
    team: str,
    team_matches: pd.DataFrame,   # now also needs home_score/away_score columns
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> FormAnalysis:
```

`team_matches` already carries `home_score`/`away_score` at the one call site
(`cli/wc2026.py`'s `completed` DataFrame, built from
`schedule[schedule["is_completed"]].dropna(subset=["home_score",
"away_score"])`), so no upstream change needed to supply them — just update
the docstring's documented column requirements.

Logic:
1. Sort `team_matches` chronologically; compute the tier-2 cache key from
   `team` + sorted ISO match dates.
2. Tier-2 cache hit → return the cached `FormAnalysis` directly. No Guardian
   calls, no LLM call.
3. Tier-2 miss → for each match (chronological order), fetch its texts via
   Task 1's tier-1 cache/fetch, build its labeled block (`Match i of N — vs
   {opponent} ({date}), {team} {won/lost/drew} {score}`), collect all blocks
   + urls, call `analyse_team_trajectory` once, cache the result under the
   tier-2 key, return it.
4. Empty `team_matches` → return `FormAnalysis.neutral(team)` without any
   fetch/LLM call (matches current no-completed-matches short-circuit in
   `cli/wc2026.py`, which can now move into this function instead of living
   in the caller).

Tests: tier-2 hit skips both tier-1 fetch and the LLM call entirely (assert
`analyse_team_trajectory` not called); tier-2 miss with all matches
tier-1-cached calls `analyse_team_trajectory` once with N labeled blocks;
adding one new match to an existing 2-match set re-fetches only the new
match's articles (tier-1 cache hit for the other 2) but still produces a
fresh tier-2 combined call; malformed tier-2 cache entry degrades to
recompute.

## Task 4 — Simplify `compute_trajectory_factor`

**Files:** `features/llm_form_feature.py`, `tests/features/test_llm_form_feature.py`

Since aggregation across matches now happens inside the LLM call, not in this
function, `compute_trajectory_factor` no longer takes a list — it becomes a
single-analysis clamp, same shape as `compute_sentiment_factor` but with
`K_TRAJECTORY`/`TRAJECTORY_MIN_CONFIDENCE`/the tighter
`[_TRAJECTORY_FACTOR_MIN, _TRAJECTORY_FACTOR_MAX]` range:

```python
def compute_trajectory_factor(
    analysis: FormAnalysis,
    k: float = K_TRAJECTORY,
    min_confidence: float = TRAJECTORY_MIN_CONFIDENCE,
) -> float:
    if analysis.confidence < min_confidence:
        return 1.0
    factor = 1.0 + k * analysis.form_score
    return max(_TRAJECTORY_FACTOR_MIN, min(_TRAJECTORY_FACTOR_MAX, factor))
```

Delete the old weighted-average logic and its multi-entry tests; replace with
single-analysis equivalents (confidence-below-threshold → 1.0, clamp range,
straightforward factor computation).

## Task 5 — Update `cli/wc2026.py` call site + logging

**Files:** `cli/wc2026.py`

Update the trajectory block (currently lines ~857-903, including the
per-team logging added earlier this session) for the new single-`FormAnalysis`
return shape:

- `analyses = get_team_trajectory(team, team_matches)` →
  `analysis = get_team_trajectory(team, team_matches)`, drop the
  `team_matches.empty` special case (now handled inside `get_team_trajectory`
  itself per Task 3 point 4).
- `compute_trajectory_factor(analyses)` → `compute_trajectory_factor(analysis)`.
- Per-team log line changes from `"{n_usable}/{len(analyses)} match(es)
  usable"` (list-based) to something reflecting the single combined read —
  e.g. `f"trajectory across {len(team_matches)} match(es)  conf
  {analysis.confidence:.2f}  λ×{factor:.3f}"` — and, mirroring how the
  sentiment section surfaces `performance_context` as a second `form_notes`
  line today, add the trajectory's own `performance_context` as a follow-up
  line so the actual narrative (not just the number) is visible in the "Form"
  section of the output.
- Update `data/ingest/trajectory.py`'s per-match log lines added earlier this
  session (`cache hit — form X conf Y` / `fetching...` / `N article(s) ->
  form X conf Y`) — these described the old per-match LLM result, which no
  longer exists per-match. Replace with tier-1-appropriate logging (article
  counts fetched/cached per match) plus one new tier-2 log line after the
  combined call: `f"[trajectory] {team}: combined analysis across {N}
  match(es) -> form {form_score:+.2f} conf {confidence:.2f}"`.

## Task 6 — Verify end-to-end

Run the full test suite; lint; manually re-run `wc2026 --next` (Spain vs
Austria, same match used earlier this session) and confirm the trajectory
log output now shows one combined narrative per team instead of a per-match
list, and that the tier-2 cache actually short-circuits on a second run with
no new completed matches (no Guardian/Ollama calls, visible via absence of
the tier-1/tier-2 fetch log lines).

## Out of scope

- Any change to the pre-match sentiment path (`fetch_team_news`,
  `analyse_team_form`, `_SYSTEM_PROMPT`/`_USER_TEMPLATE`) — untouched.
- A cap on match count per prompt (explicitly decided against this session).
- A new trend/direction field on `FormAnalysis` (explicitly decided against
  this session).
- Retuning `K_TRAJECTORY` / `TRAJECTORY_MIN_CONFIDENCE` / the factor clamp
  range — unchanged from the current shipped values.
