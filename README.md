# World Cup 2026 Prediction Model

A scientific hobby project that models football match outcomes as a joint score distribution and derives coherent probabilities for all betting markets (1X2, Over/Under, BTTS, Correct Score, Asian Handicap) from a single forward pass. Built for the 2026 FIFA World Cup.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Repository Layout](#repository-layout)
3. [Data Sources and Ingestion](#data-sources-and-ingestion)
4. [Feature Engineering](#feature-engineering)
5. [Neural Model — ScoreGridNet](#neural-model--scoregridnet)
6. [Score Grid and Market Derivation](#score-grid-and-market-derivation)
7. [Post-Hoc Adjustments](#post-hoc-adjustments)
8. [Baseline Models](#baseline-models)
9. [Ensemble](#ensemble)
10. [Training](#training)
11. [Evaluation and Backtesting](#evaluation-and-backtesting)
12. [CLI Commands](#cli-commands)
13. [Docker Scheduler and Telegram Bot](#docker-scheduler-and-telegram-bot)
14. [Environment Variables](#environment-variables)
15. [Setup](#setup)

---

## Design Philosophy

The central idea is to **predict one thing — the full joint scoreline distribution** — and derive every market from it deterministically.

The neural network outputs three parameters: `(λ_home, λ_away, ρ)`. From these, an 8×8 probability matrix `P(home_goals = i, away_goals = j)` is constructed analytically using independent Poisson distributions with a Dixon-Coles low-score correction. Every market probability is then a closed-form sum over that grid:

```
Home win   = Σ grid[i,j]  where i > j
Draw       = Σ grid[i,j]  where i = j
Over 2.5   = Σ grid[i,j]  where i + j >= 3
BTTS       = Σ grid[i,j]  where i >= 1 and j >= 1
```

This guarantees that all market probabilities are internally coherent — they are derived from the same underlying probability mass, not fitted independently.

The honest baseline for any prediction is the **bookmaker closing line** with the margin removed. Beating the closing line is rare; the project's goal is a well-calibrated model that occasionally finds value, not a systematic edge.

---

## Repository Layout

```
worldcup_prediction/
  data/
    ingest/
      results.py          # Kaggle international football results CSV loader
      wc2026.py           # WC 2026 live schedule (API-Football → openfootball → Wikipedia)
      injuries.py         # Injury/suspension data (API-Football → Transfermarkt)
      odds.py             # Historical closing odds
      odds_live.py        # Live odds from the-odds-api.com
      squads.py           # Tournament squad rosters (Wikipedia)
      goalscorers.py      # Career international goal tallies
      player_stats.py     # Club-level stats via soccerdata
      api_football.py     # API-Football client
      openfootball.py     # Openfootball.github.io client
      llm_form.py         # LLM form signal extraction from news articles
      cache.py            # Parquet-based local cache

  features/
    elo.py                # Self-computed Elo ratings with inactivity decay
    form.py               # Recent-match sequence encoder (GRU input format)
    context.py            # Match context features (neutral, rest days, tournament stage)
    squad.py              # Squad quality features
    squad_registry.py     # Tournament squad registry (top5_share, caps, goals)
    injury.py             # Injury-adjusted lambda computation
    llm_form_feature.py   # LLM sentiment → lambda multiplier

  models/
    neural.py             # ScoreGridNet + NeuralModel wrapper (main model)
    dixon_coles.py        # Classical Dixon-Coles baseline
    elo_model.py          # Plain Elo baseline
    xgb_model.py          # XGBoost / LightGBM baseline
    ensemble.py           # Weighted grid-average ensemble
    grid.py               # Score grid construction and market derivation

  eval/
    metrics.py            # RPS, NLL, Brier, margin removal
    backtest.py           # Walk-forward backtest harness, Kelly staking
    calibration.py        # Calibration / reliability diagrams

  cli/
    train.py              # Train the neural model
    predict.py            # Single-match prediction CLI
    wc2026.py             # WC 2026 match batch prediction + live odds comparison
    backtest.py           # Walk-forward backtest CLI
    export.py             # CSV export

  docker/
    Dockerfile            # CPU-only Python 3.13 image
    docker-compose.yml    # Scheduler service definition
    scheduler.py          # APScheduler-based pre-match Telegram bot

  checkpoints/
    neural.pt             # Saved NeuralModel checkpoint

  pyproject.toml
```

---

## Data Sources and Ingestion

### Training spine — match results

**Source:** Kaggle "International football results from 1872 to present" (Mart Jürisoo), loaded via `data/ingest/results.py`. Approximately 48 000 international matches with date, home/away team, score, tournament name, and neutral-venue flag.

WC 2026 matches that have leaked into the Kaggle dataset are explicitly excluded from training (`drop_wc2026` in `results.py`). The live schedule is the single authoritative source for WC 2026 results used at inference time.

### Live WC 2026 schedule

`data/ingest/wc2026.py` resolves the schedule through three sources in priority order:

1. **API-Football** (`/fixtures?league=1&season=2026`) — structured JSON with real kickoff times. Requires `API_FOOTBALL_KEY`.
2. **openfootball/worldcup** — free GitHub-hosted JSON feed, no authentication.
3. **Wikipedia scrape** — parses the 2026 FIFA World Cup page using three strategies (1×3 match tables, named columns, positional fallback). Robust against page layout changes.

Completed WC 2026 matches are injected into the training context at inference time so that form sequences and Elo ratings reflect the tournament as it unfolds.

### Injury and suspension data

`data/ingest/injuries.py` fetches absent players per team for upcoming fixtures:

1. **API-Football** `/injuries?league=1&season=2026` — returns confirmed injuries and suspensions; filters to future fixtures only.
2. **Transfermarkt** HTML scrape — fallback when no API key is available. Covers all 48 WC 2026 nations via hardcoded slug/ID mapping. Politely rate-limited (3 s between requests).

Player names are fuzzy-matched to the squad registry using NFC Unicode normalisation + difflib (cutoff 0.80) with a substring fallback, bridging API-Football name variants against squad data.

### Live odds

`data/ingest/odds_live.py` fetches upcoming match odds from `the-odds-api.com`. Results are cached for 15 minutes to avoid burning API credits on repeated runs.

### Squad rosters

`data/ingest/squads.py` loads tournament squad lists (WC 2014/2018/2022/2026, Euro 2016/2020/2024) from Wikipedia, providing player names, clubs, and caps counts. The `SquadRegistry` in `features/squad_registry.py` aggregates these into per-team features used as context inputs.

### LLM form signal

`data/ingest/llm_form.py` fetches news articles from BBC, Guardian, ESPN, and France24 RSS feeds and sends the text to a locally-running Ollama model (`qwen3.5:9b` by default) for structured extraction. No cache — always fetches fresh. See [Post-Hoc Adjustments](#post-hoc-adjustments) for how the signal is applied.

---

## Feature Engineering

All features are computed strictly as of the match date — no future leakage.

### Elo ratings (`features/elo.py`)

A self-computed Elo system trained from scratch on the full match history. Key design choices:

- **K-factor by competition** — World Cup K=60, continental championships K=50, qualifiers K=40, friendlies K=20, default K=35.
- **Home advantage** — +100 Elo points added to the home team's effective rating. Applied only for non-neutral venues.
- **Inactivity decay** — after 365 days without a match, the rating is pulled toward 1500 proportionally to years of absence. Prevents long-absent teams from carrying stale ratings.

The Elo pass runs chronologically over the full dataset. `elo_home_pre` and `elo_away_pre` (ratings before the match) are stored per row and used as features. `elo_diff = elo_home_pre - elo_away_pre` is passed as a normalised scalar to the neural network.

### Recent form sequence (`features/form.py`)

The neural model's GRU encoder receives a sequence of the team's last 10 matches before the prediction date. Each match in the sequence is a 5-dimensional vector:

```
[goals_for, goals_against, points (3/1/0), opponent_elo / 1500.0, is_home (0/1)]
```

Sequences are zero-padded at the front when fewer than 10 prior matches exist. The same shared GRU encoder processes home and away sequences independently.

### Match context (`features/context.py`)

A 13-dimensional vector fed to a small MLP:

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0 | Neutral venue flag | float (0/1) |
| 1 | Rest days (home) | / 180 |
| 2 | Rest days (away) | / 180 |
| 3 | Elo difference | / 400 |
| 4 | Is knockout tournament | float (0/1) |
| 5 | Is WC 2026 host (home team) | float (0/1) |
| 6 | Is WC 2026 host (away team) | float (0/1) |
| 7 | Top-5 league share (home) | fraction [0, 1] |
| 8 | Top-5 league share (away) | fraction [0, 1] |
| 9 | Average caps / 100 (home) | normalised |
| 10 | Average caps / 100 (away) | normalised |
| 11 | Career intl goals per cap (home) | raw ratio |
| 12 | Career intl goals per cap (away) | raw ratio |

Friendly matches receive a sample weight of 0.20 (vs 1.0 for competitive matches) during training.

### Squad quality features (`features/squad_registry.py`)

For each team in a known tournament, three features are computed from the squad roster:

- **`top5_share`** — fraction of the 26-man squad whose club plays in the Premier League, La Liga, Bundesliga, Serie A, or Ligue 1.
- **`avg_caps_norm`** — mean player caps / 100. Proxies for international experience.
- **`intl_goals_per_cap`** — total squad career international goals divided by total squad caps. Normalised by caps so a 26-player squad with a 60-goal striker doesn't dominate over a balanced squad.

---

## Neural Model — ScoreGridNet

**File:** `models/neural.py`

### Architecture

The model has three parallel branches that are concatenated and passed through a fusion MLP:

```
home_team_id  ──► Embedding(n_teams, 32) ──────────────────────────────────────────────┐
away_team_id  ──► Embedding(n_teams, 32) ──────────────────────────────────────────────┤
                                                                                         │
home_seq      ──► GRU(input=5, hidden=32, layers=1) ──► final hidden state (32) ────────┤
away_seq      ──► GRU(input=5, hidden=32, layers=1) ──► final hidden state (32) ────────┤
                                                                                         │
context (13)  ──► Linear(13→32) → ReLU → Linear(32→16) ────────────────────────────────┤
                                                                                         │
                                         concat (32+32+32+32+16 = 144) ─────────────────┘
                                         Linear(144→64) → ReLU → Dropout(0.3) → Linear(64→3)
                                                                                         │
                                    softplus(out[0]) = λ_home  (positive Poisson rate)
                                    softplus(out[1]) = λ_away  (positive Poisson rate)
                                    tanh(out[2]) * 0.2 = ρ     (low-score correlation)
```

**Team embedding branch:** Index 0 is reserved as an UNK slot for teams absent from the training vocabulary. Each known team gets a 32-dimensional learned vector — a neural generalisation of the attack/defence strength parameters in Dixon-Coles.

**GRU form branch:** A single-layer GRU (hidden size 32) processes each team's last-10-match sequence chronologically. The final hidden state is used as the form context vector. The same GRU weights are shared between home and away — it encodes "what kind of recent form does a team have", position-independently.

**Context MLP:** Two fully-connected layers (13 → 32 → 16) with ReLU activation encode the scalar match context features.

**Fusion MLP:** A 144-dimensional concatenation of all branches passes through `Linear(144→64) → ReLU → Dropout(0.3) → Linear(64→3)`. The three output scalars are transformed into distribution parameters via softplus (for positivity) and scaled tanh (for bounded ρ).

### Loss function

The model is trained with **scoreline negative log-likelihood**:

```
loss = -log P(home_goals = i, away_goals = j | λ_home, λ_away, ρ)
```

where `P(i, j)` is the Dixon-Coles corrected Poisson probability at the actual scoreline. This is a proper scoring rule that trains all market probabilities jointly — the model learns to assign high probability mass to the correct scoreline, which implicitly calibrates 1X2, Over/Under, and BTTS simultaneously.

The grid is computed entirely in PyTorch so gradients flow through `λ_home`, `λ_away`, and `ρ`:

```python
pmf_h = exp(-λ_h) * λ_h^k / k!   # (batch, 8)
pmf_a = exp(-λ_a) * λ_a^k / k!   # (batch, 8)
grid  = pmf_h ⊗ pmf_a             # outer product (batch, 8, 8)

# Dixon-Coles τ corrections (applied in-place with grad preserved)
grid[:, 0, 0] *= (1 - λ_h * λ_a * ρ)
grid[:, 1, 0] *= (1 + λ_a * ρ)
grid[:, 0, 1] *= (1 + λ_h * ρ)
grid[:, 1, 1] *= (1 - ρ)

grid /= grid.sum(dim=(1,2))   # renormalise truncation + correction
p     = grid[batch, actual_home, actual_away]
loss  = -log(p).mean()
```

### Training procedure

**Optimiser:** Adam with `lr=3e-4`, `weight_decay=1e-4`.

**Batch size:** 256, randomly shuffled each epoch.

**Validation:** Strictly chronological split. By default the last 6 months of the training corpus are held out. Form sequences in the validation set are always looked up from the full corpus (no leakage — only matches *before* each match date contribute).

**Early stopping:** Patience of 15 epochs on validation NLL. The best checkpoint is restored at the end.

**Form sequence precomputation:** All training and validation form sequences are precomputed once before the epoch loop and kept on the GPU, avoiding repeated `get_form_sequence` calls per batch.

**Production refit (`--no-holdout`):** After the holdout run reports validation RPS and selects the best epoch, a second model is trained on the entire corpus for exactly that many epochs with early stopping disabled. This is the checkpoint used at inference time.

**WC 2026 exclusion:** All WC 2026 matches are stripped from the training corpus before fitting. At inference time, completed WC 2026 matches are injected into the context DataFrame used only for form sequence lookups and Elo forward-passes — the model weights are never updated from them.

### Checkpoint format

Saved via `torch.save` as a dict:

```python
{
    "model_state": net.state_dict(),
    "team_vocab":  {team_name: int, ...},   # 1-indexed; 0 = UNK
    "n_teams":     int,
    "hparams":     {n_epochs, lr, weight_decay, batch_size, patience},
}
```

Load with `NeuralModel.load("checkpoints/neural.pt")`.

---

## Score Grid and Market Derivation

**File:** `models/grid.py`

`build_grid(λ_home, λ_away, ρ)` constructs an 8×8 probability matrix (goals 0–7 each axis):

1. Compute independent Poisson PMFs for each team.
2. Form the outer product (independent joint distribution).
3. Apply the Dixon-Coles τ correction to the four low-score cells (0-0, 1-0, 0-1, 1-1). These cells are systematically mispriced by pure independent Poisson.
4. Renormalise the grid to sum to 1.0 (accounts for truncation at 7 goals and the τ corrections).

`derive_markets(grid)` reads off all market probabilities as deterministic sums:

| Market | Grid condition |
|--------|---------------|
| Home win | i > j |
| Draw | i = j |
| Away win | i < j |
| Over/Under N.5 | i + j >= N+1 |
| BTTS yes | i >= 1 and j >= 1 |
| Asian handicap ±0.5/±1.5 | half-goal lines, no push |
| Asian handicap ±1 | full-goal lines, push on exact margin |

---

## Post-Hoc Adjustments

The neural model outputs `(λ_home, λ_away, ρ)` from historical patterns. Two exogenous signals are applied on top at inference time.

### Injury adjustment (`features/injury.py`)

The model has no historical injury labels, so injury effects are applied as a multiplicative scale:

```
λ_home_adj = λ_home * (1 - K * inj_loss_home)
λ_away_adj = λ_away * (1 - K * inj_loss_away)
```

`inj_loss` is the caps-weighted fraction of squad strength absent:

```
inj_loss = Σ caps_i (absent players) / Σ caps_j (full squad)
```

`K = 0.5` by default. The market typically prices a top-forward absence as roughly half their xG contribution (coaches compensate tactically), not the full amount.

Player name matching uses NFC Unicode normalisation + difflib fuzzy matching (cutoff 0.80) with a substring fallback, bridging API-Football name variants against squad registry names.

`ρ` is passed through unchanged — the Dixon-Coles correlation is a match-context property, not a goal-rate property.

### LLM narrative form adjustment (`features/llm_form_feature.py`)

News articles about each team are fetched from six RSS feeds in priority order:

```
BBC WC RSS  →  Guardian WC RSS  →  ESPN Soccer RSS  →  France24 Sport RSS
→  BBC Sport RSS  →  Guardian Football RSS
```

Articles are filtered by team name keyword match. BBC, Guardian, and ESPN articles get full-body HTML extraction (`<p>` tag parsing with boilerplate filtering). France24 uses RSS descriptions. Up to 6 articles per team are fetched per run.

The combined text (up to 8000 chars) is sent to a locally-running Ollama model with a structured extraction prompt. The LLM returns:

```json
{
  "form_score":          float [-1, 1],
  "performance_context": "string",
  "key_absences":        ["player names confirmed OUT"],
  "morale_signals":      ["direct phrases from text"],
  "tactical_notes":      "string",
  "confidence":          float [0, 1]
}
```

The sentiment factor is applied after the injury adjustment:

```
adj_factor = 1.0 + 0.30 * form_score * confidence
λ_adj      = λ_base * adj_factor        # clamped to [0.70, 1.30]
```

Analyses with `confidence < 0.25` are silently ignored (factor = 1.0). This means thin reads — articles that barely mention the team — have no effect, while richly-sourced reads with clear form signals can shift λ by up to ±30%.

Application order: injury adjustment first (mechanistic, well-trusted), then sentiment.

Pre-match reads are logged to `data/cache/llm_form_log.jsonl` for retrospective review.

---

## Baseline Models

Three baselines are implemented with the same `predict_batch(matches) -> DataFrame` interface as `NeuralModel`, so they can be dropped into the backtest harness or ensemble directly.

### Elo baseline (`models/elo_model.py`)

Converts the pre-match Elo ratings into 1X2 win probabilities via the Elo expected-score formula. Does not produce a score grid or over/under probabilities.

### Dixon-Coles baseline (`models/dixon_coles.py`)

Fits per-team attack (`α`) and defence (`β`) strength parameters plus a global home-advantage (`γ`) and low-score correction (`ρ`) by maximising the weighted log-likelihood of historical scorelines. Time-decay weight: `exp(-ξ * days_ago)` with `ξ = 0.0065`. Optimised with L-BFGS-B via scipy.

`α_h * β_a * γ = λ_home` (non-neutral)  
`α_a * β_h = λ_away`

The same `build_grid` / `derive_markets` pipeline produces a full score grid from these parameters.

### XGBoost baseline (`models/xgb_model.py`)

Tabular model on the same features as the neural network (Elo diff, form aggregates, rest days, squad context). Fits separate regressors for `λ_home` and `λ_away`, then passes them through `build_grid`. Gradient boosting often outperforms neural networks on tabular football data with limited observations — if the neural net can't beat XGBoost, that is a finding.

---

## Ensemble

**File:** `models/ensemble.py`

`EnsembleModel` accepts any list of fitted models and averages their predicted score grids:

```
avg_grid = Σ weight_i * grid_i
```

Markets are re-derived from the averaged grid, not averaged independently. This preserves internal coherence.

Weights can be uniform or inverse-RPS (`from_rps_scores` constructor): models with lower (better) RPS receive proportionally higher weight.

---

## Training

```bash
# Holdout evaluation only (saves best-epoch checkpoint)
train --csv-path data/raw/results.csv

# Production refit: evaluate holdout, then retrain on all data
train --csv-path data/raw/results.csv --no-holdout

# Tune epochs and learning rate
train --csv-path data/raw/results.csv --n-epochs 200 --lr 1e-4

# Retrain checkpoint location
train --csv-path data/raw/results.csv --checkpoint checkpoints/neural.pt
```

The train command:
1. Loads results, excludes WC 2026 matches.
2. Computes Elo ratings and squad registry.
3. Derives match context features.
4. Runs a chronological holdout split (last 6 months held out by default).
5. Trains `ScoreGridNet` with early stopping on validation NLL.
6. Reports validation RPS.
7. With `--no-holdout`: refits on the full corpus for the best epoch count and saves the production checkpoint.

---

## Evaluation and Backtesting

### Metrics (`eval/metrics.py`)

- **RPS (Ranked Probability Score)** — standard metric for ordered 3-outcome markets. Computed over the cumulative probability distribution; penalises confident wrong predictions more than uncertain ones. Lower is better.
- **NLL (Negative Log-Likelihood)** — scoreline NLL from the full 8×8 grid. The direct analogue of the training loss, computed on held-out data.
- **Brier score** — binary metric per market outcome.
- **Margin removal** — bookmaker implied probabilities are normalised by dividing by the overround: `p_i = (1/odds_i) / Σ(1/odds_j)`. This is the scientific yardstick for edge calculations.

### Walk-forward backtest (`eval/backtest.py`)

Strictly chronological folds:
- Minimum 36 months of training data per fold.
- Validation windows of 6 months each.
- Step size: 6 months.

For each fold: fit the model on train data, predict on the validation window, compute RPS and NLL, then simulate value betting against the closing line.

**Value betting:** a bet is flagged when `model_prob - market_implied_prob >= 0.02` (configurable). Staking is fractional Kelly (¼ Kelly): `stake = 0.25 * edge / decimal_odds`.

The backtest reports flat-bet ROI and Kelly ROI separately, so Kelly variance doesn't obscure the underlying model signal.

---

## CLI Commands

All commands are installed as entry points via `pyproject.toml`.

### `wc2026` — main prediction command

```bash
wc2026 [OPTIONS]
```

Full workflow for WC 2026 upcoming matches:

1. Load results; compute Elo; build squad registry; derive context.
2. Fetch live WC 2026 schedule; inject completed matches as context.
3. Fit the selected model (or load a checkpoint).
4. Filter to upcoming matches (optionally `--next` for next matchday only).
5. Apply injury adjustment (fetches live injury data).
6. Apply LLM narrative form adjustment (fetches news, runs Ollama).
7. Fetch live odds from the-odds-api.com.
8. Display match table with model probabilities, market probabilities, edge, and Kelly stake.

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `neural` | `dc`, `xgb`, `neural`, or `ensemble` |
| `--checkpoint` | `checkpoints/neural.pt` | Neural model checkpoint |
| `--next` | off | Only predict the next matchday |
| `--show-all` | off | Show all matches, not just value bets |
| `--no-injuries` | off | Skip injury adjustment |
| `--no-llm-form` | off | Skip LLM narrative form |
| `--llm-model` | `qwen3.5:9b` | Ollama model name |
| `--min-edge` | `0.02` | Minimum edge to flag as value bet |
| `--force-odds` | off | Bypass the 15-minute odds cache |
| `--telegram` | off | Output HTML for Telegram (used by scheduler) |

### `train` — train the neural model

```bash
train --csv-path data/raw/results.csv --no-holdout
```

### `predict` — single-match prediction

```bash
predict --home Argentina --away France --neutral
```

### `backtest` — walk-forward backtest

```bash
backtest --model neural --csv-path data/raw/results.csv
```

### `export` — export probabilities to CSV

```bash
export --output predictions.csv
```

---

## Docker Scheduler and Telegram Bot

The scheduler runs inside a Docker container and fires the `wc2026` prediction command automatically before each match, sending the result to a Telegram chat.

### How it works

`docker/scheduler.py` uses APScheduler to manage a job queue:

1. **On startup:** loads the WC 2026 schedule, registers a `DateTrigger` job per upcoming match, firing `ADVANCE_MINUTES` (default 60) before kickoff.
2. **Daily at 02:00 UTC:** refreshes the schedule to pick up any FIFA rescheduling.
3. **At each firing:** runs `wc2026 --next --show-all --telegram` as a subprocess, captures the stdout, and sends it to Telegram via the Bot API, splitting at the 4096-character limit.

The neural model checkpoint and data cache are mounted from the host as Docker volumes. GPU inference stays on the host; the container runs CPU-only PyTorch.

### Build and run

```bash
# Copy .env with required tokens (see Environment Variables)
cp .env.example .env

# Build and start
cd docker
docker compose up -d

# View logs
docker compose logs -f scheduler
```

### Dockerfile summary

- Base: `python:3.13-slim`
- CPU-only PyTorch installed first from the PyTorch CPU wheel index.
- APScheduler and requests added for scheduler-specific dependencies.
- Project installed as an editable package (`pip install -e .`).
- Entrypoint: `python docker/scheduler.py`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes (scheduler) | Telegram Bot API token |
| `TELEGRAM_CHAT_ID` | Yes (scheduler) | Target chat or channel ID |
| `THE_ODDS_API_KEY` | Optional | the-odds-api.com key; enables live odds display |
| `API_FOOTBALL_KEY` | Optional | API-Football key; enables injury data and schedule |
| `OLLAMA_HOST` | Optional | Ollama base URL (default: `http://localhost:11434`) |
| `ADVANCE_MINUTES` | Optional | Minutes before kickoff to fire prediction (default: 60) |

---

## Setup

### Prerequisites

- Python 3.11+
- PyTorch 2.0+ (CUDA optional; the model is small and trains in minutes on CPU)
- Ollama with `qwen3.5:9b` model pulled (for LLM form adjustment)
- Kaggle results CSV (`data/raw/results.csv`)

### Install

```bash
pip install -e .
```

### First run

```bash
# Train the neural model
train --csv-path data/raw/results.csv --no-holdout

# Run predictions for upcoming WC 2026 matches
wc2026 --next

# Run with all adjustments (requires Ollama running)
wc2026 --next --show-all
```

### Ollama setup (for LLM form)

```bash
ollama pull qwen3.5:9b
ollama serve
```

The LLM form step adds approximately 2 minutes per matchday batch. Disable with `--no-llm-form` for faster runs.
